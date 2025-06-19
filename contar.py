# contar.py
import os
import cv2
import json
import numpy as np
from ultralytics import YOLO
import datetime # Para manipulação de tempo para o relatório detalhado
import math     # Para usar math.floor para arredondar horas

def contar_veiculos(video_path, areas_path, model_path, show_video=True):
    """
    Função principal para contar veículos em um vídeo usando YOLOv5 e áreas predefinidas.
    Gera um relatório detalhado por hora e tipo de veículo.

    Args:
        video_path (str): Caminho para o arquivo de vídeo de entrada ou URL do stream.
        areas_path (str): Caminho para o arquivo JSON contendo as coordenadas das áreas de entrada e saída.
        model_path (str): Caminho para o arquivo do modelo YOLO (.pt).
        show_video (bool): Se True, exibe o vídeo em tempo real durante o processamento.

    Returns:
        tuple: Uma tupla contendo o caminho para o arquivo de relatório e o caminho para o vídeo de saída.
    """
    pasta_saida = "resultados"
    os.makedirs(pasta_saida, exist_ok=True)
    caminho_saida_video = os.path.join(pasta_saida, "saida_com_contagem.mp4")
    caminho_relatorio = os.path.join(pasta_saida, "relatorio.txt")

    # Carregar áreas do arquivo JSON
    try:
        with open(areas_path, "r") as f:
            areas = json.load(f)
        # Verifica se as áreas são válidas (duas listas, cada uma com pelo menos 3 pontos)
        if not (isinstance(areas, list) and len(areas) == 2 and
                all(isinstance(a, list) and len(a) >= 3 for a in areas)):
            raise ValueError("O arquivo de áreas está em um formato inválido ou incompleto.")
            
        area_entrada = np.array(areas[0], dtype=np.int32)
        area_saida = np.array(areas[1], dtype=np.int32)
    except FileNotFoundError:
        raise FileNotFoundError(f"Arquivo de áreas não encontrado: {areas_path}. Por favor, defina as áreas primeiro.")
    except json.JSONDecodeError:
        raise ValueError(f"Erro ao decodificar JSON do arquivo de áreas: {areas_path}. Verifique a sintaxe.")
    except ValueError as ve:
        raise ValueError(f"Erro de dados no arquivo de áreas: {ve}")
    except Exception as e:
        raise ValueError(f"Erro inesperado ao carregar áreas do arquivo {areas_path}: {e}")

    # Carrega o modelo YOLO usando o caminho fornecido pelo App.py
    try:
        modelo = YOLO(model_path)
    except Exception as e:
        raise RuntimeError(f"Erro ao carregar o modelo YOLO '{model_path}'. Verifique o caminho ou se o arquivo está corrompido. Erro: {e}")

    # Mapeamento de IDs de classe YOLO para nomes de veículos em português
    nome_classe = {2: "Carro", 3: "Moto", 5: "Onibus", 7: "Caminhao"}
    classes_interesse = list(nome_classe.keys())

    # Estruturas para contagem total (IDs únicos que passaram pela área)
    cont_entrada_total = {nome: 0 for nome in nome_classe.values()}
    cont_saida_total = {nome: 0 for nome in nome_classe.values()}
    ids_unicos_entrada = set()
    ids_unicos_saida = set()

    # Estruturas para rastreamento de estado de veículos e coleta de eventos para fluxo horário
    # vehicle_tracking_states: {'track_id': {'type': str, 'was_in_entry_zone': bool, 'was_in_exit_zone': bool}}
    vehicle_tracking_states = {}
    hourly_entry_events = [] # [{'time_seconds': float, 'type': str}]
    hourly_exit_events = []  # [{'time_seconds': float, 'type': str}]

    # Tenta abrir o vídeo (pode ser um arquivo local ou uma URL de stream)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Não foi possível abrir o vídeo em {video_path}. Verifique o caminho/URL ou se o vídeo está corrompido/inacessível.")

    fps = int(cap.get(cv2.CAP_PROP_FPS))
    # Define o tamanho de saída do vídeo para consistência
    frame_width_out, frame_height_out = 1280, 720

    # Inicializa o objeto VideoWriter para salvar o vídeo de saída
    video_out = cv2.VideoWriter(caminho_saida_video, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width_out, frame_height_out))
    if not video_out.isOpened():
        raise IOError(f"Não foi possível criar o arquivo de vídeo de saída em {caminho_saida_video}. Verifique permissões ou codecs de vídeo.")

    # Loop principal para processar cada frame do vídeo
    while cap.isOpened():
        ret, frame = cap.read() # Lê um frame do vídeo
        if not ret: # Se não conseguir ler o frame (fim do vídeo ou erro), sai do loop
            break

        current_frame_timestamp_ms = cap.get(cv2.CAP_PROP_POS_MSEC)
        current_seconds = current_frame_timestamp_ms / 1000.0

        frame = cv2.resize(frame, (frame_width_out, frame_height_out))

        # Desenha as áreas de interesse no frame para visualização
        # Polígonos são preenchidos com uma cor semi-transparente para melhor visualização
        overlay = frame.copy()
        cv2.fillPoly(overlay, [area_entrada], (0, 255, 0)) # Verde para entrada
        cv2.fillPoly(overlay, [area_saida], (0, 0, 255))   # Vermelho para saída
        alpha = 0.3 # Transparência
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
        # Desenha as bordas dos polígonos
        cv2.polylines(frame, [area_entrada], True, (0, 255, 0), 2)
        cv2.polylines(frame, [area_saida], True, (0, 0, 255), 2)

        # Para rastrear quais veículos foram vistos neste frame
        current_frame_track_ids = set()

        # Realiza a detecção e rastreamento de objetos no frame
        resultados = modelo.track(source=frame, tracker="botsort.yaml", persist=True, verbose=False)[0]

        # Verifica se há IDs de rastreamento nos resultados
        if resultados.boxes.id is not None:
            ids = resultados.boxes.id.int().cpu().tolist()
            boxes = resultados.boxes.xyxy.cpu().tolist()
            classes = resultados.boxes.cls.int().cpu().tolist()

            # Itera sobre cada detecção/rastreamento
            for box, cls, track_id in zip(boxes, classes, ids):
                # Se a classe detectada for uma das classes de interesse
                if cls in classes_interesse:
                    nome = nome_classe[cls]
                    x1, y1, x2, y2 = map(int, box)
                    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2 # Centro da caixa delimitadora

                    # Desenha a caixa delimitadora e o ID de rastreamento no frame
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                    cv2.putText(frame, f"{nome} ID: {track_id}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

                    # Adiciona o ID ao conjunto de IDs vistos no frame atual
                    current_frame_track_ids.add(track_id)

                    # Inicializa o estado do veículo se for a primeira vez que é visto
                    if track_id not in vehicle_tracking_states:
                        vehicle_tracking_states[track_id] = {
                            'type': nome,
                            'was_in_entry_zone': False,
                            'was_in_exit_zone': False
                        }
                    
                    # Lógica para detectar e registrar eventos de ENTRADA
                    is_in_entry_now = cv2.pointPolygonTest(area_entrada, (cx, cy), False) >= 0
                    was_in_entry_before = vehicle_tracking_states[track_id]['was_in_entry_zone']

                    if is_in_entry_now and not was_in_entry_before:
                        # O veículo acabou de ENTRAR na área de entrada
                        hourly_entry_events.append({'time_seconds': current_seconds, 'type': nome})
                        ids_unicos_entrada.add(track_id) # Para contagem total de IDs únicos
                        cont_entrada_total[nome] += 1 # Contagem total por tipo
                        vehicle_tracking_states[track_id]['was_in_entry_zone'] = True
                    elif not is_in_entry_now and was_in_entry_before:
                        # O veículo acabou de SAIR da área de entrada (resetar o estado para permitir re-entrada)
                        vehicle_tracking_states[track_id]['was_in_entry_zone'] = False

                    # Lógica para detectar e registrar eventos de SAÍDA
                    is_in_exit_now = cv2.pointPolygonTest(area_saida, (cx, cy), False) >= 0
                    was_in_exit_before = vehicle_tracking_states[track_id]['was_in_exit_zone']

                    if is_in_exit_now and not was_in_exit_before:
                        # O veículo acabou de ENTRAR na área de saída (considerado um evento de SAÍDA do sistema)
                        hourly_exit_events.append({'time_seconds': current_seconds, 'type': nome})
                        ids_unicos_saida.add(track_id) # Para contagem total de IDs únicos
                        cont_saida_total[nome] += 1 # Contagem total por tipo
                        vehicle_tracking_states[track_id]['was_in_exit_zone'] = True
                    elif not is_in_exit_now and was_in_exit_before:
                        # O veículo acabou de SAIR da área de saída (resetar o estado para permitir re-entrada)
                        vehicle_tracking_states[track_id]['was_in_exit_zone'] = False
        
        # Limpar o estado de veículos que não foram mais vistos neste frame
        # Isso é importante para permitir que um veículo que sumiu e reapareceu em uma zona
        # seja contado novamente como uma "nova" entrada/saída, se for o caso.
        keys_to_reset = [tid for tid in vehicle_tracking_states if tid not in current_frame_track_ids]
        for tid in keys_to_reset:
            # Se o veículo não está mais sendo rastreado, reseta seu estado de zona
            # para que, se ele aparecer novamente na zona, conte como uma nova entrada/saída
            # (isso é importante para fluxo horário, não para IDs únicos)
            vehicle_tracking_states[tid]['was_in_entry_zone'] = False
            vehicle_tracking_states[tid]['was_in_exit_zone'] = False

        # Adiciona a contagem atualizada (TOTAL) no canto superior esquerdo do frame
        y_offset_count = 30
        cv2.putText(frame, "Contagem TOTAL:", (10, y_offset_count), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        y_offset_count += 30
        for nome, qtd in cont_entrada_total.items():
            cv2.putText(frame, f"Entrada {nome}: {qtd}", (10, y_offset_count), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            y_offset_count += 25
        y_offset_count += 10
        for nome, qtd in cont_saida_total.items():
            cv2.putText(frame, f"Saida {nome}: {qtd}", (10, y_offset_count), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            y_offset_count += 25

        video_out.write(frame)

        # Exibe o frame em tempo real se 'show_video' for True
        if show_video:
            cv2.imshow("Processando Video - Pressione 'q' para fechar", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'): # Permite fechar a janela com a tecla 'q'
                break

    # Libera os recursos do OpenCV
    cap.release()
    video_out.release()
    if show_video:
        cv2.destroyAllWindows() # Fecha as janelas do OpenCV

    # --- Geração do Relatório Detalhado ---
    with open(caminho_relatorio, "w") as f:
        f.write("RELATÓRIO DE CONTAGEM DE VEÍCULOS\n")
        f.write("="*40 + "\n\n")

        # Se não houver eventos, indica que o vídeo era muito curto ou não havia veículos
        if not hourly_entry_events and not hourly_exit_events:
            f.write("Nenhum veículo detectado ou áreas não cruzadas.\n\n")
        else:
            # TOTAIS GERAIS
            f.write("TOTAIS GERAIS (ID ÚNICO POR ÁREA):\n")
            f.write("-" * 30 + "\n")
            f.write("ENTRADAS:\n")
            for nome, qtd in cont_entrada_total.items():
                f.write(f"  {nome}: {qtd}\n")
            f.write(f"Total IDs únicos entrada: {len(ids_unicos_entrada)}\n\n")

            f.write("SAÍDAS:\n")
            for nome, qtd in cont_saida_total.items():
                f.write(f"  {nome}: {qtd}\n")
            f.write(f"Total IDs únicos saída: {len(ids_unicos_saida)}\n\n")

            # FLUXO HORÁRIO (ENTRADAS)
            f.write("FLUXO HORÁRIO (ENTRADAS):\n")
            f.write("-" * 30 + "\n")
            if hourly_entry_events:
                # Ordena os eventos por tempo
                hourly_entry_events.sort(key=lambda x: x['time_seconds'])
                min_time = hourly_entry_events[0]['time_seconds']
                max_time = hourly_entry_events[-1]['time_seconds']
                
                # Define o início do vídeo como 00:00:00 para o relatório
                start_hour = math.floor(min_time / 3600) if min_time > 0 else 0
                end_hour_report = math.ceil(max_time / 3600) if max_time > 0 else 1 # Para garantir que inclui o último pedaço de hora

                # Cria os buckets de hora
                hourly_entry_counts = {} # {hour_segment: {type: count, '_total': count}}
                for hour_offset in range(start_hour, end_hour_report):
                    hourly_entry_counts[hour_offset] = {nome: 0 for nome in nome_classe.values()}
                    hourly_entry_counts[hour_offset]['_total'] = 0

                # Preenche os buckets
                for event in hourly_entry_events:
                    hour_segment = math.floor(event['time_seconds'] / 3600)
                    if hour_segment in hourly_entry_counts:
                        hourly_entry_counts[hour_segment][event['type']] += 1
                        hourly_entry_counts[hour_segment]['_total'] += 1
                    else: # Caso um evento esteja fora dos limites iniciais calculados (muito raro)
                        hourly_entry_counts[hour_segment] = {nome: 0 for nome in nome_classe.values()}
                        hourly_entry_counts[hour_segment]['_total'] = 0
                        hourly_entry_counts[hour_segment][event['type']] += 1
                        hourly_entry_counts[hour_segment]['_total'] += 1
                
                # Escreve no relatório
                for hour_offset in sorted(hourly_entry_counts.keys()):
                    start_timedelta = datetime.timedelta(seconds=hour_offset * 3600)
                    end_timedelta = datetime.timedelta(seconds=(hour_offset + 1) * 3600 -1) # -1 second to avoid overlapping
                    f.write(f"Hora: {str(start_timedelta).split('.')[0]} - {str(end_timedelta).split('.')[0]}\n")
                    
                    hourly_data = hourly_entry_counts[hour_offset]
                    has_data = False
                    for nome in nome_classe.values():
                        if hourly_data[nome] > 0:
                            f.write(f"  {nome}: {hourly_data[nome]}\n")
                            has_data = True
                    if has_data:
                        f.write(f"  Total Neste Horário: {hourly_data['_total']}\n")
                    else:
                        f.write("  Nenhum evento.\n")
                    f.write("\n")
            else:
                f.write("  Nenhum evento de entrada registrado.\n\n")

            # FLUXO HORÁRIO (SAÍDAS)
            f.write("FLUXO HORÁRIO (SAÍDAS):\n")
            f.write("-" * 30 + "\n")
            if hourly_exit_events:
                # Ordena os eventos por tempo
                hourly_exit_events.sort(key=lambda x: x['time_seconds'])
                min_time = hourly_exit_events[0]['time_seconds']
                max_time = hourly_exit_events[-1]['time_seconds']
                
                start_hour = math.floor(min_time / 3600) if min_time > 0 else 0
                end_hour_report = math.ceil(max_time / 3600) if max_time > 0 else 1

                hourly_exit_counts = {}
                for hour_offset in range(start_hour, end_hour_report):
                    hourly_exit_counts[hour_offset] = {nome: 0 for nome in nome_classe.values()}
                    hourly_exit_counts[hour_offset]['_total'] = 0
                
                for event in hourly_exit_events:
                    hour_segment = math.floor(event['time_seconds'] / 3600)
                    if hour_segment in hourly_exit_counts:
                        hourly_exit_counts[hour_segment][event['type']] += 1
                        hourly_exit_counts[hour_segment]['_total'] += 1
                    else:
                        hourly_exit_counts[hour_segment] = {nome: 0 for nome in nome_classe.values()}
                        hourly_exit_counts[hour_segment]['_total'] = 0
                        hourly_exit_counts[hour_segment][event['type']] += 1
                        hourly_exit_counts[hour_segment]['_total'] += 1

                for hour_offset in sorted(hourly_exit_counts.keys()):
                    start_timedelta = datetime.timedelta(seconds=hour_offset * 3600)
                    end_timedelta = datetime.timedelta(seconds=(hour_offset + 1) * 3600 -1)
                    f.write(f"Hora: {str(start_timedelta).split('.')[0]} - {str(end_timedelta).split('.')[0]}\n")
                    
                    hourly_data = hourly_exit_counts[hour_offset]
                    has_data = False
                    for nome in nome_classe.values():
                        if hourly_data[nome] > 0:
                            f.write(f"  {nome}: {hourly_data[nome]}\n")
                            has_data = True
                    if has_data:
                        f.write(f"  Total Neste Horário: {hourly_data['_total']}\n")
                    else:
                        f.write("  Nenhum evento.\n")
                    f.write("\n")
            else:
                f.write("  Nenhum evento de saída registrado.\n\n")

    return caminho_relatorio, caminho_saida_video

