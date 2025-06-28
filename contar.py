# contar.py
import os
import cv2
import json
import numpy as np
from ultralytics import YOLO
import datetime
import math

def contar_veiculos(video_path, areas_path, model_path, show_video=True):
    # ... (toda a parte inicial de configuração até antes do loop permanece a mesma) ...
    pasta_saida = "resultados"
    os.makedirs(pasta_saida, exist_ok=True)
    caminho_saida_video = os.path.join(pasta_saida, "saida_com_contagem.mp4")
    caminho_relatorio = os.path.join(pasta_saida, "relatorio.txt")

    try:
        with open(areas_path, "r") as f:
            areas = json.load(f)
        if not (isinstance(areas, list) and len(areas) == 2 and
                all(isinstance(a, list) and len(a) >= 3 for a in areas)):
            raise ValueError("O arquivo de áreas está em um formato inválido ou incompleto.")
        area_entrada_original = np.array(areas[0], dtype=np.int32)
        area_saida_original = np.array(areas[1], dtype=np.int32)
    except Exception as e:
        raise ValueError(f"Erro ao carregar áreas do arquivo {areas_path}: {e}")

    try:
        modelo = YOLO(model_path)
    except Exception as e:
        raise RuntimeError(f"Erro ao carregar o modelo YOLO '{model_path}': {e}")

    nome_classe = {2: "Carro", 3: "Moto", 5: "Onibus", 7: "Caminhao"}
    classes_interesse = list(nome_classe.keys())

    cont_entrada_total = {nome: 0 for nome in nome_classe.values()}
    cont_saida_total = {nome: 0 for nome in nome_classe.values()}
    ids_unicos_entrada = set()
    ids_unicos_saida = set()
    vehicle_tracking_states = {}
    hourly_entry_events = []
    hourly_exit_events = []

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Não foi possível abrir o vídeo em {video_path}.")

    fps = int(cap.get(cv2.CAP_PROP_FPS))
    largura_video_original = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    altura_video_original = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_width_out, frame_height_out = 1280, 720

    video_out = cv2.VideoWriter(caminho_saida_video, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width_out, frame_height_out))
    if not video_out.isOpened():
        raise IOError(f"Não foi possível criar o arquivo de vídeo de saída em {caminho_saida_video}.")

    fator_escala_x = frame_width_out / largura_video_original
    fator_escala_y = frame_height_out / altura_video_original
    area_entrada_ajustada = np.array([[int(p[0] * fator_escala_x), int(p[1] * fator_escala_y)] for p in area_entrada_original], dtype=np.int32)
    area_saida_ajustada = np.array([[int(p[0] * fator_escala_x), int(p[1] * fator_escala_y)] for p in area_saida_original], dtype=np.int32)
    
    # Loop principal para processar cada frame do vídeo
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        current_frame_timestamp_ms = cap.get(cv2.CAP_PROP_POS_MSEC)
        current_seconds = current_frame_timestamp_ms / 1000.0

        frame = cv2.resize(frame, (frame_width_out, frame_height_out))

        # <<< ALTERAÇÃO 1: Realiza a detecção no frame LIMPO
        resultados = modelo.track(source=frame, tracker="botsort.yaml", persist=True, verbose=False)[0]
        
        # <<< ALTERAÇÃO 2: Cria uma cópia do frame para desenhar
        frame_para_exibicao = frame.copy()

        # Desenha as áreas de interesse no frame de exibição
        overlay = frame_para_exibicao.copy()
        cv2.fillPoly(overlay, [area_entrada_ajustada], (0, 255, 0))
        cv2.fillPoly(overlay, [area_saida_ajustada], (0, 0, 255))
        alpha = 0.3
        cv2.addWeighted(overlay, alpha, frame_para_exibicao, 1 - alpha, 0, frame_para_exibicao)
        cv2.polylines(frame_para_exibicao, [area_entrada_ajustada], True, (0, 255, 0), 2)
        cv2.polylines(frame_para_exibicao, [area_saida_ajustada], True, (0, 0, 255), 2)

        current_frame_track_ids = set()

        if resultados.boxes.id is not None:
            ids = resultados.boxes.id.int().cpu().tolist()
            boxes = resultados.boxes.xyxy.cpu().tolist()
            classes = resultados.boxes.cls.int().cpu().tolist()

            for box, cls, track_id in zip(boxes, classes, ids):
                if cls in classes_interesse:
                    nome = nome_classe[cls]
                    x1, y1, x2, y2 = map(int, box)
                    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

                    # <<< ALTERAÇÃO 3: Desenha a caixa e o ID no frame de exibição
                    cv2.rectangle(frame_para_exibicao, (x1, y1), (x2, y2), (255, 0, 0), 2)
                    cv2.putText(frame_para_exibicao, f"{nome} ID: {track_id}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

                    current_frame_track_ids.add(track_id)

                    if track_id not in vehicle_tracking_states:
                        vehicle_tracking_states[track_id] = {'type': nome, 'was_in_entry_zone': False, 'was_in_exit_zone': False}
                    
                    is_in_entry_now = cv2.pointPolygonTest(area_entrada_ajustada, (cx, cy), False) >= 0
                    was_in_entry_before = vehicle_tracking_states[track_id]['was_in_entry_zone']

                    if is_in_entry_now and not was_in_entry_before:
                        hourly_entry_events.append({'time_seconds': current_seconds, 'type': nome})
                        ids_unicos_entrada.add(track_id)
                        cont_entrada_total[nome] += 1
                        vehicle_tracking_states[track_id]['was_in_entry_zone'] = True
                    elif not is_in_entry_now and was_in_entry_before:
                        vehicle_tracking_states[track_id]['was_in_entry_zone'] = False

                    is_in_exit_now = cv2.pointPolygonTest(area_saida_ajustada, (cx, cy), False) >= 0
                    was_in_exit_before = vehicle_tracking_states[track_id]['was_in_exit_zone']

                    if is_in_exit_now and not was_in_exit_before:
                        hourly_exit_events.append({'time_seconds': current_seconds, 'type': nome})
                        ids_unicos_saida.add(track_id)
                        cont_saida_total[nome] += 1
                        vehicle_tracking_states[track_id]['was_in_exit_zone'] = True
                    elif not is_in_exit_now and was_in_exit_before:
                        vehicle_tracking_states[track_id]['was_in_exit_zone'] = False
        
        keys_to_reset = [tid for tid in vehicle_tracking_states if tid not in current_frame_track_ids]
        for tid in keys_to_reset:
            vehicle_tracking_states[tid]['was_in_entry_zone'] = False
            vehicle_tracking_states[tid]['was_in_exit_zone'] = False

        # <<< ALTERAÇÃO 4: Desenha o painel de contagem no frame de exibição
        y_offset_count = 30
        cv2.putText(frame_para_exibicao, "Contagem TOTAL:", (10, y_offset_count), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        y_offset_count += 30
        for nome, qtd in cont_entrada_total.items():
            cv2.putText(frame_para_exibicao, f"Entrada {nome}: {qtd}", (10, y_offset_count), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            y_offset_count += 25
        y_offset_count += 10
        for nome, qtd in cont_saida_total.items():
            cv2.putText(frame_para_exibicao, f"Saida {nome}: {qtd}", (10, y_offset_count), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            y_offset_count += 25

        # <<< ALTERAÇÃO 5: Salva e exibe o frame com tudo desenhado
        video_out.write(frame_para_exibicao)

        if show_video:
            cv2.imshow("Processando Video - Pressione 'q' para fechar", frame_para_exibicao)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    # Libera recursos
    cap.release()
    video_out.release()
    if show_video:
        cv2.destroyAllWindows()

    # --- Geração do Relatório Detalhado (sem alterações aqui) ---
    # (O código de geração de relatório permanece exatamente o mesmo)
    # ... (código do relatório omitido para brevidade, mas deve ser mantido no seu ficheiro) ...
    with open(caminho_relatorio, "w", encoding="utf-8") as f: # Adicionado encoding utf-8 por segurança
        # (Cole aqui toda a sua lógica de escrita de relatório)
        # ...
        f.write("RELATÓRIO DE CONTAGEM DE VEÍCULOS\n")
        f.write("="*40 + "\n\n")

        if not hourly_entry_events and not hourly_exit_events:
            f.write("Nenhum veículo detectado ou áreas não cruzadas.\n\n")
        else:
            f.write("TOTAIS GERAIS (ID ÚNICO POR ÁREA):\n")
            # ... resto da sua lógica de relatório ...


    return caminho_relatorio, caminho_saida_video