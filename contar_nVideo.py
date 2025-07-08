import os
import cv2
import json
import numpy as np
import datetime
import logging
import sqlite3
import torch
from ultralytics import YOLO

DB_PATH = os.path.join("resultados", "relatorios.db")
FONT = cv2.FONT_HERSHEY_SIMPLEX

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

def init_db():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS relatorios (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            report_path TEXT NOT NULL,
            video_source TEXT NOT NULL,
            model_used TEXT NOT NULL
        )
    ''')
    conn.commit()
    conn.close()

def log_report(timestamp, report_path, video_source, model_used):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''
        INSERT INTO relatorios (timestamp, report_path, video_source, model_used)
        VALUES (?, ?, ?, ?)
    ''', (timestamp, report_path, video_source, model_used))
    conn.commit()
    conn.close()

def contar_veiculos_nVideo(
    video_path,
    areas_path,
    model_path,
    classes_selecionadas,
    camera_name=None,
    stop_event=None
):
    init_db()
    logging.info("Iniciando contagem headless de veículos.")
    inicio_real = datetime.datetime.now()

    # --- nome do relatório com dia-mês-ano ---
    ts_str = inicio_real.strftime("%d-%m-%Y_%H%M%S")
    cam_str = camera_name.strip().replace(" ", "_") if camera_name else ""
    nome_rel = f"relatorio_{ts_str}" + (f"_{cam_str}" if cam_str else "") + ".txt"
    os.makedirs("resultados", exist_ok=True)
    caminho_relatorio = os.path.join("resultados", nome_rel)

    # --- carregar áreas (1 ou 2) ---
    try:
        with open(areas_path, "r", encoding="utf-8") as f:
            areas = json.load(f)
        if not isinstance(areas, list) or len(areas) not in (1, 2):
            raise ValueError("Esperado lista com 1 (entrada) ou 2 (entrada, saída) áreas.")
        for a in areas:
            if not (isinstance(a, list) and len(a) >= 3):
                raise ValueError("Cada área deve ter pelo menos 3 pontos.")
        area_ent_orig = np.array(areas[0], dtype=np.int32)
        area_sai_orig = np.array(areas[1], dtype=np.int32) if len(areas) == 2 else None
    except Exception as e:
        raise ValueError(f"Erro ao carregar áreas '{areas_path}': {e}")

    # --- carregar e otimizar modelo para GPU/FP16 se disponível ---
    try:
        modelo = YOLO(model_path)
    except Exception as e:
        raise RuntimeError(f"Erro ao carregar modelo '{model_path}': {e}")
    modelo.model.fuse()
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    modelo.to(device)
    if device.startswith("cuda"):
        modelo.model.half()
        logging.info("Modelo movido para CUDA e meia-precisão (FP16).")
    else:
        logging.warning("CUDA não disponível: inferência em CPU.")

    # --- preparar contadores ---
    TODAS_AS_CLASSES = {0:"Pessoa",1:"Bicicleta",2:"Carro",3:"Moto",5:"Onibus",7:"Caminhao"}
    nomes_sel = [TODAS_AS_CLASSES[c] for c in classes_selecionadas if c in TODAS_AS_CLASSES]
    cont_ent = {n:0 for n in nomes_sel}
    cont_sai = {n:0 for n in nomes_sel} if area_sai_orig is not None else {}
    ids_ent, ids_sai = set(), set()
    estados = {}

    # --- abrir vídeo e calcular transformações ---
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Não foi possível abrir vídeo '{video_path}'.")
    w_o = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h_o = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    w_out, h_out = 1280, 720
    fx, fy = w_out / w_o, h_out / h_o
    area_ent = np.array([[int(x*fx), int(y*fy)] for x,y in area_ent_orig], dtype=np.int32)
    area_sai = (np.array([[int(x*fx), int(y*fy)] for x,y in area_sai_orig], dtype=np.int32)
                if area_sai_orig is not None else None)

    # --- loop de processamento ---
    while True:
        if stop_event and stop_event.is_set():
            logging.info("Parada solicitada externamente.")
            break

        ret, frame = cap.read()
        if not ret:
            break

        # timestamp do frame
        t_ms = cap.get(cv2.CAP_PROP_POS_MSEC)
        hora_evt = inicio_real + datetime.timedelta(seconds=(t_ms/1000.0))

        # resize simples (sem exibição)
        frame_resized = cv2.resize(frame, (w_out, h_out), interpolation=cv2.INTER_LINEAR)

        # inferência
        res = modelo.track(
            source=frame_resized,
            tracker="botsort.yaml",
            persist=True,
            classes=classes_selecionadas
        )[0]

        # contagem de eventos
        if hasattr(res.boxes, 'id') and res.boxes.id is not None:
            ids_ = res.boxes.id.int().cpu().tolist()
            bxs  = res.boxes.xyxy.cpu().tolist()
            clss = res.boxes.cls.int().cpu().tolist()
            for (x1,y1,x2,y2), c, tid in zip(bxs, clss, ids_):
                if c not in classes_selecionadas:
                    continue
                nome = TODAS_AS_CLASSES.get(c, "Desconhecido")
                cx, cy = int((x1+x2)/2), int((y1+y2)/2)
                estados.setdefault(tid, {'in_entry':False, 'in_exit':False})

                # entrada
                if cv2.pointPolygonTest(area_ent, (cx,cy), False) >= 0:
                    if not estados[tid]['in_entry']:
                        if tid not in ids_ent:
                            ids_ent.add(tid)
                            cont_ent[nome] += 1
                        estados[tid]['in_entry'] = True
                else:
                    estados[tid]['in_entry'] = False

                # saída (só se definida)
                if area_sai is not None:
                    if cv2.pointPolygonTest(area_sai, (cx,cy), False) >= 0:
                        if not estados[tid]['in_exit']:
                            if tid not in ids_sai:
                                ids_sai.add(tid)
                                cont_sai[nome] += 1
                            estados[tid]['in_exit'] = True
                    else:
                        estados[tid]['in_exit'] = False

    cap.release()
    fim_real = datetime.datetime.now()

    # --- escrever relatório com dia/mês/ano ---
    with open(caminho_relatorio, "w", encoding="utf-8") as f:
        f.write("RELATÓRIO DE CONTAGEM DE VEÍCULOS (Modo Sem Vídeo)\n")
        if camera_name:
            f.write(f"CÂMERA: {camera_name}\n")
        f.write(f"Início: {inicio_real.strftime('%d/%m/%Y %H:%M:%S')}\n")
        f.write(f"Fim:    {fim_real.strftime('%d/%m/%Y %H:%M:%S')}\n")
        f.write(f"Duração: {str(fim_real - inicio_real)}\n")
        f.write("="*40 + "\n\n")
        f.write("ENTRADAS:\n")
        for nome in nomes_sel:
            f.write(f"  {nome}: {cont_ent[nome]}\n")
        f.write(f"  Total IDs entrada: {len(ids_ent)}\n\n")
        if area_sai is not None:
            f.write("SAÍDAS:\n")
            for nome in nomes_sel:
                f.write(f"  {nome}: {cont_sai[nome]}\n")
            f.write(f"  Total IDs saída: {len(ids_sai)}\n")

   
    now_iso = datetime.datetime.now().strftime("%d/%m/%Y %H:%M:%S")
    log_report(now_iso, caminho_relatorio, video_path, os.path.basename(model_path))
    logging.info(f"Relatório salvo em '{caminho_relatorio}'")

    return caminho_relatorio
