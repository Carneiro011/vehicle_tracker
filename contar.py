# contar.py

import os
import cv2
import json
import numpy as np
from ultralytics import YOLO
import datetime
import math
import logging
import sqlite3
import tkinter as tk
from tkinter import messagebox

# --- Configuração básica de logging ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

# --- Banco de dados para histórico de relatórios ---
DB_PATH = os.path.join("resultados", "relatorios.db")

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

def contar_veiculos(video_path, areas_path, model_path, classes_selecionadas, show_video=True):
    """
    Conta veículos em um vídeo usando YOLO, exibindo em tempo real
    hora local e totais de entradas/saídas, e ao fechar a janela
    pergunta se deve salvar o relatório.
    """
    init_db()
    logging.info("Iniciando contagem de veículos.")
    inicio_real = datetime.datetime.now()

    # --- Preparar pastas e nomes ---
    os.makedirs("resultados", exist_ok=True)
    ts_str = inicio_real.strftime("%Y%m%d_%H%M%S")
    caminho_saida_video = os.path.join("resultados", f"saida_{ts_str}.mp4")
    caminho_relatorio    = os.path.join("resultados", f"relatorio_{ts_str}.txt")

    # --- Carregar áreas ---
    try:
        with open(areas_path, "r", encoding="utf-8") as f:
            areas = json.load(f)
        if not (isinstance(areas, list) and len(areas) == 2):
            raise ValueError("Esperado lista com 2 áreas (entrada, saída).")
        for a in areas:
            if not (isinstance(a, list) and len(a) >= 3):
                raise ValueError("Cada área deve ter pelo menos 3 pontos.")
        area_ent_orig = np.array(areas[0], dtype=np.int32)
        area_sai_orig = np.array(areas[1], dtype=np.int32)
    except Exception as e:
        raise ValueError(f"Erro ao carregar áreas '{areas_path}': {e}")

    # --- Carregar modelo YOLO ---
    try:
        modelo = YOLO(model_path)
    except Exception as e:
        raise RuntimeError(f"Erro ao carregar modelo '{model_path}': {e}")

    # --- Preparar contadores ---
    TODAS_AS_CLASSES = {0:"Pessoa",1:"Bicicleta",2:"Carro",3:"Moto",5:"Onibus",7:"Caminhao"}
    nomes_sel    = [TODAS_AS_CLASSES[c] for c in classes_selecionadas if c in TODAS_AS_CLASSES]
    cont_ent     = {n:0 for n in nomes_sel}
    cont_sai     = {n:0 for n in nomes_sel}
    ids_ent, ids_sai = set(), set()
    estados      = {}  # track_id → {'in_entry':bool,'in_exit':bool}
    eventos_ent  = []  # {'time': datetime, 'type': nome}
    eventos_sai  = []

    # --- Abrir vídeo e saída ---
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Não foi possível abrir vídeo '{video_path}'.")
    w_o = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h_o = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS)) if cap.get(cv2.CAP_PROP_FPS)>0 else 30
    w_out, h_out = 1280, 720

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_out = cv2.VideoWriter(caminho_saida_video, fourcc, fps, (w_out,h_out))
    if not video_out.isOpened():
        raise IOError(f"Não foi possível criar '{caminho_saida_video}'.")

    # ajustar polígonos
    fx, fy = w_out/w_o, h_out/h_o
    area_ent = np.array([[int(x*fx),int(y*fy)] for x,y in area_ent_orig], dtype=np.int32)
    area_sai = np.array([[int(x*fx),int(y*fy)] for x,y in area_sai_orig], dtype=np.int32)

    # --- Laço de processamento ---
    window_name = "Processando - pressione 'q' para sair"
    break_on_x = False

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # calcula hora real do evento
        t_ms = cap.get(cv2.CAP_PROP_POS_MSEC)
        t_s  = t_ms / 1000.0
        hora_evt = inicio_real + datetime.timedelta(seconds=t_s)

        frame = cv2.resize(frame, (w_out,h_out))
        res = modelo.track(source=frame, tracker="botsort.yaml",
                            persist=True, classes=classes_selecionadas)[0]

        disp = frame.copy()
        overlay = disp.copy()
        cv2.fillPoly(overlay, [area_ent], (0,255,0))
        cv2.fillPoly(overlay, [area_sai], (0,0,255))
        cv2.addWeighted(overlay, 0.3, disp, 0.7, 0, disp)
        cv2.polylines(disp, [area_ent], True, (0,255,0), 2)
        cv2.polylines(disp, [area_sai], True, (0,0,255), 2)

        # sobrepõe a hora atual
        cv2.putText(disp,
            hora_evt.strftime("%Y-%m-%d %H:%M:%S"),
            (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2
        )

        # processa detecções
        if hasattr(res.boxes, 'id') and res.boxes.id is not None:
            ids_ = res.boxes.id.int().cpu().tolist()
            bxs  = res.boxes.xyxy.cpu().tolist()
            clss = res.boxes.cls.int().cpu().tolist()

            for (x1,y1,x2,y2), c, tid in zip(bxs, clss, ids_):
                if c not in classes_selecionadas:
                    continue
                nome = TODAS_AS_CLASSES.get(c, "Desconhecido")
                cx, cy = int((x1+x2)/2), int((y1+y2)/2)

                if tid not in estados:
                    estados[tid] = {'in_entry':False,'in_exit':False}

                # entrada
                in_ent = cv2.pointPolygonTest(area_ent, (cx,cy), False) >= 0
                if in_ent and not estados[tid]['in_entry']:
                    eventos_ent.append({'time':hora_evt, 'type':nome})
                    if tid not in ids_ent:
                        ids_ent.add(tid)
                        cont_ent[nome] += 1
                    estados[tid]['in_entry'] = True
                elif not in_ent:
                    estados[tid]['in_entry'] = False

                # saída
                in_sai = cv2.pointPolygonTest(area_sai, (cx,cy), False) >= 0
                if in_sai and not estados[tid]['in_exit']:
                    eventos_sai.append({'time':hora_evt, 'type':nome})
                    if tid not in ids_sai:
                        ids_sai.add(tid)
                        cont_sai[nome] += 1
                    estados[tid]['in_exit'] = True
                elif not in_sai:
                    estados[tid]['in_exit'] = False

                # desenha bbox
                cv2.rectangle(disp, (int(x1),int(y1)), (int(x2),int(y2)), (255,0,0), 2)
                cv2.putText(disp, f"{nome} ID:{tid}",
                            (int(x1),int(y1)-6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 2)

        # desenha contadores dinâmicos
        y0 = 40
        cv2.putText(disp, "ENTRADAS:", (10, y0),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
        y0 += 25
        for n,q in cont_ent.items():
            cv2.putText(disp, f"{n}: {q}", (10,y0),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)
            y0 += 20

        y0 += 10
        cv2.putText(disp, "SAIDAS:", (10, y0),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)
        y0 += 25
        for n,q in cont_sai.items():
            cv2.putText(disp, f"{n}: {q}", (10,y0),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 2)
            y0 += 20

        video_out.write(disp)
        if show_video:
            cv2.imshow(window_name, disp)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            # detecta clique no X
            if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                break_on_x = True
                break

    # libera recursos
    cap.release()
    video_out.release()
    if show_video:
        cv2.destroyAllWindows()

    # se fechou via X, pergunta se salva
    save = True
    if break_on_x:
        root = tk.Tk(); root.withdraw()
        save = messagebox.askyesno("Salvar Relatório", "Deseja salvar o relatório?")
        root.destroy()

    if save:
        # gera relatório em disco
        with open(caminho_relatorio, "w", encoding="utf-8") as f:
            f.write("RELATÓRIO DE CONTAGEM DE VEÍCULOS\n")
            f.write("="*40 + "\n\n")
            # Totais gerais
            f.write("TOTAIS GERAIS (IDs únicos):\n")
            for n in nomes_sel:
                f.write(f"  Entrada {n}: {cont_ent[n]}\n")
            f.write(f"  Total IDs entrada: {len(ids_ent)}\n\n")
            for n in nomes_sel:
                f.write(f"  Saída {n}: {cont_sai[n]}\n")
            f.write(f"  Total IDs saída: {len(ids_sai)}\n\n")

            # Fluxo horário de ENTRADAS
            f.write("FLUXO HORÁRIO (ENTRADAS) — Horário Local\n")
            f.write("-"*40 + "\n")
            if eventos_ent:
                agrup = {}
                for ev in eventos_ent:
                    h = ev['time'].replace(minute=0, second=0, microsecond=0)
                    agrup.setdefault(h, {**{n:0 for n in nomes_sel}, '_total':0})
                    agrup[h][ev['type']] += 1
                    agrup[h]['_total']   += 1
                for h, dados in sorted(agrup.items()):
                    end = h + datetime.timedelta(hours=1) - datetime.timedelta(seconds=1)
                    f.write(f"{h.strftime('%Y-%m-%d %H:%M:%S')} → {end.strftime('%Y-%m-%d %H:%M:%S')}\n")
                    for n in nomes_sel:
                        if dados[n] > 0:
                            f.write(f"  {n}: {dados[n]}\n")
                    f.write(f"  Total: {dados['_total']}\n\n")
            else:
                f.write("  Nenhum evento de entrada.\n\n")

            # Fluxo horário de SAÍDAS
            f.write("FLUXO HORÁRIO (SAIDAS) — Horário Local\n")
            f.write("-"*40 + "\n")
            if eventos_sai:
                agrup = {}
                for ev in eventos_sai:
                    h = ev['time'].replace(minute=0, second=0, microsecond=0)
                    agrup.setdefault(h, {**{n:0 for n in nomes_sel}, '_total':0})
                    agrup[h][ev['type']] += 1
                    agrup[h]['_total']   += 1
                for h, dados in sorted(agrup.items()):
                    end = h + datetime.timedelta(hours=1) - datetime.timedelta(seconds=1)
                    f.write(f"{h.strftime('%Y-%m-%d %H:%M:%S')} → {end.strftime('%Y-%m-%d %H:%M:%S')}\n")
                    for n in nomes_sel:
                        if dados[n] > 0:
                            f.write(f"  {n}: {dados[n]}\n")
                    f.write(f"  Total: {dados['_total']}\n\n")
            else:
                f.write("  Nenhum evento de saída.\n\n")

        logging.info(f"Relatório gravado em '{caminho_relatorio}'")
        # registra no banco
        now_iso = datetime.datetime.now().isoformat(sep=' ', timespec='seconds')
        log_report(now_iso, caminho_relatorio, video_path, os.path.basename(model_path))
        return caminho_relatorio, caminho_saida_video
    else:
        logging.info("Usuário optou por não salvar o relatório.")
        return None, caminho_saida_video
