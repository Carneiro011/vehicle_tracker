import os
import cv2
import json
import numpy as np
from ultralytics import YOLO
import datetime
import logging
import sqlite3
import torch
import tkinter as tk
from tkinter import messagebox

# --- Configuração básica de logging ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

# --- Banco de dados para histórico de relatórios ---
DB_PATH = os.path.join("resultados", "relatorios.db")

# --- Constantes de fonte para sobreposição na tela ---
FONT       = cv2.FONT_HERSHEY_SIMPLEX
THICKNESS  = 1
LINE_TYPE  = cv2.LINE_AA

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

def contar_veiculos(
        video_path,
        areas_path,
        model_path,
        classes_selecionadas,
        show_video=True,
        camera_name=None
    ):
    """
    Conta veículos com YOLO em GPU (CUDA) e usa OpenCV CUDA para resize.
    Aceita 1 área (entrada) ou 2 áreas (entrada + saída).
    Exibe bounding boxes, IDs (se houver) e totais na tela e no relatório.
    """
    init_db()
    inicio_real = datetime.datetime.now()
    logging.info("Iniciando contagem de veículos.")

    # --- Preparar relatório ---
    os.makedirs("resultados", exist_ok=True)
    ts_str = inicio_real.strftime("%d-%m-%Y_%H%M%S")
    cam_str = camera_name.strip().replace(" ", "_") if camera_name else ""
    nome_rel = f"relatorio_{ts_str}" + (f"_{cam_str}" if cam_str else "") + ".txt"
    caminho_relatorio = os.path.join("resultados", nome_rel)

    # --- Carregar áreas ---
    with open(areas_path, "r", encoding="utf-8") as f:
        areas = json.load(f)
    if not isinstance(areas, list) or len(areas) not in (1,2):
        raise ValueError("Esperado lista com 1 (entrada) ou 2 (entrada, saída) áreas.")
    for a in areas:
        if not (isinstance(a, list) and len(a) >= 3):
            raise ValueError("Cada área deve ter pelo menos 3 pontos.")
    area_ent_orig = np.array(areas[0], dtype=np.int32)
    area_sai_orig = np.array(areas[1], dtype=np.int32) if len(areas)==2 else None

    # --- Carregar modelo na GPU/FP16 se possível ---
    modelo = YOLO(model_path)
    modelo.model.fuse()
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    modelo.to(device)
    if device.startswith("cuda"):
        modelo.model.half()
        logging.info("Modelo em CUDA e FP16.")
    else:
        logging.warning("Inferência em CPU.")

    # --- Preparar estruturas de contagem ---
    TODAS_AS_CLASSES = {0:"Pessoa",1:"Bicicleta",2:"Carro",3:"Moto",5:"Onibus",7:"Caminhao"}
    nomes_sel = [TODAS_AS_CLASSES[c] for c in classes_selecionadas if c in TODAS_AS_CLASSES]
    cont_ent, cont_sai = {n:0 for n in nomes_sel}, {n:0 for n in nomes_sel} if area_sai_orig is not None else {}
    ids_ent, ids_sai = set(), set()
    estados = {}

    # --- Abrir vídeo e escalar polígonos ---
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Não foi possível abrir vídeo '{video_path}'.")
    w_o, h_o = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    w_out, h_out = 1280, 720
    fx, fy = w_out / w_o, h_out / h_o
    area_ent = np.array([[int(x*fx),int(y*fy)] for x,y in area_ent_orig],dtype=np.int32)
    area_sai = (np.array([[int(x*fx),int(y*fy)] for x,y in area_sai_orig],dtype=np.int32)
                if area_sai_orig is not None else None)

    # --- Verificar OpenCV CUDA para resize ---
    use_cuda_resize = cv2.cuda.getCudaEnabledDeviceCount()>0
    if use_cuda_resize:
        logging.info("OpenCV CUDA disponível para resize.")

    # --- Janela de exibição ---
    window_name = "Contagem - Q para sair"
    if show_video:
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, w_out, h_out)

    break_on_x = False

    # --- Loop principal ---
    while True:
        ret, frame = cap.read()
        if not ret:
            break  # vídeo acabou

        t_ms = cap.get(cv2.CAP_PROP_POS_MSEC)
        hora_evt = inicio_real + datetime.timedelta(seconds=(t_ms/1000.0))

        # resize
        if use_cuda_resize:
            gpu = cv2.cuda.GpuMat()
            gpu.upload(frame)
            gpu = cv2.cuda.resize(gpu, (w_out,h_out))
            frame_resized = gpu.download()
        else:
            frame_resized = cv2.resize(frame, (w_out,h_out))

        # inferência/tracking
        res = modelo.track(
            source=frame_resized,
            tracker="botsort.yaml",
            persist=True,
            classes=classes_selecionadas
        )[0]

        # preparar exibição
        disp = frame_resized.copy()
        overlay = disp.copy()
        cv2.fillPoly(overlay, [area_ent], (0,255,0))
        if area_sai is not None:
            cv2.fillPoly(overlay, [area_sai], (0,0,255))
        cv2.addWeighted(overlay,0.3,disp,0.7,0,disp)
        cv2.polylines(disp,[area_ent],True,(0,255,0),2)
        if area_sai is not None:
            cv2.polylines(disp,[area_sai],True,(0,0,255),2)
        cv2.putText(disp, hora_evt.strftime("%d/%m/%Y %H:%M:%S"),
                    (10,20),FONT,0.6,(255,255,255),THICKNESS,LINE_TYPE)

        # extrair boxes, classes e ids (ou None)
        bxs  = res.boxes.xyxy.cpu().tolist()
        clss = res.boxes.cls.int().cpu().tolist()
        ids_  = (res.boxes.id.int().cpu().tolist()
                 if hasattr(res.boxes,'id') and res.boxes.id is not None
                 else [None]*len(bxs))

        # processar cada detecção
        for (x1,y1,x2,y2), c, tid in zip(bxs, clss, ids_):
            if c not in classes_selecionadas:
                continue
            nome = TODAS_AS_CLASSES.get(c,"Desconhecido")
            cx, cy = int((x1+x2)/2), int((y1+y2)/2)

            # contagem só se tivermos id
            if tid is not None:
                estados.setdefault(tid, {'in_entry':False,'in_exit':False})
                # entrada
                if cv2.pointPolygonTest(area_ent,(cx,cy),False)>=0:
                    if not estados[tid]['in_entry']:
                        ids_ent.add(tid)
                        cont_ent[nome]+=1
                        estados[tid]['in_entry']=True
                else:
                    estados[tid]['in_entry']=False
                # saída
                if area_sai is not None and cv2.pointPolygonTest(area_sai,(cx,cy),False)>=0:
                    if not estados[tid]['in_exit']:
                        ids_sai.add(tid)
                        cont_sai[nome]+=1
                        estados[tid]['in_exit']=True
                else:
                    if area_sai is not None:
                        estados[tid]['in_exit']=False

            # desenhar bbox + label (sempre)
            cv2.rectangle(disp,(int(x1),int(y1)),(int(x2),int(y2)),(255,0,0),1)
            label = nome + (f" ID:{tid}" if tid is not None else "")
            cv2.putText(disp,label,(int(x1),int(y1)-6),FONT,0.5,(255,0,0),THICKNESS,LINE_TYPE)

        # sobrepor contadores e totais
        y = 40
        cv2.putText(disp,"ENTRADAS:",(10,y),FONT,0.8,(0,255,0),THICKNESS,LINE_TYPE)
        for n,cnt in cont_ent.items():
            y+=20
            cv2.putText(disp,f"{n}: {cnt}",(10,y),FONT,0.6,(0,255,0),THICKNESS,LINE_TYPE)
        y+=20
        cv2.putText(disp,f"Total Entradas: {len(ids_ent)}",(10,y),FONT,0.6,(0,255,0),THICKNESS,LINE_TYPE)

        if area_sai is not None:
            y+=30
            cv2.putText(disp,"SAÍDAS:",(10,y),FONT,0.8,(0,0,255),THICKNESS,LINE_TYPE)
            for n,cnt in cont_sai.items():
                y+=20
                cv2.putText(disp,f"{n}: {cnt}",(10,y),FONT,0.6,(0,0,255),THICKNESS,LINE_TYPE)
            y+=20
            cv2.putText(disp,f"Total Saídas: {len(ids_sai)}",(10,y),FONT,0.6,(0,0,255),THICKNESS,LINE_TYPE)

        # exibir frame
        if show_video:
            cv2.imshow(window_name,disp)
            key = cv2.waitKey(1)&0xFF
            if key==ord('q') or cv2.getWindowProperty(window_name,cv2.WND_PROP_VISIBLE)<1:
                break_on_x=True
                break

    cap.release()
    if show_video:
        cv2.destroyAllWindows()

    fim_real = datetime.datetime.now()
    save=True
    if break_on_x:
        root=tk.Tk(); root.withdraw()
        save=messagebox.askyesno("Salvar Relatório","Deseja salvar o relatório?")
        root.destroy()

    if save:
        with open(caminho_relatorio,"w",encoding="utf-8") as f:
            f.write("RELATÓRIO DE CONTAGEM DE VEÍCULOS\n")
            if camera_name:
                f.write(f"CÂMERA: {camera_name}\n")
            f.write(f"Início: {inicio_real.strftime('%d/%m/%Y %H:%M:%S')}\n")
            f.write(f"Fim:    {fim_real.strftime('%d/%m/%Y %H:%M:%S')}\n")
            f.write(f"Duração: {fim_real - inicio_real}\n")
            f.write("="*40+"\n\n")
            f.write("ENTRADAS:\n")
            for n in nomes_sel:
                f.write(f"  {n}: {cont_ent[n]}\n")
            f.write(f"  Total IDs entrada: {len(ids_ent)}\n\n")
            if area_sai is not None:
                f.write("SAÍDAS:\n")
                for n in nomes_sel:
                    f.write(f"  {n}: {cont_sai[n]}\n")
                f.write(f"  Total IDs saída: {len(ids_sai)}\n")
        logging.info(f"Relatório gravado em '{caminho_relatorio}'")
        log_report(
            datetime.datetime.now().strftime("%d/%m/%Y %H:%M:%S"),
            caminho_relatorio,
            video_path,
            os.path.basename(model_path)
        )
        return caminho_relatorio
    else:
        logging.info("Relatório não salvo.")
        return None
