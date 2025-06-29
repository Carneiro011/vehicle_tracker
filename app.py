# app.py

import os
import threading
import logging
import sqlite3
import tkinter as tk
from tkinter import filedialog, messagebox

import customtkinter as ctk
import requests

from contar import contar_veiculos
from definir_areas import AreaSelector

# --- Config e logging ---
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# --- Constantes e paths ---
RESULT_DIR = "resultados"
DB_PATH     = os.path.join(RESULT_DIR, "relatorios.db")
AREAS_PATH  = os.path.join(RESULT_DIR, "areas.json")
MODEL_DIR   = "models"
YOLO_MODELS = {
    "yolov8n.pt": "https://github.com/ultralytics/ultralytics/releases/download/v8.0.0/yolov8n.pt",
    "yolov8s.pt": "https://github.com/ultralytics/ultralytics/releases/download/v8.0.0/yolov8s.pt",
    "yolov8m.pt": "https://github.com/ultralytics/ultralytics/releases/download/v8.0.0/yolov8m.pt",
    "yolov5nu.pt": "https://github.com/ultralytics/yolov5/releases/download/v6.0/yolov5nu.pt",
}
CLASSES_DISPONIVEIS = {
    "Pessoa": 0, "Bicicleta": 1, "Carro": 2,
    "Moto": 3, "Ônibus": 5, "Caminhão": 7
}

os.makedirs(RESULT_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)


class App(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("Sistema de Contagem de Veículos")
        self.geometry("500x800")
        self.resizable(False, False)
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("dark-blue")

        # Estado
        self.video_path = ""
        self.last_report_path = None
        self.video_source_type = ctk.StringVar(value="local")
        self.model_name = ctk.StringVar(value=list(YOLO_MODELS.keys())[0])
        self.class_vars = {name: ctk.IntVar(value=1) for name in CLASSES_DISPONIVEIS}

        # Layout
        self.grid_columnconfigure(0, weight=1)
        self.create_settings_frame()
        self.create_class_selection_frame()
        self.create_actions_frame()

        self.status_label = ctk.CTkLabel(self, text="Bem-vindo! Selecione e inicie.", text_color="gray")
        self.status_label.grid(row=4, column=0, padx=20, pady=(10,0), sticky="w")

        self.progress_bar = ctk.CTkProgressBar(self, mode='determinate')
        self.progress_bar.set(0)
        self.toggle_video_source_input()

    def create_settings_frame(self):
        frm = ctk.CTkFrame(self)
        frm.grid(row=0, column=0, padx=20, pady=20, sticky="ew")
        frm.grid_columnconfigure(1, weight=1)

        ctk.CTkLabel(frm, text="1. Fonte do Vídeo", font=ctk.CTkFont(weight="bold")).grid(
            row=0, column=0, columnspan=2, sticky="w", padx=10, pady=(10,5)
        )
        ctk.CTkRadioButton(frm, text="Arquivo Local", variable=self.video_source_type,
                           value="local", command=self.toggle_video_source_input).grid(row=1, column=0, sticky="w", padx=20)
        ctk.CTkRadioButton(frm, text="URL (Stream)", variable=self.video_source_type,
                           value="url", command=self.toggle_video_source_input).grid(row=1, column=1, sticky="w")

        self.btn_select = ctk.CTkButton(frm, text="Selecionar Vídeo", command=self.select_video_file)
        self.ent_url    = ctk.CTkEntry(frm, placeholder_text="https://.../video.mp4 ou .m3u8")

        ctk.CTkFrame(frm, height=2, fg_color="gray20").grid(
            row=4, column=0, columnspan=2, sticky="ew", padx=10, pady=15
        )
        ctk.CTkLabel(frm, text="2. Modelo YOLO", font=ctk.CTkFont(weight="bold")).grid(
            row=5, column=0, columnspan=2, sticky="w", padx=10, pady=5
        )
        ctk.CTkOptionMenu(frm, variable=self.model_name, values=list(YOLO_MODELS.keys())).grid(
            row=6, column=0, columnspan=2, sticky="ew", padx=10, pady=10
        )

    def create_class_selection_frame(self):
        frm = ctk.CTkFrame(self)
        frm.grid(row=1, column=0, padx=20, pady=10, sticky="ew")
        ctk.CTkLabel(frm, text="3. Classes para Rastrear", font=ctk.CTkFont(weight="bold")).grid(
            row=0, column=0, columnspan=3, sticky="w", padx=10, pady=(10,5)
        )
        r, c = 1, 0
        for name, var in self.class_vars.items():
            cb = ctk.CTkCheckBox(frm, text=name, variable=var)
            cb.grid(row=r, column=c, padx=10, pady=5, sticky="w")
            c += 1
            if c == 3:
                c = 0; r += 1

    def create_actions_frame(self):
        frm = ctk.CTkFrame(self, fg_color="transparent")
        frm.grid(row=2, column=0, padx=20, pady=10, sticky="ew")
        frm.grid_columnconfigure(0, weight=1)

        ctk.CTkButton(frm, text="Definir Áreas", command=self.open_define_areas).grid(
            row=0, column=0, pady=5, sticky="ew"
        )
        ctk.CTkButton(frm, text="Iniciar Contagem", command=self.start_counting,
                      font=ctk.CTkFont(weight="bold"), height=40).grid(
            row=1, column=0, pady=10, sticky="ew"
        )
        ctk.CTkButton(frm, text="Exibir Último Relatório", command=self.show_report,
                      fg_color="gray50", hover_color="gray60").grid(
            row=2, column=0, pady=5, sticky="ew"
        )
        ctk.CTkButton(frm, text="Histórico de Relatórios", command=self.show_history,
                      fg_color="#3B8ED0").grid(
            row=3, column=0, pady=5, sticky="ew"
        )

    def toggle_video_source_input(self):
        if self.video_source_type.get() == "local":
            self.ent_url.grid_forget()
            self.btn_select.grid(row=2, column=0, columnspan=2, padx=10, pady=10, sticky="ew")
            status = f"Arquivo: {os.path.basename(self.video_path)}" if self.video_path else "Selecione um vídeo local."
        else:
            self.btn_select.grid_forget()
            self.ent_url.grid(row=2, column=0, columnspan=2, padx=10, pady=10, sticky="ew")
            status = "Insira URL do vídeo (stream)."
        self.update_status(status)

    def select_video_file(self):
        path = filedialog.askopenfilename(title="Selecione um vídeo",
                                          filetypes=[("Vídeos", "*.mp4 *.avi *.mkv")])
        if path:
            self.video_path = path
            self.update_status(f"Arquivo: {os.path.basename(path)}", "success")
            logging.info(f"Vídeo selecionado: {path}")
        else:
            self.update_status("Nenhum vídeo selecionado.", "warning")

    def get_current_video_source(self):
        if self.video_source_type.get() == "local":
            if not self.video_path:
                messagebox.showerror("Erro", "Selecione um vídeo local.")
                return None
            return self.video_path
        url = self.ent_url.get().strip()
        if not url:
            messagebox.showerror("Erro", "Insira uma URL válida.")
            return None
        return url

    def open_define_areas(self):
        src = self.get_current_video_source()
        if not src:
            return
        self.update_status("Abrindo definidor de áreas...", "info")
        def run_selector():
            sel = AreaSelector()
            sel.run(video_source=src)
            self.after(0, lambda: self.update_status("Áreas definidas.", "success"))
        self.run_in_thread(run_selector)

    def get_selected_ids(self):
        return [CLASSES_DISPONIVEIS[n] for n, v in self.class_vars.items() if v.get() == 1]

    def start_counting(self):
        src = self.get_current_video_source()
        if not src:
            return
        if not os.path.exists(AREAS_PATH):
            messagebox.showerror("Erro", "Defina as áreas primeiro.")
            return
        ids = self.get_selected_ids()
        if not ids:
            messagebox.showerror("Erro", "Selecione pelo menos uma classe.")
            return

        model = self.model_name.get()
        model_path = os.path.join(MODEL_DIR, model)

        if not os.path.exists(model_path):
            resp = messagebox.askyesno(
                "Modelo Ausente",
                f"Você escolheu o modelo '{model}', mas ele não está em '{MODEL_DIR}'.\n"
                "Deseja baixá-lo agora?"
            )
            if not resp:
                self.update_status("Contagem cancelada pelo usuário.", "warning")
                return
            self.update_status(f"Baixando modelo '{model}'...", "processing")
            self.run_in_thread(
                self._download_and_start,
                args=(model, src, model_path, ids)
            )
        else:
            self.update_status("Iniciando contagem...", "info")
            self.run_in_thread(
                self.execute_counting_thread,
                args=(src, model_path, ids)
            )

    def _download_and_start(self, model_filename, video_source, model_full_path, selected_ids):
        url = YOLO_MODELS.get(model_filename)
        if not url:
            self.after(0, lambda: messagebox.showerror("Erro", f"URL para {model_filename} não encontrada."))
            return

        self.after(0, self.progress_bar.grid, {'row':5, 'column':0, 'padx':20, 'pady':5, 'sticky':"ew"})
        try:
            resp = requests.get(url, stream=True, timeout=30)
            resp.raise_for_status()
            total = int(resp.headers.get("content-length", 0)) or None
            downloaded = 0
            with open(model_full_path, "wb") as f:
                for chunk in resp.iter_content(chunk_size=8192):
                    if not chunk:
                        continue
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total:
                        frac = downloaded / total
                        self.after(0, lambda f=frac: self.progress_bar.set(f))
            self.after(0, self.progress_bar.grid_forget)
            self.after(0, lambda: self.update_status("Download concluído.", "success"))
            self.after(0, lambda: self.execute_counting_thread(video_source, model_full_path, selected_ids))
        except Exception as e:
            logging.exception("Erro no download do modelo.")
            self.after(0, self.progress_bar.grid_forget)
            self.after(0, lambda: messagebox.showerror("Erro de Download", str(e)))
            self.after(0, lambda: self.update_status("Falha no download do modelo.", "error"))

    def execute_counting_thread(self, video_source, model_path, selected_ids):
        self.after(0, lambda: self.update_status("Processando vídeo... Uma janela pode abrir.", "processing"))
        try:
            relatorio_path, saida_video_path = contar_veiculos(
                video_source, AREAS_PATH, model_path,
                classes_selecionadas=selected_ids, show_video=True
            )
            self.last_report_path = relatorio_path
            msg = f"Contagem finalizada!\nRelatório: {relatorio_path}\nVídeo: {saida_video_path}"
            self.after(0, lambda: self.update_status("Processamento concluído com sucesso!", "success"))
            self.after(0, lambda: messagebox.showinfo("Concluído", msg))
        except Exception as e:
            logging.exception("Erro na contagem de veículos.")
            self.after(0, lambda: messagebox.showerror("Erro na Contagem", str(e)))
            self.after(0, lambda: self.update_status(f"Erro na contagem: {e}", "error"))

    def show_report(self):
        if not self.last_report_path or not os.path.exists(self.last_report_path):
            messagebox.showwarning("Aviso", "Nenhum relatório disponível.")
            return
        self.open_report_file(self.last_report_path)

    def show_history(self):
        try:
            conn = sqlite3.connect(DB_PATH)
            c = conn.cursor()
            c.execute(
                "SELECT id, timestamp, report_path, video_source, model_used "
                "FROM relatorios ORDER BY timestamp DESC"
            )
            rows = c.fetchall()
            conn.close()
        except Exception as e:
            messagebox.showerror("Erro BD", f"Falha ao acessar banco:\n{e}")
            return

        win = ctk.CTkToplevel(self)
        win.title("Histórico de Relatórios")
        win.geometry("800x600")
        win.transient(self)
        win.grab_set()
        win.lift()
        win.focus_force()

        sf = ctk.CTkScrollableFrame(win)
        sf.pack(expand=True, fill="both", padx=10, pady=10)

        for rec_id, ts, rpt, src, mdl in rows:
            frm = ctk.CTkFrame(sf)
            frm.pack(fill="x", pady=5)
            info = f"[{ts}] Modelo: {mdl} | Fonte: {os.path.basename(src)}"
            ctk.CTkLabel(frm, text=info, anchor="w").grid(row=0, column=0, sticky="w", padx=5)
            ctk.CTkButton(frm, text="Ver Relatório", width=120,
                          command=lambda p=rpt: self.open_report_file(p)).grid(
                row=0, column=1, padx=5
            )

    def open_report_file(self, path):
        if not os.path.exists(path):
            messagebox.showerror("Erro", f"Arquivo não encontrado:\n{path}")
            return

        with open(path, "r", encoding="utf-8") as f:
            content = f.read()

        w = ctk.CTkToplevel(self)
        w.title(f"Relatório — {os.path.basename(path)}")
        w.geometry("600x400")
        w.transient(self)
        w.grab_set()
        w.lift()
        w.focus_force()

        tb = ctk.CTkTextbox(w, wrap="word", font=("Courier New", 12))
        tb.pack(expand=True, fill="both", padx=10, pady=10)
        tb.insert("1.0", content)
        tb.configure(state="disabled")

    def update_status(self, text, level="info"):
        colors = {
            "info": "gray", "success": "#2ECC71",
            "warning": "#F1C40F", "error": "#E74C3C"
        }
        self.status_label.configure(text=text, text_color=colors.get(level, "gray"))

    def run_in_thread(self, func, args=()):
        threading.Thread(target=func, args=args, daemon=True).start()


if __name__ == "__main__":
    app = App()
    app.mainloop()
