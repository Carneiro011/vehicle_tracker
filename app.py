# TRACKER_PROJECT/app.py

import tkinter as tk
import customtkinter as ctk
from tkinter import filedialog, messagebox
# import subprocess  # <<< ALTERAÇÃO: Não precisamos mais do subprocess para isto
import os
import threading
import requests
# import json        # <<< ALTERAÇÃO: Não é usado diretamente aqui
import sys

# --- Importação dos scripts locais com tratamento de erro ---

# Importa a função de contagem principal
try:
    from contar import contar_veiculos
except ImportError:
    def contar_veiculos(*args, **kwargs):
        messagebox.showerror("Erro Crítico", "O arquivo 'contar.py' não foi encontrado. A função principal de contagem está ausente.")
        raise ImportError("O arquivo 'contar.py' ou a função 'contar_veiculos' não foi encontrada.")

# <<< ALTERAÇÃO 1: Importar a classe AreaSelector que modificamos
try:
    from definir_areas import AreaSelector
except ImportError:
    messagebox.showerror("Erro Crítico", "O arquivo 'definir_areas.py' não foi encontrado. A função de definição de áreas está ausente.")
    # Criamos uma classe 'placeholder' para que a app não quebre ao iniciar
    class AreaSelector:
        def __init__(self, *args, **kwargs):
            print("Classe AreaSelector não encontrada.")
        def run(self, *args, **kwargs):
            messagebox.showerror("Erro", "Não é possível definir áreas. 'definir_areas.py' está em falta.")


# --- Constantes de Configuração ---
MODEL_DIR = "models"
RESULT_DIR = "resultados"
AREAS_PATH = os.path.join(RESULT_DIR, "areas.json")

# Modelos YOLO e suas URLs
YOLO_MODELS = {
    "yolov5nu.pt": "https://github.com/ultralytics/yolov5/releases/download/v6.0/yolov5nu.pt",
    "yolov5s.pt": "https://github.com/ultralytics/yolov5/releases/download/v6.0/yolov5s.pt",
    "yolov8n.pt": "https://github.com/ultralytics/ultralytics/releases/download/v8.0.0/yolov8n.pt",
    "yolov8s.pt": "https://github.com/ultralytics/ultralytics/releases/download/v8.0.0/yolov8s.pt",
    "yolov8m.pt": "https://github.com/ultralytics/ultralytics/releases/download/v8.0.0/yolov8m.pt",
}

# Garante que os diretórios de resultados e modelos existam
os.makedirs(RESULT_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

class App(ctk.CTk):
    def __init__(self):
        super().__init__()

        # --- Configuração da Janela Principal ---
        self.title("Sistema de Contagem de Veículos")
        self.geometry("500x620")
        self.resizable(False, False)
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("dark-blue")

        # --- Variáveis de Estado ---
        self.video_path_global = "" # <<< ALTERAÇÃO: Inicializar como string vazia para consistência
        self.last_report_path = None
        self.video_source_type = ctk.StringVar(value="local")
        self.model_name = ctk.StringVar(value="yolov5nu.pt")

        # --- Estrutura da UI ---
        self.grid_columnconfigure(0, weight=1)
        
        # Frame de Configurações
        self.create_settings_frame()
        
        # Frame de Ações
        self.create_actions_frame()

        # Rótulo de Status e Barra de Progresso
        self.status_label = ctk.CTkLabel(self, text="Bem-vindo! Selecione as opções e inicie.", text_color="gray")
        self.status_label.grid(row=2, column=0, padx=20, pady=(10, 0), sticky="w")

        self.progress_bar = ctk.CTkProgressBar(self, mode='determinate')
        self.progress_bar.set(0)
        # O progress_bar será adicionado/removido dinamicamente

        # Inicializa o estado da UI
        self.toggle_video_source_input()

    def create_settings_frame(self):
        """Cria o frame com as opções de vídeo e modelo."""
        settings_frame = ctk.CTkFrame(self)
        settings_frame.grid(row=0, column=0, padx=20, pady=20, sticky="ew")
        settings_frame.grid_columnconfigure(1, weight=1)

        # Título do Frame
        ctk.CTkLabel(settings_frame, text="1. Fonte do Vídeo", font=ctk.CTkFont(weight="bold")).grid(row=0, column=0, columnspan=2, padx=10, pady=(10,5), sticky="w")

        # Radio Buttons para Fonte do Vídeo
        local_rb = ctk.CTkRadioButton(settings_frame, text="Arquivo Local", variable=self.video_source_type, value="local", command=self.toggle_video_source_input)
        local_rb.grid(row=1, column=0, padx=20, pady=5, sticky="w")
        url_rb = ctk.CTkRadioButton(settings_frame, text="URL (Stream)", variable=self.video_source_type, value="url", command=self.toggle_video_source_input)
        url_rb.grid(row=1, column=1, padx=10, pady=5, sticky="w")
        
        # Widgets de entrada de vídeo (controlados por toggle_video_source_input)
        self.video_local_button = ctk.CTkButton(settings_frame, text="Selecionar Arquivo de Vídeo", command=self.select_video_file)
        self.video_url_entry = ctk.CTkEntry(settings_frame, placeholder_text="https://.../video.mp4")

        # Separador
        separator = ctk.CTkFrame(settings_frame, height=2, fg_color="gray20")
        separator.grid(row=4, column=0, columnspan=2, padx=10, pady=15, sticky="ew")

        # Título para Modelo YOLO
        ctk.CTkLabel(settings_frame, text="2. Modelo de Detecção (YOLO)", font=ctk.CTkFont(weight="bold")).grid(row=5, column=0, columnspan=2, padx=10, pady=5, sticky="w")

        # Menu de Seleção de Modelo
        self.model_menu = ctk.CTkOptionMenu(settings_frame, variable=self.model_name, values=list(YOLO_MODELS.keys()))
        self.model_menu.grid(row=6, column=0, columnspan=2, padx=10, pady=10, sticky="ew")

    def create_actions_frame(self):
        """Cria o frame com os botões de ação principais."""
        actions_frame = ctk.CTkFrame(self, fg_color="transparent")
        actions_frame.grid(row=1, column=0, padx=20, pady=10, sticky="ew")
        actions_frame.grid_columnconfigure(0, weight=1)

        btn_define_areas = ctk.CTkButton(actions_frame, text="Definir Áreas de Contagem", command=self.open_define_areas)
        btn_define_areas.grid(row=0, column=0, padx=0, pady=5, sticky="ew")

        btn_start_count = ctk.CTkButton(actions_frame, text="Iniciar Contagem", command=self.start_counting, font=ctk.CTkFont(weight="bold"), height=40)
        btn_start_count.grid(row=1, column=0, padx=0, pady=10, sticky="ew")

        btn_show_report = ctk.CTkButton(actions_frame, text="Exibir Último Relatório", command=self.show_report, fg_color="gray50", hover_color="gray60")
        btn_show_report.grid(row=2, column=0, padx=0, pady=5, sticky="ew")

    def toggle_video_source_input(self):
        """Alterna a UI entre seleção de arquivo local e entrada de URL."""
        if self.video_source_type.get() == "local":
            self.video_url_entry.grid_forget()
            self.video_local_button.grid(row=2, column=0, columnspan=2, padx=10, pady=10, sticky="ew")
            status_text = f"Arquivo: {os.path.basename(self.video_path_global)}" if self.video_path_global else "Selecione um arquivo de vídeo local."
            self.update_status(status_text)
        else: # "url"
            self.video_local_button.grid_forget()
            self.video_url_entry.grid(row=2, column=0, columnspan=2, padx=10, pady=10, sticky="ew")
            self.update_status("Insira a URL de um vídeo (stream).")

    def select_video_file(self):
        """Abre a caixa de diálogo para selecionar um vídeo."""
        path = filedialog.askopenfilename(title="Selecione um vídeo", filetypes=[("Vídeos", "*.mp4 *.avi *.mov *.mkv")])
        if path:
            self.video_path_global = path
            self.update_status(f"Arquivo: {os.path.basename(path)}", "success")
        else:
            self.video_path_global = ""
            self.update_status("Nenhum vídeo selecionado.", "warning")
            
    # <<< ALTERAÇÃO 2: Função totalmente reescrita para integrar com `definir_areas.py`
    def open_define_areas(self):
        """
        Abre o seletor de áreas usando o vídeo selecionado na UI, sem usar subprocess.
        """
        # 1. Determina qual a fonte do vídeo com base na seleção da UI
        source_type = self.video_source_type.get()
        video_source = ""

        if source_type == "local":
            video_source = self.video_path_global
            if not video_source:
                messagebox.showerror("Erro", "Por favor, selecione um arquivo de vídeo local primeiro.")
                return
        else: # "url"
            video_source = self.video_url_entry.get().strip()
            if not video_source:
                messagebox.showerror("Erro", "Por favor, insira uma URL de vídeo primeiro.")
                return

        # 2. Executa o seletor de áreas numa thread para não congelar a UI
        self.update_status("Abrindo janela para definir áreas...")
        
        # A função alvo da thread
        def run_area_selector():
            try:
                selector = AreaSelector()
                selector.run(video_source=video_source)
                # self.after(0, lambda: self.update_status("Janela de áreas fechada.", "info"))
            except Exception as e:
                # self.after(0, lambda: messagebox.showerror("Erro", f"Falha ao abrir definidor de áreas: {e}"))
                print(f"Erro na thread de AreaSelector: {e}")

        # Inicia a thread
        self.run_in_thread(run_area_selector)
        
    def get_current_video_source(self):
        """Valida e retorna a fonte de vídeo atual (local ou URL)."""
        if self.video_source_type.get() == "local":
            if not self.video_path_global:
                messagebox.showerror("Erro de Validação", "Selecione um arquivo de vídeo local.")
                return None
            return self.video_path_global
        else: # URL
            final_video_path = self.video_url_entry.get().strip()
            if not (final_video_path.startswith("http://") or final_video_path.startswith("https://")):
                messagebox.showerror("Erro de Validação", "Por favor, insira uma URL de vídeo válida.")
                return None
            return final_video_path
            
    def start_counting(self):
        """Inicia a validação e o processo de contagem."""
        # Validação do vídeo
        final_video_path = self.get_current_video_source()
        if not final_video_path:
            return

        # Validação do arquivo de áreas
        if not os.path.exists(AREAS_PATH):
            messagebox.showerror("Erro de Validação", "Arquivo 'areas.json' não encontrado. Use 'Definir Áreas' primeiro.")
            return

        # Validação e download do modelo
        selected_model = self.model_name.get()
        model_path = os.path.join(MODEL_DIR, selected_model)

        if not os.path.exists(model_path):
            if messagebox.askyesno("Modelo Ausente", f"O modelo '{selected_model}' não foi encontrado. Deseja baixá-lo agora?"):
                self.run_in_thread(self._download_and_start, args=(selected_model, final_video_path, model_path))
            else:
                self.update_status("Contagem cancelada. Modelo necessário.", "warning")
        else:
            self.update_status("Iniciando contagem em segundo plano...")
            self.run_in_thread(self.execute_counting_thread, args=(final_video_path, model_path))
            
    def _download_and_start(self, model_filename, video_source, model_full_path):
        """Baixa o modelo e, se bem-sucedido, inicia a contagem."""
        url = YOLO_MODELS.get(model_filename)
        self.show_progress_bar()
        self.update_status(f"Baixando {model_filename}...")
        try:
            response = requests.get(url, stream=True, timeout=30)
            response.raise_for_status()
            total_size = int(response.headers.get('content-length', 0))
            
            with open(model_full_path, 'wb') as f:
                downloaded = 0
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total_size > 0:
                        progress = downloaded / total_size
                        self.update_progress(progress)
            
            self.update_status(f"Modelo {model_filename} baixado!", "success")
            self.hide_progress_bar()
            self.execute_counting_thread(video_source, model_full_path)
        except requests.exceptions.RequestException as e:
            self.hide_progress_bar()
            self.update_status("Falha no download.", "error")
            messagebox.showerror("Erro de Download", f"Não foi possível baixar o modelo: {e}")
            if os.path.exists(model_full_path): os.remove(model_full_path)

    def execute_counting_thread(self, video_source, model_path):
        """Executa a função de contagem e atualiza a UI."""
        self.update_status("Processando vídeo... Uma janela pode ser aberta.", "processing")
        try:
            relatorio_path, saida_video_path = contar_veiculos(video_source, AREAS_PATH, model_path, show_video=True)
            self.last_report_path = relatorio_path
            msg = f"Contagem finalizada!\nRelatório: {relatorio_path}\nVídeo: {saida_video_path}"
            self.update_status("Processamento concluído com sucesso!", "success")
            messagebox.showinfo("Concluído", msg)
        except Exception as e:
            messagebox.showerror("Erro na Contagem", f"Ocorreu um erro: {e}")
            self.update_status(f"Erro na contagem: {e}", "error")
            
    def show_report(self):
        """Exibe o último relatório gerado."""
        if not (self.last_report_path and os.path.exists(self.last_report_path)):
            messagebox.showwarning("Relatório Não Encontrado", "Nenhum relatório foi gerado ou o arquivo foi movido.")
            return

        content = None
        try:
            with open(self.last_report_path, "r", encoding="utf-8") as f: content = f.read()
        except UnicodeDecodeError:
            try:
                with open(self.last_report_path, "r", encoding="latin-1") as f: content = f.read()
            except Exception as e:
                messagebox.showerror("Erro de Leitura", f"Não foi possível ler o arquivo: {e}")
                return
        except Exception as e:
             messagebox.showerror("Erro Inesperado", f"Ocorreu um erro: {e}")
             return

        if content:
            report_window = ctk.CTkToplevel(self)
            report_window.title("Relatório de Contagem")
            report_window.geometry("600x400")
            
            textbox = ctk.CTkTextbox(report_window, wrap="word", font=("Courier New", 12))
            textbox.pack(expand=True, fill="both", padx=10, pady=10)
            textbox.insert("1.0", content)
            textbox.configure(state="disabled")

    # --- Funções Utilitárias da UI ---
    def update_status(self, text, level="info"):
        """Atualiza o texto e a cor do rótulo de status."""
        colors = {"info": "gray", "success": "#2ECC71", "warning": "#F1C40F", "error": "#E74C3C", "processing": "#3498DB"}
        self.status_label.configure(text=text, text_color=colors.get(level, "gray"))

    def show_progress_bar(self):
        """Mostra e reinicia a barra de progresso."""
        self.progress_bar.set(0)
        self.progress_bar.grid(row=3, column=0, padx=20, pady=5, sticky="ew")

    def hide_progress_bar(self):
        """Esconde a barra de progresso."""
        self.progress_bar.grid_forget()

    def update_progress(self, value):
        """Atualiza o valor da barra de progresso."""
        self.progress_bar.set(value)

    def run_in_thread(self, target_func, args=()):
        """Executa uma função em uma nova thread para não bloquear a UI."""
        thread = threading.Thread(target=target_func, args=args, daemon=True)
        thread.start()

if __name__ == "__main__":
    app = App()
    app.mainloop()