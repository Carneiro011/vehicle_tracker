import cv2
import numpy as np
import json
from tkinter import Tk, filedialog, messagebox
import os

class AreaSelector:
    """
    Uma classe para criar uma interface interativa com OpenCV para definir
    polígonos em um frame de vídeo. Permite desenhar, editar (arrastar pontos),
    reiniciar, carregar e salvar áreas.
    """
    def __init__(self, window_name="Definir Areas"):
        # --- Configurações da UI ---
        self.window_name = window_name
        self.FONT = cv2.FONT_HERSHEY_SIMPLEX
        self.COLORS = {
            "entry": (0, 255, 0),       # Verde
            "exit": (0, 0, 255),        # Vermelho
            "drawing": (0, 255, 255),   # Amarelo
            "highlight": (255, 100, 0), # Azul claro
            "text": (255, 255, 255),     # Branco
            "button_bg": (80, 80, 80),
            "button_text": (255, 255, 255)
        }
        self.WINDOW_WIDTH = 1280
        self.WINDOW_HEIGHT = 720

        # --- Estado da Aplicação ---
        self.areas = [[], []]  # [[area_entrada], [area_saida]]
        self.current_area_index = 0
        self.drawing_points = []
        self.dragging_point_index = -1
        self.highlighted_point_index = -1
        self.original_frame = None
        self.display_frame = None
        self.original_dims = None # (width, height) do vídeo original

    def _mouse_callback(self, event, x, y, flags, param):
        """Callback principal para todos os eventos do mouse."""
        # --- Destaque de Ponto (Hover) ---
        self.highlighted_point_index = -1
        for i, p in enumerate(self.areas[self.current_area_index]):
            if np.linalg.norm(np.array(p) - np.array((x, y))) < 10:
                self.highlighted_point_index = i
                break
        
        # --- Iniciar Arraste ---
        if event == cv2.EVENT_LBUTTONDOWN and self.highlighted_point_index != -1:
            self.dragging_point_index = self.highlighted_point_index
        
        # --- Parar Arraste ---
        elif event == cv2.EVENT_LBUTTONUP:
            self.dragging_point_index = -1

        # --- Adicionar Ponto ---
        elif event == cv2.EVENT_LBUTTONDOWN and self.dragging_point_index == -1:
            self.areas[self.current_area_index].append([x, y])

        # --- Finalizar Polígono (auto-fecha) ---
        elif event == cv2.EVENT_RBUTTONDOWN and len(self.areas[self.current_area_index]) >= 3:
            if self.current_area_index == 0:
                self.current_area_index = 1
                messagebox.showinfo("Próxima Área", "Área de ENTRADA definida. Agora defina a área de SAÍDA.")
            else:
                messagebox.showinfo("Concluído", "Área de SAÍDA definida. Pressione 'S' para salvar.")

        # --- Lógica de Arrastar ---
        if self.dragging_point_index != -1:
            self.areas[self.current_area_index][self.dragging_point_index] = [x, y]

    def _draw(self):
        """Desenha tudo na tela: frame, polígonos e UI."""
        self.display_frame = self.original_frame.copy()

        # Desenha ambos os polígonos (entrada e saída)
        for i, area_points in enumerate(self.areas):
            if not area_points:
                continue
            
            color = self.COLORS["entry"] if i == 0 else self.COLORS["exit"]
            
            # Preenchimento semi-transparente
            overlay = self.display_frame.copy()
            cv2.fillPoly(overlay, [np.array(area_points)], color)
            cv2.addWeighted(overlay, 0.3, self.display_frame, 0.7, 0, self.display_frame)
            
            # Contorno do polígono
            cv2.polylines(self.display_frame, [np.array(area_points)], True, color, 2)
            
            # Pontos do polígono
            for j, p in enumerate(area_points):
                pt_color = self.COLORS["highlight"] if j == self.highlighted_point_index and i == self.current_area_index else color
                cv2.circle(self.display_frame, tuple(p), 7, pt_color, -1)
        
        self._draw_ui()
        cv2.imshow(self.window_name, self.display_frame)

    def _draw_ui(self):
        """Desenha os textos de instrução e botões na tela."""
        # --- Textos de Instrução ---
        area_name = "ENTRADA" if self.current_area_index == 0 else "SAIDA"
        color = self.COLORS["entry"] if self.current_area_index == 0 else self.COLORS["exit"]
        
        cv2.putText(self.display_frame, f"Definindo Area: {area_name}", (20, 40), self.FONT, 1, color, 2)
        
        instructions = [
            "Clique ESQUERDO para adicionar ou arrastar pontos.",
            "Clique DIREITO para finalizar a área atual.",
            "Z: Reiniciar area ATUAL | R: Reiniciar TUDO",
            "S: Salvar e Sair | ESC: Sair sem Salvar"
        ]
        for i, text in enumerate(instructions):
            cv2.putText(self.display_frame, text, (20, self.WINDOW_HEIGHT - 100 + i*25), self.FONT, 0.6, self.COLORS["text"], 1)

    def _select_video(self):
        """Abre a caixa de diálogo para selecionar o vídeo."""
        root = Tk()
        root.withdraw()
        video_path = filedialog.askopenfilename(
            title="Selecione um vídeo para obter o frame de fundo",
            filetypes=[("Vídeos", "*.mp4 *.avi *.mov")]
        )
        return video_path
    
    def _load_areas(self):
        """Carrega áreas de um arquivo JSON se existir."""
        path = os.path.join("resultados", "areas.json")
        if os.path.exists(path):
            if messagebox.askyesno("Carregar Áreas", "Arquivo 'areas.json' encontrado. Deseja carregar e editar as áreas existentes?"):
                with open(path, 'r') as f:
                    loaded_areas = json.load(f)
                
                # Escala os pontos carregados para as dimensões da janela de exibição
                sx = self.WINDOW_WIDTH / self.original_dims[0]
                sy = self.WINDOW_HEIGHT / self.original_dims[1]
                
                self.areas[0] = [[int(p[0] * sx), int(p[1] * sy)] for p in loaded_areas[0]]
                self.areas[1] = [[int(p[0] * sx), int(p[1] * sy)] for p in loaded_areas[1]]
                return True
        return False
        
    def _save_areas(self):
        """Salva as áreas no arquivo JSON, escalando de volta para as dimensões originais."""
        if not self.areas[0] or not self.areas[1]:
            messagebox.showwarning("Aviso", "Ambas as áreas (ENTRADA e SAIDA) precisam ser definidas para salvar.")
            return

        # Escala os pontos da janela de exibição para as dimensões do vídeo original
        sx = self.original_dims[0] / self.WINDOW_WIDTH
        sy = self.original_dims[1] / self.WINDOW_HEIGHT

        scaled_areas = [
            [[int(p[0] * sx), int(p[1] * sy)] for p in self.areas[0]],
            [[int(p[0] * sx), int(p[1] * sy)] for p in self.areas[1]]
        ]

        path = os.path.join("resultados", "areas.json")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w') as f:
            json.dump(scaled_areas, f, indent=2)
        
        messagebox.showinfo("Salvo", f"Áreas salvas com sucesso em:\n{os.path.abspath(path)}")

    def run(self):
        """Inicia e gerencia o loop principal da aplicação."""
        video_path = self._select_video()
        if not video_path:
            return

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            messagebox.showerror("Erro", "Não foi possível abrir o vídeo.")
            return

        ret, frame = cap.read()
        if not ret:
            messagebox.showerror("Erro", "Não foi possível ler o primeiro frame do vídeo.")
            cap.release()
            return
        
        self.original_dims = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        cap.release()

        self.original_frame = cv2.resize(frame, (self.WINDOW_WIDTH, self.WINDOW_HEIGHT))
        
        cv2.namedWindow(self.window_name)
        cv2.setMouseCallback(self.window_name, self._mouse_callback)

        self._load_areas() # Tenta carregar áreas existentes

        while True:
            self._draw()
            key = cv2.waitKey(1) & 0xFF

            if key == 27: # ESC
                if messagebox.askyesno("Sair", "Tem certeza que deseja sair sem salvar?"):
                    break
            
            elif key == ord('s'): # Salvar
                self._save_areas()
                break
                
            elif key == ord('z'): # Reiniciar área ATUAL
                self.areas[self.current_area_index] = []
            
            elif key == ord('r'): # Reiniciar TUDO
                if messagebox.askyesno("Reiniciar Tudo", "Tem certeza que deseja apagar TODAS as áreas?"):
                    self.areas = [[], []]
                    self.current_area_index = 0

        cv2.destroyAllWindows()


if __name__ == "__main__":
    app = AreaSelector()
    app.run()
