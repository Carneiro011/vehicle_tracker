import cv2
import numpy as np
import json
from tkinter import Tk, filedialog, messagebox
import os

class AreaSelector:
    def __init__(self, window_name="Definir Areas"):
        self.window_name = window_name
        self.FONT = cv2.FONT_HERSHEY_SIMPLEX
        self.COLORS = {
            "entry": (0, 255, 0),
            "exit": (0, 0, 255),
            "drawing": (0, 255, 255),
            "highlight": (255, 100, 0),
            "text": (255, 255, 255)
        }
        self.WINDOW_WIDTH = 1280
        self.WINDOW_HEIGHT = 720
        self.areas = [[], []]
        self.current_area_index = 0
        self.dragging_point_index = -1
        self.highlighted_point_index = -1
        self.original_frame = None
        self.display_frame = None
        self.original_dims = None

    def _create_tk_root(self):
        root = Tk()
        root.withdraw()
        root.wm_attributes("-topmost", False)
        return root

    def _refocus_window(self):
        cv2.setWindowProperty(self.window_name, cv2.WND_PROP_TOPMOST, 1)
        cv2.setWindowProperty(self.window_name, cv2.WND_PROP_TOPMOST, 0)

    def _mouse_callback(self, event, x, y, flags, param):
        self.highlighted_point_index = -1
        for i, p in enumerate(self.areas[self.current_area_index]):
            if np.linalg.norm(np.array(p) - np.array((x, y))) < 10:
                self.highlighted_point_index = i
                break

        if event == cv2.EVENT_LBUTTONDOWN and self.highlighted_point_index != -1:
            self.dragging_point_index = self.highlighted_point_index
        elif event == cv2.EVENT_LBUTTONUP:
            self.dragging_point_index = -1
        elif event == cv2.EVENT_LBUTTONDOWN and self.dragging_point_index == -1:
            self.areas[self.current_area_index].append([x, y])
        elif event == cv2.EVENT_RBUTTONDOWN and len(self.areas[self.current_area_index]) >= 3:
            root = self._create_tk_root()
            if self.current_area_index == 0:
                if messagebox.askyesno("Área de Saída", "Deseja definir uma área de SAÍDA também?", parent=root):
                    self.current_area_index = 1
                else:
                    messagebox.showinfo("Pronto", "Área de ENTRADA definida. Pressione 'S' para salvar.", parent=root)
            else:
                messagebox.showinfo("Concluído", "Área de SAÍDA definida. Pressione 'S' para salvar.", parent=root)
            self._refocus_window()

        if self.dragging_point_index != -1:
            self.areas[self.current_area_index][self.dragging_point_index] = [x, y]

    def _draw(self):
        self.display_frame = self.original_frame.copy()
        for i, area_points in enumerate(self.areas):
            if not area_points:
                continue
            color = self.COLORS["entry"] if i == 0 else self.COLORS["exit"]
            overlay = self.display_frame.copy()
            cv2.fillPoly(overlay, [np.array(area_points)], color)
            cv2.addWeighted(overlay, 0.3, self.display_frame, 0.7, 0, self.display_frame)
            cv2.polylines(self.display_frame, [np.array(area_points)], True, color, 2)
            for j, p in enumerate(area_points):
                pt_color = self.COLORS["highlight"] if j == self.highlighted_point_index and i == self.current_area_index else color
                cv2.circle(self.display_frame, tuple(p), 7, pt_color, -1)
        self._draw_ui()
        cv2.imshow(self.window_name, self.display_frame)

    def _draw_ui(self):
        area_name = "ENTRADA" if self.current_area_index == 0 else "SAIDA"
        color = self.COLORS["entry"] if self.current_area_index == 0 else self.COLORS["exit"]
        cv2.putText(self.display_frame, f"Definindo Area: {area_name}", (20, 40), self.FONT, 1, color, 2)

        instructions = [
            "Clique ESQUERDO para adicionar ou arrastar pontos.",
            "Clique DIREITO para finalizar a area atual.",
            "Z: Reiniciar area ATUAL | R: Reiniciar TUDO",
            "S: Salvar e Sair | ESC: Sair sem Salvar"
        ]
        for i, text in enumerate(instructions):
            cv2.putText(self.display_frame, text, (20, self.WINDOW_HEIGHT - 100 + i*25), self.FONT, 0.6, self.COLORS["text"], 1)

    def _select_video(self):
        root = self._create_tk_root()
        video_path = filedialog.askopenfilename(
            title="Selecione um vídeo para obter o frame de fundo",
            filetypes=[("Vídeos", "*.mp4 *.avi *.mov")],
            parent=root
        )
        self._refocus_window()
        return video_path

    def _load_areas(self):
        path = os.path.join("resultados", "areas.json")
        if os.path.exists(path):
            root = self._create_tk_root()
            if messagebox.askyesno("Carregar Áreas", "Arquivo 'areas.json' encontrado. Deseja carregar e editar as áreas existentes?", parent=root):
                with open(path, 'r') as f:
                    loaded_areas = json.load(f)
                sx = self.WINDOW_WIDTH / self.original_dims[0]
                sy = self.WINDOW_HEIGHT / self.original_dims[1]
                self.areas[0] = [[int(p[0] * sx), int(p[1] * sy)] for p in loaded_areas[0]]
                if len(loaded_areas) > 1:
                    self.areas[1] = [[int(p[0] * sx), int(p[1] * sy)] for p in loaded_areas[1]]
                self._refocus_window()
                return True
        return False

    def _save_areas(self):
        root = self._create_tk_root()
        if not self.areas[0]:
            messagebox.showwarning("Aviso", "A área de ENTRADA precisa ser definida para salvar.", parent=root)
            self._refocus_window()
            return

        sx = self.original_dims[0] / self.WINDOW_WIDTH
        sy = self.original_dims[1] / self.WINDOW_HEIGHT

        scaled_areas = [
            [[int(p[0] * sx), int(p[1] * sy)] for p in self.areas[0]],
            [[int(p[0] * sx), int(p[1] * sy)] for p in self.areas[1]] if self.areas[1] else []
        ]

        path = os.path.join("resultados", "areas.json")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w') as f:
            json.dump(scaled_areas, f, indent=2)

        msg = f"Área(s) salva(s) com sucesso em:\n{os.path.abspath(path)}"
        if not self.areas[1]:
            msg += "\n\n*Observação: Apenas a área de ENTRADA foi definida.*"
        messagebox.showinfo("Salvo", msg, parent=root)
        self._refocus_window()

    def run(self, video_source=None):
        video_path = video_source or self._select_video()
        if not video_path:
            print("Nenhum vídeo selecionado. Encerrando.")
            return

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            root = self._create_tk_root()
            messagebox.showerror("Erro", f"Não foi possível abrir o vídeo:\n{video_path}", parent=root)
            self._refocus_window()
            return

        ret, frame = cap.read()
        if not ret:
            root = self._create_tk_root()
            messagebox.showerror("Erro", "Não foi possível ler o primeiro frame do vídeo.", parent=root)
            cap.release()
            self._refocus_window()
            return

        self.original_dims = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        cap.release()

        self.original_frame = cv2.resize(frame, (self.WINDOW_WIDTH, self.WINDOW_HEIGHT))

        cv2.namedWindow(self.window_name)
        cv2.setMouseCallback(self.window_name, self._mouse_callback)
        self._refocus_window()

        self._load_areas()

        while True:
            # Detecta se o usuário clicou no X para fechar
            if cv2.getWindowProperty(self.window_name, cv2.WND_PROP_VISIBLE) < 1:
                root = self._create_tk_root()
                if messagebox.askyesno("Sair", "\nDeseja sair sem salvar?", parent=root):
                    break
                else:
                    self._refocus_window()
                    continue

            self._draw()
            key = cv2.waitKey(1) & 0xFF

            if key == 27:
                root = self._create_tk_root()
                if messagebox.askyesno("Sair", "Tem certeza que deseja sair sem salvar?", parent=root):
                    break
                self._refocus_window()
            elif key == ord('s'):
                self._save_areas()
                break
            elif key == ord('z'):
                self.areas[self.current_area_index] = []
            elif key == ord('r'):
                root = self._create_tk_root()
                if messagebox.askyesno("Reiniciar Tudo", "Tem certeza que deseja apagar TODAS as áreas?", parent=root):
                    self.areas = [[], []]
                    self.current_area_index = 0
                self._refocus_window()

        cv2.destroyAllWindows()


if __name__ == "__main__":
    app = AreaSelector()
    app.run()
