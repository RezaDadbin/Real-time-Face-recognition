import tkinter as tk
from .config import load_config
from .models import FaceModels
from .ui_label import labeling_ui
from .ui_predict import prediction_ui

def main(cfg_path: str = "configs/default.yaml"):
    cfg = load_config(cfg_path)
    cfg.database_path.mkdir(parents=True, exist_ok=True)

    models = FaceModels(device=cfg.device)
    embeddings_db = {}  # global-ish container shared between UIs

    root = tk.Tk()
    root.title(cfg.window_titles.get("root", "Face Recognition System"))
    tk.Button(root, text="Labeling UI",
              command=lambda: labeling_ui(cfg, models, embeddings_db),
              font=("Arial", 16)).pack(pady=10)
    tk.Button(root, text="Prediction UI",
              command=lambda: prediction_ui(cfg, models, embeddings_db),
              font=("Arial", 16)).pack(pady=10)
    root.mainloop()

if __name__ == "__main__":
    main()
