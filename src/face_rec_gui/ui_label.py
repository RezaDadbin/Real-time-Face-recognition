import tkinter as tk
from tkinter import simpledialog, messagebox
import cv2, os
from pathlib import Path
from .config import AppConfig
from .db import load_embeddings
from .models import FaceModels

def labeling_ui(cfg: AppConfig, models: FaceModels, embeddings_database: dict) -> dict:
    def capture_and_save():
        label = simpledialog.askstring("Input Label", "Enter the name for this person:")
        if not label:
            messagebox.showerror("Error", "Label cannot be empty!")
            return

        save_path = Path(cfg.database_path) / label
        save_path.mkdir(parents=True, exist_ok=True)

        cap = cv2.VideoCapture(cfg.camera_index)
        if not cap.isOpened():
            messagebox.showerror("Camera Error", f"Cannot open camera index {cfg.camera_index}")
            return

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            cv2.imshow("Press 's' to Save | 'q' to Quit", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('s'):
                img_name = f"{label}_{len(os.listdir(save_path))}.jpg"
                img_path = save_path / img_name
                cv2.imwrite(str(img_path), frame)
                messagebox.showinfo("Saved", f"Image: {img_path}")
            elif key == ord('q'):
                break
        cap.release()
        cv2.destroyAllWindows()

        # refresh embeddings
        embeddings_database.clear()
        embeddings_database.update(load_embeddings(models, cfg.database_path))
        messagebox.showinfo("Info", f"Loaded embeddings for {len(embeddings_database)} classes.")

    app = tk.Toplevel() if isinstance(tk._default_root, tk.Tk) else tk.Tk()
    app.title(cfg.window_titles.get("label", "Face Labeling"))
    tk.Button(app, text="Start Labeling", command=capture_and_save, font=("Arial", 16)).pack(pady=20)
    app.mainloop()
    return embeddings_database
