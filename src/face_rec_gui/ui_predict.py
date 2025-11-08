import tkinter as tk
from tkinter import messagebox
import cv2, numpy as np
from .config import AppConfig
from .db import load_embeddings
from .models import FaceModels

def _nearest_label(emb: np.ndarray, embeddings_db: dict, threshold: float) -> tuple[str, float]:
    best_label, best_dist = "Unknown", float("inf")
    for label, vecs in embeddings_db.items():
        d = float(np.linalg.norm(vecs - emb, axis=1).min())
        if d < best_dist:
            best_dist, best_label = d, label
    return (best_label if best_dist < threshold else "Unknown", best_dist)

def prediction_ui(cfg: AppConfig, models: FaceModels, embeddings_database: dict):
    def predict():
        if not embeddings_database:
            embeddings_database.update(load_embeddings(models, cfg.database_path))
        if not embeddings_database:
            messagebox.showerror("No data", "No labeled faces found. Use Labeling first.")
            return

        cap = cv2.VideoCapture(cfg.camera_index)
        if not cap.isOpened():
            messagebox.showerror("Camera Error", f"Cannot open camera index {cfg.camera_index}")
            return

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            boxes, _ = models.mtcnn.detect(frame_rgb)

            if boxes is not None:
                for box in boxes:
                    x1, y1, x2, y2 = [int(b) for b in box]
                    face = frame_rgb[y1:y2, x1:x2]
                    emb = models.face_to_embedding(face)
                    if emb is None:
                        continue
                    label_text, dist = _nearest_label(emb, embeddings_database, cfg.recognition_threshold)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
                    cv2.putText(frame, f"{label_text} ({dist:.2f})", (x1, y1-10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

            cv2.imshow("Prediction (Press 'q' to Quit)", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

    app = tk.Toplevel() if isinstance(tk._default_root, tk.Tk) else tk.Tk()
    app.title(cfg.window_titles.get("predict", "Face Recognition"))
    tk.Button(app, text="Start Prediction", command=predict, font=("Arial", 16)).pack(pady=20)
    app.mainloop()
