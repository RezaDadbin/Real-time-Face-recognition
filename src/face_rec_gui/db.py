from __future__ import annotations
from pathlib import Path
import os, cv2, numpy as np
from typing import Dict
from .models import FaceModels

EmbDB = Dict[str, np.ndarray]  # label -> (N,512) array

def load_embeddings(models: FaceModels, database_path: Path) -> EmbDB:
    db: EmbDB = {}
    if not database_path.is_dir():
        return db
    for label in os.listdir(database_path):
        label_path = database_path / label
        if not label_path.is_dir():
            continue
        vecs = []
        for img_name in os.listdir(label_path):
            img_path = label_path / img_name
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            boxes, _ = models.mtcnn.detect(img_rgb)
            if boxes is None or len(boxes) == 0:
                continue
            x1, y1, x2, y2 = [int(b) for b in boxes[0]]
            face = img_rgb[y1:y2, x1:x2]
            emb = models.face_to_embedding(face)
            if emb is not None:
                vecs.append(emb)
        if vecs:
            db[label] = np.stack(vecs, axis=0)
    return db
