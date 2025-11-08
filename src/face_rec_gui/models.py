from __future__ import annotations
import numpy as np
import torch
import cv2
from facenet_pytorch import MTCNN, InceptionResnetV1

class FaceModels:
    def __init__(self, device: str = "cpu"):
        self.device = torch.device(device)
        self.mtcnn = MTCNN(keep_all=True, device=self.device)
        self.resnet = InceptionResnetV1(pretrained='vggface2').eval().to(self.device)

    @torch.inference_mode()
    def face_to_embedding(self, face_rgb: np.ndarray) -> np.ndarray | None:
        """Return L2-normalized 512-D embedding from an RGB face crop."""
        if face_rgb is None or face_rgb.size == 0:
            return None
        face_resized = cv2.resize(face_rgb, (160, 160))
        tensor = torch.from_numpy(face_resized).permute(2, 0, 1).float() / 255.0
        tensor = tensor.unsqueeze(0).to(self.device)
        emb = self.resnet(tensor).squeeze(0).cpu().numpy()
        n = np.linalg.norm(emb)
        return emb / n if n > 0 else None
