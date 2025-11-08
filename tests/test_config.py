from src.face_rec_gui.config import load_config

def test_load_config_defaults():
    cfg = load_config("configs/default.yaml")
    assert cfg.database_path is not None
    assert cfg.recognition_threshold > 0
