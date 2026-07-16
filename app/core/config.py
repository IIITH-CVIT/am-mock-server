import os
from dataclasses import dataclass, field

import yaml

CONFIG_PATH = os.environ.get("CONFIG_PATH", "config.yaml")

@dataclass
class ServerConfig:
    host: str = "0.0.0.0"
    port: int = 8000

@dataclass
class DatabaseConfig:
    path: str = "./data/db.sqlite"

@dataclass
class ModelsConfig:
    face_detector_path: str = "./models/face_detection_yunet_2026may.onnx"
    face_recognizer_path: str = "./models/face_recognition_sface_2021dec.onnx"
    face_recognizer_auraface_path: str = "./models/aurar100.onnx"
    # sface (128-dim, native cv2.FaceRecognizerSF) | auraface (512-dim, onnxruntime)
    embedder_model: str = "sface"
    # initial square size passed to cv2.FaceDetectorYN.create(); actual
    # detection runs at each image's real dimensions via setInputSize()
    face_detector_input_size: int = 640
    face_detector_score_threshold: float = 0.5

@dataclass
class IdentifyConfig:
    # Max normalized-L2 distance between query and stored vector to count as a
    # match. Carried over from the mobilefacenet pipeline — may need empirical
    # retuning now that vectors come from SFace's/AuraFace's embedding space.
    face_recognition_threshold: float = 0.8
    fingerprint_recognition_threshold: float = 0.7
    default_n: int = 10

@dataclass
class Settings:
    server: ServerConfig = field(default_factory=ServerConfig)
    database: DatabaseConfig = field(default_factory=DatabaseConfig)
    models: ModelsConfig = field(default_factory=ModelsConfig)
    identify: IdentifyConfig = field(default_factory=IdentifyConfig)

def load_settings(path: str = CONFIG_PATH) -> Settings:
    if not os.path.exists(path):
        print(
            f"WARNING: config file not found at {path!r} - using built-in defaults."
            f"If this is unexpected, check CONFIG_PATH and that config.yaml sits next to the binary."
        )
        raw = {}
    else:
        with open(path, "r") as f:
            raw = yaml.safe_load(f) or {}

    return Settings(
        server=ServerConfig(**raw.get("server", {})),
        database=DatabaseConfig(**raw.get("database", {})),
        models=ModelsConfig(**raw.get("models", {})),
        identify=IdentifyConfig(**raw.get("identify", {})),
    )

settings = load_settings()