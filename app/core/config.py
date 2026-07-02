import os
from dataclasses import dataclass, field

import yaml

CONFIG_PATH = os.environ.get("CONFIG_PATH", "/app/config.yaml")

@dataclass
class ServerConfig:
    host: str = "0.0.0.0"
    port: int = 8000

@dataclass
class DatabaseConfig:
    path: str = "/app/data/db.sqlite"

@dataclass
class ModelsConfig:
    face_detector_path: str = "/app/models/face_detection_yunet_2023mar.onnx"
    face_recognizer_path: str = "/app/models/mobilefacenet.onnx"
    # square letterbox size fed to YuNet; the model has no built-in NMS, so
    # there's no separate nms_threshold to configure
    face_detector_input_size: int = 640
    face_detector_score_threshold: float = 0.5

@dataclass
class IdentifyConfig:
    # max normalized-L2 distance between query and stored vector to count as a match
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
            f"WARNING: config file not found at {path|r} - using built-in defaults."
            f"If this is unexpected, check CONFIG_PATH and the bind mount in compose.yml"
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