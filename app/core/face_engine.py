import cv2
import numpy as np
import onnxruntime

from app.core.config import settings

# ArcFace/AuraFace 112x112 reference landmarks, used to warp a detected face
# into a canonical frontal pose before the auraface embedder.
_REF_LANDMARKS = np.array(
    [
        [38.2946, 51.6963],
        [73.5318, 51.5014],
        [56.0252, 71.7366],
        [41.5493, 92.3655],
        [70.7299, 92.2041],
    ],
    dtype=np.float32,
)


class NoFaceDetectedError(Exception):
    pass


class FaceEngine:
    """Detects a face via cv2.FaceDetectorYN (YuNet) and computes an embedding
    via one of two never-mixed pairings, selected by cfg.embedder_model:

    - sface     — cv2.FaceRecognizerSF (native OpenCV 5, no onnxruntime),
                  128-dim L2-normalised, aligned via alignCrop()
    - auraface  — aurar100.onnx (ArcFace-style, via onnxruntime), 512-dim
                  L2-normalised, aligned via the 112x112 ArcFace reference
                  landmark warp (estimateAffinePartial2D + warpAffine)
    """

    ARCFACE_INPUT_SIZE = 112
    CROP_MARGIN = 0.20

    def __init__(self) -> None:
        cfg = settings.models
        size = cfg.face_detector_input_size
        self.detector = cv2.FaceDetectorYN.create(
            cfg.face_detector_path, "", (size, size),
            score_threshold=cfg.face_detector_score_threshold,
        )

        self.model_name = cfg.embedder_model
        if self.model_name == "auraface":
            self.recognizer = None
            self.session = onnxruntime.InferenceSession(
                cfg.face_recognizer_auraface_path, providers=["CPUExecutionProvider"]
            )
            self._input_name = self.session.get_inputs()[0].name
            self.embedding_dim = int(self.session.get_outputs()[0].shape[-1])
        else:
            self.session = None
            self.recognizer = cv2.FaceRecognizerSF.create(cfg.face_recognizer_path, "")
            self.embedding_dim = 128

    def _detect_face(self, image: np.ndarray) -> np.ndarray:
        h, w = image.shape[:2]
        self.detector.setInputSize((w, h))
        _, faces = self.detector.detect(image)
        if faces is None or len(faces) == 0:
            raise NoFaceDetectedError("no face detected in image")
        return faces[int(np.argmax(faces[:, 14]))]

    def _align_auraface(self, image: np.ndarray, face_row: np.ndarray) -> np.ndarray:
        landmarks = np.array(
            [[face_row[4 + 2 * k], face_row[5 + 2 * k]] for k in range(5)], dtype=np.float32
        )
        transform, _ = cv2.estimateAffinePartial2D(landmarks, _REF_LANDMARKS, method=cv2.LMEDS)
        if transform is not None:
            return cv2.warpAffine(
                image, transform, (self.ARCFACE_INPUT_SIZE, self.ARCFACE_INPUT_SIZE),
                flags=cv2.INTER_LINEAR, borderValue=0,
            )

        h, w = image.shape[:2]
        x, y, bw, bh = face_row[0:4]
        x1, y1 = max(0, int(x)), max(0, int(y))
        x2, y2 = min(w, int(x + bw)), min(h, int(y + bh))
        mx, my = int((x2 - x1) * self.CROP_MARGIN), int((y2 - y1) * self.CROP_MARGIN)
        crop = image[max(0, y1 - my):min(h, y2 + my), max(0, x1 - mx):min(w, x2 + mx)]
        if crop.size == 0:
            crop = image
        return cv2.resize(crop, (self.ARCFACE_INPUT_SIZE, self.ARCFACE_INPUT_SIZE))

    def embed(self, image_bytes: bytes) -> list[float]:
        arr = np.frombuffer(image_bytes, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("could not decode image")

        face_row = self._detect_face(img)

        if self.model_name == "auraface":
            aligned = self._align_auraface(img, face_row)
            rgb = cv2.cvtColor(aligned, cv2.COLOR_BGR2RGB)
            blob = (rgb.astype(np.float32) - 127.5) / 128.0
            blob = np.transpose(blob, (2, 0, 1))[np.newaxis, ...]
            vector = self.session.run(None, {self._input_name: blob})[0].flatten().astype(np.float32)
        else:
            aligned = self.recognizer.alignCrop(img, face_row)
            vector = self.recognizer.feature(aligned).flatten().astype(np.float32)

        norm = np.linalg.norm(vector)
        if norm > 0:
            vector = vector / norm
        return vector.tolist()


face_engine = FaceEngine()
