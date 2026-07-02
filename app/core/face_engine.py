import cv2
import numpy as np
import onnxruntime

from app.core.config import settings

# ArcFace/MobileFaceNet 112x112 reference landmarks, used to warp a detected
# face into a canonical frontal pose before embedding.
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
    """Detects a face (YuNet) and computes an embedding (MobileFaceNet) via onnxruntime.

    Mirrors the am-master-server auraface pipeline: manual multi-scale YuNet
    decode (letterbox to a square input, strides 8/16/32), 5-point landmark
    alignment into the 112x112 ArcFace pose, BGR->RGB, (x-127.5)/128 normalize.
    """

    ARCFACE_INPUT_SIZE = 112
    YUNET_STRIDES = (8, 16, 32)
    CROP_MARGIN = 0.20

    def __init__(self) -> None:
        cfg = settings.models
        self.yunet_input_size = cfg.face_detector_input_size
        self.score_threshold = cfg.face_detector_score_threshold

        self.detector_session = onnxruntime.InferenceSession(
            cfg.face_detector_path, providers=["CPUExecutionProvider"]
        )
        self.detector_input_name = self.detector_session.get_inputs()[0].name
        self.detector_output_names = [o.name for o in self.detector_session.get_outputs()]

        self.session = onnxruntime.InferenceSession(
            cfg.face_recognizer_path, providers=["CPUExecutionProvider"]
        )
        self.model_name = "mobilefacenet"
        self._input_name = self.session.get_inputs()[0].name
        self.embedding_dim = int(self.session.get_outputs()[0].shape[-1])

    def _yunet_preprocess(self, image: np.ndarray) -> dict:
        """Letterbox resize into a square canvas, BGR uint8 as float, NCHW, no normalization."""
        h, w = image.shape[:2]
        scale = min(self.yunet_input_size / w, self.yunet_input_size / h)
        nw, nh = int(w * scale), int(h * scale)

        resized = cv2.resize(image, (nw, nh))
        canvas = np.zeros((self.yunet_input_size, self.yunet_input_size, 3), dtype=np.uint8)
        canvas[:nh, :nw, :] = resized

        blob = canvas.astype(np.float32).transpose(2, 0, 1)
        batched = np.expand_dims(blob, axis=0)
        return {"input": batched, "scale": scale, "img_shape": (h, w)}

    def _yunet_postprocess(self, outputs_by_name: dict, scale: float, img_shape: tuple) -> list[dict]:
        h, w = img_shape
        results = []

        for stride in self.YUNET_STRIDES:
            cls = outputs_by_name[f"cls_{stride}"][0]
            obj = outputs_by_name[f"obj_{stride}"][0]
            bbox = outputs_by_name[f"bbox_{stride}"][0]
            kps = outputs_by_name[f"kps_{stride}"][0]

            fm_width = self.yunet_input_size // stride

            cls_scores = np.clip(cls[:, 0], 0.0, 1.0)
            obj_scores = np.clip(obj[:, 0], 0.0, 1.0)
            scores = np.sqrt(cls_scores * obj_scores)

            for i in np.where(scores > self.score_threshold)[0]:
                col = int(i % fm_width)
                row = int(i // fm_width)

                cx = (col + bbox[i, 0]) * stride
                cy = (row + bbox[i, 1]) * stride
                bw = np.exp(bbox[i, 2]) * stride
                bh = np.exp(bbox[i, 3]) * stride

                x1 = (cx - bw / 2.0) / scale
                y1 = (cy - bh / 2.0) / scale
                x2 = (cx + bw / 2.0) / scale
                y2 = (cy + bh / 2.0) / scale

                bbox_out = [
                    max(0, min(float(x1), w)),
                    max(0, min(float(y1), h)),
                    max(0, min(float(x2), w)),
                    max(0, min(float(y2), h)),
                ]

                landmarks = []
                for k in range(5):
                    lx = (col + kps[i, 2 * k]) * stride / scale
                    ly = (row + kps[i, 2 * k + 1]) * stride / scale
                    landmarks.append(
                        [max(0, min(float(lx), w)), max(0, min(float(ly), h))]
                    )

                results.append({"bbox": bbox_out, "landmarks": landmarks, "score": float(scores[i])})

        return results

    def _detect_face(self, image: np.ndarray) -> dict:
        inputs = self._yunet_preprocess(image)
        outputs = self.detector_session.run(
            self.detector_output_names, {self.detector_input_name: inputs["input"]}
        )
        outputs_by_name = dict(zip(self.detector_output_names, outputs))
        results = self._yunet_postprocess(outputs_by_name, inputs["scale"], inputs["img_shape"])
        if not results:
            raise NoFaceDetectedError("no face detected in image")
        return max(results, key=lambda r: r["score"])

    def _align_face(self, image: np.ndarray, landmarks: list) -> np.ndarray | None:
        lm = np.array(landmarks, dtype=np.float32)
        transform, _ = cv2.estimateAffinePartial2D(lm, _REF_LANDMARKS, method=cv2.LMEDS)
        if transform is None:
            return None
        return cv2.warpAffine(
            image,
            transform,
            (self.ARCFACE_INPUT_SIZE, self.ARCFACE_INPUT_SIZE),
            flags=cv2.INTER_LINEAR,
            borderValue=0,
        )

    def _crop_and_align(self, image: np.ndarray, detection: dict) -> np.ndarray:
        h, w = image.shape[:2]
        x1, y1, x2, y2 = (int(v) for v in detection["bbox"])
        if x2 < x1:
            x1, x2 = x2, x1
        if y2 < y1:
            y1, y2 = y2, y1
        x1, y1 = max(0, min(x1, w)), max(0, min(y1, h))
        x2, y2 = max(0, min(x2, w)), max(0, min(y2, h))
        if x2 <= x1 or y2 <= y1:
            raise ValueError("invalid face bbox")

        margin_x = int((x2 - x1) * self.CROP_MARGIN)
        margin_y = int((y2 - y1) * self.CROP_MARGIN)
        crop = image[
            max(0, y1 - margin_y) : min(h, y2 + margin_y),
            max(0, x1 - margin_x) : min(w, x2 + margin_x),
        ]
        if crop.size == 0:
            raise ValueError("invalid face crop")

        landmarks = detection.get("landmarks")
        if landmarks and len(landmarks) == 5:
            aligned = self._align_face(image, landmarks)
            if aligned is not None:
                return aligned
        return cv2.resize(crop, (self.ARCFACE_INPUT_SIZE, self.ARCFACE_INPUT_SIZE))

    def embed(self, image_bytes: bytes) -> list[float]:
        arr = np.frombuffer(image_bytes, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("could not decode image")

        detection = self._detect_face(img)
        face_img = self._crop_and_align(img, detection)

        face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
        blob = (face_img.astype(np.float32) - 127.5) / 128.0
        blob = np.transpose(blob, (2, 0, 1))[np.newaxis, ...]

        output = self.session.run(None, {self._input_name: blob})[0]
        vector = output.flatten().astype(np.float32)
        norm = np.linalg.norm(vector)
        if norm > 0:
            vector = vector / norm
        return vector.tolist()


face_engine = FaceEngine()
