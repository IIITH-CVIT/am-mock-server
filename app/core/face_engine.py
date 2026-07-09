import os

import cv2
import numpy as np
import onnxruntime

from app.core.config import settings

# dlib + its bundled model weights are imported lazily so the server still boots
# (with only the mobilefacenet backend) if the dlib wheel isn't installed — e.g.
# in the unit-test environment, which has no dlib. When present, they give us the
# second, dlib backend used to enroll a 128-dim descriptor alongside the 512-dim
# mobilefacenet one. See DlibEngine below.
try:
    import dlib as _dlib
    import face_recognition_models as _face_rec_models
except ModuleNotFoundError:  # pragma: no cover - depends on install profile
    _dlib = None
    _face_rec_models = None

# dlib ResNet descriptors are always 128-dim (raw, not L2-normalized). Exposed as
# a module constant so identify() can recognize a dlib query by its length without
# needing a live DlibEngine instance.
DLIB_EMBEDDING_DIM = 128

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

        self.detector_session = self._load_onnx_session(
            cfg.face_detector_path, "Face detector (YuNet)"
        )
        self.detector_input_name = self.detector_session.get_inputs()[0].name
        self.detector_output_names = [o.name for o in self.detector_session.get_outputs()]
        self._validate_detector_outputs(cfg.face_detector_path)

        self.session = self._load_onnx_session(
            cfg.face_recognizer_path, "Face recognizer (MobileFaceNet)"
        )
        self.model_name = "mobilefacenet"
        self._input_name = self.session.get_inputs()[0].name
        self.embedding_dim = int(self.session.get_outputs()[0].shape[-1])

    @staticmethod
    def _load_onnx_session(path: str, label: str) -> onnxruntime.InferenceSession:
        """Load an ONNX model, failing fast with an actionable message.

        FaceEngine() is instantiated at import time (module scope), so a missing
        or unreadable model file otherwise crashes container boot with a raw
        onnxruntime error. The overwhelmingly common cause on a fresh checkout is
        the ./models bind mount not being wired up — name that explicitly, and
        distinguish it from a present-but-corrupt file.
        """
        try:
            return onnxruntime.InferenceSession(path, providers=["CPUExecutionProvider"])
        except Exception as exc:
            if not os.path.exists(path):
                raise RuntimeError(
                    f"{label} model not found at {path!r}. Check that ./models is "
                    f"bind-mounted into the container (see compose.yml / README) and "
                    f"that the matching models.*_path in config.yaml points at the "
                    f"correct file."
                ) from exc
            raise RuntimeError(
                f"Failed to load {label} model at {path!r}: {exc}. The file exists "
                f"but onnxruntime could not load it — it may be corrupt or not a "
                f"valid ONNX export."
            ) from exc

    def _validate_detector_outputs(self, model_path: str) -> None:
        """Fail fast at startup, not mid-request, if the detector ONNX export doesn't have the tensor names _yunet_postprocess expects."""

        expected = {
            f"{prefix}_{stride}"
            for stride in self.YUNET_STRIDES
            for prefix in ("cls", "obj", "bbox", "kps")
        }

        missing = expected - set(self.detector_output_names)

        if missing:
            raise RuntimeError(
                f"Face detector model at {model_path!r} is missing expected "
                f"output tensor(s) {sorted(missing)}. This model export doesn't "
                f"match the YuNet variant this engine expects (strides "
                f"{self.YUNET_STRIDES}, actual outputs: {self.detector_output_names}). "
                f"Check config.yaml's models.face_detector_path points at the "
                f"correct model file."
            )

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


class DlibUnavailableError(RuntimeError):
    """Raised when a DlibEngine is requested but the dlib packages aren't installed."""


class DlibEngine:
    """Detects a face (dlib HOG) and computes a 128-dim descriptor (dlib ResNet).

    Mirrors am-master-server's DlibBackend and am-mock-client's DlibFaceDetector +
    DlibEmbedder exactly: HOG detection with number_of_times_to_upsample=1, a
    5-point shape predictor for alignment, and a raw (NOT L2-normalized) 128-dim
    ResNet descriptor. Enrolling this alongside the mobilefacenet embedding is what
    lets a client using the default dlib pairing match against this server.

    The two .dat weight files ship inside the `face_recognition_models` pip
    package (not ./models), so nothing extra needs to be bind-mounted.
    """

    DIM = DLIB_EMBEDDING_DIM

    def __init__(self, num_upsamples: int = 1, num_jitters: int = 1, threshold: float = 0.3) -> None:
        if _dlib is None or _face_rec_models is None:
            raise DlibUnavailableError(
                "The dlib backend needs the 'dlib' and 'face_recognition_models' "
                "packages, which aren't installed. Rebuild the image so the "
                "Containerfile installs them, or accept mobilefacenet-only enrollment."
            )
        self.model_name = "dlib"
        self.embedding_dim = self.DIM
        self.num_upsamples = num_upsamples
        self.num_jitters = num_jitters
        self.threshold = threshold

        models_dir = os.path.join(os.path.dirname(_face_rec_models.__file__), "models")
        resnet_path = os.path.join(models_dir, "dlib_face_recognition_resnet_model_v1.dat")
        predictor_path = os.path.join(models_dir, "shape_predictor_5_face_landmarks.dat")
        for label, path in (
            ("dlib face-recognition ResNet weights", resnet_path),
            ("dlib 5-point shape predictor", predictor_path),
        ):
            if not os.path.exists(path):
                raise DlibUnavailableError(
                    f"{label} not found at {path!r}. These ship inside the "
                    f"'face_recognition_models' package; reinstall it "
                    f"(pip install --force-reinstall face_recognition_models)."
                )

        self._face_encoder = _dlib.face_recognition_model_v1(resnet_path)
        self._shape_predictor = _dlib.shape_predictor(predictor_path)
        self._detector = _dlib.get_frontal_face_detector()

    def embed(self, image_bytes: bytes) -> list[float]:
        arr = np.frombuffer(image_bytes, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("could not decode image")

        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        dets, scores, _ = self._detector.run(rgb, self.num_upsamples, self.threshold)
        if not dets:
            raise NoFaceDetectedError("no face detected in image")

        best = int(np.argmax(scores))
        shape = self._shape_predictor(rgb, dets[best])
        descriptor = self._face_encoder.compute_face_descriptor(rgb, shape, self.num_jitters)
        # RAW descriptor — do NOT L2-normalize (matches DlibBackend's raw-L2 cutoff).
        return np.asarray(descriptor, dtype=np.float32).tolist()


face_engine = FaceEngine()

# The dlib engine is optional: if the dlib packages aren't installed the server
# still runs, enrolling only the mobilefacenet embedding. When it loads, every
# registration gets BOTH a 512-dim mobilefacenet and a 128-dim dlib embedding.
try:
    dlib_engine: DlibEngine | None = DlibEngine()
except DlibUnavailableError as exc:
    dlib_engine = None
    print(
        f"WARNING: dlib backend unavailable — registrations will store only the "
        f"mobilefacenet (512-dim) embedding, not the dlib (128-dim) one. Reason: {exc}"
    )
