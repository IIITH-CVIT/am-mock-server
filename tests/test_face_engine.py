from unittest.mock import MagicMock, patch

import cv2
import numpy as np
import pytest

from app.core.face_engine import FaceEngine, NoFaceDetectedError


def _fake_detector(faces=None):
    detector = MagicMock()
    detector.detect.return_value = (None, faces)
    return detector


def _encode_dummy_image(size=100):
    """A real, decodable JPEG so embed() exercises the actual cv2.imdecode path."""
    img = np.zeros((size, size, 3), dtype=np.uint8)
    ok, buf = cv2.imencode(".jpg", img)
    assert ok
    return buf.tobytes()


def _face_row(size=100):
    """A YuNet-shaped detection row: [x, y, w, h, 5x(landmark x, y), score]."""
    row = np.zeros(15, dtype=np.float32)
    row[0:4] = [size * 0.2, size * 0.2, size * 0.6, size * 0.6]
    landmarks = [
        (size * 0.35, size * 0.4), (size * 0.65, size * 0.4),
        (size * 0.5, size * 0.55), (size * 0.4, size * 0.7), (size * 0.6, size * 0.7),
    ]
    for k, (lx, ly) in enumerate(landmarks):
        row[4 + 2 * k], row[5 + 2 * k] = lx, ly
    row[14] = 0.99
    return row


def test_sface_engine_loads_native_recognizer():
    """Default embedder_model (sface) uses cv2.FaceRecognizerSF, not onnxruntime."""
    with patch("cv2.FaceDetectorYN.create", return_value=_fake_detector()), \
         patch("cv2.FaceRecognizerSF.create", return_value=MagicMock()) as mock_sface, \
         patch("onnxruntime.InferenceSession") as mock_ort:
        engine = FaceEngine()
        assert engine.model_name == "sface"
        assert engine.embedding_dim == 128
        mock_sface.assert_called_once()
        mock_ort.assert_not_called()


def test_auraface_engine_loads_onnxruntime_session(monkeypatch):
    """embedder_model=auraface loads aurar100.onnx via onnxruntime instead of cv2.FaceRecognizerSF."""
    from app.core import face_engine as fe_module

    monkeypatch.setattr(fe_module.settings.models, "embedder_model", "auraface")

    session = MagicMock()
    session.get_inputs.return_value = [MagicMock(name="input")]
    session.get_inputs.return_value[0].name = "input"
    session.get_outputs.return_value = [MagicMock(shape=[1, 512])]

    with patch("cv2.FaceDetectorYN.create", return_value=_fake_detector()), \
         patch("cv2.FaceRecognizerSF.create") as mock_sface, \
         patch("onnxruntime.InferenceSession", return_value=session):
        engine = FaceEngine()
        assert engine.model_name == "auraface"
        assert engine.embedding_dim == 512
        mock_sface.assert_not_called()


def test_no_face_detected_raises():
    with patch("cv2.FaceDetectorYN.create", return_value=_fake_detector(faces=None)), \
         patch("cv2.FaceRecognizerSF.create", return_value=MagicMock()):
        engine = FaceEngine()
        with pytest.raises(NoFaceDetectedError):
            engine._detect_face(np.zeros((10, 10, 3), dtype=np.uint8))


def test_embed_raises_valueerror_on_undecodable_image():
    with patch("cv2.FaceDetectorYN.create", return_value=_fake_detector()), \
         patch("cv2.FaceRecognizerSF.create", return_value=MagicMock()):
        engine = FaceEngine()
        with pytest.raises(ValueError, match="could not decode image"):
            engine.embed(b"not a real image")


def test_sface_embed_returns_normalized_128dim_vector():
    face_row = _face_row()
    recognizer = MagicMock()
    recognizer.alignCrop.return_value = np.zeros((112, 112, 3), dtype=np.uint8)
    # A non-unit vector, to prove embed() normalizes it rather than passing it through.
    recognizer.feature.return_value = np.array([[3.0, 4.0] + [0.0] * 126], dtype=np.float32)

    with patch("cv2.FaceDetectorYN.create", return_value=_fake_detector(faces=np.array([face_row]))), \
         patch("cv2.FaceRecognizerSF.create", return_value=recognizer):
        engine = FaceEngine()
        vector = engine.embed(_encode_dummy_image())

    assert len(vector) == 128
    assert vector[0] == pytest.approx(0.6)  # 3/5
    assert vector[1] == pytest.approx(0.8)  # 4/5
    assert np.linalg.norm(vector) == pytest.approx(1.0)


def test_auraface_embed_warps_landmarks_and_normalizes(monkeypatch):
    from app.core import face_engine as fe_module

    monkeypatch.setattr(fe_module.settings.models, "embedder_model", "auraface")

    session = MagicMock()
    session.get_inputs.return_value = [MagicMock(name="input")]
    session.get_inputs.return_value[0].name = "input"
    session.get_outputs.return_value = [MagicMock(shape=[1, 512])]
    session.run.return_value = [np.ones((1, 512), dtype=np.float32)]

    face_row = _face_row()
    with patch("cv2.FaceDetectorYN.create", return_value=_fake_detector(faces=np.array([face_row]))), \
         patch("cv2.FaceRecognizerSF.create"), \
         patch("onnxruntime.InferenceSession", return_value=session):
        engine = FaceEngine()
        vector = engine.embed(_encode_dummy_image())

    assert len(vector) == 512
    assert np.linalg.norm(vector) == pytest.approx(1.0)
    session.run.assert_called_once()


def test_align_auraface_falls_back_to_crop_when_transform_fails():
    """If estimateAffinePartial2D can't solve a transform (e.g. degenerate landmarks),
    _align_auraface must still return a usable 112x112 crop instead of raising."""
    with patch("cv2.FaceDetectorYN.create", return_value=_fake_detector()), \
         patch("cv2.FaceRecognizerSF.create", return_value=MagicMock()), \
         patch("cv2.estimateAffinePartial2D", return_value=(None, None)):
        engine = FaceEngine()
        img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        aligned = engine._align_auraface(img, _face_row())

    assert aligned.shape == (112, 112, 3)
