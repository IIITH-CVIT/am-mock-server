import numpy as np
import pytest
from unittest.mock import MagicMock, patch

from app.core.face_engine import FaceEngine, NoFaceDetectedError


def _fake_detector(faces=None):
    detector = MagicMock()
    detector.detect.return_value = (None, faces)
    return detector


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
