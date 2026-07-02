import pytest 
from unittest.mock import MagicMock, patch 
from app.core.face_engine import FaceEngine 

def _fake_session_with_outputs(names):
    session = MagicMock()
    session.get_inputs.return_value = [MagicMock(name = "input")]
    session.get_outputs.return_value = [MagicMock(name = n) for n in names]
    for out, n in zip(session.get_outputs.return_value, name):
        out.name = n
    session.get_inputs.return_value[0].name = "input"

    return session 

def test_missing_output_name_fails_fast():
    incomplete = _fake_session_with_outputs(["cls_8", "obj_8", "bbox_8", "kps_8"])  # missing stride 16/32
    with patch("onnxruntime.InferenceSession", return_value=incomplete):
        with pytest.raises(RuntimeError, match="missing expected output tensor"):
            FaceEngine()

def test_correct_output_names_pass():
    names = [f"{p}_{s}" for s in (8, 16, 32) for p in ("cls", "obj", "bbox", "kps")]
    good_detector = _fake_session_with_outputs(names)
    good_recognizer = _fake_session_with_outputs(["embedding"])
    good_recognizer.get_outputs.return_value[0].shape = [1, 512]
    with patch("onnxruntime.InferenceSession", side_effect=[good_detector, good_recognizer]):
        engine = FaceEngine()  # should not raise
        assert engine.embedding_dim == 512