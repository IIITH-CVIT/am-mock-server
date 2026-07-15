import yaml

from app.core.config import Settings


def test_dataclass_defaults_match_config_yaml():
    """Guards against the two config files drifting apart silently. If this fails, it means
    that either config.yaml or config.py has been changed, and the dataclasses haven't been synced"""

    with open("config.yaml") as f:
        yaml_values = yaml.safe_load(f)
    defaults = Settings()
    identify, models = yaml_values["identify"], yaml_values["models"]

    assert defaults.database.path == yaml_values["database"]["path"]
    assert defaults.identify.face_recognition_threshold == identify["face_recognition_threshold"]
    assert (
        defaults.identify.fingerprint_recognition_threshold
        == identify["fingerprint_recognition_threshold"]
    )
    assert defaults.identify.default_n == identify["default_n"]
    assert defaults.models.embedder_model == models["embedder_model"]
    assert defaults.models.face_detector_score_threshold == models["face_detector_score_threshold"]

def test_missing_config_file_warns(capsys, tmp_path):
    from app.core.config import load_settings
    load_settings(path = str(tmp_path / "does-not-exist.yaml"))
    assert "WARNING" in capsys.readouterr().out