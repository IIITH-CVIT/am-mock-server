import json


def test_malformed_face_vector_returns_400(client):
    resp = client.post("api/v1/identify/", data = {
        "type": "face",
        "face_vector": "{not valid json}",
    })
    assert resp.status_code == 400
    assert "invalid vector format" in resp.json()["detail"]

def test_empty_face_vector_returns_400(client):
    resp = client.post("/api/v1/identify/", data = {"type": "face", "face_vector": "[]"})
    assert resp.status_code == 400

def test_face_vector_wrong_dimension_returns_400(client):
    resp = client.post("/api/v1/identify/", data={
        "type": "face",
        "face_vector": json.dumps([0.1] * 256),  # not 128 (sface)
    })
    assert resp.status_code == 400
    assert "expected" in resp.json()["detail"]

def test_face_vector_correct_dimension_reaches_search(client):
    resp = client.post("/api/v1/identify/", data={
        "type": "face",
        "face_vector": json.dumps([0.1] * 128),
    })
    assert resp.status_code == 200  # reaches the search, whatever the match outcome

def test_vector_exceeding_max_length_returns_400(client):
    resp = client.post("/api/v1/identify/", data={
        "type": "fingerprint",
        "fingerprint_vector": json.dumps([0.1] * 5000),  # over MAX_VECTOR_LENGTH
    })
    assert resp.status_code == 400
    assert "exceeds max allowed" in resp.json()["detail"]

def test_vector_at_max_length_boundary_ok(client):
    # exactly at the cap should not be rejected. Catches an off-by-one on the > vs >= comparison
    resp = client.post("/api/v1/identify/", data={
        "type": "fingerprint",
        "fingerprint_vector": json.dumps([0.1] * 4096),
    })
    assert resp.status_code == 200