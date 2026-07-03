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