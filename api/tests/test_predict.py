def test_predict_positive_sentiment(client):
    response = client.post("/predict", json={"text": "I love this product, it's amazing!"})
    assert response.status_code == 200
    body = response.json()
    assert body["label"] in ["POSITIVE", "NEGATIVE"]
    assert 0.0 <= body["score"] <= 1.0


def test_predict_negative_sentiment(client):
    response = client.post("/predict", json={"text": "This is the worst experience ever."})
    assert response.status_code == 200
    assert response.json()["label"] == "NEGATIVE"


def test_predict_rejects_empty_text(client):
    response = client.post("/predict", json={"text": ""})
    assert response.status_code == 422


def test_predict_rejects_missing_field(client):
    response = client.post("/predict", json={})
    assert response.status_code == 422