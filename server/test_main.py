from fastapi.testclient import TestClient
import main

client = TestClient(main.app)

def test_read_main():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "Hello App"}

def test_predict_route():
    file_name = r'C:\Users\posttoska\Documents\detr-detection-project\server\images\cups.jpg'

    with open(file_name, "rb") as f:
        response = client.post("/predict",files = {"file": ("cups.jpg", f, "image/jpeg")})

    assert response.status_code == 200
    data = response.json()

    # fixed part of the contract
    assert data["message"] == "Hello from predict"
    assert "predictions" in data

    # collect & validate variable‑length outputs
    preds = data["predictions"]
    boxes = preds["boxes"]
    scores = preds["scores"]
    labels = preds["labels"]

    # shape/type invariants (do not depend on exact count)
    assert isinstance(boxes, list) and isinstance(scores, list) and isinstance(labels, list)
    assert len(boxes) == len(scores) == len(labels)
    assert all(isinstance(b, list) and len(b) == 4 for b in boxes)  # [x1,y1,x2,y2]
    assert all(isinstance(s, (float, int)) and 0.0 <= float(s) <= 1.0 for s in scores)
    assert all(isinstance(l, int) and 0 <= l < 23 for l in labels)  # 23 classes
    assert len(boxes) <= 25
