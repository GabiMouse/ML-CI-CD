from app import app
import pytest


@pytest.fixture
def client():
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client


def test_predictions_not_none(client):

    payload = {
        "sepal_length": 5.1,
        "sepal_width": 3.5,
        "petal_length": 1.4,
        "petal_width": 0.2
    }

    response = client.post('/predict', json=payload)
    assert response.status_code == 200
    pred = response.get_json()
    assert pred is not None

def test_predictions_length(client):

    payload = {
        "sepal_length": 5.1,
        "sepal_width": 3.5,
        "petal_length": 1.4,
        "petal_width": 0.2
    }

    response = client.post('/predict', json=payload)
    assert response.status_code == 200
    pred = response.get_json()
    assert len(pred) == 2

#udało się zrealizować dwa testy
