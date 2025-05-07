from flask import Flask, jsonify, request
import pickle
from pydantic import BaseModel, ValidationError
import numpy as np

with open("model.pkl", "rb") as f:
    scaler, model = pickle.load(f)

app = Flask(__name__)

class IrisInput(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float

@app.route('/')
def hello_world():  # put application's code here
    return {"message": "Witaj w NTPD3!"}

@app.post("/predict")
def predict():
    try:

        data = request.get_json()

        # Sprawdzenie wymaganych pol
        required_fields = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
        missing_fields = [field for field in required_fields if field not in data]

        if missing_fields:
            return jsonify({
                "error": "Brak wymaganych pól",
                "missing_fields": missing_fields
            }), 400

        #walidacja danych
        iris = IrisInput(**data)

        #konwersja danych na tablicę NumPy
        input_data = np.array([[iris.sepal_length, iris.sepal_width, iris.petal_length, iris.petal_width]])

        #standaryzacja
        input_data_scaled = scaler.transform(input_data)

        #predykcja
        prediction = model.predict(input_data_scaled)
        predicted_class = int(prediction[0])

        #mapowanie klasy
        class_names = ["setosa", "versicolor", "virginica"]
        predicted_species = class_names[predicted_class]

        return jsonify({"predicted_class": predicted_class, "predicted_species": predicted_species})

    except ValidationError as e:
        return jsonify({"error": "Nieprawidłowe dane wejściowe", "details": e.errors()}), 400

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/info', methods=["GET"])
def model_info():
    info = {
        "model_type": type(model).__name__,
        "num_features": len(model.coef_[0]) if hasattr(model, "coef_") else "Unknown",
        "scaler_type": type(scaler).__name__,
        "class_names": ["setosa", "versicolor", "virginica"]
    }
    return jsonify(info)

@app.route('/health', methods=["GET"])
def health_check():
    return jsonify({"status": "ok"})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
