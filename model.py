import pickle
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

iris = load_iris()
X, y = iris.data, iris.target

#podział na zbiory
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#normalizacja
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

#inicjalizacja+trening
model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)

#predykcja
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

with open("model.pkl", "wb") as f:
    pickle.dump((scaler, model), f)

print("Model został zapisany!")