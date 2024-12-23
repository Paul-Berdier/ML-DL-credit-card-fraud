import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib

def train_ml_model(input_file, model_file):
    # Charger les données d'entraînement
    data = pd.read_csv(input_file)

    # Séparer les caractéristiques (features) et la cible (target)
    X = data.drop(columns=['Class'])
    y = data['Class']

    # Diviser les données en ensembles d'entraînement et de test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialiser et entraîner le modèle Random Forest
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Évaluer les performances sur l'ensemble de test
    predictions = model.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, predictions))
    print("Rapport de classification:\n", classification_report(y_test, predictions))

    # Sauvegarder le modèle entraîné
    joblib.dump(model, model_file)
    print(f"Modèle sauvegardé sous : {model_file}")
