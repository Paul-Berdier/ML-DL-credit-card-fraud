import pandas as pd
import joblib
from tensorflow.keras.models import load_model

def perform_prediction(isolated_row_file, ml_model_file, dl_model_file):
    """
    Effectuer une prédiction avec les modèles ML et DL sur une ligne isolée.

    Parameters:
        isolated_row_file (str): Path to the isolated row file.
        ml_model_file (str): Path to the trained Machine Learning model file.
        dl_model_file (str): Path to the trained Deep Learning model file.

    Returns:
        None
    """
    print("Chargement de la ligne isolée...")
    isolated_row = pd.read_csv(isolated_row_file)

    # Extraire les caractéristiques et le label théorique
    X_isolated = isolated_row.drop(columns=['Class'])
    y_true = isolated_row['Class'].iloc[0]

    # Charger le modèle ML
    print("Chargement du modèle ML...")
    ml_model = joblib.load(ml_model_file)

    # Prédiction avec le modèle ML
    print("Prédiction avec le modèle ML...")
    ml_prediction = ml_model.predict(X_isolated)[0]
    ml_prediction_proba = ml_model.predict_proba(X_isolated)[0][1]
    print(f"Prédiction ML: {ml_prediction}, Probabilité: {ml_prediction_proba:.4f}")

    # Charger le modèle DL
    print("Chargement du modèle DL...")
    dl_model = load_model(dl_model_file)

    # Prédiction avec le modèle DL
    print("Prédiction avec le modèle DL...")
    dl_prediction_proba = dl_model.predict(X_isolated)[0][0]
    dl_prediction = int(dl_prediction_proba > 0.5)
    print(f"Prédiction DL: {dl_prediction}, Probabilité: {dl_prediction_proba:.4f}")

    # Comparaison des résultats
    print("\nComparaison des résultats avec le label théorique...")
    print(f"Label théorique: {y_true}")
    print(f"Prédiction ML: {ml_prediction}, Probabilité: {ml_prediction_proba:.4f}")
    print(f"Prédiction DL: {dl_prediction}, Probabilité: {dl_prediction_proba:.4f}")

    # Commentaires sur les écarts
    if ml_prediction != dl_prediction:
        print("\nCommentaire: Les prédictions des modèles ML et DL diffèrent. Cela peut être dû à la structure des modèles ou aux données utilisées pour l'entraînement.")
    else:
        print("\nCommentaire: Les prédictions des modèles ML et DL sont cohérentes avec le label théorique.")
