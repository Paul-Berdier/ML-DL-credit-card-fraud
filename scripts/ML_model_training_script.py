import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

# Fonction pour entraîner un modèle de Machine Learning
def train_ml_model(prepared_data_file, target_column='Class'):
    """
    Charge les données préparées, entraîne un modèle de classification, et affiche les métriques de performance.

    Parameters:
        prepared_data_file (str): Chemin vers le fichier des données préparées (normalisées).
        target_column (str): Nom de la colonne cible (par défaut 'Class').

    Returns:
        None
    """
    print("Chargement des données...")
    data = pd.read_csv(prepared_data_file)

    # Séparer les features (X) et la cible (y)
    X = data.drop(columns=[target_column])
    y = data[target_column]

    # Diviser les données en ensembles d'entraînement et de test
    print("Division des données en ensembles d'entraînement et de test...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Initialiser le modèle
    print("Entraînement du modèle Random Forest...")
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)

    # Prédictions
    print("Évaluation du modèle...")
    y_pred = model.predict(X_test)

    # Afficher les métriques
    print("\nMétriques de performance :")
    print(classification_report(y_test, y_pred, target_names=['Légitime', 'Fraude']))

    # Matrice de confusion
    print("\nMatrice de confusion :")
    print(confusion_matrix(y_test, y_pred))