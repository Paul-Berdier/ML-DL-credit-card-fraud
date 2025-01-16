import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import joblib

# Fonction pour entraîner un modèle de Machine Learning avancé
def train_ml_model(prepared_data_file, target_column='Class', output_model_file='model.joblib'):
    """
    Charge les données préparées, effectue une analyse exploratoire, entraîne plusieurs modèles de classification,
    et affiche les métriques de performance.

    Parameters:
        prepared_data_file (str): Chemin vers le fichier des données préparées (normalisées).
        target_column (str): Nom de la colonne cible (par défaut 'Class').
        output_model_file (str): Chemin pour sauvegarder le meilleur modèle entraîné.

    Returns:
        None
    """
    print("Chargement des données...")
    data = pd.read_csv(prepared_data_file)

    # Analyse exploratoire basique
    print("\nStatistiques descriptives :")
    print(data.describe())

    # Séparer les features (X) et la cible (y)
    X = data.drop(columns=[target_column])
    y = data[target_column]

    # Diviser les données en ensembles d'entraînement et de test
    print("\nDivision des données en ensembles d'entraînement et de test...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Initialiser les modèles
    print("\nEntraînement des modèles...")
    models = {
        "Random Forest": RandomForestClassifier(random_state=42),
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Gradient Boosting": GradientBoostingClassifier(random_state=42)
    }

    best_model = None
    best_auc = 0

    for model_name, model in models.items():
        print(f"\nModèle en cours : {model_name}")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]

        # Calculer l'AUC-ROC
        auc = roc_auc_score(y_test, y_pred_proba)
        print(f"AUC-ROC pour {model_name}: {auc:.4f}")

        # Afficher les métriques
        print("\nMétriques de classification :")
        print(classification_report(y_test, y_pred, target_names=['Légitime', 'Fraude']))

        # Mettre à jour le meilleur modèle
        if auc > best_auc:
            best_auc = auc
            best_model = model

        # Courbe ROC
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        plt.plot(fpr, tpr, label=f"{model_name} (AUC = {auc:.2f})")

    # Sauvegarder le meilleur modèle
    print(f"\nMeilleur modèle : {best_model.__class__.__name__} avec AUC = {best_auc:.4f}")
    joblib.dump(best_model, output_model_file)
    print(f"Modèle sauvegardé sous : {output_model_file}")

    # Afficher la courbe ROC
    plt.plot([0, 1], [0, 1], 'k--', label='Random Guess')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Courbe ROC')
    plt.legend()
    plt.savefig('roc_curve.png')
    plt.show()

    # Matrice de confusion
    print("\nMatrice de confusion pour le meilleur modèle :")
    y_best_pred = best_model.predict(X_test)
    print(confusion_matrix(y_test, y_best_pred))