import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from scripts.visualisation_script import plot_learning_curve
import matplotlib.pyplot as plt
import joblib

# Fonction pour entraîner un modèle de Machine Learning avancé
def train_ml_model(prepared_data_file, output_roc_curve, output_learning_curve, target_column='Class', output_model_file='model_ML.joblib'):
    """
    Train a Machine Learning model with multiple techniques and evaluate its performance.

    Parameters:
        prepared_data_file (str): Path to the prepared dataset.
        output_roc_curve (str): Path to save the ROC curve plot.
        target_column (str): Name of the target column.
        output_model_file (str): Path to save the trained model.

    Returns:
        None
    """
    print("Chargement des données...")
    data = pd.read_csv(prepared_data_file)

    # Séparer les caractéristiques (X) et la cible (y)
    X = data.drop(columns=[target_column])
    y = data[target_column]

    # Diviser les données en ensembles d'entraînement et de test
    print("Division des données en ensembles d'entraînement et de test...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Initialisation des modèles
    models = {
        "Random Forest": RandomForestClassifier(random_state=42),
        "Gradient Boosting": GradientBoostingClassifier(random_state=42),
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42)
    }

    # Optimisation des hyperparamètres
    param_grids = {
        "Random Forest": {
            "n_estimators": [100, 200],
            "max_depth": [10, 20, None],
            "min_samples_split": [2, 5]
        },
        "Gradient Boosting": {
            "learning_rate": [0.01, 0.1],
            "n_estimators": [100, 200],
            "max_depth": [3, 5]
        },
        "Logistic Regression": {
            "C": [0.1, 1, 10]
        }
    }

    best_estimators = {}

    for model_name, model in models.items():
        print(f"Optimisation des hyperparamètres pour {model_name}...")
        grid_search = GridSearchCV(model, param_grids[model_name], scoring="accuracy", cv=3)
        grid_search.fit(X_train, y_train)
        best_estimators[model_name] = grid_search.best_estimator_
        print(f"Meilleurs paramètres pour {model_name}: {grid_search.best_params_}")

    # Entraînement d'un modèle ensembliste (VotingClassifier)
    print("Entraînement du modèle VotingClassifier...")
    voting_model = VotingClassifier(
        estimators=[(name, estimator) for name, estimator in best_estimators.items()],
        voting='soft'
    )
    voting_model.fit(X_train, y_train)

    # Évaluation du modèle
    print("Évaluation des performances...")
    y_pred = voting_model.predict(X_test)
    y_pred_proba = voting_model.predict_proba(X_test)[:, 1]

    # Rapport de classification
    print("\nRapport de classification :")
    print(classification_report(y_test, y_pred))

    # Matrice de confusion
    print("\nMatrice de confusion :")
    print(confusion_matrix(y_test, y_pred))

    # AUC-ROC
    auc_score = roc_auc_score(y_test, y_pred_proba)
    print(f"AUC-ROC: {auc_score:.4f}")

    # Courbe ROC
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    plt.figure()
    plt.plot(fpr, tpr, label=f"VotingClassifier (AUC = {auc_score:.2f})")
    plt.plot([0, 1], [0, 1], 'k--', label="Random Guess")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    plt.grid()
    plt.savefig(output_roc_curve)
    plt.close()

    # Sauvegarder le modèle
    joblib.dump(voting_model, output_model_file)
    print(f"Modèle sauvegardé sous : {output_model_file}")

    # Courbe d'apprentissage
    print("Génération des courbes d'apprentissage...")
    plot_learning_curve(voting_model, X_train, y_train, "Courbe d'apprentissage", output_learning_curve)
