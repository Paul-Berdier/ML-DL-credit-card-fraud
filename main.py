import os
import pandas as pd
from scripts.cleanig_data_script import reduce_transaction, normalize_data, compute_statistics, correlation_matrix, isolate_random_row
from scripts.visualisation_script import visualisation_V1_V2, visualisation_correlation_matrix
from scripts.ML_model_training_script import train_ml_model
from scripts.DL_model_training_script import train_deep_learning_model  # Nouvelle fonction pour le DL
from scripts.prediction_script import perform_prediction  # Nouvelle fonction pour la prédiction
from colorama import Fore, Style

def display_message(message):
    print(Fore.GREEN + message.upper() + Style.RESET_ALL)

# Définir les chemins des fichiers
credit_card_data_file = 'data/creditcard.csv'
reduced_data_file = 'data/reduced_creditcard.csv'
correlation_matrix_file = 'data/correlation_matrix_creditcard.csv'
output_image_V1_V2 = 'docs/V1_V2.png'
output_image_corr_before = 'docs/correlation_matrix_before.png'
output_image_corr_after = 'docs/correlation_matrix_after.png'
output_scaled_file = 'data/scaled_creditcard.csv'
output_stats_file = 'data/stats_summary.csv'
ml_model_file = 'models/ml_model.joblib'
dl_model_file = 'models/deep_model.keras'
isolated_row_file = 'data/isolated_row.csv'
output_roc_curve = 'docs/roc_curve.png'
reformed_data_file = 'data/reformed_creditcard.csv'
output_learning_curve = 'docs/learning_curve.png'
dl_learning_curve = 'docs/deep_learning_curve.png'
output_loss_curve = 'docs/loss_curve.png'

# Options pour le choix de l'utilisateur
def main():
    display_message("Bienvenue dans le programme de traitement des données et d'entraînement de modèle")
    options = {
        "1": "Réduire les données",
        "2": "Visualiser les données (V1 vs V2)",
        "3": "Générer la matrice de corrélation",
        "4": "Normaliser les données",
        "5": "Calculer les statistiques",
        "6": "Séparer une ligne du dataset",
        "7": "Entraîner un modèle Machine Learning",
        "8": "Entraîner un modèle Deep Learning",
        "9": "Effectuer une prédiction",
    }

    for key, value in options.items():
        print(f"{key}. {value}")

    choice = input(Fore.CYAN + "\nEntrez les numéros des étapes à exécuter (séparés par des virgules) ou appuyez sur Entrée pour tout exécuter : " + Style.RESET_ALL)

    if not choice.strip():
        choice = list(map(int, options.keys()))
    else:
        choice = list(map(int, choice.split(',')))

    if 1 in choice:
        pourcentage_reduction = input(
            "\nA combien de pourcents voulez-vous réduire les ligne non frauduleuse du dataset ? (entre 0 et 1 avec 0.01 de base) : ")
        try:
            pourcentage_reduction = float(pourcentage_reduction)  # Convertir en flottant
            if not (0 < pourcentage_reduction <= 1):
                raise ValueError("Le pourcentage doit être un nombre entre 0 et 1.")
        except ValueError as e:
            print(Fore.RED + "Erreur : " + str(e) + Style.RESET_ALL)
            return  # Terminer le programme si l'entrée est invalide

        display_message("Réduction des transactions en cours...")
        reduce_transaction(credit_card_data_file, reduced_data_file, pourcentage_reduction)
        display_message("Réduction des transactions terminée.")

    if 2 in choice:
        display_message("Génération du graphique de dispersion (V1 vs V2)...")
        reduced_data = pd.read_csv(reduced_data_file)
        visualisation_V1_V2(reduced_data, output_image_V1_V2)
        display_message(f"Graphique enregistré sous : {output_image_V1_V2}")

    if 3 in choice:
        display_message("Génération de la matrice de corrélation...")
        reduced_data = pd.read_csv(reduced_data_file)
        visualisation_correlation_matrix(reduced_data, output_image_corr_before)
        display_message(f"Matrice de corrélation enregistrée sous : {output_image_corr_before}")
        correlation_matrix(reduced_data_file, correlation_matrix_file)
        display_message(f"Matrice de corrélation sauvegardée sous : {correlation_matrix_file}")
        visualisation_correlation_matrix(pd.read_csv(correlation_matrix_file), output_image_corr_after)
        display_message(f"Matrice de corrélation enregistrée sous : {output_image_corr_after}")

    if 4 in choice:
        display_message("Normalisation des données en cours...")
        normalize_data(correlation_matrix_file, output_scaled_file)
        display_message(f"Données normalisées et sauvegardées sous : {output_scaled_file}")

    if 5 in choice:
        display_message("Calcul des statistiques essentielles...")
        compute_statistics(output_scaled_file, output_stats_file)
        display_message(f"Statistiques essentielles sauvegardées sous : {output_stats_file}")

    if 6 in choice:
        display_message("Isolation d'une ligne aléatoire pour la prédiction finale...")
        isolate_random_row(output_scaled_file, reformed_data_file, isolated_row_file)
        display_message(f"Ligne isolée sauvegardée sous : {isolated_row_file}")

    if 7 in choice:
        display_message("Entraînement du modèle de Machine Learning...")
        train_ml_model(reformed_data_file, output_roc_curve, output_learning_curve, target_column='Class', output_model_file=ml_model_file)
        display_message(f"Modèle ML sauvegardé sous : {ml_model_file}")

    if 8 in choice:
        display_message("Entraînement du modèle Deep Learning...")
        train_deep_learning_model(reformed_data_file, output_loss_curve, target_column='Class', output_model_file=dl_model_file, output_learning_curve=dl_learning_curve)
        display_message(f"Modèle DL sauvegardé sous : {dl_model_file}")

    if 9 in choice:
        display_message("Effectuer une prédiction avec les modèles ML et DL...")
        perform_prediction(isolated_row_file, ml_model_file, dl_model_file)

if __name__ == '__main__':
    main()
