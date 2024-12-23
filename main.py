import os
import pandas as pd
from scripts.cleanig_data_script import reduce_transaction, normalize_data, compute_statistics, correlation_matrix
from scripts.visualisation_script import visualisation_V1_V2, visualisation_correlation_matrix
from scripts.isolate_row_script import isolate_random_row
from scripts.ml_model_script import train_ml_model

# Définir les chemins des fichiers
credit_card_data_file = 'data/creditcard.csv'
reduced_data_file = 'data/reduced_creditcard.csv'
correlation_matrix_file = 'data/correlation_matrix_creditcard.csv'
output_image_V1_V2 = 'data/V1_V2.png'
output_image_corr = 'data/correlation_matrix.png'
output_scaled_file = 'data/scaled_creditcard.csv'
output_stats_file = 'docs/stats_summary.csv'
isolated_row_file = 'data/isolated_row.csv'
train_data_file = 'data/train_data.csv'
ml_model_file = 'models/ml_model.pkl'

if __name__ == '__main__':

    ######################## NETTOYAGE ET REDUCTION DU DATASET #############################

    # Vérifier si le fichier réduit existe déjà
    if os.path.exists(reduced_data_file):
        print(f"Le fichier '{reduced_data_file}' existe déjà. Chargement des données...")
        reduced_data = pd.read_csv(reduced_data_file)
    else:
        print("Réduction des transactions en cours...")
        reduce_transaction(credit_card_data_file, reduced_data_file)
        print("Réduction des transactions terminée.")
        reduced_data = pd.read_csv(reduced_data_file)

    # Compter et afficher le nombre de lignes
    print(f"Le dataset réduit contient {len(reduced_data)} lignes.")

    # Compter et afficher le nombre de fraudes et de non-fraudes
    class_counts = reduced_data['Class'].value_counts()
    print(f"Transactions légitimes : {class_counts[0]}")
    print(f"Transactions frauduleuses : {class_counts[1]}")

    # Visualisation des colonnes V1 et V2
    print("Génération du graphique de dispersion (V1 vs V2)...")
    visualisation_V1_V2(reduced_data, output_image_V1_V2)
    print(f"Graphique enregistré sous : {output_image_V1_V2}")

    # Générer la matrice de corrélation
    print("Génération de la matrice de corrélation...")
    visualisation_correlation_matrix(reduced_data, output_image_corr)
    print(f"Matrice de corrélation enregistrée sous : {output_image_corr}")

    # Réduction de dimensionnalité basée sur la corrélation
    print("Calcul et enregistrement de la matrice de corrélation des variables...")
    correlation_matrix(reduced_data_file, correlation_matrix_file)
    print(f"Matrice de corrélation sauvegardée sous : {correlation_matrix_file}")

    # Normaliser les données
    print("Normalisation des données en cours...")
    normalize_data(correlation_matrix_file, output_scaled_file)
    print(f"Données normalisées et sauvegardées sous : {output_scaled_file}")

    # Calculer et sauvegarder les statistiques essentielles
    print("Calcul des statistiques essentielles...")
    compute_statistics(output_scaled_file, output_stats_file)
    print(f"Statistiques essentielles sauvegardées sous : {output_stats_file}")

    ##############################################################################

    # Isoler une ligne pour la prédiction finale
    print("Isolation d'une ligne pour la prédiction finale...")
    isolate_random_row(output_scaled_file, train_data_file, isolated_row_file)
    print(f"Ligne isolée sauvegardée sous : {isolated_row_file}")
    print(f"Données d'entraînement mises à jour sous : {train_data_file}")

    print("Traitement terminé. Les résultats sont disponibles dans les fichiers de sortie.")

    ########################### MACHINE LEARNING ################################

    # Entraîner un modèle de Machine Learning
    print("Entraînement d'un modèle de Machine Learning...")
    train_ml_model(train_data_file, ml_model_file)
    print(f"Modèle ML sauvegardé sous : {ml_model_file}")

    print("Traitement terminé. Les résultats sont disponibles dans les fichiers de sortie.")
