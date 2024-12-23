import os
import pandas as pd
from scripts.cleanig_data_script import reduce_transaction, reduce_dimensions, normalize_data, compute_statistics
from scripts.visualisation_script import visualisation_V1_V2, visualisation_correlation_matrix

# Définir les chemins des fichiers
input_file = 'data/creditcard.csv'
output_file = 'data/reduced_creditcard.csv'
output_image_V1_V2 = 'data/V1_V2.png'
output_image_corr = 'data/correlation_matrix.png'
output_scaled_file = 'data/scaled_creditcard.csv'
output_stats_file = 'docs/stats_summary.csv'

if __name__ == '__main__':
    # Vérifier si le fichier réduit existe déjà
    if os.path.exists(output_file):
        print(f"Le fichier '{output_file}' existe déjà. Chargement des données...")
        reduced_data = pd.read_csv(output_file)
    else:
        print("Réduction des transactions en cours...")
        reduce_transaction(input_file, output_file)
        reduced_data = pd.read_csv(output_file)

    # Compter et afficher le nombre de lignes
    print(f"Le dataset réduit contient {len(reduced_data)} lignes.")

    # Compter et afficher le nombre de fraudes et de non-fraudes
    class_counts = reduced_data['Class'].value_counts()
    print(f"Transactions légitimes : {class_counts[0]}")
    print(f"Transactions frauduleuses : {class_counts[1]}")

    # Visualisation des colonnes V1 et V2
    visualisation_V1_V2(reduced_data, output_image_V1_V2)

    # Générer la matrice de corrélation
    visualisation_correlation_matrix(reduced_data, output_image_corr)

    # Réduction de dimensionnalité basée sur la corrélation
    reduced_dim_data = reduce_dimensions(output_file, threshold=0.1)

    # Normaliser les données
    scaled_data = normalize_data(output_file)
    scaled_data.to_csv(output_scaled_file, index=False)

    # Calculer et sauvegarder les statistiques essentielles
    compute_statistics(output_scaled_file, output_stats_file)

    print("Traitement terminé. Les résultats sont disponibles dans les fichiers de sortie.")
