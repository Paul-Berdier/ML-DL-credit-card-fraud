import os
import pandas as pd
from utils.reduce_transaction import reduce_transaction
from utils.visualisation_V1_V2 import visualisation_V1_V2

# Définir les chemins des fichiers
input_file = 'data/creditcard.csv'
output_file = 'data/reduced_creditcard.csv'

if __name__ == '__main__':
    # Vérifier si le fichier réduit existe déjà
    if os.path.exists(output_file):
        print(f"Le fichier '{output_file}' existe déjà. Chargement des données...")
        reduced_data = pd.read_csv(output_file)
    else:
        print("Réduction des transactions en cours...")
        reduce_transaction(input_file, output_file)

    # Compter et afficher le nombre de lignes
    reduced_data = pd.read_csv(output_file)
    print(f"Le dataset réduit contient {len(reduced_data)} lignes.")

    # Compter et afficher le nombre de fraudes et de non-fraudes
    class_counts = reduced_data['Class'].value_counts()
    print(f"Transactions légitimes : {class_counts[0]}")
    print(f"Transactions frauduleuses : {class_counts[1]}")

    #Visualisation des colonnes V1 et V2
    visualisation_V1_V2(reduced_data)




