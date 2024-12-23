import numpy as np
import pandas as pd

def reduce_transaction(input_file,output_file):
    # Charger le fichier CSV
    data = pd.read_csv(input_file)

    # Séparer les transactions frauduleuses et non frauduleuses
    fraud_data = data[data['Class'] == 1]
    non_fraud_data = data[data['Class'] == 0]

    # Calculer la variance pour chaque colonne (hors 'Class') pour les non frauduleuses
    variances = non_fraud_data.drop(columns=['Class']).var()

    # Créer une mesure de variance combinée pour chaque ligne
    variance_weights = non_fraud_data.drop(columns=['Class']).apply(lambda row: np.dot(row, variances), axis=1)

    # Ajouter cette mesure dans le dataset
    non_fraud_data.loc[:, 'VarianceScore'] = variance_weights

    # Trier les non frauduleuses par ordre décroissant de la mesure de variance
    non_fraud_sorted = non_fraud_data.sort_values(by='VarianceScore', ascending=False)

    # Conserver seulement 1% des lignes les plus variées pour les non frauduleuses
    reduced_non_fraud = non_fraud_sorted.head(int(len(non_fraud_data) * 0.01))

    # Supprimer la colonne 'VarianceScore'
    reduced_non_fraud = reduced_non_fraud.drop(columns=['VarianceScore'])

    # Combiner les données réduites avec toutes les transactions frauduleuses
    reduced_data = pd.concat([reduced_non_fraud, fraud_data])

    # Sauvegarder le nouveau dataset
    reduced_data.to_csv(output_file, index=False)




