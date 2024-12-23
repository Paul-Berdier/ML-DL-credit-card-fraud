import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

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

# Fonction pour réduire la dimensionnalité
def reduce_dimensions(input_file, threshold=0.1):
    # Charger le fichier CSV
    data = pd.read_csv(input_file)

    correlation_matrix = data.corr()
    correlated_features = correlation_matrix['Class'][abs(correlation_matrix['Class']) > threshold].index.tolist()
    return data[correlated_features]

# Fonction pour normaliser les données
def normalize_data(input_file):
    # Charger le fichier CSV
    data = pd.read_csv(input_file)

    scaler = StandardScaler()
    scaled_data = pd.DataFrame(scaler.fit_transform(data), columns=data.columns)
    return scaled_data

# Fonction pour calculer des statistiques essentielles
def compute_statistics(input_file, output_file):
    # Charger le fichier CSV
    data = pd.read_csv(input_file)

    stats = data.describe().T
    stats['median'] = data.median()
    stats['IQR'] = stats['75%'] - stats['25%']
    stats.to_csv(output_file)
    return stats
