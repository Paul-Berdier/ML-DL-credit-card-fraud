import numpy as np

def reduce_transaction(data):
    # Calculer la variance pour chaque colonne (hors 'Class')
    variances = data.drop(columns=['Class']).var()

    # Créer une mesure de variance combinée pour chaque ligne
    variance_weights = data.drop(columns=['Class']).apply(lambda row: np.dot(row, variances), axis=1)

    # Ajouter cette mesure dans le dataset
    data['VarianceScore'] = variance_weights

    # Trier le dataset par ordre décroissant de la mesure de variance
    data_sorted = data.sort_values(by='VarianceScore', ascending=False)

    # Conserver seulement 1% des lignes les plus variées (environ)
    reduced_data = data_sorted.head(int(len(data) * 0.01))

    # Supprimer la colonne 'VarianceScore' après la réduction
    reduced_data = reduced_data.drop(columns=['VarianceScore'])

    print(f"Réduction terminée. Le nouveau dataset contient {len(reduced_data)} lignes.")
    return reduced_data
    # output_path = 'data/reduced_creditcard.csv'
    # reduced_data.to_csv(output_path, index=False)



