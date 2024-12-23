import pandas as pd
import numpy as np

def isolate_random_row(input_file, output_file_train, output_file_test):
    # Charger les données
    data = pd.read_csv(input_file)

    # Isoler une ligne aléatoirement
    random_index = np.random.randint(0, len(data))
    test_row = data.iloc[random_index]

    # Enregistrer la ligne isolée pour la prédiction finale
    test_row.to_csv(output_file_test, index=False)

    # Supprimer la ligne isolée du dataset d'entraînement
    train_data = data.drop(index=random_index)

    # Sauvegarder les données d'entraînement mises à jour
    train_data.to_csv(output_file_train, index=False)

    print(f"Ligne isolée pour la prédiction sauvegardée sous : {output_file_test}")
    print(f"Données d'entraînement sauvegardées sous : {output_file_train}")
