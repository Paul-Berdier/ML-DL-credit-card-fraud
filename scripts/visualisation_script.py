import seaborn as sns
import matplotlib.pyplot as plt
import os

def visualisation_V1_V2(input_data, output_image):
    # Visualisation des composantes principales V1 et V2
    sns.scatterplot(data=input_data, x='V1', y='V2', hue='Class', alpha=0.5, palette={0: 'blue', 1: 'red'})
    plt.title('Distribution des transactions sur V1 et V2')
    if os.path.exists(output_image):
        print(f"Le fichier '{output_image}' existe déjà.")
    else:
        print(f"Graphique sauvegardé sous : {output_image}")
        plt.savefig(output_image)
    plt.show()

# Fonction pour générer une matrice de corrélation
def visualisation_correlation_matrix(data, output_image):
    plt.figure(figsize=(12, 10))
    correlation_matrix = data.corr()
    sns.heatmap(correlation_matrix, cmap='coolwarm', annot=False, fmt='.2f')
    plt.title('Matrice de corrélation des variables')
    plt.savefig(output_image)
    plt.show()