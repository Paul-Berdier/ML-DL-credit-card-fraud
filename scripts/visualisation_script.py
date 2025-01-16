import seaborn as sns
import matplotlib.pyplot as plt
import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

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

def plot_learning_curve(estimator, X_train, y_train, title, output_file):
    """
    Plot the learning curve for a given model.

    Parameters:
        estimator: The model to evaluate.
        X_train: Training data features.
        y_train: Training data labels.
        title: Title for the plot.
        output_file: Path to save the learning curve plot.
    """
    train_sizes = np.linspace(0.1, 1.0, 10)
    train_scores = []
    test_scores = []

    for train_size in train_sizes:
        X_partial, _, y_partial, _ = train_test_split(
            X_train, y_train, train_size=train_size, random_state=42
        )
        estimator.fit(X_partial, y_partial)
        train_scores.append(accuracy_score(y_partial, estimator.predict(X_partial)))
        test_scores.append(accuracy_score(y_train, estimator.predict(X_train)))

    plt.figure()
    plt.plot(train_sizes, train_scores, label="Training Accuracy")
    plt.plot(train_sizes, test_scores, label="Validation Accuracy")
    plt.title(title)
    plt.xlabel("Training Set Size")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid()
    plt.savefig(output_file)
    plt.close()