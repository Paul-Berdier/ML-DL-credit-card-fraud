import seaborn as sns
import matplotlib.pyplot as plt

def visualisation_V1_V2(data):
    # Visualisation des composantes principales V1 et V2
    sns.scatterplot(data=data, x='V1', y='V2', hue='Class', alpha=0.5, palette={0: 'blue', 1: 'red'})
    plt.title('Distribution des transactions sur V1 et V2')
    plt.show()

