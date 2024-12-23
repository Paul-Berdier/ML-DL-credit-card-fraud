import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

def visualisation_V1_V2():
    file_path = 'creditcard.csv'
    data = pd.read_csv(file_path)

    # Visualisation des composantes principales V1 et V2
    sns.scatterplot(data=data, x='V1', y='V2', hue='Class', alpha=0.5, palette={0: 'blue', 1: 'red'})
    plt.title('Distribution des transactions sur V1 et V2')
    plt.show()

