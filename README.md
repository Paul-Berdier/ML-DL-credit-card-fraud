# Projet de Détection de Fraude par Carte de Crédit

## Objectif du Projet
Ce projet vise à détecter les fraudes dans un jeu de données de transactions par carte de crédit en utilisant des techniques de Machine Learning et Deep Learning. L'objectif est d'analyser les données, de réduire leur taille tout en conservant les informations critiques, et de visualiser des modèles pour améliorer la compréhension des comportements frauduleux.

---

## Structure des fichiers

1. **main.py**
   - Script principal orchestrant l'ensemble des opérations : réduction des données et visualisation.
   - Affiche le nombre de transactions légitimes et frauduleuses après traitement.
   - Appelle les fonctions des autres modules pour la réduction et la visualisation.

2. **reduce_transaction.py**
   - Fonction pour réduire la taille du dataset en ne conservant que 1 % des transactions non frauduleuses tout en gardant toutes les transactions frauduleuses.
   - Méthode basée sur la variance des colonnes pour sélectionner les lignes les plus variées.

3. **visualisation_V1_V2.py**
   - Script pour générer un graphique de dispersion (scatterplot) des colonnes V1 et V2, coloré en fonction des classes (légitime ou fraude).

4. **data/**
   - **creditcard.csv** : Dataset original contenant les transactions avant traitement.
   - **reduced_creditcard.csv** : Dataset réduit généré après exécution du script principal.

---

## Consignes et Étapes du TP

### 1. Choix et Présentation du Dataset
- **Dataset** : "Credit Card Fraud Detection" (Kaggle).
- **Nombre de lignes** : 284 807.
- **Colonnes** : 31 (V1 à V28 issues d'une PCA, Time, Amount, Class).
- **Type de problème** : Classification binaire (0 = légitime, 1 = fraude).

### 2. Nettoyage et Préparation des Données
- Suppression des valeurs extrêmes et standardisation des colonnes.
- Réduction des transactions non frauduleuses pour équilibrer les classes.

### 3. Analyse Exploratoire
- Graphiques de dispersion sur les colonnes V1 et V2 pour observer la distribution des classes.
- Calcul des statistiques descriptives.

### 4. Modélisation et Apprentissage
- Préparation pour entraîner des modèles de Machine Learning et Deep Learning.
- Optimisation des hyperparamètres et évaluation des performances.

### 5. Prédictions
- Test sur une ligne isolée.
- Comparaison des prédictions des modèles ML et DL.

---

## Exemple de Graphique Généré

Le fichier **visualisation_V1_V2.py** génère un graphique de dispersion comme ci-dessous :

![Graphique V1 vs V2](docs/images/scatter_v1_v2.png)

---

## Installation et Exécution

### Prérequis
- Python 3.8+
- Bibliothèques nécessaires :
  ```bash
  pip install pandas numpy seaborn matplotlib
  ```

### Lancer le projet
```bash
python main.py
```

---

## Résultats obtenus
- Dataset initial : 284 807 lignes.
- Dataset réduit : Environ 3 335 lignes (incluant toutes les fraudes).
- Transactions légitimes : Affichées dans la console.
- Transactions frauduleuses : Affichées dans la console.

---

## Notes supplémentaires
- Ce projet peut être étendu avec des modèles d'apprentissage supervisé.
- La visualisation peut être enrichie avec d'autres dimensions (ex. V3, V4).
- Ajout potentiel d'un modèle de Deep Learning avec Keras/TensorFlow.

---

## Auteurs
- **Nom** : Paul Berdier
- **Formation** : Master 1 - Data Science et Intelligence Artificielle
- **Année** : 2024/2025

