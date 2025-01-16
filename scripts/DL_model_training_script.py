import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
import joblib

# Fonction pour construire et entraîner un modèle de Deep Learning
def train_deep_learning_model(prepared_data_file, output_loss_curve, target_column='Class', output_model_file='deep_model.pkl', output_learning_curve='docs/deep_learning_curve.png'):
    """
    Train a Deep Learning model using TensorFlow/Keras and evaluate its performance.

    Parameters:
        prepared_data_file (str): Path to the prepared dataset.
        target_column (str): Name of the target column.
        output_model_file (str): Path to save the trained model.
        output_learning_curve (str): Path to save the learning curve plot.

    Returns:
        None
    """
    print("Chargement des données...")
    data = pd.read_csv(prepared_data_file)

    # Séparer les caractéristiques (X) et la cible (y)
    X = data.drop(columns=[target_column])
    y = data[target_column]

    # Diviser les données en ensembles d'entraînement et de test
    print("Division des données en ensembles d'entraînement et de test...")
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Construire le modèle
    print("Construction du modèle...")
    model = Sequential([
        Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
        Dropout(0.3),
        Dense(32, activation='relu'),
        Dropout(0.3),
        Dense(1, activation='sigmoid')
    ])

    # Compiler le modèle
    print("Compilation du modèle...")
    model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])

    # Entraîner le modèle avec Early Stopping
    print("Entraînement du modèle...")
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=50,
        batch_size=32,
        callbacks=[early_stopping]
    )

    # Sauvegarder le modèle avec joblib
    print("Sauvegarde du modèle avec joblib...")
    joblib.dump(model, output_model_file)
    print(f"Modèle sauvegardé sous : {output_model_file}")

    # Évaluation du modèle
    print("Évaluation des performances...")
    results = model.evaluate(X_test, y_test, verbose=0)
    print(f"Loss: {results[0]:.4f}, Accuracy: {results[1]:.4f}")

    # Générer les courbes d'apprentissage
    print("Génération des courbes d'apprentissage...")
    plt.figure()
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Courbe d\'apprentissage - Précision')
    plt.grid()
    plt.savefig(output_learning_curve)
    plt.close()

    # Courbes d'erreur
    plt.figure()
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Courbe d\'erreur - Loss')
    plt.grid()
    plt.savefig(output_loss_curve)
    plt.close()