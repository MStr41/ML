# -*- coding: utf-8 -*-

from importlib.abc import FileLoader
from msilib import Directory
import sys
import os
from pathlib import Path
import pandas as pd
import json
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense, Dropout
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.metrics import MeanAbsolutePercentageError
from tensorflow.keras import regularizers
from keras.models import load_model
from keras.saving import register_keras_serializable





directory = os.getcwd()

dataPath = os.path.join(directory, 'bla.parquet')

data = pd.read_parquet(dataPath)

print(data)

if 0 == 1:

    data['value'] = pd.to_numeric(data['value'], errors='coerce')

    data = data.dropna(subset=['value'])

    nameModell = "DeepLearning" + "bla.parquet" + ".keras"

    pathFolder = os.path.join(directory, nameModell)

    #Zeitstempel in numerische Werte umwandeln, falls sie noch nicht im richtigen Format sind
    data['time'] = pd.to_datetime(data['time'])
        
    data['time'] = data['time'].astype(np.int64)

    letzteDatum = np.max(data['time'].values)

    #dataRead = dataRead.sample(frac=0.00001, random_state=42)

    #Datei bearbeiten

    # Target, der Punkt an dem der Sensorwert den Grenzwert erreicht
    data['target'] = np.where(data['value'] == grenzwert, data['time'], time_value)

    # Feature und Ziel trennen
    X = data[['time', 'value']].values
    y = data['target'].values

    # Daten normalisieren
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Train- und Testdaten
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    @register_keras_serializable()
    def rmse_loss(y_true, y_pred):
        return tf.sqrt(tf.reduce_mean(tf.square(y_pred - y_true)))

    if os.path.exists(pathFolder):

            #loadModel = load_model(pathFolderPlusDatei)
            loadModel = load_model(pathFolder, custom_objects={'rmse_loss': rmse_loss})
            prediction = loadModel.predict(X_test)

    else:

            #Modell erstellen
            model = Sequential([
            Input(shape=(X_train.shape[1],)),  # Eingabeschicht
            Dense(64, kernel_regularizer=regularizers.l2(0.01)),  # L2 Regularisierung
            LeakyReLU(alpha=0.1),
            Dropout(0.2),  # Eingabeschicht
            Dense(32, kernel_regularizer=regularizers.l2(0.01)),  # L2 Regularisierung
            LeakyReLU(alpha=0.1),
            Dropout(0.2),
            Dense(16, kernel_regularizer=regularizers.l2(0.01)),  # L2 Regularisierung
            LeakyReLU(alpha=0.1),
            Dropout(0.2),
            Dense(1)  # Ausgabeschicht
            ])

            #Modell komplieren
            model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),loss=rmse_loss)

            #Early Stop
            early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

            def data_generator(batch_size, X_data, y_data):
                size = len(X_data)
                while True:
                    for i in range(0, size, batch_size):
                        batch_x = X_data[i:i+batch_size]
                        batch_y = y_data[i:i+batch_size]
                        yield batch_x, batch_y

            #Modell trainieren
            model.fit(data_generator(512,X_train, y_train), epochs = 1, validation_data = (X_test, y_test), steps_per_epoch = len(X_train) // 512, callbacks=[early_stopping])

            # Modell speichern
            model.save(pathFolder)

            #Vorhersage
            prediction = model.predict(X_test)

    print(prediction)