# -*- coding: utf-8 -*-

from importlib.abc import FileLoader
from msilib import Directory
import sys
import os
from pathlib import Path
from matplotlib.lines import lineStyles
import pandas as pd
import json
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
import matplotlib.pyplot as plt

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
dataPath = os.path.join(directory, 'time_series_15min_singleindex.csv')
data = pd.read_csv(dataPath, delimiter=",")
data.rename(columns={'utc_timestamp': 'time'}, inplace=True)
data.rename(columns={'DE_tennet_wind_generation': 'value'}, inplace=True)
data = data.dropna(subset=['value'])

#Datei bearbeiten
data['value'] = pd.to_numeric(data['value'], errors='coerce')
data = data.dropna(subset=['value'])
nameModell = "DeepLearning" + "time_series_15min_singleindex.csv" + ".keras"
pathFolder = os.path.join(directory, nameModell)

#Zeitstempel in numerische Werte umwandeln, falls sie noch nicht im richtigen Format sind
data['time'] = pd.to_datetime(data['time'])
data['time'] = data['time'].astype(np.int64) // 10**6


k = 5
all_losses_mean = pd.DataFrame()
all_val_losses_mean = pd.DataFrame()
path_csv_loss = os.path.join(directory, 'csv_loss.csv')
path_csv_val_loss = os.path.join(directory, 'csv_val_loss.csv')

if not(os.path.exists(path_csv_loss) and os.path.exists(path_csv_val_loss)):

    for i in range(1, 11, 1):

        #Downsampling
        percent = i/10
        print(len(data))
        downSampling = int(len(data) * percent)
        dataDownSampling = data.iloc[:downSampling]
        print(len(dataDownSampling))

        all_losses_collect = pd.DataFrame()
        all_val_losses_collect = pd.DataFrame()

        for j in range(0,k,1):
            print(percent)
            # Feature und Ziel trennen
            X = dataDownSampling[['time', 'value']].iloc[:-1].values
            y = dataDownSampling['value'].iloc[1:].values

            #Daten normalisieren
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)

            y = y.reshape(-1, 1)  #falls nötig
            scaler_y = StandardScaler()
            y_scaled = scaler_y.fit_transform(y)


            #Train- und Testdaten
            X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled, test_size=0.2, random_state=42)

            @register_keras_serializable()
            def rmse_loss(y_true, y_pred):
                return tf.sqrt(tf.reduce_mean(tf.square(y_pred - y_true)))

            if 0==1:#os.path.exists(pathFolder):

                    #loadModel = load_model(pathFolderPlusDatei)
                    loadModel = load_model(pathFolder, custom_objects={'rmse_loss': rmse_loss})
                    prediction = loadModel.predict(X_test)

            else:

                    #Modell erstellen
                    model = Sequential([
                    Input(shape=(X_train.shape[1],)),  # Eingabeschicht
                    Dense(32, kernel_regularizer=regularizers.l2(0.01)),  # L2 Regularisierung
                    LeakyReLU(alpha=0.1),
                    Dropout(0.1),  # Eingabeschicht
                    Dense(16, kernel_regularizer=regularizers.l2(0.01)),  # L2 Regularisierung
                    LeakyReLU(alpha=0.1),
                    Dropout(0.1),
                    Dense(1)  # Ausgabeschicht
                    ])

                    #Modell kompilieren
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
                    history  = model.fit(data_generator(2048,X_train, y_train), epochs = 128, validation_data = (X_test, y_test), steps_per_epoch = len(X_train) // 2048, callbacks=[early_stopping])


                    

                    all_losses = pd.DataFrame(history.history['loss'])
                    all_val_losses = pd.DataFrame(history.history['val_loss'])

                    print('Test-------------------------------------------------')

                    print(all_losses)
                    print(all_val_losses)

                    all_losses_collect = pd.concat([all_losses_collect, all_losses], axis=1)
                    all_val_losses_collect = pd.concat([all_val_losses_collect, all_val_losses], axis=1)
                    print(all_losses_collect)
                    print(all_val_losses_collect)

        print('----------------------------------------')
        all_losses_collect = all_losses_collect.mean(axis=1).to_frame(name=f'{i*10}%')
        all_val_losses_collect = all_val_losses_collect.mean(axis=1).to_frame(name=f'{i*10}%')
        all_losses_mean = pd.concat([all_losses_mean, all_losses_collect], axis=1)
        print(all_losses_mean)
        all_val_losses_mean = pd.concat([all_val_losses_mean, all_val_losses_collect], axis=1)
        print(all_val_losses_mean)



            #Modell speichern
            #model.save(pathFolder)

            #Vorhersage
            #prediction_scaled = model.predict(X_test)
            #prediction = scaler_y.inverse_transform(prediction_scaled)

            
    #Daten in CSV-Datei speichern
    all_losses_mean.to_csv('csv_loss.csv', index=True)
    all_val_losses_mean.to_csv('csv_val_loss.csv', index=True)

    
csv_loss = pd.read_csv('csv_loss.csv', delimiter=",")
csv_val_loss = pd.read_csv('csv_val_loss.csv', delimiter=",")

#Plotten von loss und val_loss

for k in range(0,2,1):
        plt.figure(figsize=(10, 5))
        for i in range(1+k*1,len(csv_loss.columns),2):
            plt.plot(csv_loss.index, csv_loss.iloc[:,i], label= 'Trainings-Loss ' + csv_loss.columns[i],linestyle='-')
        for j in range(1+k*1,len(csv_val_loss.columns),2):
           plt.plot(csv_val_loss.index, csv_val_loss.iloc[:,j], label= 'Validierungs-Loss ' + csv_val_loss.columns[j], linestyle='-')

        plt.xlabel('Epochs')
        plt.ylabel('Loss (RMSE)')
        plt.title('Trainings- und Validierungsverlust')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(fname=f'Trainings- und Validierungsverlust_{k}')
        plt.show(block=False)
        plt.pause(0.1)

plt.show()


#print(prediction)