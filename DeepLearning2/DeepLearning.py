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
from tensorflow.keras.layers import BatchNormalization
from sklearn.preprocessing import MinMaxScaler




directory = os.getcwd()
dataPath = os.path.join(directory, 'household_power_consumption.csv')
data = pd.read_csv(dataPath, delimiter=";", low_memory=False)

#Dateien benennen
data["time"] = pd.to_datetime(data['Date'] + " " + data['Time'], dayfirst=True)
data.rename(columns={'Global_active_power': 'value'}, inplace=True)
data = data.dropna(subset=['value'])

#Datei bearbeiten 
data['value'] = pd.to_numeric(data['value'], errors='coerce')
data = data.dropna(subset=['value'])

#Zeitstempel in numerische Werte umwandeln, falls sie noch nicht im richtigen Format sind
data['time'] = data['time'].astype(np.int64) // 10**6


k = 5
all_losses_mean = pd.DataFrame()
all_val_losses_mean = pd.DataFrame()
path_csv_loss = os.path.join(directory, 'csv_loss2.csv')
path_csv_val_loss = os.path.join(directory, 'csv_val_loss2.csv')

if 1==1:#(os.path.exists(path_csv_loss) and os.path.exists(path_csv_val_loss)):

    for i in range(1, 11, 1):

        percent = i/10
        downSampling = int(len(data) * percent)
        dataDownSampling = data.iloc[:downSampling]

        all_losses_collect = pd.DataFrame()
        all_val_losses_collect = pd.DataFrame()

        for j in range(0,k,1):
            print(len(dataDownSampling))
            print(i)

            nameModell = "DeepLearning2" + f"household_power_consumption.csv_d{i}_v{j}_v2" + ".keras"
            pathFolder = os.path.join(directory, nameModell)

            #print('------------------')
            print(percent)
            # Feature und Ziel trennen
            X = dataDownSampling[['time', 'value']].iloc[:-1].values
            y = dataDownSampling['value'].iloc[1:].values

            """def sequence_generator(downSampling, window_size, batch_size, scaler_X=None, scaler_y=None):
                size = len(data) - window_size
                i = 0

                # Wenn keine Skalierer übergeben werden, neue erstellen
                scaler_X = scaler_X or StandardScaler()
                scaler_y = scaler_y or StandardScaler()

                # Fit nur einmal zu Beginn auf den gesamten Datenbereich
                full_X = data[['time', 'value']].values
                full_y = data[['value']].values
                scaler_X.fit(full_X)
                scaler_y.fit(full_y)

                while True:
                    X_list, y_list = [], []
                    for _ in range(batch_size):
                        if i >= size:
                            i = 0
                        X_window = data[['time', 'value']].iloc[i:i+window_size].values
                        y_value = data[['value']].iloc[i+window_size]

                        # Skalieren
                        X_scaled = scaler_X.transform(X_window)
                        #y_scaled = scaler_y.transform([[y_value]])[0][0]
                        y_scaled = scaler_y.transform(np.array([y_value]).reshape(1, -1))[0][0]
                        # Glätten
                        X_list.append(X_scaled.flatten())
                        y_list.append(y_scaled)

                        i += 1

                    # Spaltennamen erzeugen
                    col_names = [f'{col}_{j}' for j in range(window_size) for col in ['time', 'value']]
                    X = pd.DataFrame(X_list, columns=col_names)
                    y = pd.DataFrame(y_list, columns=['value'])

                    yield X.values, y.values

            def create_sequences(data, window_size, scaler_X, scaler_y):
                X, y = [], []
                for i in range(len(data) - window_size):
                    X_window = data[['time', 'value']].iloc[i:i+window_size].values
                    y_value = data['value'].iloc[i+window_size]

                    X_scaled = scaler_X.transform(X_window)
                    y_scaled = scaler_y.transform([[y_value]])[0][0]

                    X.append(X_scaled.flatten())
                    y.append(y_scaled)

                X_df = pd.DataFrame(X, columns=[f'{col}_{j}' for j in range(window_size) for col in ['time', 'value']])
                y_df = pd.DataFrame(y, columns=['value'])
                return X_df, y_df"""


            #Daten normalisieren
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)

            y = y.reshape(-1, 1)  #falls n�tig
            scaler_y = StandardScaler()
            y_scaled = scaler_y.fit_transform(y)

            #scaler_X, scaler_y = StandardScaler(), StandardScaler()

            #train_gen = sequence_generator(dataDownSampling, window_size=60, batch_size=128)
            #sample_X, _ = next(train_gen)
            #input_dim = sample_X.shape[1]

            # Aufteilen der Daten für Validierung
            #split = int(len(dataDownSampling) * percent)
            #train_data = dataDownSampling.iloc[:split]
            #test_data = dataDownSampling.iloc[split:]
            #val_data = train_data.iloc[-1440:]  # Übergang erhalten
            #train_data = train_data.iloc[:-1440]

            #Train- und Testdaten
            X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled, test_size=0.2, random_state=42)

            @register_keras_serializable()
            def rmse_loss(y_true, y_pred):
                return tf.sqrt(tf.reduce_mean(tf.square(y_pred - y_true)))

            if 0==1:#os.path.exists(pathFolder):

                    #loadModel = load_model(pathFolderPlusDatei)
                    print('hello')
                    loadModel = load_model(pathFolder, custom_objects={'rmse_loss': rmse_loss})
                    #prediction = loadModel.predict(X_test)

            else:

                    #Modell erstellen
                    model = Sequential([
                    Input(shape=(X_train.shape[1],)),  # Eingabeschicht
                    Dense(16, kernel_regularizer=regularizers.l2(0.001)),  # L2 Regularisierung
                    #BatchNormalization(),
                    LeakyReLU(alpha=0.1),
                    Dropout(0.1),  # Eingabeschicht
                    Dense(8, kernel_regularizer=regularizers.l2(0.001)),  # L2 Regularisierung
                    #BatchNormalization(),
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
                    history  = model.fit(data_generator((60),X_train, y_train), epochs = 128, validation_data = (X_test, y_test), steps_per_epoch = len(X_train) // (60), callbacks=[early_stopping])

                    #val_gen = sequence_generator(val_data, window_size=60, batch_size=128)

                    #history = model.fit(train_gen, steps_per_epoch=20, epochs=128, validation_data=(val_gen), validation_steps=20, callbacks=[early_stopping])

                    #Modell speichern
                    model.save(pathFolder)

                    #scaler_X = StandardScaler()
                    #scaler_y = StandardScaler()

                    #X_test, y_test = create_sequences(test_data, window_size=60)
                    #X_test_scaled = scaler.transform(X_test)
                    #y_test_scaled = scaler_y.transform(y_test.reshape(-1,1))
                    
                    #Vorhersage
                    prediction_scaled = model.predict(X_test)
                    prediction = scaler_y.inverse_transform(prediction_scaled)
                    df = pd.DataFrame(prediction)
                    df.to_csv(f'predictionDeeplearning2_d{i}_v{j}_v2.csv', index=True)

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

        if i == 10 :
            print('-----------------')
            print(i)
            

            
        if i == 10 :
            print('-----------------')
            
            

            
    #Daten in CSV-Datei speichern
    all_losses_mean.to_csv('csv_loss2_v2.csv', index=True)
    all_val_losses_mean.to_csv('csv_val_loss2_v2.csv', index=True)

    
csv_loss = pd.read_csv('csv_loss2_v2.csv', delimiter=",")
csv_val_loss = pd.read_csv('csv_val_loss2_v2.csv', delimiter=",")
print(csv_loss)
print(csv_val_loss)

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
        plt.savefig(fname=f'Trainings- und Validierungsverlust2_{k}_v2')
        plt.show(block=False)
        plt.pause(0.1)

plt.show()


#print(prediction)