import pandas as pd
import numpy as np
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.metrics import root_mean_squared_error
from scipy import stats
import os
import matplotlib.pyplot as plt

x_values = [48, 96, 192, 672, 1344]
#y_values = [48, 96, 192, 672, 1344]
all_exist = all(os.path.exists(f'mape{x_values}_f{x_values}.csv') for x in x_values)
all_exist2 = all(os.path.exists(f'rmse{x_values}_f{x_values}.csv') for x in x_values)
#directory = os.getcwd()
#path_mape = os.path.join(directory, f'mape{*}.csv')
#path_rmse = os.path.join(directory, f'rmse{*}.csv')


if 0==1:# not(all_exist and all_exist2):

    

    # CSV laden
    data = pd.read_csv('household_power_consumption.csv', delimiter=';', low_memory=False)

    #Dateien benennen
    data['time'] = pd.to_datetime(data['Date'] + " " + data['Time'], dayfirst=True)
    data.rename(columns={'Global_active_power': 'value'}, inplace=True)
    data = data.dropna(subset=['value'])
    



    # Zeit und Werte bereinigen
    data['value'] = pd.to_numeric(data['value'], errors='coerce')
    data = data.dropna(subset=['value'])
    data = data[data['value'] > 0]  # wichtig für multiplicative Methoden
    data.set_index('time', inplace=True)  # nötig für Forecasts

    data = data.sort_index()
    print(data.index.to_series().diff().value_counts())
    # Optional: Ausreißer entfernen
    z_scores = np.abs(stats.zscore(data['value']))

    data = data[z_scores < 3]
    print(data.index.to_series().diff().value_counts())
    # Resampling falls nötig (z. B. auf 15-minütige Intervalle)
    data = data[['value']].resample('1min').mean()


    data = data.dropna(subset=['value'])
    data = data[data['value'] > 0]

    #mape_ = pd.DataFrame()
    #rmse_ = pd.DataFrame()
    print(data.index.to_series().diff().value_counts())

    for sp in [15, 60, 1440]:
        for k in [15, 60, 1440, 10080]:
            mape_collect = pd.DataFrame()
            rmse_collect = pd.DataFrame()
            for i in range(1, 11, 1):

                #Downsampling
                percent = i/50
                print(len(data))
                downSampling = int(len(data) * percent)
                dataDownSampling = data.iloc[:downSampling]
                print(len(dataDownSampling))

                # Trainingsdaten
                train_df = dataDownSampling.tail(int(len(data) * 0.8))


                # Modelltraining
                model = ExponentialSmoothing(train_df['value'],
                                             trend='mul',
                                             seasonal='mul',
                                             seasonal_periods=sp,
                                             initialization_method="estimated").fit()



                # Vorhersage für 1000 Schritte (~10 Tage)
                forecast = model.forecast(k)
                forecast.to_csv(f'ExponentialSmoothingDatas2/forecast{sp}_f{k}_d{i}_v2.csv', index=True)

                #last_time = train_df.index[-1]
                #forecast.index = pd.date_range(start=last_time + pd.Timedelta(minutes=15), periods=1344, freq='15min')
    

                # MAE berechnen
                fitted = model.fittedvalues
                rmse = root_mean_squared_error(train_df['value'].loc[fitted.index], fitted)
                print(rmse/np.mean(forecast)*100)
                actual_values = train_df['value'].loc[fitted.index]
                mape = np.mean(np.abs((actual_values - fitted) / actual_values)) * 100
                print(f"MAPE: {mape:.2f}%")

                history = mape
                history2 = rmse*100

                mape_list = []
                rmse_list = []

                mape_list.append(mape)
                rmse_list.append(rmse/np.mean(forecast))

                mape_ = pd.DataFrame(mape_list, columns=[f'{i*10}'])
                rmse_ = pd.DataFrame(rmse_list, columns=[f'{i*10}'])

                mape_collect = pd.concat([mape_collect, mape_], axis=1)
                rmse_collect = pd.concat([rmse_collect, rmse_], axis=1)

            print(mape_collect)
            print(rmse_collect)

            mape_collect.to_csv(f'ExponentialSmoothingDatas2/mape{sp}_f{k}_v2.csv', index=True)
            rmse_collect.to_csv(f'ExponentialSmoothingDatas2/rmse{sp}_f{k}_v2.csv', index=True)

# Fehlerverlauf über die Zeit plotten

for sp in [15, 60, 1440]:
    plt.figure(figsize=(10, 5))
    for k in [15, 60, 1440, 10080]:
        #print(sp)
        #print(k)
        mape_read = pd.read_csv(f'ExponentialSmoothingDatas2/mape{sp}_f{k}_v2.csv', delimiter=",")
        rmse_read = pd.read_csv(f'ExponentialSmoothingDatas2/rmse{sp}_f{k}_v2.csv', delimiter=",")

        #print(mape_read)
        #print(rmse_read)

        
        plt.plot(mape_read.columns, mape_read.iloc[0], label= f'Mape_f{k}', marker='o')
        plt.plot(rmse_read.columns, rmse_read.iloc[0], label= f'RMSE_f{k}', marker='o')
        plt.xlabel('Downsampling in Prozent')
        plt.ylabel('Loss (RMSE & MAPE)')
        plt.title(f'Trainings- und Validierungsverlust_{sp}')
        plt.legend()
        plt.yticks(np.arange(0,15,1))
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(fname=f'ExponentialSmoothingDatas2/ExponentialSmoothing_{sp}_v2')
        plt.show(block=False)
        plt.pause(1)

plt.show()


