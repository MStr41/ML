# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import os
import json
import re
import pandas as pd


directory = os.getcwd()

jsonPath = os.path.join(directory,'jsonDatas', 'metric_results.json')
print(jsonPath)


with open (jsonPath) as file:

    data = json.load(file)
    i = 0
    
    df = pd.DataFrame(data) 
       
    #df.iloc[:,i].to_csv(datas + '.csv', index=True)

    plt.figure(figsize=(10, 5))
       
    plt.plot((df.index), df, label= df.columns,linestyle='-')

    plt.xlabel('Prozentualer Anteil')
    plt.ylabel('Quali')
    plt.title('Trainings- und Validierungsverlust')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
       #plt.savefig(fname=f'Trainings- und Validierungsverlust_{k}')
    plt.show(block=False)
    plt.pause(0.1)
       
    i+=1

plt.show()