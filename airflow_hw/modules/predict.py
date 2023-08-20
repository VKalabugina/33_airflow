import dill
import pandas as pd
import os
import json
from datetime import datetime


path = os.environ.get('PROJECT_PATH', '..')


def predict():
    predictions = []
    car_id = []
    mypath1 = f'{path}/data/models/'
    mypath2 = f'{path}/data/test/'

    for filename in os.listdir(mypath1):
        with open(os.path.join(mypath1, filename), 'rb') as file:
            model = dill.load(file)
            break

    for filename in os.listdir(mypath2):
        with open(os.path.join(mypath2, filename), 'r') as data_file:
            data = json.load(data_file)
        df = pd.json_normalize(data)
        y = model.predict(df)
        predictions.append(y[0])
        car_id.append(filename.split('.')[0])


    df_pred = pd.DataFrame(
        {'car_id': car_id,
         'pred': predictions
         })

    pred_filename = f'{path}/data/predictions/preds__{datetime.now().strftime("%Y%m%d%H%M")}.csv'
    df_pred.to_csv(pred_filename, index=False)


if __name__ == '__main__':
    predict()
