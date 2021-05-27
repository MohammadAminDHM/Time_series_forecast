import os
import numpy as np
from sklearn.metrics import mean_squared_error
from Core.Prepare_Data import Prepare_Data
from Core.Model import Model
from Core.SMAPE_error import SMAPE_error
from Core.Visualize_evaluation import plot_data
from pandas import read_csv
import json

def main():
    configs = json.load(open('config.json', 'r'))

    # Prepare Data

    data = Prepare_Data(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'Data', configs['data']['filename_in']),
                        configs['sequence_number'],
                        configs['feature_number'],
                        configs['data']['test_number'])
    X_train, y_train, X_test, y_test, transform = data.get_train_test_data()

    # Model

    model = Model()
    model.Build_model(configs)
    model.Train_model(
        X_train,
        y_train,
        epochs=configs['training']['epochs'],
        batch_size=configs['training']['batch_size']
    )

    # Model Evaluate

    y_train_predict = model.Evaluate_model(X_train)[:, 0]
    y_test_predict = model.Evaluate_model(X_test)[:, 0]

    plot_data(y_train, y_train_predict, "Train Data")
    plot_data(y_test, y_test_predict, "Test Data")

    print('smape score train data : \t', str(SMAPE_error(y_train_predict, y_train)))
    print('smape score test data : \t', str(SMAPE_error(y_test_predict, y_test)))
    print('MSE train data : \t', str(mean_squared_error(y_train_predict, y_train)))
    print('MSE test data : \t', str(mean_squared_error(y_test_predict, y_test)))

    # Forecast future day sale with obtained model

    forecast = read_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'Data', configs['data']['filename_out']))
    start = np.array([X_test[-1]])
    a = model.Forecast_model(start, 30, transform)
    forecast.sale = a
    forecast.to_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                 'Data', 'Complete_output.csv'), index=False, header=True)

if __name__ == '__main__':
    main()

