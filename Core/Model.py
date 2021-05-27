import os
from keras.models import Sequential, load_model
from keras.layers.core import Dense, Dropout
from keras.layers.recurrent import LSTM
import datetime as dt
import numpy as np

class Model:
    def __init__(self):
        self.model = Sequential()

    def Build_model(self, configs):

        for layer in configs['model']['layers']:
            neurons = layer['neurons'] if 'neurons' in layer else None
            dropout_rate = layer['rate'] if 'rate' in layer else None
            activation = layer['activation'] if 'activation' in layer else None
            return_seq = layer['return_seq'] if 'return_seq' in layer else None
            input_timesteps = layer['input_timesteps'] if 'input_timesteps' in layer else None
            input_dim = layer['input_dim'] if 'input_dim' in layer else None

            if layer['type'] == 'dense':
                self.model.add(Dense(neurons, activation=activation))
            if layer['type'] == 'lstm':
                self.model.add(LSTM(neurons, input_shape=(input_timesteps, input_dim), return_sequences=return_seq))
            if layer['type'] == 'dropout':
                self.model.add(Dropout(dropout_rate))

        self.model.compile(loss=configs['model']['loss'], optimizer=configs['model']['optimizer'], metrics=['accuracy'])

    def Train_model(self,  x, y, epochs, batch_size):
        os.chdir(os.path.dirname(os.path.abspath(__file__)))
        if not os.path.exists("Save_model"):
            os.mkdir("Save_model")
        save_filename = os.path.join('Save_model',
                                     '%s-e%s.h5' % (dt.datetime.now().strftime('(%d_%m_%Y)-(%H_%M_%S)'),
                                     str(epochs)))
        self.model.fit(x,
                       y,
                       epochs=epochs,
                       batch_size=batch_size)
        self.model.save(save_filename)

    def Load_model(self, filepath):
        self.model = load_model(filepath)

    def Evaluate_model(self, x):
        out = self.model.predict(x)
        return out

    def Forecast_model(self, x, time_future, transform):
        """
        :param x: input data with shape (1,feature,sequence)
        :param future: number of forecast data is integer
        :param transform: transform of normalize data
        :return: list of forecast data

        """
        out = np.zeros(time_future)
        feature = x.shape[1]
        sequence = x.shape[2]
        for i in range(time_future):
            forecast_next_day = self.model.predict(x)[:, 0]
            forecast_next_day_main = transform.inverse_transform(forecast_next_day.reshape(-1, 1))
            out[i] = forecast_next_day_main[0][0]
            x = np.delete(x, 0)
            x = np.append(x, forecast_next_day)
            x = x.reshape((1, feature, sequence))
        return out
