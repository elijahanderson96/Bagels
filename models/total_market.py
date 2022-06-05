import pandas as pd
import tensorflow as tf
import logging
import os
from sklearn.preprocessing import MinMaxScaler
from dotenv import load_dotenv
from datetime import datetime
from iex_api.transforms import fetch_dataset
from tensorboard.plugins.hparams import api as hp

load_dotenv()

logging.basicConfig(format='%(name)s - %(levelname)s - %(message)s', level=logging.DEBUG)
logger = logging.getLogger(__name__)


class Model:
    def __init__(self, model_type='total', batch_size=8, epochs=50, learning_rate=0.00001):
        self.model = None
        self.model_type= model_type
        self.batch_size = batch_size
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.optimizer  = tf.keras.optimizers.Adam(learning_rate=0.00001)
        self.early_stopping = tf.keras.callbacks.EarlyStopping(monitor='loss',verbose=5,mode='min',patience=4)

        self.dataset = fetch_dataset(self.model_type)

        self.num_units = hp.Hparam('num_units', hp.Discrete([32,64,128,256]))
        self.drop_out  = hp.Hparam('dropout',hp.RealInterval(.1, .2, .3))
        self.metric = hp.Metric('loss',display_name='Loss')

        with tf.summary.create_file_writer('logs/hparam_tuning').as_default():
            hp.params_config(
                hparams=[self.num_units,self.drop_out],
                metrics=[self.metric]
            )

    def train_test_model()
                                       
                                
    def one_hot_encode(self):
        dum_df = pd.get_dummies(self.dataset['symbol'],columns=['symbol'])
        dum_df_prediction = pd.get_dummies(self.prediction_dataset['symbol'],columns=['symbol'])
        self.dataset = self.dataset.join(dum_df)
        self.dataset = self.dataset.join(self.close)
        self.prediction_dataset = self.prediction_dataset.join(dum_df_prediction)
        self.prediction_dataset = self.prediction_dataset.join(self.close.head(len(self.prediction_dataset))) # super jank
        logger.info(f'The training dataset now has shape {self.dataset.shape} after one hot encoding')
        logger.info(f'The prediction dataset now has shape {self.prediction_dataset.shape} after one hot encoding')
    
    def normalize(self):
        self.dataset.fillna(0)
        self.prediction_dataset.fillna(0)
        self.dataset = self.dataset._get_numeric_data()
        self.prediction_dataset = self.prediction_dataset._get_numeric_data()
        self.columns = self.dataset.columns.to_list()
        self.scaler = MinMaxScaler()
        self.normalized_dataset = self.scaler.fit_transform(self.dataset)
        self.normalized_prediction_dataset = self.scaler.transform(self.prediction_dataset)

    def batch(self):
        self.training_data = tf.data.Dataset.from_tensor_slices((self.normalized_dataset[:,:-1],
        self.normalized_dataset[:,-1]))
        self.training_dataset = self.training_data.batch(8,drop_remainder=True).batch(1,drop_remainder=True)
        self.prediction_dataset_batched = tf.data.Dataset.from_tensor_slices((self.normalized_prediction_dataset[:,:-1],
        self.normalized_prediction_dataset[:,-1]))
        self.prediction_dataset_batched = self.prediction_dataset_batched.batch(1,drop_remainder=True).batch(1, drop_remainder=True)

    def train(self):
        self.model = tf.keras.models.Sequential([
                     tf.keras.layers.LSTM(64, input_shape=(None,self.dataset.shape[1] - 1)),
                     tf.keras.layers.Dropout(.20),
                     tf.keras.layers.Dense(64, activation='relu'),
                     tf.keras.layers.Dense(1)])
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.00001)
        self.early_stopping = tf.keras.callbacks.EarlyStopping(monitor='loss',verbose=5,mode='min',patience=4)
        self.model.compile(optimizer=self.optimizer,loss='mse')
        self.model.fit(self.training_dataset,verbose=2,epochs=1,callbacks=self.early_stopping)

    def predict(self):
        self.training_preds = self.training_data.batch(1,drop_remainder=True).batch(1,drop_remainder=True)
        self.training_points = self.model.predict(self.training_preds)
        self.predictions = self.model.predict(self.prediction_dataset_batched)

    def renormalize(self):
        x = list(self.training_preds.as_numpy_iterator()) # do not renormalize any training data thats not in batches of 1
        z = list(self.prediction_dataset_batched.as_numpy_iterator())
        train_df = {}
        prediction_df = {}

        for i in range(len(x)):
            values = x[i][0][0][0]
            train_df.update({i: values})
        for i in range(len(z)):
            values = z[i][0][0][0]
            prediction_df.update({i: values})
        
        df_train = pd.DataFrame(train_df).transpose()
        renormalize_train = pd.DataFrame(self.training_points)

        df_prediction = pd.DataFrame(prediction_df).transpose()
        renormalize_prediction = pd.DataFrame(self.predictions)

        dataframe_to_inverse_transform_train = pd.concat([df_train, renormalize_train], axis=1, ignore_index=True)
        renormalized_training_values = self.scaler.inverse_transform(dataframe_to_inverse_transform_train)
        dataframe_to_inverse_transform_actual = pd.concat([df_prediction, renormalize_prediction], axis=1, ignore_index=True)
        renormalized_predicted_values = self.scaler.inverse_transform(dataframe_to_inverse_transform_actual)

        self.df_train = pd.DataFrame(renormalized_training_values[:, :], columns=self.columns).rename(
        columns={'marketCap': 'marketCap_Predicted'})
        self.df_actual = pd.DataFrame(renormalized_predicted_values[:, :], columns=self.columns).rename(
        columns={'marketCap': 'marketCap_Predicted'})

        self.df_train['marketCap'] = self.close
        logger.info(f'{self.df_train.head(5)}')
        logger.info(f'{self.df_actual.head(5)}')

    def marketCap_to_sharePrice(self):
        df = pd.DataFrame()
        self.shares_outstanding = pd.read_sql('SELECT symbol,"sharesOutstanding" FROM stock_prices;',
                                     self.db_con).drop_duplicates()
                                     
        for symbol in self.shares_outstanding['symbol']:
            shares_outstanding = self.shares_outstanding.loc[self.shares_outstanding['symbol']==symbol]['sharesOutstanding']
            temp = self.df_train.loc[self.df_train[symbol] == 1]
            temp_pred = self.df_actual.loc[self.df_actual[symbol] == 1]
            temp_pred['Price_Predicted'] = temp_pred['marketCap_Predicted'].apply(lambda x: x / shares_outstanding)
            temp['Price_Predicted'] = temp['marketCap_Predicted'].apply(lambda x: x / shares_outstanding)
            temp['Price_Actual']    = temp['marketCap'].apply(lambda x: x / shares_outstanding)
            df = pd.concat([df, temp, temp_pred])

        
        df['dateOffset'] = self.Date
        df.to_sql('test',self.db_con,if_exists='replace')
        


if __name__=='__main__':
    Model = Model()
    Model.one_hot_encode()
    Model.normalize()
    Model.batch()
    Model.train()
    Model.predict()
    Model.renormalize()
    Model.marketCap_to_sharePrice()

