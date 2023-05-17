from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd

from database import db_connector

db_connector.connect()

df = db_connector.run_query('SELECT * FROM transform.rates')

# Assume we have a dataframe `df` with features and labels

# Convert categorical variable into dummy/indicator variables.
df = pd.get_dummies(df, columns=['symbol'])

# Assume 'Label' is the column with the labels
labels = df.pop('Label')

# Scale the features to be between 0 and 1
scaler = MinMaxScaler()
df = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)

# Convert the dataframe into a 3D array (samples, timesteps, features) for LSTM
# Here we're treating each single row as a sequence.
# If you want to consider a sequence of several rows (e.g., several quarterly reports), you'll need to adjust this.
data = np.expand_dims(df.values, axis=1)

# Split data into training and testing datasets
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)

# Define the model architecture
model = Sequential()
model.add(LSTM(50, activation='relu', input_shape=(data.shape[1], data.shape[2])))
model.add(Dense(1, activation='sigmoid'))

# Compile the model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Fit the model
history = model.fit(X_train, y_train, epochs=20, validation_data=(X_test, y_test))

# Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test, verbose=2)

print("\nTest accuracy:", accuracy)
