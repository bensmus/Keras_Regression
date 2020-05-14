'''
This program aims to create a continuous relationship between temperature 
and time (in a geographical region) given yearly weather data from:
https://climatedata.ca/download/
'''

# Keras is a high level neural network API 
from tensorflow import keras

# Graphing
import matplotlib.pyplot as plt

# MATLAB-like arrays
import numpy as np

# Reading csv
import csv

# Date, Latitude, Longitude, RCP 2.6 Range (low), RCP 2.6 Median, RCP 2.6 Range (high), RCP 4.5 Range (low), RCP 4.5 Median, RCP 4.5 Range (high), RCP 8.5 Range (low), RCP 8.5 Median, RCP 8.5 Range (high)

max_temp = []
years = []

with open('max_temp_data.csv', newline='') as data:
    f = list(csv.reader(data))

    # use slices to not include the csv header
    for i, row in enumerate(f[1:]):

        # the fifth element in the row is RCP 2.6 Median
        max_temp.append(float(row[5]))
        years.append(i)

# creating the neural net
inputs = keras.Input(shape=(1))
h = keras.layers.Dense(10, activation='sigmoid')(inputs)
outputs = keras.layers.Dense(1)(h)
model = keras.Model(inputs=inputs, outputs=outputs)
model.summary()

# having an accuracy metric is pointless since the accuracy will most likely be zero
# accuracy makes sense for classication problems

model.compile(optimizer='sgd', loss='mse')
model.fit(years, max_temp, epochs=100)

plt.xlabel('years since 1950')
plt.ylabel('maximum temperatures in YVR region')

plt.scatter(years, max_temp, label='given')

times = np.linspace(0, 150, 10000)
test_max_temp = model.predict(times)
plt.plot(times, test_max_temp, label='interpolated')
plt.legend()

plt.show()

