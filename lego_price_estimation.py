# Keras is a high level neural network API 
from tensorflow import keras

# Numerical python and graphing
import numpy as np
import matplotlib.pyplot as plt

# Getting the dataset from brickset.com
import csv

pieces = []
price = []

headers = [
    'SetID', 
    'Number', 
    'Variant', 
    'Theme', 
    'Subtheme', 
    'Year', 
    'Name', 
    'Minifigs', 
    'Pieces', 
    'UKPrice', 
    'USPrice', 
    'CAPrice', 
    'EUPrice', 
    'ImageURL', 
    'OwnedBy', 
    'WantedBy'
]

# make it a dictionary
headers = dict(zip(headers, list(range(len(headers)))))

with open('lego_price_data.csv', newline='') as data:
    f = csv.reader(data)
    for i, row in enumerate(f):
        if i != 0:
            if row[headers['CAPrice']] != '' and int(row[headers['Pieces']]) > 10:
                pieces.append(float(row[headers['Pieces']])) 
                price.append(float(row[headers['CAPrice']]))

# make a dictionary of piece count and price
pairs = dict(zip(pieces, price))

# sort the inputs in ascending order and convert to numpy
pieces = np.array(sorted(pieces))
price = np.array([pairs[a] for a in pieces])

# creating the neural net
inputs = keras.Input(shape=(1))
h = keras.layers.Dense(10, activation='relu')(inputs)
outputs = keras.layers.Dense(1)(h)
model = keras.Model(inputs=inputs, outputs=outputs)
model.summary()

model.compile(optimizer='adam', loss='mean_absolute_percentage_error', metrics='accuracy')
model.fit(pieces, price, epochs=100)

''''
predicted_prices = model.predict([500, 1000, 2000])
print(predicted_prices)
'''

plt.xlabel('piece count')
plt.ylabel('set price')

plt.scatter(pieces, price, label='given data')

test_pieces = np.linspace(0, 10000)
test_prices = model.predict(test_pieces)
plt.plot(test_pieces, test_prices, label='predicted')
plt.legend()

plt.show()