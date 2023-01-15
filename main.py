import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import regularizers
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers.experimental import preprocessing

url_dataset = "https://junwin.github.io/data/housepriceclean2.csv"
housePrices = pd.read_csv(url_dataset).sample(frac=1)
housePrices.pop('ClosedDate')

housePrices['Zip'] = housePrices['Zip'].astype(str)

train, test = train_test_split(housePrices, test_size=0.2)
train, val = train_test_split(train, test_size=0.2)
print(len(train), 'train examples')
print(len(val), 'validation examples')
print(len(test), 'test examples')

housePrices_features = train.copy()
housePrices_labels = housePrices_features.pop('SoldPr')
housePrices_labels = housePrices_labels / 100000

val_features = val.copy()
val_labels = val.pop('SoldPr')
val_labels = val_labels / 100000

inputs = {}

for name, column in housePrices_features.items():
    dtype = column.dtype
    if dtype == object:
        dtype = tf.string
    else:
        dtype = tf.float32

    inputs[name] = tf.keras.Input(shape=(1,), name=name, dtype=dtype)

numeric_inputs = {name: input for name, input in inputs.items()
                  if input.dtype == tf.float32}

x = layers.Concatenate()(list(numeric_inputs.values()))
norm = preprocessing.Normalization()
norm.adapt(np.array(housePrices[numeric_inputs.keys()]))
all_numeric_inputs = norm(x)

preprocessed_inputs = [all_numeric_inputs]

for name, input in inputs.items():
    if input.dtype == tf.float32:
        continue

    lookup = preprocessing.StringLookup(vocabulary=np.unique(housePrices_features[name]))
    one_hot = preprocessing.CategoryEncoding(max_tokens=lookup.vocab_size())

    x = lookup(input)
    x = one_hot(x)
    preprocessed_inputs.append(x)

preprocessed_inputs_cat = layers.Concatenate()(preprocessed_inputs)

housePrices_preprocessing = tf.keras.Model(inputs, preprocessed_inputs_cat)

housePrices_features_dict = {name: np.array(value) for name, value in housePrices_features.items()}

features_dict = {name: values[:1] for name, values in housePrices_features_dict.items()}


def housePrices_model(preprocessing_head, inputs):
    body = tf.keras.Sequential([
        layers.Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.001)),
        layers.Dense(64, activation='relu'),
        layers.Dense(1)
    ])

    preprocessed_inputs = preprocessing_head(inputs)
    result = body(preprocessed_inputs)
    model = tf.keras.Model(inputs, result)

    model.compile(loss='mse', optimizer='adam', metrics=['mae'])
    return model


housePrices_model = housePrices_model(housePrices_preprocessing, inputs)

val_features_dict = {name: np.array(value) for name, value in val.items()}
history_1 = housePrices_model.fit(x=housePrices_features_dict, y=housePrices_labels, epochs=250,
                                  validation_data=(val_features_dict, val_labels))
housePrices_model.save('model.h5')