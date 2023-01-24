import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

def polynomialEq(x):
    return 5*x**2 + 3*x +1

#define inputs and outputs
inputs = np.arange(0,1,0.001)
outputs = []
for input in inputs:
    outputs.append( polynomialEq(input))

scaling = np.max(outputs)
outputs = outputs/scaling


#define model

model = keras.Sequential([
  keras.layers.Input(1, name='input'),
  keras.layers.Dense(10),
  keras.layers.Dense(10),
  keras.layers.Dense(10),
keras.layers.Dense(10),
  keras.layers.Dense(10),
  keras.layers.Dense(10),
  keras.layers.Dense(10),
  keras.layers.Dense(10),
  keras.layers.Dense(10),
  keras.layers.Dense(1, name='output'),
])
model.build()
model.summary()

testing = False
epochs = 5

#compile model
model.compile(
    optimizer="adam",
    loss=tf.keras.losses.BinaryCrossentropy(),
    metrics=None,
    loss_weights=None,
    weighted_metrics=None,
    run_eagerly=None,
    steps_per_execution=None,
    jit_compile=None,)


#fit the model
model.fit(inputs, outputs, epochs=epochs)

#evaluate the model
test_acc = model.evaluate(inputs, outputs)

#print results
print('\nTest accuracy: {}'.format(test_acc))
x_array =  np.arange(0,1,0.01)
y_actual = (polynomialEq(x_array))
y_model = model.predict(x_array)*scaling
plt.plot(x_array,y_actual, label = "polynomial output")
plt.plot(x_array,y_model, label = "machine learning output")
plt.legend()
plt.show()