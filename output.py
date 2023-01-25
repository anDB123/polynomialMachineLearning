import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt


def f(x,par):
   return par[0] + par[1]*x + par[2] * x**2
 
Par = (0.5, 1, 0.2)
sigma = 0.2
 
n = 200
a = np.zeros([n,2])
for i in range(n):
   a[i][0] = 10*(np.random.random_sample() - 0.5)
   a[i][1] = f( a[i][0], Par ) + np.random.normal(0, sigma)


#define inputs and outputs



#define model

model = keras.Sequential([
  keras.layers.Input(1, name='input'),
  keras.layers.Dense(100, activation = "relu"),
  keras.layers.Dense(1, name='output'),
])


testing = False
epochs = 5

#compile model
model.compile(
     optimizer = tf.optimizers.Adam(learning_rate=0.01),
      metrics = ['accuracy'],
      loss = 'mse'   # mean square error\
      )

print ("shape 1 = ")
print (a[:,0].shape)
print ("shape 2 =")
print (a[:,1].shape)
#fit the model
model.fit(
      a[:,0], a[:,1],
      epochs = 100, # number of iteration
      verbose = 0
)

#evaluate the model
model.summary()
test_acc = model.evaluate(a[:,0], a[:,1])

#print results
print('\nTest accuracy: {}'.format(test_acc))
x_array =  a[:,0]
y_actual = a[:,1]
x_model = np.arange(np.min(x_array),np.max(x_array),0.001)
y_model = model.predict(x_model)
plt.scatter(x_array,y_actual, label = "polynomial output",s=1)
plt.plot(x_model,y_model, label = "machine learning output", color="green")
plt.legend()
plt.show()