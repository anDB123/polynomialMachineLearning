#polynomialOutput.py
import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from copy import deepcopy

minX=0
minToMax = 10
n=100
numberOfConstants = 4
sigma = 0.2
constantRange = 1

modelFile = './kerasPythonQuadratic'
inputDataFile = './inputData.npy'
outputDataFile = './outputData.npy'

def printModel(model,n,numberOfConstants,constantRange,ax1):
    test_array = np.zeros([n,2])
    params = np.zeros([numberOfConstants])
    for k in range(numberOfConstants):
        params[k] = np.random.normal(0,constantRange)
    for i in range(n):
        test_array[i][0] = minToMax*i/n+minX
        test_array[i][1]= f(test_array[i][0], params ) + np.random.normal(0, sigma)

    x_values = test_array[:,0]
    y_values = test_array[:,1]
    test_array = test_array[None, :]

    model_params = np.asarray(model(test_array))

    print("real params  = ")
    print(params)
    print("model params  = ")
    print(model_params[0])
    y_model = np.zeros([n,1])
    for i in range(n):
        y_model[i] = f(x_values[i],model_params[0])
    ax1.scatter(x_values,y_values, label = "polynomial output",s=1)
    ax1.plot(x_values,y_model, label = "machine learning output", color="green")

def f(x,pars):
    sum = 0
    pow = 0
    for par in pars:
        sum+=par* x**pow
        pow+=1
    return sum

model = keras.Sequential()
inputVectors = []
outputVectors = []
if(Path(inputDataFile).is_file() and Path(outputDataFile).is_file()):
    inputVectors = np.load(inputDataFile)
    outputVectors = np.load(outputDataFile)
else:
    #generate data
    numberOfSamples = 1000
    for j in range(0,numberOfSamples):
        pars = np.zeros([numberOfConstants])
        for k in range(numberOfConstants):
            pars[k] = np.random.normal(0,constantRange)
        inputVector = np.zeros([n,2])
        for i in range(n):
            inputVector[i][0] = minToMax*i/n+minX
            inputVector[i][1] = f(inputVector[i][0], pars ) + np.random.normal(0, sigma)
        print("completed sample %d"%j)
        inputVectors.append(inputVector)
        outputVectors.append(pars)
    np.save(inputDataFile,inputVectors)
    np.save(outputDataFile,outputVectors)
    
if(Path(modelFile).is_dir()):
    model = keras.models.load_model(modelFile)
else:
    #generate model
    
    model = keras.Sequential([
        keras.layers.Flatten(input_shape=[n,2]),
        keras.layers.Dense(8*n),
        keras.layers.Dense(8*n, activation = "sigmoid"),
        keras.layers.Dense(8*n),
        keras.layers.Dense(numberOfConstants, name='output'),
    ])
    model.compile(optimizer = "adam",
        metrics = ['accuracy'],
        loss = 'mse'   # mean square error\
    )

epochs = 500
inputVectors = np.array(inputVectors)
outputVectors = np.array(outputVectors)
print(inputVectors)
#compile model

model.fit(inputVectors,outputVectors,
    epochs = epochs, # number of iteration
)

model.save(modelFile)

rows = 3
cols = 4
fig, axs = plt.subplots(rows, cols)

for row in range (rows):
    for col in range(cols):
        printModel(model,n,numberOfConstants,constantRange,axs[row, col])
#fig.legend()
plt.show()