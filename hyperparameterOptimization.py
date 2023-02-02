#polynomialOutput.py
import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from copy import deepcopy

minX=-10
minToMax = 20
n=10
numberOfConstants = 4
sigma = 0.2
constantRange = 1


inputDataFile = './inputData.npy'
outputDataFile = './outputData.npy'


def f(x,pars):
    sum = 0
    pow = 0
    for par in pars:
        sum+=par* x**pow
        pow+=1
    return sum




inputVectors = []
outputVectors = []
if(Path(inputDataFile).is_file() and Path(outputDataFile).is_file()):
    inputVectors = np.load(inputDataFile)
    outputVectors = np.load(outputDataFile)
else:
    #generate data
    numberOfSamples = 100000
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


modelArray = []
historyArray = []
modelParamArray = []
layerRange = 6
sizeRange = 6
model = keras.Sequential()
for hiddenLayers in range(1,layerRange):
    for hiddenSize in range(1,sizeRange):
        modelFile = './kerasPythonQuadraticHL{}HS{}'.format(hiddenLayers, hiddenSize)
        if(Path(modelFile).is_dir()):
            model = keras.models.load_model(modelFile)
        else:
            #generate model
            
            model = keras.Sequential()
            model.add(keras.layers.Flatten(input_shape=[n,2]))
            for layer in range(hiddenLayers):
                model.add(keras.layers.Dense(hiddenSize*n))
            model.add(keras.layers.Dense(numberOfConstants, name='output'))
            
            model.compile(optimizer = "adam",
                metrics = ['accuracy'],
                loss = 'mse'   # mean square error\
            )

        epochs = 50
        inputVectors = np.array(inputVectors)
        outputVectors = np.array(outputVectors)
        #compile model

        history = model.fit(inputVectors,outputVectors,
            epochs = epochs, # number of iteration
            validation_split = 0.2
        )

        model.save(modelFile)
        modelArray.append(model)
        historyArray.append(history)
        modelParamArray.append([hiddenLayers,hiddenSize])



accuracyArray = np.zeros(shape=[len(historyArray)])
for i in range(len(historyArray)):
    accuracy = historyArray[i].history['accuracy']
    print(accuracy)
    accuracyArray[i] = accuracy[0]
    
accuracyArray = np.reshape(accuracyArray, (layerRange-1, sizeRange-1))
plt.imshow(accuracyArray, cmap='hot', interpolation='nearest')
plt.show()