import tensorflow as tf
from tensorflow import keras
print("Hello World")

def polynomialEq(x):
    return 5*x**2 + 3*x +1

inputs = [0,1,2,3,4,5,6,7,8,9]
outputs = []
for input in inputs:
    outputs.append( polynomialEq(input))

#get data
fashion_mnist = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

# scale data
train_images = train_images / 255.0
test_images = test_images / 255.0

# reshape from 2d to 1d array
train_images = train_images.reshape(train_images.shape[0], 28, 28, 1)
test_images = test_images.reshape(test_images.shape[0], 28, 28, 1)

#define labels
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

print('\ntrain_images.shape: {}, of {}'.format(train_images.shape, train_images.dtype))
print('test_images.shape: {}, of {}'.format(test_images.shape, test_images.dtype))


#define model
model = keras.Sequential([
  keras.layers.Conv2D(input_shape=(28,28,1), filters=8, kernel_size=3, 
                      strides=2, activation='relu', name='Conv1'),
  keras.layers.Flatten(),
  keras.layers.Dense(10, name='Dense')
])
model.summary()

testing = False
epochs = 5

#compile model
model.compile(optimizer='adam', 
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=[keras.metrics.SparseCategoricalAccuracy()])

#fit the model
model.fit(train_images, train_labels, epochs=epochs)

#evaluate the model
test_loss, test_acc = model.evaluate(test_images, test_labels)

#print results
print('\nTest accuracy: {}'.format(test_acc))