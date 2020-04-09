import keras
from keras.datasets import mnist
batch_size = 128
num_classes = 10
epochs = 12

# input image dimensions
img_rows, img_cols = 28, 28

# the data, split between train and test sets
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.reshape(60000, 28, 28, 1)
x_test = x_test.reshape(10000, 28, 28, 1)

print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)
model = keras.equential()
model.add(keras.Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(28,28,1)))
model.add(keras.Conv2D(64, (3, 3), activation='relu'))
model.add(keras.MaxPooling2D(pool_size=(2, 2)))
model.add(keras.Dropout(0.25))
model.add(keras.Flatten())
model.add(keras.Dense(128, activation='relu'))
model.add(keras.Dropout(0.5))
model.add(keras.Dense(num_classes, activation='softmax'))
