# Tensorflow Speech Recognition Challenge

Kaggle: https://www.kaggle.com/c/tensorflow-speech-recognition-challenge

#### Task

Build an algorithm to understand simple speech commands. 

#### Dataset

A subset of 5 speech commands (i.e. classes) were used. There was a total of 6101 observations across these 5 classes.

## Result

A decent validation accuracy of 0.93202 was achieved. The model used consisted of 4 convolutional layers, 1 flatten layer and 2 fully connected layers.

```
inputs = Input(shape=(8000,1))

x = Conv1D(8,13, padding='valid', activation='relu', strides=1)(inputs)
x = MaxPooling1D(3)(x)
x = Dropout(0.3)(x)

x = Conv1D(16, 11, padding='valid', activation='relu', strides=1)(x)
x = MaxPooling1D(3)(x)
x = Dropout(0.3)(x)

x = Conv1D(32, 9, padding='valid', activation='relu', strides=1)(x)
x = MaxPooling1D(3)(x)
x = Dropout(0.3)(x)

x = Conv1D(64, 7, padding='valid', activation='relu', strides=1)(x)
x = MaxPooling1D(3)(x)
x = Dropout(0.3)(x)

x = Flatten()(x)

x = Dense(256, activation='relu')(x)
x = Dropout(0.3)(x)

x = Dense(128, activation='relu')(x)
x = Dropout(0.3)(x)

outputs = Dense(len(labels), activation='softmax')(x)

model = Model(inputs, outputs)
```
