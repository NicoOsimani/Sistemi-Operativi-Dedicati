import ssl
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import cifar10
from sklearn.metrics import classification_report

# epochs num
epochs = 50

# number of times the train is done
repetitions = 10

# path to model to save
model_path = './vgg.h5'

# define which dataset to use
def dataset():
    ssl._create_default_https_context = ssl._create_unverified_context

    # loading the dataset
    (x_train,y_train),(x_test,y_test)=cifar10.load_data()

    # setting class names
    class_names=['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']

    # normalize the test data
    x_test=x_test/255.0
    
    # creating dataset object
    train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    train_ds = (
        train_ds
        .shuffle(1024)
        .map(scale)
        .map(augment)
        .batch(128)
    )

    test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test))
    test_ds = (
        test_ds
        .batch(128)
    )

    return train_ds, test_ds, y_test, class_names

# define model
def _model():
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same', input_shape=(32, 32, 3)))
    model.add(tf.keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
    model.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))
    model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
    model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
    model.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))
    model.add(tf.keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
    model.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(128, activation='relu', kernel_initializer='he_uniform'))
    model.add(tf.keras.layers.Dense(10, activation='softmax'))

    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    # define callbacks
    train_callbacks = [
    tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=10,
        restore_best_weights=True
    ),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss", factor=0.5,
        patience=3, verbose=1
    )
    ]

    return model, train_callbacks

# utility to plot training and validation history
def plot_history(history):
    x_epochs = []
    for i in range(0,len(history.history["accuracy"])):
        x_epochs.append(i+1)	
    fig = plt.figure()
    plt.title('Train and Validation')
    x = list(range(0, len(history.history["accuracy"]), 1))
    plt.xticks(x, x_epochs)
    plt.plot(history.history["loss"], label="Training Loss")
    plt.plot(history.history["val_loss"], label="Validation Loss")
    plt.plot(history.history["accuracy"], label="Training Accuracy")
    plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
    plt.legend()
    plt.grid()
    plt.show()
    return fig

# scale image
def scale(image, label):
    image = tf.image.convert_image_dtype(image, tf.float32)
    label = tf.cast(label, tf.int32)
    return image, label

# data augmentation
def augment(image,label):
    image = tf.image.resize_with_crop_or_pad(image, 40, 40) # Add 8 pixels of padding
    image = tf.image.random_crop(image, size=[32, 32, 3]) # Random crop back to 32x32
    image = tf.image.random_brightness(image, max_delta=0.5) # Random brightness
    image = tf.clip_by_value(image, 0., 1.)
    return image, label

model, train_callbacks = _model()
print()
model.summary()
best_acc = -1.0
for i in range (0, repetitions):
    train_ds, test_ds, y_test, class_names = dataset()
    model, train_callbacks = _model()
    print("\nTrain " + str(i + 1) + ":")
    print("\nTrain and Validation:")
    history = model.fit(train_ds,
                      epochs=epochs,
                      callbacks=train_callbacks,
                      validation_data=test_ds)
    test_loss, test_accuracy = model.evaluate(test_ds)
    if (test_accuracy > best_acc):
        best_acc = test_accuracy
        best_model = model
        best_history = history
fig = plot_history(best_history)
fig.savefig("./Train and Validation.png")
best_model.save(model_path)
print("\nTest:")
y_prob = best_model.predict(test_ds)
y_pred = [np.argmax(x) for x in y_prob]
print(classification_report(y_test, y_pred, target_names=class_names, digits=4))
