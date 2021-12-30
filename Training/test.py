import ssl
import numpy as np
from tensorflow.keras.datasets import cifar10
from sklearn.metrics import classification_report
from tensorflow import keras

# path to model to load
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
    
    return x_test, y_test, class_names

x_test, y_test, class_names = dataset()
model = keras.models.load_model(model_path)
print()
model.summary()
print("\nTest:")
y_prob = model.predict(x_test, batch_size=128)
y_pred = [np.argmax(x) for x in y_prob]
print(classification_report(y_test, y_pred, target_names=class_names, digits=4))
