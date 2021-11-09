import cv2
import numpy as np
import os
import tflearn
import tensorflow as tf
import matplotlib.pyplot as plt

from random import shuffle
from random import choice
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression
from tqdm import tqdm

from .config import TRAIN_DIR
from .config import TEST_DIR
from .config import ARTISTS
from .config import ART_BASE_DIR
from .config import LR
from .config import IMG_SIZE
from .config import CONV_LAYERS
from .config import MODEL_NAME
from .config import TRAIN_DATA_FILE
from .config import TEST_DATA_FILE
from .config import TRAIN_RATIO


def label_img(img_filename):
    # Label definition:
    # dog       [0,1]
    # not_dog   [1,0]
    if img_filename[0] == 'n': # dog img files follow format nXXXXXXX
        return [0,1]
    return [1,0]


def create_train_data():
    print("Loading Train Data")
    if os.path.exists(TRAIN_DATA_FILE):
        print('Loading from file')
        return np.load(TRAIN_DATA_FILE, allow_pickle=True)

    training_data = []
    for img in tqdm(os.listdir(TRAIN_DIR)): # TQDM TO MAKE IT PRETTY WHILE LOADING
        label = label_img(img)
        path = os.path.join(TRAIN_DIR, img)
        img = resize_image(path)
        if not len(img):
            continue
        training_data.append([np.array(img), np.array(label)])
    shuffle(training_data)
    # TO RUN THE FUNCTION JUST ONCE
    np.save(TRAIN_DATA_FILE, training_data)
    print('Training with %d data size' % len(training_data))
    return training_data


def process_test_data():
    testing_data = []
    for img_filename in tqdm(os.listdir(TEST_DIR)):
        path = os.path.join(TEST_DIR, img_filename)
        img = resize_image(path)
        if not len(img):
            continue
        testing_data.append([np.array(img), img_filename])
    shuffle(testing_data)
    print('Testing with %d data size' % len(testing_data))
    np.save(TEST_DATA_FILE, testing_data)
    return testing_data


def show_results(test_data, model):
    fig = plt.figure()
    pos = 1
    for data in test_data:
        img_data = data[0]
        orig = img_data
        data = img_data.reshape(IMG_SIZE, IMG_SIZE, 1)
        
        model_out = model.predict([data])[0]

        if np.argmax(model_out) == 1:
            pos += 1
            y = fig.add_subplot(8, 8, pos)
            str_label = '[DOG]'
            y.imshow(orig, cmap='gray')
            plt.title(str_label)
            y.axes.get_xaxis().set_visible(False)
            y.axes.get_yaxis().set_visible(False)        
    plt.show()


def get_prediction_results(test_data, model):
    results = []
    for data in test_data:
        img_file = data[1]
        img_data = data[0]
        data = img_data.reshape(IMG_SIZE, IMG_SIZE, 1)
        
        model_out = model.predict([data])[0]

        if np.argmax(model_out) == 1:
            results.append(img_file)
    
    return results


def create_model():
    print('Creating Convolutional DNN with %d layers' % CONV_LAYERS)
    convnet = input_data(shape=[None, IMG_SIZE, IMG_SIZE, 1], name='input')

    # Convolutional 2D layers
    # Alternate between 32 and 64 conv filters
    for layer in range(CONV_LAYERS):
        if layer % 2 == 0:
            convnet = conv_2d(convnet, 32, 2, activation='relu')
            convnet = max_pool_2d(convnet, 2)
        else:
            convnet = conv_2d(convnet, 64, 2, activation='relu')
            convnet = max_pool_2d(convnet, 2)

    convnet = fully_connected(convnet, 1024, activation='relu')
    convnet = dropout(convnet, 0.8)

    convnet = fully_connected(convnet, 2, activation='softmax')
    convnet = regression(
        convnet, 
        optimizer='adam', 
        learning_rate=LR, 
        loss='categorical_crossentropy', 
        name='targets'
    )

    model = tflearn.DNN(convnet, tensorboard_dir='log')

    if os.path.exists('{}.meta'.format(MODEL_NAME)):
        print('Loading model data')
        model.load(MODEL_NAME)
    
    return model


def filter_artists_data():
    testing_data = []
    for img_filename in tqdm(os.listdir(ART_BASE_DIR)):
        for artist in ARTISTS:
            if artist in img_filename:
                path = os.path.join(ART_BASE_DIR, img_filename)
                img = resize_image(path)
                if not len(img):
                    continue
                testing_data.append([np.array(img), img_filename])
    shuffle(testing_data)
    print('Testing with %d data size' % len(testing_data))
    return testing_data


def resize_image(image_path):
    try:
        return cv2.resize(cv2.imread(
            image_path, cv2.IMREAD_GRAYSCALE), (IMG_SIZE, IMG_SIZE))
    except Exception:
        return []  # Ignore bad files


def train_model(model, train_data):
    # Create train and test collections
    train_size = int(len(train_data) * TRAIN_RATIO)
    print('Train size: %d' % train_size)

    train = train_data[:train_size]
    test = train_data[train_size:]

    print('Crerating attributes for train and test sets')
    X = np.array([i[0] for i in train]).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
    test_x = np.array([i[0] for i in test]).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
    # Labels
    Y = [i[1] for i in train]
    test_y = [i[1] for i in test]

    print('Training model')
    model.fit(
        {'input': X}, {'targets': Y},
        n_epoch=5,
        validation_set=({'input': test_x}, {'targets': test_y}),
        snapshot_step=500,
        show_metric=True,
        run_id=MODEL_NAME
    )
    print('Saving trained model')
    model.save(MODEL_NAME)

    return model


def get_predictions_for_artist(artist_name):

    print(artist_name)
    model = create_model()

    testing_data = []
    for img_filename in tqdm(os.listdir(ART_BASE_DIR)):
        if artist_name in img_filename:
            path = os.path.join(ART_BASE_DIR, img_filename)
            img = resize_image(path)
            if not len(img):
                continue
            testing_data.append([np.array(img), img_filename])
    shuffle(testing_data)

    return get_prediction_results(testing_data, model)


# Run ML analysis in tensorboard using:
# tensorboard --logdir='/Users/amapola/Documents/S/WGU/5th term/Capstone/Project_Twinky/log'
