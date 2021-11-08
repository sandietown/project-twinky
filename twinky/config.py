from pathlib import Path

BASE_PATH = Path(__file__).parent.absolute()


SECRET='thisisasecretsecret'
TRAIN_DIR = '{}/../Datasets/Train'.format(BASE_PATH)
TEST_DIR = '{}/../Datasets/Test'.format(BASE_PATH)
# TEST_DIR = '{}/../Datasets/perritos_met'.format(BASE_PATH)
ARTISTS = ['Diego_Velazquez', 'Gustave_Courbet',
           'Henri_de_Toulouse-Lautrec', 'Titian']
ART_BASE_DIR = '{}/../Datasets/artworks/resized/'.format(BASE_PATH)
ART_PATH = '/artworks/resized/'

# Model parameters
LR = 1e-3  # LEARNING RATE
IMG_SIZE = 150
CONV_LAYERS = 10  # N convolutional layers
MODEL_NAME = 'ml_model/art_dogs-{}-{}convDNN-{}p'.format(
    LR, CONV_LAYERS, IMG_SIZE)
TRAIN_DATA_FILE = '{}_train_data.npy'.format(MODEL_NAME)
TEST_DATA_FILE = '{}_test_data.npy'.format(MODEL_NAME)

# Training parameters
TRAIN_RATIO = 0.85
