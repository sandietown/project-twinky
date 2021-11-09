from pathlib import Path

BASE_PATH = Path(__file__).parent.absolute()


SECRET='thisisasecretsecret'
TRAIN_DIR = '{}/../Datasets/Train'.format(BASE_PATH)
TEST_DIR = '{}/../Datasets/Test'.format(BASE_PATH)
ARTISTS = ['Albrecht_Dürer', 'Diego_Rivera', 'Diego_Velazquez', 'Edgar_Degas', 'Edouard_Manet',
           'Francisco_Goya', 'Giotto_di_Bondone', 'Gustave_Courbet', 'Titian', 'Vincent_van_Gogh', 'Others']
ART_BASE_DIR = '{}/../Datasets/Test/'.format(BASE_PATH)
ART_PATH = '/Test/'

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
