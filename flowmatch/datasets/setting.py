'''
Settings for generating synthetic images using code for
Cut, Paste, and Learn Paper
'''

# Paths
BACKGROUND_DIR = '/scratch/jnan1/background/TRAIN'
BACKGROUND_GLOB_STRING = '*.png'
INVERTED_MASK = False # Set to true if white pixels represent background

# Parameters for generator
NUMBER_OF_WORKERS = 4
BLENDING_LIST = ['gaussian', 'none', 'box', 'motion']

# Parameters for images
MIN_NO_OF_OBJECTS = 1
MAX_NO_OF_OBJECTS = 3
MIN_NO_OF_DISTRACTOR_OBJECTS = 1
MAX_NO_OF_DISTRACTOR_OBJECTS = 3
WIDTH = 960
HEIGHT = 720
OBJECT_SIZE = 256
MAX_ATTEMPTS_TO_SYNTHESIZE = 20

# Parameters for objects in images
MIN_SCALE = 0.25 # min scale for scale augmentation
MAX_SCALE = 0.6 # max scale for scale augmentation
MAX_DEGREES = 30 # max rotation allowed during rotation augmentation
MAX_TRUNCATION_FRACTION = 0 # max fraction to be truncated = MAX_TRUNCACTION_FRACTION*(WIDTH/HEIGHT)
MAX_ALLOWED_IOU = 0.75 # IOU > MAX_ALLOWED_IOU is considered an occlusion
MIN_WIDTH = 6 # Minimum width of object to use for data generation
MIN_HEIGHT = 6 # Minimum height of object to use for data generation