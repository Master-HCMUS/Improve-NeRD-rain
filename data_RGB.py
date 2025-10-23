from dataset_RGB import *


def get_training_data(rgb_dir, img_options):
    if not os.path.exists(rgb_dir):
        raise FileNotFoundError(f"Training data directory not found: {rgb_dir}")
    return DataLoaderTrain(rgb_dir, img_options)

def get_validation_data(rgb_dir, img_options):
    if not os.path.exists(rgb_dir):
        raise FileNotFoundError(f"Validation data directory not found: {rgb_dir}")
    return DataLoaderVal(rgb_dir, img_options)

def get_test_data(rgb_dir, img_options):
    if not os.path.exists(rgb_dir):
        raise FileNotFoundError(f"Test data directory not found: {rgb_dir}")
    return DataLoaderTest(rgb_dir, img_options)
