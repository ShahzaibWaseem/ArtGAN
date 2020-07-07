import os
import numpy as np
from PIL import Image

IMAGE_SIZE = 64
IMAGE_CHANNELS = 1
DATASETS_PATH = "../Datasets/"
DATASET_SELECTED = "PolarPlots"
# SUB_DATASET_SELECTED = "Cubism/"

# Defining image dir path. Change this if you have different directory
images_path = os.path.join(DATASETS_PATH, DATASET_SELECTED)

if not os.path.exists(os.path.join(DATASETS_PATH, DATASET_SELECTED + "_28")):
	os.makedirs(os.path.join(DATASETS_PATH, DATASET_SELECTED + "_28"))
	print("Making", os.path.join(DATASETS_PATH, DATASET_SELECTED + "_28"))

training_data = []

# Iterating over the images inside the directory and resizing them using
# Pillow's resize method.
print('Resizing...')

for filename in os.listdir(images_path):
	path = os.path.join(images_path, filename)
	print(os.path.join(images_path + "_28", filename))
	image = Image.open(path).resize((IMAGE_SIZE, IMAGE_SIZE), Image.ANTIALIAS)
	image.save(os.path.join(images_path + "_28", filename))

	# training_data.append(np.asarray(image))

# training_data = np.reshape(
# 	training_data, (-1, IMAGE_SIZE, IMAGE_SIZE, IMAGE_CHANNELS))
# training_data = training_data / 127.5 - 1

# print("Data shape", training_data.shape)

# print('saving file...')
# np.save(os.path.join('dataFiles', 'cubism_grayscale_data.npy'), training_data)