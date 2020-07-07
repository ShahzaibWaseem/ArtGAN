import os
import time
import cv2
import numpy as np
from PIL import Image
from glob import glob
import matplotlib.pyplot as plt
import tensorflow.keras as keras
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, LeakyReLU, Reshape, Conv2D, Conv2DTranspose, Flatten, Dropout
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

NUM_EPOCHS = 1000
BATCH_SIZE = 32

LATENT_DIM = 64
HEIGHT = 64
WIDTH = 64
CHANNELS = 3

PREVIEW_ROWS = 4
PREVIEW_COLS = 7
PREVIEW_MARGIN = 4

IMAGE_SAVE_FREQ = 10
MODEL_SAVE_FREQ = 100

DATASETS_PATH=os.path.join("..")
GENERATED_IMAGES_PATH=os.path.join("..", "GeneratedImages")
DATASET_SELECTED="Datasets"
SUB_DATASET_SELECTED="william-morris-glitched"
CHECKPOINTS_PATH="./training_checkpoints"
WEIGHTS_PATH="weights"
LOSS_FILE=os.path.join("loss", SUB_DATASET_SELECTED + "losses.csv")

if not os.path.exists(os.path.join("loss")):
	print("Making\t", os.path.join("loss"))
	os.makedirs(os.path.join("loss"))

if not os.path.exists(os.path.join(GENERATED_IMAGES_PATH, SUB_DATASET_SELECTED)):
	print("Making\t", os.path.join(GENERATED_IMAGES_PATH, SUB_DATASET_SELECTED))
	os.makedirs(os.path.join(GENERATED_IMAGES_PATH, SUB_DATASET_SELECTED))

def buildGenerator(latentShape=(LATENT_DIM,)):
	generatorInput = Input(shape=latentShape)
	model = Dense(units=128 * 32 * 32)(generatorInput)
	model = LeakyReLU()(model)
	model = Reshape((32, 32, 128))(model)
	model = Conv2D(filters=256, kernel_size=5, padding="same")(model)
	model = LeakyReLU()(model)
	model = Conv2DTranspose(filters=256, kernel_size=4, strides=2, padding="same")(model)
	model = LeakyReLU()(model)
	model = Conv2D(filters=256, kernel_size=5, padding="same")(model)
	model = LeakyReLU()(model)
	model = Conv2D(filters=256, kernel_size=5, padding="same")(model)
	model = LeakyReLU()(model)
	model = Conv2D(filters=CHANNELS, kernel_size=7, activation="tanh", padding="same")(model)

	generator = Model(inputs=generatorInput, outputs=model, name="Generator")
	return generator

def buildDiscriminator(inputShape=(HEIGHT, WIDTH, CHANNELS)):
	discInput = Input(inputShape)
	model = Conv2D(filters=128, kernel_size=3)(discInput)
	model = LeakyReLU()(model)
	model = Conv2D(filters=128, kernel_size=4, strides=2)(model)
	model = LeakyReLU()(model)
	model = Conv2D(filters=128, kernel_size=4, strides=2)(model)
	model = LeakyReLU()(model)
	model = Conv2D(filters=128, kernel_size=4, strides=2)(model)
	model = LeakyReLU()(model)
	model = Flatten()(model)
	model = Dropout(0.4)(model)
	model = Dense(units=1, activation="sigmoid")(model)

	discriminator = Model(inputs=discInput, outputs=model, name="Discriminator")
	return discriminator

def save_images(generated_images, epoch):
	image_array = np.full((
		PREVIEW_MARGIN + (PREVIEW_ROWS * (WIDTH + PREVIEW_MARGIN)),
		PREVIEW_MARGIN + (PREVIEW_COLS * (HEIGHT + PREVIEW_MARGIN)), 3),
		255, dtype=np.uint8)

	image_count = 0
	for row in range(PREVIEW_ROWS):
		for col in range(PREVIEW_COLS):
			r = row * (WIDTH + PREVIEW_MARGIN) + PREVIEW_MARGIN
			c = col * (HEIGHT + PREVIEW_MARGIN) + PREVIEW_MARGIN
			image_array[r:r + WIDTH, c:c + HEIGHT] = generated_images[image_count] * 255
			image_count += 1

	filename = os.path.join(GENERATED_IMAGES_PATH, SUB_DATASET_SELECTED, "epoch_" + str(epoch) + ".jpg")
	im = Image.fromarray(image_array)
	im.save(filename)

def datasetLoader(datasetPath=os.path.join(DATASETS_PATH, DATASET_SELECTED, SUB_DATASET_SELECTED, "*.jpg")):
	dataset=[]
	for filename in glob(datasetPath):
		image = cv2.imread(filename)
		dataset.append(np.asarray(image))

	dataset = np.reshape(dataset, (-1, HEIGHT, WIDTH, CHANNELS))
	dataset = dataset / 127.5 - 1

	return dataset

generator = buildGenerator()
discriminator = buildDiscriminator()

generator.summary()
discriminator.summary()

disc_opt = keras.optimizers.RMSprop(lr=0.0008, clipvalue=1.0, decay=1e-8)
discriminator.compile(optimizer=disc_opt, loss="binary_crossentropy")

# adversarial
# discriminator.trainable = False

gan_input = Input(shape=(LATENT_DIM,))
gan_output = discriminator(generator(gan_input))

gan = Model(inputs=gan_input, outputs=gan_output, name="Adversarial")
gan_optimizer = keras.optimizers.RMSprop(lr=4e-04, clipvalue=1.0, decay=1e-8)
gan.compile(optimizer=gan_optimizer, loss="binary_crossentropy")

customDataset=datasetLoader()
print("\nDataset used:", os.path.join(DATASETS_PATH, DATASET_SELECTED, SUB_DATASET_SELECTED))
print("Data Shape:", customDataset.shape, "\n")

# print("Loading Weights...\n")
# gan.load_weights(os.path.join(WEIGHTS_PATH, "gan.1"))

print("Running...\n")

if not os.path.isfile(LOSS_FILE):
	print("Creating a loss file (For Graphing Later on)...", LOSS_FILE)
	open(LOSS_FILE, "w").close()

if os.stat(LOSS_FILE).st_size == 0:
	lossFile=open(LOSS_FILE, "w")
	lossFile.write("Epochs,Discriminator Loss,Adversarial Loss\n")
	lossFile.close()

try:
	lossFile=open(LOSS_FILE, "r")
	epochsAlreadyRun=int(lossFile.readlines()[-1].split(",")[0])
	lossFile.close()
except Exception as e:
	print("File not found or no data is in it.\nStarting from the start")
	epochsAlreadyRun=0

print("Epochs already run:", str(epochsAlreadyRun))

try:
	imagesAlreadyPresent=glob(os.path.join(GENERATED_IMAGES_PATH, SUB_DATASET_SELECTED, "epoch_*.jpg"))
	imagesAlreadyPresent=int(imagesAlreadyPresent[-1].split("_")[-1].split(".")[0])
except Exception as e:
	imagesAlreadyPresent=0

print("_" * 93)
start = 0

for epoch in range(NUM_EPOCHS):
	startTime = time.time()
	random_latent_vectors = np.random.normal(size=(BATCH_SIZE, LATENT_DIM))

	generated_images = generator.predict(random_latent_vectors)
	stop = start + BATCH_SIZE
	real_images = customDataset[start: stop]
	combined_images = np.concatenate([generated_images, real_images])
	labels = np.concatenate([np.ones((BATCH_SIZE, 1)), np.zeros((BATCH_SIZE, 1))])
	labels += 0.05 * np.random.random(labels.shape)

	# train discriminator
	d_loss = discriminator.train_on_batch(combined_images, labels)
	random_latent_vectors = np.random.normal(size=(BATCH_SIZE, LATENT_DIM))

	# train generator
	misleading_targets = np.zeros((BATCH_SIZE, 1))
	a_loss = gan.train_on_batch(random_latent_vectors,misleading_targets)
	start += BATCH_SIZE

	print("| Epoch: {:04} | Discriminator Loss: {:.4f} | Adversarial Loss: {:.4f} | Time Taken: {:.4f} |".format(MODEL_SAVE_FREQ*(epochsAlreadyRun//MODEL_SAVE_FREQ)+epoch+1, d_loss, a_loss, time.time() - startTime), end=" ")

	lossFile = open(LOSS_FILE, "a")
	lossFile.write("{},{},{}\n".format(epochsAlreadyRun+epoch+1, d_loss, a_loss))
	lossFile.close()

	if start > len(customDataset) - BATCH_SIZE:
		start = 0

	if epoch % IMAGE_SAVE_FREQ == 0:
		print("<", end="")
		save_images(generated_images, imagesAlreadyPresent + epoch)

	if epoch + 1 % MODEL_SAVE_FREQ == 0:
		print("<", end="")
		gan.save_weights(os.path.join(WEIGHTS_PATH, "gan.1"))

	print()
print("_" * 93)