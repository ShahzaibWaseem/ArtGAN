import tensorflow as tf
import numpy as np
from PIL import Image
from glob import glob
import time
import os
import cv2
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Preview image Frame
PREVIEW_ROWS = 4
PREVIEW_COLS = 7
PREVIEW_MARGIN = 4
SAVE_FREQ = 10
# Size vector to generate images from
NOISE_SIZE = 100
# Configuration
EPOCHS = 10000 # number of iterations
BATCH_SIZE = 32
GENERATE_RES = 3
IMAGE_SIZE = 64 # rows/cols
IMAGE_CHANNELS = 3

DATASET_SELECTED="Datasets"
SUB_DATASET_SELECTED="PolarPlots_28"
DATASETS_PATH=".."
GENERATED_IMAGES_PATH="../GeneratedImages/"
CHECKPOINTS_PATH="./training_checkpoints"
LOSS_FILE=os.path.join("loss", SUB_DATASET_SELECTED + "losses.csv")

if not os.path.exists(os.path.join("loss")):
	print("Making\t", os.path.join("loss"))
	os.makedirs(os.path.join("loss"))

if not os.path.exists(os.path.join(GENERATED_IMAGES_PATH, SUB_DATASET_SELECTED)):
	print("Making\t", os.path.join(GENERATED_IMAGES_PATH, SUB_DATASET_SELECTED))
	os.makedirs(os.path.join(GENERATED_IMAGES_PATH, SUB_DATASET_SELECTED))

def datasetLoader(datasetPath=os.path.join(DATASETS_PATH, DATASET_SELECTED, SUB_DATASET_SELECTED, "*.jpg")):
	dataset=[]
	for filename in glob(datasetPath):
		image = cv2.imread(filename)
		dataset.append(np.asarray(image))

	dataset = np.reshape(dataset, (-1, IMAGE_SIZE, IMAGE_SIZE, IMAGE_CHANNELS))
	dataset = dataset / 127.5 - 1

	return dataset

training_data = datasetLoader()

def build_discriminator(image_shape):
	model = tf.keras.Sequential()
	model.add(tf.keras.layers.Conv2D(32, kernel_size=3, strides=2,input_shape=image_shape, padding="same"))
	model.add(tf.keras.layers.LeakyReLU(alpha=0.2))
	model.add(tf.keras.layers.Dropout(0.25))
	model.add(tf.keras.layers.Conv2D(64, kernel_size=3, strides=2, padding="same"))
	model.add(tf.keras.layers.ZeroPadding2D(padding=((0, 1), (0, 1))))
	model.add(tf.keras.layers.BatchNormalization(momentum=0.8))
	model.add(tf.keras.layers.LeakyReLU(alpha=0.2))
	model.add(tf.keras.layers.Dropout(0.25))
	model.add(tf.keras.layers.Conv2D(128, kernel_size=3, strides=2, padding="same"))
	model.add(tf.keras.layers.BatchNormalization(momentum=0.8))
	model.add(tf.keras.layers.LeakyReLU(alpha=0.2))
	model.add(tf.keras.layers.Dropout(0.25))
	model.add(tf.keras.layers.Conv2D(256, kernel_size=3, strides=1, padding="same"))
	model.add(tf.keras.layers.BatchNormalization(momentum=0.8))
	model.add(tf.keras.layers.LeakyReLU(alpha=0.2))
	model.add(tf.keras.layers.Dropout(0.25))
	model.add(tf.keras.layers.Conv2D(512, kernel_size=3, strides=1, padding="same"))
	model.add(tf.keras.layers.BatchNormalization(momentum=0.8))
	model.add(tf.keras.layers.LeakyReLU(alpha=0.2))
	model.add(tf.keras.layers.Dropout(0.25))
	model.add(tf.keras.layers.Flatten())
	model.add(tf.keras.layers.Dense(1, activation="sigmoid"))
	input_image = tf.keras.Input(shape=image_shape)
	validity = model(input_image)
	model.summary()

	return tf.keras.Model(input_image, validity)

def build_generator(noise_size, channels):
	model = tf.keras.Sequential()
	model.add(tf.keras.layers.Dense(2 * 2 * 256, activation="relu", input_dim=noise_size))
	model.add(tf.keras.layers.Reshape((2, 2, 256)))
	model.add(tf.keras.layers.UpSampling2D())
	model.add(tf.keras.layers.Conv2D(256, kernel_size=3, padding="same"))
	model.add(tf.keras.layers.BatchNormalization(momentum=0.8))
	model.add(tf.keras.layers.Activation("relu"))
	model.add(tf.keras.layers.UpSampling2D())
	model.add(tf.keras.layers.Conv2D(256, kernel_size=3, padding="same"))
	model.add(tf.keras.layers.BatchNormalization(momentum=0.8))
	model.add(tf.keras.layers.Activation("relu"))
	for i in range(GENERATE_RES):
		model.add(tf.keras.layers.UpSampling2D())
		model.add(tf.keras.layers.Conv2D(256, kernel_size=3, padding="same"))
		model.add(tf.keras.layers.BatchNormalization(momentum=0.8))
		model.add(tf.keras.layers.Activation("relu"))
	model.add(tf.keras.layers.Conv2D(channels, kernel_size=3, padding="same"))
	model.add(tf.keras.layers.Activation("tanh"))
	input = tf.keras.Input(shape=(noise_size,))
	generated_image = model(input)

	model.summary()

	return tf.keras.Model(input, generated_image)

def save_images(cnt, noise):
	image_array = np.full((
		PREVIEW_MARGIN + (PREVIEW_ROWS * (IMAGE_SIZE + PREVIEW_MARGIN)),
		PREVIEW_MARGIN + (PREVIEW_COLS * (IMAGE_SIZE + PREVIEW_MARGIN)), 3),
		255, dtype=np.uint8)
	generated_images = generator.predict(noise)
	generated_images = 0.5 * generated_images + 0.5
	image_count = 0
	for row in range(PREVIEW_ROWS):
		for col in range(PREVIEW_COLS):
			r = row * (IMAGE_SIZE + PREVIEW_MARGIN) + PREVIEW_MARGIN
			c = col * (IMAGE_SIZE + PREVIEW_MARGIN) + PREVIEW_MARGIN
			image_array[r:r + IMAGE_SIZE, c:c + IMAGE_SIZE] = generated_images[image_count] * 255
			image_count += 1

	filename = os.path.join(GENERATED_IMAGES_PATH, SUB_DATASET_SELECTED, f"trained-{cnt}.png")
	im = Image.fromarray(image_array)
	im.save(filename)

# randomly flip some labels
def noisy_labels(y, p_flip):
	n_select = int(p_flip * y.shape[0])
	flip_ix = np.random.choice([i for i in range(y.shape[0])], size=n_select)
	y[flip_ix] = 1 - y[flip_ix]
	return y

image_shape = (IMAGE_SIZE, IMAGE_SIZE, IMAGE_CHANNELS)
optimizer = tf.keras.optimizers.Adam(1.5e-4, 0.5, epsilon=1e-8)

discriminator = build_discriminator(image_shape)
discriminator.compile(loss="binary_crossentropy", optimizer=optimizer, metrics=["accuracy"])

generator = build_generator(NOISE_SIZE, IMAGE_CHANNELS)
random_input = tf.keras.Input(shape=(NOISE_SIZE,))
generated_image = generator(random_input)
discriminator.trainable = False
validity = discriminator(generated_image)

combined = tf.keras.Model(random_input, validity)
combined.compile(loss="binary_crossentropy", optimizer=optimizer, metrics=["accuracy"])
# Noisy Label Technique
y_real = np.ones((BATCH_SIZE, 1))
# y_real = noisy_labels(y_real, 0.05)

y_fake = np.zeros((BATCH_SIZE, 1))
# y_fake = noisy_labels(y_fake, 0.05)

fixed_noise = np.random.normal(0, 1, (PREVIEW_ROWS * PREVIEW_COLS, NOISE_SIZE))

checkpoint_path = os.path.join(CHECKPOINTS_PATH, DATASET_SELECTED, SUB_DATASET_SELECTED)
checkpoint_prefix = os.path.join(CHECKPOINTS_PATH, DATASET_SELECTED, SUB_DATASET_SELECTED,  "ckpt")

checkpoint = tf.train.Checkpoint(model=combined)

# checkpoint.restore(tf.train.latest_checkpoint(checkpoint_path))
epochs_done=1
# epochs_done=int(sorted(glob(os.path.join(GENERATED_IMAGES_PATH, DATASET_SELECTED, "*.png")), key=os.path.getmtime)[-1].split("-")[-1].split(".")[0])

cnt = 1

if not os.path.isfile(LOSS_FILE):
	print("Creating a loss file (For Graphing Later on)...", LOSS_FILE)
	open(LOSS_FILE, "w").close()

if os.stat(LOSS_FILE).st_size == 0:
	lossFile=open(LOSS_FILE, "w")
	lossFile.write("Epochs,Discriminator Accuracy,Adversarial Accuracy\n")
	lossFile.close()

for epoch in range(EPOCHS):
	startTime = time.time()
	idx = np.random.randint(0, training_data.shape[0], BATCH_SIZE)
	x_real = training_data[idx]
	# Label Smoothing
	# x_real = x_real + np.random.normal(-0.5, 0.5, x_real.shape)

	noise= np.random.normal(0, 1, (BATCH_SIZE, NOISE_SIZE))
	x_fake = generator.predict(noise)
	# x_fake = x_fake + np.random.normal(-0.5, 0.5, x_fake.shape)

	discriminator_metric_real = discriminator.train_on_batch(x_real, y_real)
	discriminator_metric_generated = discriminator.train_on_batch(x_fake, y_fake)

	# Average
	discriminator_metric = 0.5 * np.add(discriminator_metric_real, discriminator_metric_generated)
	generator_metric = combined.train_on_batch(noise, y_real)

	print(f"{epoch + 1} epoch | discriminator : {100 *  discriminator_metric[1]}, generator: {100 * generator_metric[1]}\tTime Taken: {time.time() - startTime}")

	lossFile = open(LOSS_FILE, "a")
	lossFile.write("{},{},{}\n".format(epoch+1, 100 *  discriminator_metric[1], 100 *  generator_metric[1]))
	lossFile.close()

	if epoch % SAVE_FREQ == 0:
		save_images(epoch, fixed_noise)
		checkpoint.save(file_prefix = checkpoint_prefix)