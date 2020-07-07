import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tqdm.autonotebook import tqdm
from IPython import display
import pandas as pd
import cv2
from glob import glob
from sklearn.model_selection import train_test_split
import math
from dataLoader import cropImage

WIKIART_DATASETS=["Action_painting", "Analytical_Cubism", "Color_Field_Painting", "Contemporary_Realism", "Cubism", "Early_Renaissance", "Fauvism", "Mannerism_Late_Renaissance", "Pointillism", "Post_Impressionism", "Rococo", "Synthetic_Cubism",]
DATASET_SELECTED=os.path.join("jpg_processed_images")
SUB_DATASET_SELECTED="FelicitationArabicFeasts_28"
DATASETS_PATH="../Datasets/"
DATASETS=["MNIST", "Morris(reduced)", "WilliamMorris", "Icons", "PolarPlots"]
CHECKPOINTS_PATH="./training_checkpoints"
GENERATED_IMAGES_PATH="../GeneratedImages/"
GIF_FILE="wgan-gp.gif"
LOSS_FILE=os.path.join("loss", SUB_DATASET_SELECTED + "_WGAN_losses.csv")

TRAIN_BUF=30150
TEST_BUF=14850
BATCH_SIZE=32
REQUIRED_DIMS = (28,28,1)
N_TRAIN_BATCHES =int(TRAIN_BUF/BATCH_SIZE)
N_TEST_BATCHES = int(TEST_BUF/BATCH_SIZE)

genAlpha=5e-6
disAlpha=5e-6

customDataset = []

if not os.path.exists(os.path.join(GENERATED_IMAGES_PATH, SUB_DATASET_SELECTED)):
	os.makedirs(os.path.join(GENERATED_IMAGES_PATH, SUB_DATASET_SELECTED))
	print("Making", os.path.join(GENERATED_IMAGES_PATH, SUB_DATASET_SELECTED))

if not os.path.isfile(LOSS_FILE):
	print("Creating a loss file (For Graphing Later on)...", LOSS_FILE)
	open(LOSS_FILE, "w").close()

if os.stat(LOSS_FILE).st_size == 0:
	lossFile=open(LOSS_FILE, "w")
	lossFile.write("Epochs,Discriminator Loss,Generator Loss\n")
	lossFile.close()

numFiles=len(glob(os.path.join(DATASETS_PATH, "WilliamMorris800", "*.jpg")))
print("Number of Files in WilliamMorris800:\t", numFiles)
for filename in glob(os.path.join(DATASETS_PATH, "WilliamMorris800", "*.jpg")):
	image = cv2.imread(filename, 0)
	images=cropImage(image)
	customDataset.extend(images)

numFiles=len(glob(os.path.join(DATASETS_PATH, "WilliamMorrisOthers", "*.jpg")))
print("Number of Files in WilliamMorrisOthers:\t", numFiles)
for filename in glob(os.path.join(DATASETS_PATH, "WilliamMorrisOthers", "*.jpg")):
	image = cv2.imread(filename, 0)
	images=cropImage(image)
	customDataset.extend(images)

customDataset=np.load(os.path.join("dataFiles", "cubism_data.npy"))
customDataset=np.array(customDataset)

def datasetLoader(datasetPath=os.path.join(DATASETS_PATH, DATASET_SELECTED, SUB_DATASET_SELECTED, "*.jpg")):
	dataset=[]
	for filename in glob(datasetPath):
		image = cv2.imread(filename, 1)
		dataset.append(image)

	dataset=np.asarray(dataset)
	print(dataset.shape)
	dataset = np.reshape(dataset, (-1, REQUIRED_DIMS[0], REQUIRED_DIMS[1], REQUIRED_DIMS[2])).astype("float32")
	dataset = dataset / 127.5 - 1

	return dataset

customDataset=datasetLoader()

train_images, test_images = train_test_split(customDataset, test_size=0.33, random_state=42)

# train_images = train_images.astype("float32")
# test_images = test_images.astype("float32")

# train_images = train_images.reshape(train_images.shape[0], REQUIRED_DIMS[0], REQUIRED_DIMS[1], 1).astype("float32")/255.0
# test_images = test_images.reshape(test_images.shape[0], REQUIRED_DIMS[0], REQUIRED_DIMS[1], 1).astype("float32")/255.0

print(customDataset.shape, train_images.shape, test_images.shape)

# batch datasets
train_dataset = (
	tf.data.Dataset.from_tensor_slices(train_images)
	.shuffle(TRAIN_BUF)
	.batch(BATCH_SIZE)
)
test_dataset = (
	tf.data.Dataset.from_tensor_slices(test_images)
	.shuffle(TEST_BUF)
	.batch(BATCH_SIZE)
)

"""### Define the network as tf.keras.model object"""

class WGAN(tf.keras.Model):
	"""[summary]
	I used github/LynnHo/DCGAN-LSGAN-WGAN-GP-DRAGAN-Tensorflow-2/ as a reference on this.

	Extends:
		tf.keras.Model
	"""
	def __init__(self, **kwargs):
		super(WGAN, self).__init__()
		self.__dict__.update(kwargs)

		self.gen = tf.keras.Sequential(self.gen)
		self.disc = tf.keras.Sequential(self.disc)

	def generate(self, z):
		return self.gen(z)

	def discriminate(self, x):
		return self.disc(x)

	def compute_loss(self, x):
		""" passes through the network and computes loss
		"""
		### pass through network
		# generating noise from a uniform distribution

		z_samp = tf.random.normal([x.shape[0], 1, 1, self.n_Z])

		# run noise through generator
		x_gen = self.generate(z_samp)
		# discriminate x and x_gen
		logits_x = self.discriminate(x)
		logits_x_gen = self.discriminate(x_gen)

		# gradient penalty
		d_regularizer = self.gradient_penalty(x, x_gen)
		### losses
		disc_loss = (
			tf.reduce_mean(logits_x)
			- tf.reduce_mean(logits_x_gen)
			+ d_regularizer * self.gradient_penalty_weight
		)

		# losses of fake with label "1"
		gen_loss = tf.reduce_mean(logits_x_gen)

		return disc_loss, gen_loss

	def compute_gradients(self, x):
		""" passes through the network and computes loss
		"""
		### pass through network
		with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
			disc_loss, gen_loss = self.compute_loss(x)

		# compute gradients
		gen_gradients = gen_tape.gradient(gen_loss, self.gen.trainable_variables)

		disc_gradients = disc_tape.gradient(disc_loss, self.disc.trainable_variables)

		return gen_gradients, disc_gradients

	def apply_gradients(self, gen_gradients, disc_gradients):

		self.gen_optimizer.apply_gradients(
			zip(gen_gradients, self.gen.trainable_variables)
		)
		self.disc_optimizer.apply_gradients(
			zip(disc_gradients, self.disc.trainable_variables)
		)

	def gradient_penalty(self, x, x_gen):
		epsilon = tf.random.uniform([x.shape[0], 1, 1, 1], 0.0, 1.0)
		x_hat = epsilon * x + (1 - epsilon) * x_gen
		with tf.GradientTape() as t:
			t.watch(x_hat)
			d_hat = self.discriminate(x_hat)
		gradients = t.gradient(d_hat, x_hat)
		ddx = tf.sqrt(tf.reduce_sum(gradients ** 2, axis=[1, 2]))
		d_regularizer = tf.reduce_mean((ddx - 1.0) ** 2)
		return d_regularizer

	@tf.function
	def train(self, train_x):
		gen_gradients, disc_gradients = self.compute_gradients(train_x)
		self.apply_gradients(gen_gradients, disc_gradients)

"""### Define the network architecture"""

N_Z = 64
generator = [
    tf.keras.layers.Dense(units=7 * 7 * 64, activation="relu"),
    tf.keras.layers.Reshape(target_shape=(7, 7, 64)),
    tf.keras.layers.Conv2DTranspose(
        filters=64, kernel_size=3, strides=(2, 2), padding="SAME", activation="relu"
    ),
    tf.keras.layers.Conv2DTranspose(
        filters=32, kernel_size=3, strides=(2, 2), padding="SAME", activation="relu"
    ),
    tf.keras.layers.Conv2DTranspose(
        filters=REQUIRED_DIMS[2], kernel_size=3, strides=(1, 1), padding="SAME", activation="sigmoid"
    ),
]

discriminator = [
    tf.keras.layers.InputLayer(input_shape=REQUIRED_DIMS),
    tf.keras.layers.Conv2D(
        filters=32, kernel_size=3, strides=(2, 2), activation="relu"
    ),
    tf.keras.layers.Conv2D(
        filters=64, kernel_size=3, strides=(2, 2), activation="relu"
    ),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(units=1, activation="sigmoid"),
]

# generator = [
# 	tf.keras.layers.Dense(units=16 * 16 * 64),
# 	tf.keras.layers.LeakyReLU(alpha=0.2),
# 	tf.keras.layers.Reshape(target_shape=(16, 16, 64)),

# 	tf.keras.layers.Conv2D(filters=64, kernel_size=3, strides=(2, 2), padding="SAME"),
# 	tf.keras.layers.BatchNormalization(),
# 	tf.keras.layers.LeakyReLU(alpha=0.2),

# 	tf.keras.layers.Conv2DTranspose(filters=128, kernel_size=3, strides=(2, 2), padding="SAME"),
# 	tf.keras.layers.BatchNormalization(),
# 	tf.keras.layers.LeakyReLU(alpha=0.2),

# 	tf.keras.layers.Conv2D(filters=64, kernel_size=3, strides=(2, 2), padding="SAME"),
# 	tf.keras.layers.BatchNormalization(),
# 	tf.keras.layers.LeakyReLU(alpha=0.2),

# 	tf.keras.layers.Conv2DTranspose(filters=64, kernel_size=3, strides=(2, 2), padding="SAME"),
# 	tf.keras.layers.BatchNormalization(),
# 	tf.keras.layers.LeakyReLU(alpha=0.2),

# 	tf.keras.layers.Conv2DTranspose(filters=32, kernel_size=3, strides=(2, 2), padding="SAME"),
# 	tf.keras.layers.BatchNormalization(),
# 	tf.keras.layers.LeakyReLU(alpha=0.2),

# 	tf.keras.layers.Conv2DTranspose(filters=32, kernel_size=3, strides=(2, 2), padding="SAME"),
# 	tf.keras.layers.BatchNormalization(),
# 	tf.keras.layers.LeakyReLU(alpha=0.2),
# 	tf.keras.layers.Conv2DTranspose(filters=REQUIRED_DIMS[2], kernel_size=3, strides=(1, 1), padding="SAME", activation="tanh"),
# ]

# generator=[
#     tf.keras.layers.Dense(units=4 * 4 * 256, activation="relu"),
#     tf.keras.layers.Reshape(target_shape=(4, 4, 256)),
#     tf.keras.layers.UpSampling2D(),
#     tf.keras.layers.Conv2D(256, kernel_size=3, padding="same"),
#     tf.keras.layers.BatchNormalization(momentum=0.8),
#     tf.keras.layers.Activation("relu"),
#     tf.keras.layers.UpSampling2D(),
#     tf.keras.layers.Conv2D(256, kernel_size=3, padding="same"),
#     tf.keras.layers.BatchNormalization(momentum=0.8),
#     tf.keras.layers.Activation("relu"),

#     tf.keras.layers.UpSampling2D(),
#     tf.keras.layers.Conv2D(256, kernel_size=3, padding="same"),
#     tf.keras.layers.BatchNormalization(momentum=0.8),
#     tf.keras.layers.Activation("relu"),

#     tf.keras.layers.UpSampling2D(),
#     tf.keras.layers.Conv2D(256, kernel_size=3, padding="same"),
#     tf.keras.layers.BatchNormalization(momentum=0.8),
#     tf.keras.layers.Activation("relu"),

#     tf.keras.layers.UpSampling2D(),
#     tf.keras.layers.Conv2D(256, kernel_size=3, padding="same"),
#     tf.keras.layers.BatchNormalization(momentum=0.8),
#     tf.keras.layers.Activation("relu"),

#     tf.keras.layers.Conv2D(REQUIRED_DIMS[2], kernel_size=3, padding="same"),
#     tf.keras.layers.Activation("sigmoid"),
# ]

# discriminator = [
# 	tf.keras.layers.InputLayer(input_shape=REQUIRED_DIMS),
# 	tf.keras.layers.Conv2D(filters=32, kernel_size=3, strides=(2, 2)),
# 	tf.keras.layers.BatchNormalization(),
# 	tf.keras.layers.LeakyReLU(alpha=0.2),

# 	tf.keras.layers.Conv2D(filters=32, kernel_size=3, strides=(2, 2)),
# 	tf.keras.layers.BatchNormalization(),
# 	tf.keras.layers.LeakyReLU(alpha=0.2),

# 	tf.keras.layers.Conv2D(filters=64, kernel_size=3, strides=(2, 2)),
# 	tf.keras.layers.BatchNormalization(),
# 	tf.keras.layers.LeakyReLU(alpha=0.2),

# 	tf.keras.layers.Conv2D(filters=64, kernel_size=3, strides=(2, 2)),
# 	tf.keras.layers.BatchNormalization(),
# 	tf.keras.layers.LeakyReLU(alpha=0.2),

# 	tf.keras.layers.Conv2D(filters=128, kernel_size=3, strides=(2, 2)),
# 	tf.keras.layers.BatchNormalization(),
# 	tf.keras.layers.LeakyReLU(alpha=0.2),

# 	tf.keras.layers.Conv2D(filters=128, kernel_size=3, strides=(2, 2)),
# 	tf.keras.layers.BatchNormalization(),
# 	tf.keras.layers.LeakyReLU(alpha=0.2),

# 	tf.keras.layers.Flatten(),
# 	tf.keras.layers.Dropout(0.4),
# 	tf.keras.layers.Dense(units=1, activation="sigmoid"),
# ]

# discriminator = [
# 	tf.keras.layers.InputLayer(input_shape=REQUIRED_DIMS),
# 	tf.keras.layers.Conv2D(32, kernel_size=3, strides=2, padding="same"),
# 	tf.keras.layers.LeakyReLU(alpha=0.2),
# 	tf.keras.layers.Dropout(0.25),
# 	tf.keras.layers.Conv2D(64, kernel_size=3, strides=2, padding="same"),
# 	tf.keras.layers.ZeroPadding2D(padding=((0, 1), (0, 1))),
# 	tf.keras.layers.BatchNormalization(momentum=0.8),
# 	tf.keras.layers.LeakyReLU(alpha=0.2),
# 	tf.keras.layers.Dropout(0.25),
# 	tf.keras.layers.Conv2D(128, kernel_size=3, strides=2, padding="same"),
# 	tf.keras.layers.BatchNormalization(momentum=0.8),
# 	tf.keras.layers.LeakyReLU(alpha=0.2),
# 	tf.keras.layers.Dropout(0.25),
# 	tf.keras.layers.Conv2D(256, kernel_size=3, strides=1, padding="same"),
# 	tf.keras.layers.BatchNormalization(momentum=0.8),
# 	tf.keras.layers.LeakyReLU(alpha=0.2),
# 	tf.keras.layers.Dropout(0.25),
# 	tf.keras.layers.Conv2D(512, kernel_size=3, strides=1, padding="same"),
# 	tf.keras.layers.BatchNormalization(momentum=0.8),
# 	tf.keras.layers.LeakyReLU(alpha=0.2),
# 	tf.keras.layers.Dropout(0.25),
# 	tf.keras.layers.Flatten(),
# 	tf.keras.layers.Dense(1, activation="sigmoid"),
# ]


"""### Create Model"""

# optimizers
# gen_optimizer = tf.keras.optimizers.Adam(genAlpha, beta_1=0.5, epsilon=1e-8)
# disc_optimizer = tf.keras.optimizers.RMSprop(disAlpha)# train the model

gen_optimizer = tf.keras.optimizers.Adam(learning_rate=genAlpha, beta_1=0.5, epsilon=1e-07)
disc_optimizer = tf.keras.optimizers.RMSprop(learning_rate=disAlpha, epsilon=1e-07)
# model
model = WGAN(
	gen = generator,
	disc = discriminator,
	gen_optimizer = gen_optimizer,
	disc_optimizer = disc_optimizer,
	n_Z = N_Z,
	gradient_penalty_weight = 10.0
)

"""### Train the model"""

# exampled data for plotting results
def plot_reconstruction(model, epoch, nex=8, zm=2):
	samples = model.generate(tf.random.normal(shape=(BATCH_SIZE, N_Z)))
	fig, axs = plt.subplots(ncols=nex, nrows=1, figsize=(zm * nex, zm))
	for axi in range(nex):
		axs[axi].matshow(
					samples.numpy()[axi].squeeze(), cmap=plt.cm.Greys, vmin=0, vmax=1
				)
		axs[axi].axis('off')

	fig.savefig(os.path.join(GENERATED_IMAGES_PATH, SUB_DATASET_SELECTED, "epoch_{:03d}.png".format(epoch+1)), dpi=fig.dpi)
	plt.close()
	# plt.show()

# a pandas dataframe to save the loss information to
losses = pd.DataFrame(columns = ['disc_loss', 'gen_loss'])
checkpoint_path = os.path.join(CHECKPOINTS_PATH, SUB_DATASET_SELECTED)
checkpoint_prefix = os.path.join(CHECKPOINTS_PATH, SUB_DATASET_SELECTED, "ckpt")

checkpoint = tf.train.Checkpoint(model=model)

# checkpoint.restore(tf.train.latest_checkpoint(checkpoint_path))

n_epochs = 2500
for epoch in range(n_epochs):

	# train
	for batch, train_x in tqdm(
		zip(range(N_TRAIN_BATCHES), train_dataset), total=N_TRAIN_BATCHES
	):
		model.train(train_x)
	# test on holdout
	loss = []
	for batch, test_x in tqdm(
		zip(range(N_TEST_BATCHES), test_dataset), total=N_TEST_BATCHES
	):
		loss.append(model.compute_loss(train_x))
	losses.loc[len(losses)] = np.mean(loss, axis=0)
	# plot results
	display.clear_output()
	print(
		"Epoch: {} | disc_loss: {} | gen_loss: {}".format(
			epoch+1 , losses.disc_loss.values[-1], losses.gen_loss.values[-1]
		)
	)

	lossFile = open(LOSS_FILE, "a")
	lossFile.write("{},{},{}\n".format(epoch, losses.disc_loss.values[-1], losses.gen_loss.values[-1]))
	lossFile.close()

	# Save the model every 15 epochs
	if (epoch + 1) % 15 == 0:
		checkpoint.save(file_prefix = checkpoint_prefix)

	plot_reconstruction(model, epoch)

plt.plot(losses.gen_loss.values)
plt.plot(losses.disc_loss.values)