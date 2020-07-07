import os
import sys
import numpy
from PIL import Image
from glob import glob

DATASETS_PATH=os.path.join("..", "Datasets")
DATASETS_SELECTED=os.path.join(DATASETS_PATH, "william-morris")
GLITCH_DATASET_PATH=os.path.join(DATASETS_PATH, "william-morris-glitched")

if not os.path.exists(os.path.join(GLITCH_DATASET_PATH)):
	os.makedirs(os.path.join(GLITCH_DATASET_PATH))
	print("Making", os.path.join(GLITCH_DATASET_PATH))

def pixelize(infile, output_height=0, output_width=0, output_size=512, sampling_factor=16):
	"""Pixelate an imput image into glitch art for use as an identicon or other avatar"""

	if not os.path.exists(infile):
		sys.exit('File {} does not exist!'.format(infile))

	image = Image.open(infile)

	try:
		orig_height, orig_width, num_channels = image.size
	except:
		orig_height, orig_width = image.size

	if output_height and not output_width:
		output_width = int(orig_width * (orig_height / output_height))
	elif output_width and not output_height:
		output_height = int(orig_height * (orig_width / output_width))
	elif not any([output_height, output_width]):
		output_height = output_width = output_size

	image = image.resize((output_width // sampling_factor or 1, output_height // sampling_factor or 1), Image.LANCZOS)
	image = image.rotate(90 * 3)
	image = Image.fromarray(numpy.sort(image, axis=0))
	image = image.rotate(90 * 2)
	image = image.resize((output_width, output_height), Image.NEAREST)

	filename_parts = os.path.basename(infile).split('.')
	filename_parts.insert(-1, 'processed')

	processed_image_filename = os.path.join(GLITCH_DATASET_PATH, infile.split("/")[-1].split(".")[0])

	image.save(processed_image_filename + ".jpg")

def main():
	for imagePath in glob(os.path.join(DATASETS_SELECTED, "*.jpg")):
		print(imagePath)
		pixelize(imagePath, output_size=64, sampling_factor=1)

if __name__ == '__main__':
	main()