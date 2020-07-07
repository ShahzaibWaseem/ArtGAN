from PIL import Image, ImageFont, ImageDraw
import os
import random
import string
import matplotlib.pyplot as plt

DATASET_SIZE=15000
IMAGE_SIZE=128
FONT_DIRECTORY=os.path.join("..", "fonts")
FONT_NAME="FelicitationArabicFeasts.ttf"
FONT_SIZE=80
DATASET_SELECTED="jpg_processed_images"
DATASET_PATH=os.path.join("..", "Datasets", DATASET_SELECTED, FONT_NAME.split(".")[0]+"_"+str(IMAGE_SIZE))

if not os.path.exists(DATASET_PATH):
	print("Making\t", DATASET_PATH)
	os.makedirs(DATASET_PATH)

# paperBackground = Image.open(os.path.join(FONT_DIRECTORY, "paperBackground.jpg"))
# # paperBackground = paperBackground.resize((64, 64), Image.ANTIALIAS)
# # paperBackground.save(os.path.join(FONT_DIRECTORY, "paperBackground.jpg"))

font = ImageFont.truetype(os.path.join(FONT_DIRECTORY, FONT_NAME), FONT_SIZE)
# draw = ImageDraw.Draw(paperBackground)
# draw.text((20,0), random.choice(string.ascii_letters), font=font, fill=(0, 0, 0))

genImageName=0
for genImage in range(DATASET_SIZE):
	paperBackground = Image.new('RGB', (IMAGE_SIZE, IMAGE_SIZE), color = (0, 0, 0))
	draw = ImageDraw.Draw(paperBackground)
	draw.text((0,0), random.choice(string.ascii_letters), font=font, fill=(255,255,255))
	paperBackground.save(os.path.join(DATASET_PATH, str(genImageName) + ".jpg"))
	print(os.path.join(DATASET_PATH, str(genImageName) + ".jpg"))
	genImageName+=1