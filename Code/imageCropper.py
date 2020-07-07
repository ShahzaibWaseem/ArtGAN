import os
import cv2
import math
import numpy as np
from glob import glob

DATASET_SELECTED="WilliamMorris800/"
DATASETS_PATH="../Datasets/"

REQUIRED_DIMS = (100,100,1)

def cropImage(image):
	row, col = image.shape
	croppedImages=[]

	rowLoop = math.ceil(row/REQUIRED_DIMS[0])
	colLoop = math.ceil(col/REQUIRED_DIMS[1])

	for i in range(rowLoop):
		for j in range(colLoop):
			if REQUIRED_DIMS[0]*(i+1) <= row and REQUIRED_DIMS[1]*(j+1) <= col:
				croppedImage=image[REQUIRED_DIMS[0]*i:REQUIRED_DIMS[0]*(i+1), REQUIRED_DIMS[1]*j:REQUIRED_DIMS[1]*(j+1)]
				croppedImages.append(croppedImage)
	return croppedImages

def main():
	numFiles=len(glob(os.path.join(DATASETS_PATH, DATASET_SELECTED, "*.jpg")))
	print("Number of Files in this Directory:", str(numFiles))
	customDataset = []

	for filename in glob(os.path.join(DATASETS_PATH, DATASET_SELECTED, "*.jpg")):
		image = cv2.imread(filename, 0)
		images=cropImage(image, filename)
		customDataset.extend(images)

	customDataset=np.array(customDataset)
	print(customDataset.shape)

if __name__ == '__main__':
	main()