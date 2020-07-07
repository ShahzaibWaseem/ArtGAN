import os

NUM_OF_TIMES=10

print("Content\t\t\tImage\t-\tStyle\t\tImage\tTime")
for run in range(NUM_OF_TIMES):
	os.system("python3 styleTransfer.py")