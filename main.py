import os
import numpy as numpy
from features import calculateFeatures

## Creating array of labels and directories to calculate features
painters = []
paintings = []
directory = "paintings"
for folder in os.listdir(directory):
	folderRoute = os.path.join(directory, folder)
	if os.path.isdir(folderRoute):
		for fileName in os.listdir(folderRoute):
			if not fileName.startswith('.'):
				painters.append(folder)
				paintings.append(os.path.join(folderRoute, fileName))

paintersArray = numpy.array([painters])
paintingFeatures = None

for painting in paintings:
	features = calculateFeatures(painting)
	if paintingFeatures == None:
		paintingFeatures = features
	else:
		paintingFeatures = numpy.vstack((paintingFeatures, features))
print(paintingFeatures)

