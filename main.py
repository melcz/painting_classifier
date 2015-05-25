import os
import numpy as numpy
from features import calculateFeatures
from classifier import calculateCluster
from sklearn.datasets import load_digits
from sklearn.preprocessing import scale

def fillFeaturesMatrix (paintings):
	paintingFeatures = None

	for painting in paintings:
		features = calculateFeatures(painting)
		if paintingFeatures == None:
			paintingFeatures = features
		else:
			paintingFeatures = numpy.vstack((paintingFeatures, features))
	return paintingFeatures

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
#featuresMatrix = fillFeaturesMatrix(paintings)

digits = load_digits()
data = scale(digits.data)
#calculateCluster (data, digits)