import os
import numpy as numpy
from features import calculateFeatures
from classifier import calculateCluster

from sklearn.datasets import load_digits
from sklearn.preprocessing import scale
from sklearn import metrics
import pickle
import sys
import random

def fillFeaturesMatrix (paintings):
	paintingFeatures = numpy.array([])
	for painting in paintings:
		features = calculateFeatures(painting)
		if paintingFeatures.size == 0:
			paintingFeatures = features
		else:
			paintingFeatures = numpy.vstack((paintingFeatures, features))
		sys.stdout.write('.')
		sys.stdout.flush()
	print
	return paintingFeatures

def processData(directory):
	labels = []
	datapoints = []
	for folder in os.listdir(directory):
		folderRoute = os.path.join(directory, folder)
		if os.path.isdir(folderRoute):
			for fileName in os.listdir(folderRoute):
				if not fileName.startswith('.'):
					labels.append(folder)
					datapoints.append(os.path.join(folderRoute, fileName))

	labelsArray = numpy.array([labels])
	dataMatrix = fillFeaturesMatrix(datapoints)
	data = scale(dataMatrix)

	return (data, labelsArray)

def initializePopulation(size, features):
	initialPopulation = []
	for i in range(0,size):
		chromosome = []
		for j in range (0,features):
			chromosome.append(random.randint(0, 1))
		initialPopulation.append(chromosome)
	return initialPopulation


dataName = "data.sav"
labelsName = "labels.sav"
data = []
labels = []
## Load data if exists and train model
if os.path.isfile(dataName) and os.path.isfile(labelsName):
	data = pickle.load(open(dataName, 'rb'))
	labels = pickle.load(open(labelsName, 'rb'))
else:
	print("No data found, loading")
	data, labels = processData("paintings")
	pickle.dump(data, open(dataName, 'wb'))
	pickle.dump(labels, open(labelsName, 'wb'))

populationSize = 10

population = initializePopulation(populationSize, len(data[0]))
results = []

for i in range(0, populationSize):
	individual = data
	for j in reversed(range(0, len(data[0]))):
		if population[i][j] ==  0:
			individual = numpy.delete(individual, j, 1)
	kmeans = calculateCluster (individual, labels, False)

# kmeans = calculateCluster (data, labels, False)

# metrics.v_measure_score(labels[0], kmeans.labels_)
# metrics.silhouette_score(data, kmeans.labels_, metric='euclidean',sample_size=len(data))
