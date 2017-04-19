import os
import numpy as numpy
import math
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
		chromosome[random.randint(0, features-1)] = 1;
		initialPopulation.append(chromosome)
	return initialPopulation

def evaluateGeneration(population):
	results = []

	for i in range(0, populationSize):
		individual = data
		for j in reversed(range(0, len(data[0]))):
			if population[i][j] ==  0:
				individual = numpy.delete(individual, j, 1)
		kmeans = calculateCluster (individual, labels, False)
		vmeasure = metrics.v_measure_score(labels[0], kmeans.labels_)
		silhouette = metrics.silhouette_score(data, kmeans.labels_, metric='euclidean',sample_size=len(data))
		results.append(vmeasure+silhouette/2)

	return results

def generatePool(population, generation):
	pool = []
	for i in range(0,len(generation)):
		times = int(generation[i]*100)
		for j in range(0,times):
			pool.append(population[i])
	return pool

def elitistSelection(population, generation):
	average = reduce(lambda x, y: x + y, generation) / len(generation)
	pool = []
	for i in range(0,len(generation)):
		if generation[i] > average:
			times = int(generation[i]*100)
			for j in range(0,times):
				pool.append(population[i])
	return pool


def getNewGeneration(pool, populationSize, crossoverRate, mutationRate):
	rate = crossoverRate*100
	rateM = mutationRate*100
	rateB = biasRate*100
	offspring = []
	while len(offspring) != populationSize:
		parent1 = pool[random.randint(0, len(pool)-1)]
		parent2 = pool[random.randint(0, len(pool)-1)]
		if uniform:
			for i in range(0, len(pool[0])-1):
				if random.randint(0, 100) < rate:
					temp = parent1[i]
					parent1[i] = parent2[i]
					parent2[i] = temp
		elif random.randint(0, 100) < rate:
			point = random.randint(0, len(pool[0])-1)
			temp = parent1[point:]
			parent1[point:] = parent2[point:]
			parent2[point:] = temp
			# print("crossed over at "+str(point))
		if random.randint(0, 100) < rateM:
			pos = random.randint(0, len(pool[0])-1)
			parent1[pos] = 0 if parent1[pos] == 1 else 1
			# print("mutated 1 at: "+str(pos))
		if random.randint(0, 100) < rateM:
			pos = random.randint(0, len(pool[0])-1)
			parent2[pos] = 0 if parent2[pos] == 1 else 1
			# print("mutated 2 at: "+str(pos))
		if random.randint(0, 100) < rateB:
			for i in biasedIndices:
				pos = random.randint(0, len(biasedIndices)-1)
				if random.randint(0, 100) < 50:
					parent1[pos] = 1
				else:
					parent1[pos] = 1
			# print("mutated 1 at: "+str(pos))
		offspring.append(parent1)
		offspring.append(parent2)
	return offspring

def printGeneration(generation, fileOut):
	for fitness in generation:
		fileOut.write(str(fitness))
		fileOut.write(",")
	fileOut.write("\n")

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

populationSize = 50
numGenerations = 100
crossoverRate = 0.5
mutationRate = 0.05
biasedIndices = [6,7]
biasRate = 0.0
uniform = False

population = initializePopulation(populationSize, len(data[0]))
fileOut = open("out.txt", "w")

for i in range(0,numGenerations):
	print(".")
	generation = evaluateGeneration(population)
	pool = elitistSelection(population,generation)
	printGeneration(generation, fileOut)
	# pool = generatePool(population, generation)
	population = getNewGeneration(pool, populationSize, crossoverRate, mutationRate)

print("\n")
fileOut.close()
