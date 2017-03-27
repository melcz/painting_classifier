# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as numpy
import skimage as skimage
from skimage.morphology import disk
from skimage.util import img_as_ubyte
from skimage.feature import blob_dog
from skimage.io import imread
from fractal import fractal_dimension

def calculateFeatures( imagePath ):
	image = img_as_ubyte(imread(imagePath, as_grey=False, plugin=None, flatten=None))
	img_gray = skimage.color.rgb2gray(image)
	img_hsv = skimage.color.rgb2hsv(image)

	imageEntropy = skimage.filters.rank.entropy(img_gray, disk(5))

	huePreMatrix = []
	for row in img_hsv:
		hueRow = []
		for pixel in row:
			hueRow.append(pixel[0])
		huePreMatrix.append(hueRow)

	hueMatrix = numpy.matrix(huePreMatrix)
	hueRound = numpy.round(hueMatrix, 4)
	hueCount = numpy.subtract(hueRound, numpy.round(hueRound.mean(), 4))

	brightMatrix = numpy.where( img_gray > img_gray.mean() )

	meanEntropy = imageEntropy.mean()
	maxEntropy = imageEntropy.max()
	meanIntensity = img_gray.mean()
	meanHue = hueMatrix.mean()
	countOfAverageHuePixels = hueMatrix.size - numpy.count_nonzero(hueCount)
	percentageOfLightPixels = float(numpy.count_nonzero(brightMatrix)) / img_gray.size

	mbFractalDimension = fractal_dimension(img_gray)
	hFractalDimension = (numpy.log(3)/numpy.log(2))

	blobCount = len(blob_dog(img_gray))

	featureVector = numpy.array([meanEntropy, maxEntropy, meanIntensity, meanHue, countOfAverageHuePixels, percentageOfLightPixels, mbFractalDimension, hFractalDimension, blobCount])

	return featureVector
