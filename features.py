# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as numpy
import skimage as skimage
from skimage import feature
from skimage.exposure import is_low_contrast
from skimage.morphology import disk
from skimage.util import img_as_ubyte
from skimage.feature import blob_dog
from skimage.io import imread
from fractal import fractal_dimension
from skimage.feature import corner_shi_tomasi, corner_peaks
from skimage.morphology import disk
from skimage.filters.rank import gradient
from skimage.filters.rank import maximum
from skimage.filters.rank import geometric_mean




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
	edgesCanny = feature.canny(img_gray)
	edges = numpy.sum(edgesCanny)

	tomasiCorners = len(corner_peaks(corner_shi_tomasi(img_gray), min_distance=1))
	grad = numpy.sum(gradient(img_gray, disk(5)))
	maxi = numpy.sum(maximum(img_gray, disk(5)))
	geoMean = numpy.sum(geometric_mean(img_gray, disk(5)))

	histo = numpy.histogram(image, bins=5)
	histoFirst = histo[0][0]
	histo2 = histo[0][1]
	histoMiddle = histo[0][2]
	histo3 = histo[0][3]
	histoLast = histo[0][4]

	featureVector = numpy.array([meanEntropy, maxEntropy, meanIntensity, meanHue, countOfAverageHuePixels, percentageOfLightPixels, mbFractalDimension, hFractalDimension, blobCount, edges, tomasiCorners, grad, maxi, geoMean, histoFirst, histoMiddle, histoLast, histo2, histo3])

	return featureVector
