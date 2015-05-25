"""
=======
Entropy
=======

Image entropy is a quantity which is used to describe the amount of information
coded in an image.

"""
import matplotlib.pyplot as plt
import numpy as numpy

from skimage import data, color
from skimage.filters.rank import entropy
from skimage.morphology import disk
from skimage.util import img_as_ubyte
from skimage.io import imread

def calculateFeatures( imagePath ):
	image = img_as_ubyte(imread(imagePath, as_grey=False, plugin=None, flatten=None))
	img_gray = color.rgb2gray(image)
	img_hsv = color.rgb2hsv(image)

	fig, (ax0, ax1) = plt.subplots(ncols=2, figsize=(10, 4))

	img0 = ax0.imshow(img_gray, cmap=plt.cm.gray)
	ax0.set_title('Image')
	ax0.axis('off')
	fig.colorbar(img0, ax=ax0)

	imageEntropy = entropy(img_gray, disk(5))
	img1 = ax1.imshow(imageEntropy, cmap=plt.cm.jet)
	ax1.set_title('Entropy')
	ax1.axis('off')
	fig.colorbar(img1, ax=ax1)

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

	featureVector = numpy.array([imageEntropy.mean(), imageEntropy.max(), img_gray.mean(), hueMatrix.mean(), hueMatrix.size - numpy.count_nonzero(hueCount), float(numpy.count_nonzero(brightMatrix)) / img_gray.size])

	#print ("Mean entropy: "+repr(featureVector[0]))
	#print ("Max entropy: "+repr(featureVector[1]))
	#print ("Mean intensity: "+repr(featureVector[2]))
	#print ("Mean hue: "+repr(featureVector[3]))
	#print ("Count of average hue pixels: "+repr(featureVector[4]))
	#print ("Percentage of light pixels: "+repr(featureVector[5]))

	#plt.show()

	return featureVector
