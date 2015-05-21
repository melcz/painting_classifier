import os
import numpy as numpy
import features.py

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

for painting in paintings:
	pass


