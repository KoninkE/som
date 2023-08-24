import matplotlib.pyplot as plot
from mpl_toolkits.axes_grid1 import ImageGrid
import numpy as np
import math
import random
import time

class Neuron:
    def __init__(self, numWeights, xCoord, yCoord, learning_rate, lmda):
        self.xCoord = xCoord
        self.yCoord = yCoord
        self.numWeights = numWeights
        self.learning_rate = learning_rate
        self.lmda = lmda

        
        
    def initializeWeights(self, lowBound, highBound):
        temp = np.random.rand(self.numWeights)
        temp = temp / np.linalg.norm(temp)
        self.weights = temp.tolist()


    def euclideanDist(self, inputVect):
        mySum = 0
        
        for i in range(0, self.numWeights):
            mySum = mySum + ((inputVect[i] - self.weights[i]) ** 2)

        return math.sqrt(mySum)

    def updateWeights(self, inputVector, iterNum, dist):
        lmda = self.lmda * math.exp((-1 * iterNum) / 225)
        
        theta = math.exp(-1 * ((dist ** 2) / (2 * (lmda ** 2))))
        LR = self.learning_rate * math.exp((-1 * iterNum) / 225)

##        if((self.xCoord == 0) and (self.yCoord == 0)):
##            print(LR)
        
        for i in range(0, self.numWeights):
            self.weights[i] = self.weights[i] + (theta * LR * (inputVector[i] - self.weights[i]))
            

# Parent class for Neuron class
class Grid:
    def __init__(self, gridLength, gridWidth, numInputs, learning_rate, lmda):
        self.neurons = [[0] * gridWidth for i in range(gridLength)]

        for i in range(0, gridLength):
            for j in range(0, gridWidth):
                self.neurons[i][j] = Neuron(numInputs, j, i, learning_rate, lmda)
            

    def randNodeWeights(self, lowBound, highBound):
        for i in range(0, len(self.neurons)):
            for j in range(0, len(self.neurons[0])):
                self.neurons[i][j].initializeWeights(lowBound, highBound)

    def pickBMU(self, inputVect):
        #minDist is initialized as the diagnal length of the grid
        minDist = math.sqrt((len(self.neurons) ** 2) + (len(self.neurons) ** 2))
        self.bmuX = 0
        self.bmuY = 0
        
        for i in range(0, len(self.neurons)):
            for j in range(0, len(self.neurons[0])):
                neuronDist = self.neurons[i][j].euclideanDist(inputVect)
                #print("Min Dist: " + str(minDist) + " Current Iter Dist: " + str(neuronDist))

                if(neuronDist < minDist):
                    minDist = neuronDist
                    self.bmuX = j
                    self.bmuY = i

        print("(" + str(self.bmuX) + ", " + str(self.bmuY) + ")")
        self.bmuDist = minDist


    #updates the weights of the neurons around the bmu
    #takes the image iteration and acceptable diagnal tolerance to bmu
    def updateNeighborhood(self, inputVector, iterNum):
        gridLength = len(self.neurons)
        gridWidth = len(self.neurons[0])

        x1 = self.bmuX
        y1 = self.bmuY

        for i in range(0, len(self.neurons)):
            for j in range(0, len(self.neurons[0])):
                x2 = self.neurons[i][j].xCoord
                y2 = self.neurons[i][j].yCoord
                d = math.sqrt(((x2 - x1) ** 2) + ((y2 - y1) ** 2))
                #d = abs(x2 - x1)
                self.neurons[i][j].updateWeights(inputVector, iterNum, d)
                 
   
#reads a text file containing a 2D list- assumes the data is supposed
#to be of type float (because they are most likely weights)    
def readFiles(fileName):
    fid = open(fileName, "r")
    list2D = []

    for line in fid.readlines():
        list2D.append(line.split("\t"))

    for i in range(0, len(list2D)):
        for j in range(0, len(list2D[0])):
            list2D[i][j] = float(list2D[i][j])
        
    return list2D

#Randomizes a 2D list (normally a list of images)
def randomizeInputs(inputs):
    randInputs = inputs
    
    for i in range(0, len(inputs)):
        r = random.randint(0, len(inputs) - 1)
        temp = randInputs[i]
        randInputs[i] = randInputs[r]
        randInputs[r] = temp

    return randInputs


def inputSubset(inputs, numInputsToUse):
    if(numInputsToUse > len(inputs)):
        raise ValueError('Error: Input subset cannot be larger than the original input vector.')
    
    subset = []

    for i in range(0, numInputsToUse):
        subset.append(inputs[i])

    return subset

def plotSOFM(myGrid, imageLength, imageWidth):
    gridLength = len(myGrid.neurons)
    gridWidth = len(myGrid.neurons[0])
    weights = []
    images = []

    for i in range(0, gridLength):
        for j in range(0, gridWidth):
            weights = myGrid.neurons[i][j].weights
            image = np.array(weights).reshape((imageWidth, imageLength))
            image = np.rot90(image, k=1, axes=(0, 1))
            image = np.rot90(image, k=1, axes=(0, 1))
            image = np.rot90(image, k=1, axes=(0, 1))
            image = np.fliplr(image)
            images.append(image)
            
    
    fig = plot.figure(figsize = (gridWidth, gridLength))
    imGrid = ImageGrid(fig, 111, nrows_ncols=(gridLength, gridWidth), axes_pad = 0.05)

    for ax, im in zip(imGrid, images):
        ax.imshow(im, cmap = 'Greys')

    plot.axis('off')

    plot.show()


inputs = readFiles("trainingset.txt")

inputs = randomizeInputs(inputs)

inputs = inputSubset(inputs, 1600)

numInputs = len(inputs[0])

myGrid = Grid(8, 8, numInputs, 0.65, 0.995)

myGrid.randNodeWeights(0, 1)

for i in range(0, len(inputs)):
    myGrid.pickBMU(inputs[i])
    myGrid.updateNeighborhood(inputs[i], i)

plotSOFM(myGrid, 28, 28)


