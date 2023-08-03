import numpy as np
import pandas as pd
from funcs import *

class Network:  # Creating class Network for binary classification
    def __init__(self,numInputs,numHidden): # for class initialization we pass number of neurons
        self.numInputs = numInputs          # of input and hidden layer
        self.numHidden = numHidden
        self.inputs = np.ones(numInputs)    # init matrix of output neurons

        self.hiddenLayerWeights = np.zeros( (numInputs,numHidden) ) # init matrix for hidden layer weights
        self.hiddenLayerStimuli = np.zeros(numHidden)               # init matrix for hidden layer stimuli
        self.hiddenLayerOutputs = np.zeros(numHidden)               # init matrix for hidden layer outputs


        self.outputLayerWeights = np.zeros(numHidden)               # init matrix for output layer weights
        self.outputLayerStimuli = np.array([0])                     # init matrix for output layer stimuli

        self.output = 0                                             # init output
        self.correctOutput = 0                                      # init correct output
        self.learningRate = 0.1                                     # learning rate

        self.dataSet = None                                         # init dataset

    def randomizeWeights(self, range):  ## Assign random weights to hidden and output layers from a normal 
                                        ## distribution within the specified range {-range, range}
        self.hiddenLayerWeights = np.random.uniform(-range, range, size=(self.numInputs,self.numHidden) )
        self.outputLayerWeights = np.random.uniform(-range, range, size=(self.numHidden) )

    def feedData(self, data):           # The passing of the dataset to the input layer
        self.dataSet = data             
    
    def forwardPropagate(self):     # Data propagation
        self.hiddenLayerStimuli = np.matmul(self.hiddenLayerWeights.T, self.inputs) # Calculation of activations in the hidden layer
        self.hiddenLayerOutputs = leakyRELU(self.hiddenLayerStimuli)                # Calculation of outputs in the hidden layer
                                                                                    # With the utilization of the Leaky ReLU function
        self.outputLayerStimuli = np.matmul(self.outputLayerWeights, self.hiddenLayerOutputs)   # Calculation of the activation in the output neuron
        self.output = sigmoid(self.outputLayerStimuli)          # Calculation of the output of the output neuron using the sigmoid function

    def changeWeights(self):    # Weight correction based on errors
        temp = np.subtract(self.correctOutput, self.output) # Error (correct_output - output)
        outputError = np.multiply(temp, sigmoid_der(self.outputLayerStimuli))   # Calculation of the error signal for the output layer
                                                                                # With the utilization of the derivative of the sigmoid function
        temp1 = np.multiply(self.outputLayerWeights,outputError)    
        hiddenLayerError = np.multiply( temp1, leakyRELU_der(self.hiddenLayerStimuli))  # Calculation of the error signal for the hidden layer
                                                                                        # With the utilization of the derivative of the Leaky ReLU function
        # The line below: value of the change in weights for the output neuron
        outputLayerWeightChange = np.multiply(self.outputLayerStimuli, outputError) * self.learningRate
        # The line below: value of the change in weights for the neurons in the hidden layer
        hiddenLayerWeightChange = np.matmul( hiddenLayerError.reshape((self.numHidden,1)) , self.inputs.reshape((self.numInputs,1)).T ).T  * self.learningRate

        self.hiddenLayerWeights = np.add(self.hiddenLayerWeights, hiddenLayerWeightChange)  # Weight adjustment for the neurons in the hidden layer
        self.outputLayerWeights = np.add(self.outputLayerWeights, outputLayerWeightChange)  # Weight adjustment for the output neuron
    
    def getWeights(self):   # A function returning the weights of neurons in the hidden and output layers
        return [self.hiddenLayerWeights, self.outputLayerWeights]

    def feedWeights(self, hiddenWeights, outputWeights):    # A function used to pass the final weights of neurons in the hidden and output layers
        self.hiddenLayerWeights = hiddenWeights             # And using them during testing of the network with test data
        self.outputLayerWeights = outputWeights

    def oneEpoch(self):                                     # Performing an iterations
        self.dataSet = self.dataSet.sample(frac = 1)        # Shuffling the dataset
        numberRows = self.dataSet.shape[0]
        for i in range(numberRows):
            data = self.dataSet.iloc[i].tolist()            # A variable holding a data row
            self.correctOutput = data.pop()                 # Saving the correct output and removing it from the data row

            self.inputs = np.array(data)                    # Passing the data row to the input layer
            self.forwardPropagate()                         # Propagation
            self.changeWeights()                            # Weight update
    
    def checkAccuracy(self):                                # Checking accuracy and diagnostic classifiers
        false_pos = 0
        false_neg = 0
        true_pos  = 0
        true_neg  = 0
        numberRows = self.dataSet.shape[0]
        for index in range(numberRows):
            data = self.dataSet.iloc[index].tolist()
            self.correctOutput = data.pop()
            outcome = self.correctOutput

            self.inputs = np.array(data)
            self.forwardPropagate()                        # The final propagation without changing weights for testing the network
            output = self.output
            if( output >= 0.5 and outcome == 1):
                true_pos += 1
            elif( output < 0.5 and outcome == 0):
                true_neg += 1
            elif(output >= 0.5 and outcome == 0):
                false_neg += 1
            elif(output < 0.5 and outcome == 1):
                false_pos += 1
            else:
                pass
                                                                                                # Results analysis
        print(f"Correct classifications = {100*round((true_pos+true_neg)/numberRows,4)} %")     # Calculation of accuracy
        if (false_neg+true_pos != 0):
            print(f"Sensitivity = {round(true_pos/(false_neg+true_pos),4)}")                        # Calculation of true positive rate (TPR)
        if (true_neg+false_pos != 0):
            print(f"Specificity = {round(true_neg/(true_neg+false_pos),4)}")                  # Calculation of true negative rate (TNR)
        print("----")
