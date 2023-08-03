import numpy as np
import pandas as pd


def leakyRELU(input):
    return np.maximum(0.01*input , input)
    
def leakyRELU_der(x):
    data = [1 if value>0 else 0.01 for value in x]
    return np.array(data, dtype=float)

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def sigmoid_der(input):
    dummy1 = sigmoid(input)
    dummy2 = (1-sigmoid(input))
    return dummy1*dummy2

def normalize(frame):
    dummy=[]
    for header in frame.columns.values:
        col = frame[header]
        min = frame[header].min()
        max = frame[header].max()
        dummy.append((col-min)/(max-min))
    return dummy