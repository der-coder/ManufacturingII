#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 24 00:14:11 2020

@author: isaac
"""

from matplotlib import pyplot as plt
import numpy as np
from scipy import integrate

def updateWeight(weights, outputDesired, outputPerceptron, inputs, learningRate):
    outputError = outputDesired - outputPerceptron
    I = np.ones((1, 4))
    W = weights + learningRate * (np.kron(outputError, I)) @ inputs
    return W

def activateNeuron(weights, inputs):
    s = np.zeros((4))
    for i in range(4):
        s[i] = np.dot(weights[i], inputs[i])
    return s

def outputNeuron(s):
    y = np.zeros(4)
    for i in range(4):
        if s[i] <= 0:
            y[i] = 0
        else:
            y[i] = 1
    y = y.reshape((4,1))
    return y

if __name__ == '__main__':
    kBias = 0
    kInputs = 0.5
    weights = np.concatenate((kBias * np.ones((4,1)), kInputs * np.ones((4,2))), axis= 1)
    
    
    bias = np.ones((4,1))
    dataInputs = np.array([[0,0],
                           [0,1],
                           [1,0],
                           [1,1]
                           ])
    
    data = np.concatenate((bias, dataInputs),axis=1)
    
    learningRate = 1
    
    yOR = np.array([
        [0],
        [1],
        [1],
        [1]
        ])
    
    yAND = np.array([
        [0],
        [0],
        [0],
        [1]
        ])
    
    s = activateNeuron(weights, data)
    y = outputNeuron(s)
    print(y)
    
    w = updateWeight(weights, yAND, y, data, learningRate)
    
    print(w)