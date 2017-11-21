from __future__ import print_function
import collections
import numpy as np
from matplotlib import pyplot as plt
print("UBitName:"+ "anantram")
print("personNumber:" + "50249127")
print("UBitName: pdeshmuk")
print("personNumber: 50247649")
print("UBit_Name: hsokhey" )
print("personNumber: 50247213")

config = {"no_hidden_layers": 2,"no_hidden_units": 300,"batch_size": 128,"epochs": 10,
"learning_rate": 0.001,"no_samples": 1,"pi": 0.25,"sigma_p": 1.0,"sigma_p1": 0.75,"sigma_p2": 0.1,}


lgpi = np.log(2.0 * np.pi)

def gaussianlog(x, mu, sigma):
    return -0.5 * lgpi - nd.log(sigma) - (x - mu) ** 2 / (2 * sigma ** 2)

def p_gaussian(x):
    sigma_p = nd.array([config['sigma_p']], ctx=ctx)
    return nd.sum(gaussianlog(x, 0., sigma_p))

def logliklihood_softmax(yh, y):
    return nd.nansum(y * nd.log_softmax(yh), axis=0, exclude=True)
	
def gaus(x, mu, sigma):
    scal = 1.0 / nd.sqrt(2.0 * np.pi * (sigma ** 2))
    b = nd.exp(- (x - mu) ** 2 / (2.0 * sigma ** 2))
    return scal * b