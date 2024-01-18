import numpy as np
def intialise(a,b):
    epsilon=0.5
    c = np.random.rand(a,b+1)*(2*epsilon)-epsilon
    return c
