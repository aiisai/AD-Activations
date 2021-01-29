import torch
from torch.nn import Module
import numpy as np


#I_Sigmoid
offset_p = 1/(1+np.exp(-0.01))
offset_n = 1/(1+np.exp(0.01))

class I_Sigmoid(Module):
    def __init__(self, inplace=False):
        super(I_Sigmoid, self).__init__()
        self.inplace = inplace
    def forward(self, x):
        y0 = torch.sigmoid(x)
        yn = 1.0 * (x + 0.01) + offset_n
        y = y0.clone()
        y[x < -0.01] = yn[x < -0.01]
        yp = 1.0 * (x - 0.01) + offset_p
        y[x > 0.01] = yp[x > 0.01]
        return y