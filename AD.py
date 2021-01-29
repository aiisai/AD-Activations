import torch
from torch.nn import Module


#AD_ReLUs
class ad_relu1(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, a):
        ctx.save_for_backward(x, a)
        return x.clamp(min=0)
    @staticmethod
    def backward(ctx, grad_output):
        x,a = ctx.saved_tensors
        grad_x = grad_output.clone()
        grad_x = torch.mul(grad_x, torch.sigmoid(a * x))
        return grad_x, None

class AD_ReLU1(Module):
    def __init__(self, a, inplace=False):
        super(AD_ReLU1, self).__init__()
        self.inplace = inplace
        self.a = a
    def forward(self, input):
        return ad_relu1.apply(input, self.a)


class ad_relu2(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, a):
        ctx.save_for_backward(x, a)
        return x.clamp(min=0)
    @staticmethod
    def backward(ctx, grad_output):
        x, a = ctx.saved_tensors
        grad_x = grad_output.clone()
        d1 = (a * x + 0.5).clamp(min=0., max=1.)
        grad_x = torch.mul(grad_x, d1)
        return grad_x, None

class AD_ReLU2(Module):
    def __init__(self, a, inplace=False):
        super(AD_ReLU2, self).__init__()
        self.inplace = inplace
        self.a = a
    def forward(self, input):
        return ad_relu2.apply(input, self.a)


#APD_Sigmoids
class ad_sigmoid1(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        y = torch.sigmoid(x)
        ctx.save_for_backward(x, y)
        return y
    @staticmethod
    def backward(ctx, grad_output):
        x, y, = ctx.saved_tensors
        grad_x = grad_output.clone()
        d1 = torch.min(y, (1 - y))
        grad_x = torch.mul(grad_x, d1)
        return grad_x, None


class AD_Sigmoid1(Module):
    def __init__(self, inplace=False):
        super(AD_Sigmoid1, self).__init__()
        self.inplace = inplace
    def forward(self, input):
        return ad_sigmoid1.apply(input)

class ad_sigmoid2(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        y = torch.sigmoid(x)
        ctx.save_for_backward(x, y)
        return y
    @staticmethod
    def backward(ctx, grad_output):
        x, y, = ctx.saved_tensors
        grad_x = grad_output.clone()
        d1 = (2 * torch.min(y, (1 - y))).clamp_min(0.001)
        grad_x = torch.mul(grad_x, d1)
        return grad_x, None

class AD_Sigmoid2(Module):
    def __init__(self, inplace=False):
        super(AD_Sigmoid2, self).__init__()
        self.inplace = inplace
    def forward(self, input):
        return ad_sigmoid2.apply(input)