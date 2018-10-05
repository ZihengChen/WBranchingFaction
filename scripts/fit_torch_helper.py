import torch as tc
import torch.nn.functional as F
from   torch.autograd import Variable
from torch.nn import Parameter
import utility_common as common
from pylab import *
from torch.autograd import grad



def autograd2nd(loss, model):
    grads = tc.autograd.grad(loss, model.parameters(), create_graph=True, allow_unused=True)
    grads = tc.cat([g.view(-1) for g in grads])

    temp = []
    for g in grads:
        grad2 = tc.autograd.grad(g, model.parameters(), create_graph=True, allow_unused=True)
        temp.append(grad2)

    hess = []
    for i in temp:
        ihess = []
        for j in i:
            ihess.append(j.data)
        hess.append(ihess)
    hess = np.array(hess)
    return hess

def showLossHistory(losses):
    plt.figure(facecolor='w',figsize=(6,4))
    plt.plot(losses, lw=2)
    plt.xlabel('Steps')
    plt.ylabel('Loss')
    plt.grid(True,linestyle='--',alpha=0.6)
