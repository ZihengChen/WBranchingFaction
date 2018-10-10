import torch as tc
import torch.nn.functional as F
from   torch.autograd import Variable
from torch.nn import Parameter
from torch.nn import init
import utility_common as common
from pylab import *
from torch.autograd import grad

device = tc.device("cuda" if tc.cuda.is_available() else "cpu")

def autograd2nd(loss, model):
    grads = tc.autograd.grad(loss, model.parameters(), create_graph=True, allow_unused=True)
    grads = tc.cat([g.view(-1) for g in grads])

    temp = []
    for g in grads:
        grad2 = tc.autograd.grad(g, model.parameters(), create_graph=True, allow_unused=True)
        temp.append(tc.cat(grad2))

    hess = []
    for i in temp:
        ihess = []
        for j in i:
            ihess.append(float(j.data))
        hess.append(ihess)
    hess = np.array(hess)
    return hess


def showLossHistory(losses):
    losses = losses[10:]
    plt.figure(facecolor='w',figsize=(6,4))
    plt.plot(losses, lw=2)
    plt.xlabel('Steps')
    plt.ylabel('Loss')
    plt.grid(True,linestyle='--',alpha=0.6)


def templateVariation(region, shaping):
        
        baseDir = common.getBaseDirectory()
        
        # config
        if shaping:
            folderType = 'shaping'
        else:
            folderType = 'counting'

        # nominal - dwon
        dx_list = []
        for variation in ['EPt','MuPt','TauPt']:
            x0 = np.load(baseDir + "data/templates/{}_{}Region/X_{}.npy".format(folderType,region,''))
            x1 = np.load(baseDir + "data/templates/{}_{}Region/X_{}Down.npy".format(folderType,region,variation))
            dx = tc.from_numpy((x0-x1)).type(tc.FloatTensor).to(device)
            dx_list.append(dx)

        # (up-down)/2
        for variation in ['JES','JER','BTag','Mistag']:
            x1 = np.load(baseDir + "data/templates/{}_{}Region/X_{}Down.npy".format(folderType,region,variation))
            x2 = np.load(baseDir + "data/templates/{}_{}Region/X_{}Up.npy"  .format(folderType,region,variation))
            dx = tc.from_numpy((x2-x1)/2).type(tc.FloatTensor).to(device)
            dx_list.append(dx)


        return dx_list