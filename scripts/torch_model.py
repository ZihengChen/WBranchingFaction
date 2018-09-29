from torch_modellayer import *


class PredictiveModel(tc.nn.Module):
    def __init__(self):
        super(PredictiveModel,self).__init__()

        # define layers
        self.layer_beta = PertLayer_Beta()
        self.layer_xs   = PertLayer_XS()
        self.layer_eff  = PertLayer_Eff()
        self.layer_shape= PertLayer_Shape()
        
    def forward(self, x):
        # forward prop layers
        h1,regu1 = self.layer_beta  (x)
        h2,regu2 = self.layer_xs    (h1)
        h3,regu3 = self.layer_eff   (h2)
        h4,regu4 = self.layer_shape (h3)
        
        # output summation
        y = tc.sum(h4,1)
        regu = regu1+regu2+regu3+regu4

        return y,regu



def calculate_hessian(loss, model):
    grads = tc.autograd.grad(loss, model.parameters(), create_graph=True)
    grads = tc.cat([g.view(-1) for g in grads])

    temp = []
    for g in grads:
        grad2 = tc.autograd.grad(g, model.parameters(), create_graph=True)
        temp.append(grad2)

    hess = []
    for i in temp:
        ihess = []
        for j in i:
            ihess.append(j.data)
        hess.append(ihess)
    hess = np.array(hess)
    return hess
    