from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import numpy as np
import ot
import torch
import torch.nn as nn



def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        m.weight.data.normal_(0.0, 0.02)
        m.bias.data.fill_(0)




class MLP(nn.Module):
    def __init__(self, nz, ndf, ngpu,d):
        super(MLP, self).__init__()
        self.ngpu = ngpu

        main = nn.Sequential(
            # Z goes into a linear of size: ndf
            nn.Linear(d, ndf),
            nn.ReLU(True),
            nn.Linear(ndf , ndf // 2),
            nn.ReLU(True),
            nn.Linear(ndf // 2, 1),
        )
        self.main = main
        self.nz = nz

    def forward(self, input):
        input = input.view(input.size(0),input.size(1))
        if isinstance(input.data, torch.cuda.FloatTensor) and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            output = self.main(input)
        output = output.mean(0)
        return output.view(1)



 