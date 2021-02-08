""" This is the code to compute the robust estimation of the Wasserstein distance using Medians-of-Means principle. 
Three estimators may be computed, MoM-MoM, MoU-diag, MoU estimators in reference to the paper:

"when OT meets MoM: a robust estimation of the Wasserstein distance, G. Staerman, P. Laforgue, P. Mozharovskyi, F. d'Alché-buc. AISTATS 2020."

Author: Guillaume Staerman
"""

from __future__ import print_function
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.utils as vutils
from torch import autograd
from tqdm import tqdm


from models.utils import weights_init
from models.utils import MLP
from models.MoM import partition_blocks

def MoM_Wasserstein(X,Y, estimator='MoM-MoM', K=10, ngpu=1, ndf=128, nz=10, lr=0.00005, device='cuda', random_state=0):
	"""
    Median-of-Means Wasserstein distance.

    Parameters
    ----------
    X: Array-like, (n_samples,dimension)
    	the source distribution.
    Y: Array-like, (n_samples,dimension)
    	the target distribution.


    estimator: str
    	three possible choice of estimators: MoM-MoM/MoU-diag/MoU

    K: int, default=10
        The number of blocks.

    ngpu: int,default=1.
		The number of gpu.

    ndf: int, default=128

    nz : int, default=10
		The size of le latent space.	

    lr : float, default=0.00005
		The learning rate.

	device: str, defaults='cuda'
        The choice to compute the distance on gpu or cpu. Gpu is highly recommended.

    random_state : int, RandomState instance, default=0

    Returns
    ----------

    loss: float
    	the distance values.
"""

	
	niter = K * 500
	d = X.shape[1]
	random.seed(random_state)
	torch.manual_seed(random_state)
	np.random.seed(random_state)

	if ngpu > 0:
		torch.cuda.manual_seed_all(random_state)


	z = torch.FloatTensor(niter)


	net = MLP(nz, ndf, ngpu, d)
	input_1 = torch.FloatTensor(X).clone().to(device) 
	input_2 = torch.FloatTensor(Y).clone().to(device)
	one = torch.FloatTensor([1]).to(device)
	mone = one * -1
	net.to(device)
	net.train()
	optimizer = optim.RMSprop(net.parameters(), lr=lr)

	
	for epoch in tqdm(range(niter),'iter'):
		for p in net.parameters():# reset requires_grad
			p.requires_grad = True

		# clamp parameters to a cube
		for p in net.parameters():
			p.data.clamp_(-0.01, 0.01)

		net.zero_grad()



		
		if estimator == 'MoM-MoM':

			#Form a partition of the dataset:
			sigma_1, B_1 = partition_blocks(input_1, K)
			sigma_2, B_2 = partition_blocks(input_2, K)

			# Find the median block:
			blocks_values_X = [net(input_1[sigma_1[l]]) for l in range(K)]
			blocks_values_Y = [net(input_2[sigma_2[l]]) for l in range(K)]

			idx_med_X = sigma_1[np.argsort(blocks_values_X)[K // 2]]
			idx_med_Y = sigma_2[np.argsort(blocks_values_Y)[K // 2]]



			err_X = net(input_1[idx_med_X])
			err_X.backward()

			err_Y = net(input_2[idx_med_Y])
			err_Y.backward(mone)

			err = err_X - err_Y

			z[epoch] = -err.data.item()
			optimizer.step()

		elif estimator == 'MoU-diag':

			#Form a partition of the dataset:
			sigma, B_1 = partition_blocks(input_1, K)

			# Find the median block:
			blocks_values = [net(input_1[sigma[l]]) - net(input_2[sigma[l]])  for l in range(K)]
			idx_med = sigma[np.argsort(blocks_values)[K // 2]]


			err_X = net(input_1[idx_med])
			err_X.backward()

			err_Y = net(input_2[idx_med])
			err_Y.backward(mone)


			err = err_X - err_Y


			z[epoch] = -err.data.item()

			optimizer.step()

		elif estimator == 'MoU':

			#Form a partition of the dataset:
			sigma_1, B_1 = partition_blocks(input_1, K)
			sigma_2, B_2 = partition_blocks(input_2, K)

			# Find the median block:
			blocks_values = [net(input_1[sigma_1[l]]) - net(input_2[sigma_2[s]])  for l in range(K)  for s in range(K)]		

			med = np.argsort(blocks_values)[K ** 2 // 2] # 

			idx_med_X = sigma_1[med // K]
			idx_med_Y = sigma_1[med % K]


			err_X = net(input_1[idx_med_X])
			err_X.backward()

			err_Y = net(input_2[idx_med_Y])
			err_Y.backward(mone)

			err = err_X - err_Y

			z[epoch] = -err.data
			optimizer.step()
			

	#Average on the last 100 iterations to have a stable value.
	loss = z[niter-100:].mean()
	return loss



