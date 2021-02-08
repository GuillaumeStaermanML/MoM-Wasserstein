"""MoMWGAN code. It is highly inspired from the original WGAN code: https://github.com/martinarjovsky/WassersteinGAN.
This code is related to the section 4.3 of the following paper: 

When OT meets MoM: a robust estimation of the Wasserstein distance, G. Staerman, P. Laforgue, P. Mozharovskyi, F. d'Alché-Buc. AISTATS 2021.

Author: Guillaume Staerman
"""


from __future__ import print_function
import argparse
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch import autograd
import os
import json
import csv
from models.MoM import partition_blocks
from models.MoM import get_split_FM


import models.dcgan as dcgan
import models.mlp as mlp

if __name__=="__main__":




    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', required=True, help='cifar10 | celebA | lsun | imagenet | folder | lfw ')
    parser.add_argument('--noise', default=False, help='Noised real dataset')
    parser.add_argument('--anom_class', type=int, default=5, help='the choosen class to be added as anomalies (small subset)')
    parser.add_argument('--alpha', type=float, default=0.04, help='the proportion of (gaussian) anomalies')
    parser.add_argument('--proba', type=float, default=0.5, help='the probability to add an anomaly to training batches')      
    parser.add_argument('--K', type=int, default=12, help='the number of blocks')    
    parser.add_argument('--dataroot', required=True, help='path to dataset')
    parser.add_argument('--workers', type=int, help='number of data loading workers', default=2)
    parser.add_argument('--batchSize', type=int, default=64, help='input batch size')
    parser.add_argument('--imageSize', type=int, default=64, help='the height / width of the input image to network')
    parser.add_argument('--nc', type=int, default=3, help='input image channels')
    parser.add_argument('--nz', type=int, default=100, help='size of the latent z vector')
    parser.add_argument('--ngf', type=int, default=64)
    parser.add_argument('--ndf', type=int, default=64)
    parser.add_argument('--niter', type=int, default=25, help='number of epochs to train for')
    parser.add_argument('--lrD', type=float, default=0.00005, help='learning rate for Critic, default=0.00005')
    parser.add_argument('--lrG', type=float, default=0.00005, help='learning rate for Generator, default=0.00005')
    parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
    parser.add_argument('--ngpu'  , type=int, default=1, help='number of GPUs to use')
    parser.add_argument('--netG', default='', help="path to netG (to continue training)")
    parser.add_argument('--netD', default='', help="path to netD (to continue training)")
    parser.add_argument('--clamp_lower', type=float, default=-0.01)
    parser.add_argument('--clamp_upper', type=float, default=0.01)
    parser.add_argument('--Diters', type=int, default=5, help='number of D iters per each G iter')
    parser.add_argument('--noBN', action='store_true', help='use batchnorm or not (only for DCGAN)')
    parser.add_argument('--mlp_G', action='store_true', help='use MLP for G')
    parser.add_argument('--mlp_D', action='store_true', help='use MLP for D')
    parser.add_argument('--n_extra_layers', type=int, default=0, help='Number of extra layers on gen and disc')
    parser.add_argument('--experiment', default=None, help='Where to store samples and models')
    parser.add_argument('--adam', action='store_true', help='Whether to use adam (default is rmsprop)')
    parser.add_argument('--save_loss', required=True, help='directory to save the losses')
    opt = parser.parse_args()
    print(opt)

    if opt.experiment is None:
        opt.experiment = 'samples_WGAN'

    os.system('mkdir {0}'.format(opt.experiment)) 

    opt.manualSeed = 0 
    print("Random Seed: ", opt.manualSeed)
    random.seed(opt.manualSeed)
    torch.manual_seed(opt.manualSeed)
    np.random.seed(opt.manualSeed)
    if opt.ngpu > 0:
        torch.cuda.manual_seed_all(opt.manualSeed)
###########################################

#It enables benchmark mode in cudnn.
#benchmark mode is good whenever your input sizes for your network do not vary. This way, cudnn will look for the optimal set of algorithms for that particular configuration (which takes some time). This usually leads to faster runtime.
#But if your input sizes changes at each iteration, then cudnn will benchmark every time a new size appears, possibly leading to worse runtime performances.

###########################################
    cudnn.benchmark = True
###########################################
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #if torch.cuda.is_available() and not opt.cuda:
    #    print("WARNING: You have a CUDA device, so you should probably run with --cuda")



    if opt.dataset in ['imagenet', 'folder', 'lfw', 'celebA']:
        # folder dataset
        dataset = dset.ImageFolder(root=opt.dataroot,
                                transform=transforms.Compose([
                                    transforms.Resize(opt.imageSize), 
                                    transforms.CenterCrop(opt.imageSize),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                ]))
    elif opt.dataset == 'lsun':
        dataset = dset.LSUN(root=opt.dataroot, classes=['bedroom_train'],
                            transform=transforms.Compose([
                                transforms.Resize(opt.imageSize),
                                transforms.CenterCrop(opt.imageSize),
                                transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                            ]))
    elif opt.dataset == 'cifar10' or opt.dataset == 'cifar10_anom':
        dataset = dset.CIFAR10(root=opt.dataroot, download=True,
                            transform=transforms.Compose([
                                transforms.Resize(opt.imageSize),
                                transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                            ])
        )
    elif opt.dataset == 'Fashion-MNIST' or opt.dataset == 'Fashion-MNIST_anom':
        dataset = dset.FashionMNIST(root=opt.dataroot, download=True,
                            transform=transforms.Compose([
                                transforms.Resize(opt.imageSize),
                                transforms.ToTensor(),
                                transforms.Normalize((0.5,), (0.5,)),
                            ])
        )
    assert dataset

    if opt.dataset == 'cifar10_anom' or opt.dataset == 'Fashion-MNIST_anom':
        dataloader, anom = get_split_FM(dataset, anom_class=opt.anom_class, batch_size=opt.batchSize)
    else:
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batchSize,
                                            shuffle=True, num_workers=int(opt.workers))  

    ngpu = int(opt.ngpu)
    nz = int(opt.nz)
    ngf = int(opt.ngf)
    ndf = int(opt.ndf)
    nc = int(opt.nc)
    n_extra_layers = int(opt.n_extra_layers)

    # write out generator config to generate images together wth training checkpoints (.pth)
    generator_config = {"imageSize": opt.imageSize, "nz": nz, "nc": nc, "ngf": ngf, "ngpu": ngpu, "n_extra_layers": n_extra_layers, "noBN": opt.noBN, "mlp_G": opt.mlp_G}
    with open(os.path.join(opt.experiment, "generator_config.json"), 'w') as gcfg:
        gcfg.write(json.dumps(generator_config)+"\n")

    # custom weights initialization called on netG and netD
    def weights_init(m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            m.weight.data.normal_(0.0, 0.02)
        elif classname.find('BatchNorm') != -1:
            m.weight.data.normal_(1.0, 0.02)
            m.bias.data.fill_(0)



    if opt.noBN:
        netG = dcgan.DCGAN_G_nobn(opt.imageSize, nz, nc, ngf, ngpu, n_extra_layers)
    elif opt.mlp_G:               
        netG = mlp.MLP_G(opt.imageSize, nz, nc, ngf, ngpu)                                                    
    else:
        netG = dcgan.DCGAN_G(opt.imageSize, nz, nc, ngf, ngpu, n_extra_layers)

    # write out generator config to generate images together wth training checkpoints (.pth)
    generator_config = {"imageSize": opt.imageSize, "nz": nz, "nc": nc, "ngf": ngf, "ngpu": ngpu, "n_extra_layers": n_extra_layers, "noBN": opt.noBN, "mlp_G": opt.mlp_G}
    with open(os.path.join(opt.experiment, "generator_config.json"), 'w') as gcfg:
        gcfg.write(json.dumps(generator_config)+"\n")

    netG.apply(weights_init)
    if opt.netG != '': # load checkpoint if needed
        netG.load_state_dict(torch.load(opt.netG))
    print(netG)



    if opt.mlp_D:
        netD = mlp.MLP_D(opt.imageSize, nz, nc, ndf, ngpu)
    else:
        netD = dcgan.DCGAN_D(opt.imageSize, nz, nc, ndf, ngpu, n_extra_layers)
        netD.apply(weights_init)

    if opt.netD != '':
        netD.load_state_dict(torch.load(opt.netD))
    print(netD)

    input = torch.FloatTensor(opt.batchSize, nc, opt.imageSize, opt.imageSize).to(device)
    noise = torch.FloatTensor(opt.batchSize, nz, 1, 1).to(device)
    fixed_noise = torch.FloatTensor(opt.batchSize, nz, 1, 1).normal_(0, 1).to(device) 
    one = torch.FloatTensor([1]).to(device)
    mone = one * -1

    netD.to(device)
    netG.to(device)



    # setup optimizer
    if opt.adam:
        optimizerD = optim.Adam(netD.parameters(), lr=opt.lrD, betas=(opt.beta1, 0.999))
        optimizerG = optim.Adam(netG.parameters(), lr=opt.lrG, betas=(opt.beta1, 0.999))
    else:
        optimizerD = optim.RMSprop(netD.parameters(), lr = opt.lrD)
        optimizerG = optim.RMSprop(netG.parameters(), lr = opt.lrG)

    gen_iterations = 0
    plot_loss_D = []
    plot_loss_G = []
    for epoch in range(opt.niter):
        data_iter = iter(dataloader) 
        if opt.dataset == 'cifar10_anom' or opt.dataset == 'Fashion-MNIST_anom':
            anom_iter = iter(anom)
        i = 0
        while i < len(dataloader):
            ############################
            # (1) Update D network
            ###########################
            for p in netD.parameters(): # reset requires_grad
                p.requires_grad = True # they are set to False below in netG update

            # train the discriminator Diters times
            if gen_iterations < 25 or gen_iterations % 500 == 0: 
                Diters = 100
            else:
                Diters = opt.Diters
            j = 0
            while j < Diters and i < len(dataloader):
                j += 1
                

                # clamp parameters to a cube
                for p in netD.parameters():
                    p.data.clamp_(opt.clamp_lower, opt.clamp_upper)


                data = data_iter.next() 
                i += 1

                # train with real
                real_cpu, _ = data 
                netD.zero_grad() 

                # Adding anomalies for polluted Cifar10: 
                if (opt.noise and np.random.rand() > (1 - opt.proba)):
                    prop = opt.alpha * torch.rand(1)
                    anom = -1 + 2 * torch.rand(size=(int(prop * opt.batchSize), nc, opt.imageSize, opt.imageSize), dtype=torch.float) 
                    real_cpu = torch.cat((real_cpu,anom), 0)                                                                                    

                # Adding anomalies for polluted Fashion-MNIST: 
                if (opt.dataset == 'Fashion-MNIST_anom' and np.random.rand() > (1 - opt.proba)):
                        anom_, _ = anom_iter.next()
                        real_cpu = torch.cat((real_cpu, anom_), 0)


                batch_size = real_cpu.size(0)
                print(batch_size)


                
                real_cpu = real_cpu.to(device)
                input.resize_as_(real_cpu).copy_(real_cpu)
                input = input.to(device) 

                #Form a partition of the dataset:
                sigma, B = partition_blocks(input, opt.K)


                # Find the median block:
                blocks_values = [netD(input[sigma[k]]) for k in range(opt.K)]
                idx_med = sigma[np.argsort(blocks_values)[opt.K // 2]]
                



                #Applying gradients on the median block only
                errD_real = netD(input[idx_med])
                errD_real.backward()


                # train with fake
                noise.resize_(B, nz, 1, 1).normal_(0, 1)
                with torch.no_grad(): 
                    noisev = noise



                fake = netG(noisev.to(device)).data
                errD_fake = netD(fake)
                errD_fake.backward(mone)

                errD =  errD_real - errD_fake  
                
                optimizerD.step()


            ############################
            # (2) Update G network
            ###########################
            for p in netD.parameters():
                p.requires_grad = False # to avoid computation

            netG.zero_grad() 


            noise.resize_(B, nz, 1, 1).normal_(0, 1)
            fake = netG(noise)
            errG = netD(fake)
            errG.backward()
            optimizerG.step()
            gen_iterations += 1



            print('[%d/%d][%d/%d][%d] Loss_D: %f Loss_G: %f Loss_D_real: %f Loss_D_fake %f'
                % (epoch, opt.niter, i, len(dataloader), gen_iterations,
                errD.data[0], errG.data[0], errD_real.data[0], errD_fake.data[0]))
            plot_loss_D.append(errD.data[0].item())
            plot_loss_G.append(errG.data[0].item())
            


                
            if gen_iterations % 500 == 0:
                real_cpu = real_cpu.mul(0.5).add (0.5)
                vutils.save_image(real_cpu, '{0}/real_samples.png'.format(opt.experiment))
                with torch.no_grad(): 
                    fixed_noisev = fixed_noise
                fake = netG(fixed_noisev)
                fake.data = fake.data.mul(0.5).add(0.5)
                vutils.save_image(fake.data, '{0}/fake_samples_{1}.png'.format(opt.experiment, gen_iterations))


        torch.save(netG.state_dict(), '{0}/netG_epoch_{1}.pth'.format(opt.experiment, epoch))
    os.chdir(opt.save_loss)
    with open('loss_D.csv', 'w') as f:
            writer = csv.writer(f)
            writer.writerow(plot_loss_D)
    with open('loss_G.csv', 'w') as f:
            writer = csv.writer(f)
            writer.writerow(plot_loss_G)
