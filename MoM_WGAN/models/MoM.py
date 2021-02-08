import numpy as np
import torch
import torch.utils.data



def partition_blocks(X, K):
    """Partition X into K disjoint blocks as large as possible
    """
    # get largest block size (should be at least 1)
    n = len(X)
    B = n // K
    if B == 0:
        raise ValueError("Invalid number of blocks %s, "
                         "larger than number of samples %s" % (K, n))

    # create and return blocks (plus block size)
    #np.random.shuffle(X)
    sigma = np.random.permutation(n)
    sigma = sigma[:K * B]
    #X = X[sigma]
    #np.split(X, K),



    return  np.split(sigma, K), B



def get_split_FM(data_train, anom_class, batch_size=64, shuffle=True):

    targets_train = data_train.targets
    targets_train = torch.tensor(targets_train)
    target_train_idx =  (targets_train < anom_class)
    anom_train_idx = (targets_train == anom_class)
    print(target_train_idx)
    print(anom_train_idx)
    


    print('bb')
    train_loader = torch.utils.data.DataLoader(torch.utils.data.dataset.Subset(data_train, np.where(target_train_idx==1)[0]), batch_size=batch_size)
    anom_loader = torch.utils.data.DataLoader(torch.utils.data.dataset.Subset(data_train, np.where(anom_train_idx==1)[0]), batch_size=2)
    print('aa')
    return train_loader, anom_loader