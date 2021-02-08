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
