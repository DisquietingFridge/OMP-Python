## Patrick Foley, 21/Mar/21
## This file is intended to implement the "Orthogonal Matching Pursuit" algorithm
## For reference, the algorithm is described in these links:
## https://www.sciencedirect.com/topics/engineering/orthogonal-matching-pursuit
## https://youtu.be/ZG0PCzsA4XY

import numpy as np
from scipy import linalg
import multiprocessing as mp
from tqdm import tqdm, trange
import time

if __name__ == '__main__':
    mp.freeze_support()


def Batch_OMP (patches, D, error_margin = 2.3):

    '''
    Orthogonal Matching Pursuit algorithm.
    Similar to OMP function, but processes all patch columns
    Inputs:
    patches
    D-  a.k.a the "Dictionary", the 2D compressive sensing matrix
    error_margin- optional, L2_norm^2 error margin to terminate at
    Output:
    gammaMatrix- dictionary representation columns
    '''
    # https://www.cs.technion.ac.il/~ronrubin/Publications/KSVD-OMP-v2.pdf
    # refer to page 7 of this paper for algorithm steps. This has been implemente

    gammaMatrix = np.empty((D.shape[1],patches.shape[1])) #initialize output matrix
    G = D.T @ D
    print("Performing Batch OMP")
    time.sleep(1)
    for i in trange(len(patches.T)): #for each ith column x in patches
        x = patches.T[i]

        alphaZero = D.T.dot(x)
        epsilon = np.inner(x,x)
        I = np.array([],dtype=int)
        L = np.array([[1]])
        gamma = np.zeros(D.shape[1])
        alpha = alphaZero
        deltaZero = 0
        delta = deltaZero
        n = 1

        while (epsilon > error_margin):
            k = np.argmax(np.abs(alpha))
            if (n > 1): 

                #w = linalg.solve_triangular(L,G[I,k],lower = True, check_finite = False) # https://vene.ro/blog/optimizing-orthogonal-matching-pursuit-code-in-numpy-part-1.html
                w,_ = linalg.lapack.dtrtrs(L,G[I,k],lower = 1)

                #augment L matrix
                L = np.append(L,w[np.newaxis,:], axis = 0) #append row
                L = np.append(L,np.zeros((L.shape[0],1)),axis = 1) #append column of zeros
                L[-1,-1] = np.sqrt(1 - np.inner(w,w)) # add corner element

            I = np.append(I,k) # line 10

            #exploits simple way to solve triangular matrices
            #Lty = linalg.solve_triangular(L, alphaZero[I], lower=True,check_finite = False) # https://vene.ro/blog/optimizing-orthogonal-matching-pursuit-code-in-numpy-part-1.html
            Lty,_ = linalg.lapack.dtrtrs(L, alphaZero[I], lower= 1)
            #gamma[I] = linalg.solve_triangular(L, Lty, trans=1, lower=True, overwrite_b=True,check_finite = False)
            gamma[I],_ = linalg.lapack.dtrtrs(L, Lty, trans=1, lower= 1,overwrite_b=1)

            beta = G[:,I].dot(gamma[I]) # line 12

            alpha = alphaZero - beta # line 13

            deltaPrevious = delta
            delta = np.inner(gamma[I],beta[I]) # line 14

            epsilon = epsilon - delta + deltaPrevious # line 15

            #print("epsilon is ", epsilon)
            n = n + 1 # line 16

        gammaMatrix[:,i] = gamma # populate column
        #print("column ", i, "done")

    print("Batch OMP completed")
    time.sleep(1)
    return(gammaMatrix)


def OMP(Y, D, error_margin = 2.3):
    data_n = Y.shape[0]
    if len(Y.shape) == 1:
        data = np.array([Y])
    elif len(Y.shape) == 2:
        data = Y
    else:
        raise ValueError("Input must be a vector or a matrix.")

    # analyze dimensions
    n, K = D.shape
    if not n == data_n:
        raise ValueError("Dimension mismatch: %s != %s" % (n, data_n))

    alphas = []
    print("Performing OMP")
    for y in tqdm(data.T):
        residual = y

        index = np.zeros((y.shape[0]), dtype=int)

        alpha = np.zeros((D.shape[1]))
        for l in range(y.shape[0]):
            product = np.fabs(np.dot(D.T, residual))
            pos = np.argmax(product)
            if np.isclose(product[pos], 0):
                break
            index[l] = pos

            D_sub_pinv = np.linalg.pinv(D[:, index[:l + 1]])
            al = np.dot(D_sub_pinv, y)
            residual = y - np.dot(D[:, index[:l + 1]], al)

            if np.linalg.norm(residual) <= error_margin:
                break

        alpha[index[:l + 1]] = al

        alphas.append(alpha)

    alphas = np.stack(alphas, axis=0)
    res = np.transpose(alphas)
    print("OMP completed")
    return res


def old_OMP (y, A, error_margin = 2.3):

    '''
    Orthogonal Matching Pursuit algorithm.
    Returns x such that ||y - Ax|| is minimized using fewest columns of A.
    Inputs:
    y-  a.k.a the image we want to sparsely represent
    A-  a.k.a the "Dictionary", the 2D compressive sensing matrix
    error_margin- optional, L2_norm^2 error margin to terminate at. Defaults to 2.3, suitable for value range 0-255.
    Output:
    x- a.k.a "atom" vector corresponding to the "dictionary".
     '''

    #values that are used each loop, so it's good to pre-compute them once.

    Acol_norm_reciprocal = np.reciprocal(np.linalg.norm(A,2,axis=0)) # the reciprocal of each column's norm-2 value.
    At = A.T # used to update vector 'g'
    length = len(Acol_norm_reciprocal) # length of x vector

    #initializing variables

    #Aug = np.empty(A.shape) # progressively augmented with "active" columns from A
    support = np.empty(length,dtype=int) # Contains indices of the "active" columns of A
    r = y #residual vector ("error" component we are trying to minimize).

    #loop until there are no more columns to activate,
    #or until error margin is achieved
    for i in range(length):

         #Calculate the "updating vector" from the residual:
        g = At.dot(r)

         # find the column of A that has maximum correlation with r
         # This is the index j where (g_j / ||A_j||_2) is maximized
         # this identifies the next "active" column of A.
        support[i] = np.argmax(np.multiply(np.fabs(g),Acol_norm_reciprocal))

         #Aug[:,i] = A[:,support[i]] # Update the augment-matrix by inserting new active column

         #solve for y as best we can, combining only our active columns
         #nonsparse_x contains the weights of our active columns
        Ainv = np.linalg.pinv(A[:,support[:i + 1]])
        nonsparse_x = Ainv.dot(y)

         #Update the residual vector, the error in our solution so far:
        r = y - (A[:,support[:i + 1]]).dot(nonsparse_x)

         # exit if error amount is low enough
        if (np.inner(r,r) < error_margin):
            break

     # We found a suitable solution for y from most active columns of 'A'
     # those column weights are in nonsparse_x
     # now we insert those weights into the correct indices of x

    x = np.zeros(length) # weights of columns we didn't activate will be 0
    x[support[:(i+1)]] = nonsparse_x

    print("OMP complete")
    return(x)




