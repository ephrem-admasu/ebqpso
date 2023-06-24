import numpy as np
from utilities import transposon_operator

def EBQPSO(fun, D, nPop, lb, ub, maxit, lambda_):
   """
    Perform a quantum-behaved particle swarm optimization (QPSO)
   
    Parameters
    ==========
    func  :  function
             The function to be minimized
    D     :  int
             The dimension of a particle
    nPop  :  int
             Size of the swarm (Population size)
    lb    :  array
             The lower bounds of the design variable(s)
    ub    :  array
             The upper bounds of the design variable(s)
    maxit :  int
             The maximum number of iterations for the swarm to search (Default: 100)
    labda_: int
            A predefined parameter for controlling the frequency of the elitist breeding  In every λ iteration, the breeding operation will be performed.
   
    Returns
    =======
    gbest : array
            The swarm's best known position (optimal design)
    fbest : float
            The global best value at ``gbest``
    hist  : array
            The array of global best values at each iteration
    """
    
    w1 = 0.5
    w2 = 1.0

    c1 = 1.5
    c2 = 1.5

    # Initializing solution
    x = np.random.uniform(lb,ub,(nPop,D))

    # Evaluate initial population
    pbest = x.copy()



    f_x = np.array([fun(xi) for xi in x])

    f_pbest = f_x.copy()

    g = np.argmin(f_pbest)
    gbest = pbest[g,:]
    f_gbest = f_pbest[g]

    it = 1
    
    fitnesses = []
    

    while it <= maxit:

        alpha = (w2 - w1) * (maxit - it)/maxit + w1
        mbest = np.sum(pbest, axis=0)/nPop

        # Check if elitist criterion is met
        if it % lambda_ == 0:
            epool = np.zeros((nPop+1, D))
            for i in range(nPop):
                epool[i, :] = pbest[i, :]
            epool[nPop, :] = gbest

            epool_eb = transposon_operator(epool, jrate=.3)
            f_elit = np.array([fun(elit) for elit in epool_eb])

            for i in range(nPop):
                if f_elit[i] < f_pbest[i]:
                    f_pbest[i] = f_elit[i]
                    pbest[i, :] = epool_eb[i, :]
            f_gbest = np.min(f_pbest)
        
        for i in range(nPop):

            fi = np.random.random(D)

            p = (c1*fi*pbest[i, :] + c2*(1-fi)*gbest)/(c1 + c2)

            u = np.random.random(D)

            b = alpha*abs(x[i, :] - mbest)
            v = np.log(1/u)

            if np.random.random() < 0.5:
                x[i,:] = p + np.multiply(b, v)
            else:
                x[i,:] = p - np.multiply(b, v)

            # Keeping bounds
            x[i, :] = np.maximum(x[i,:], lb)
            x[i, :] = np.minimum(x[i,:], ub)

            f_x[i] = fun(x[i, :])


            if f_x[i] < f_pbest[i]:
                pbest[i, :] = x[i, :]
                f_pbest[i] = f_x[i]

            if f_pbest[i] < f_gbest:
                gbest = pbest[i, :]
                f_gbest = f_pbest[i]
                
        fitnesses.append(f_gbest)
        it = it + 1

    xmin = gbest
    fmin = f_gbest


    
    
    return xmin, fmin, fitnesses
