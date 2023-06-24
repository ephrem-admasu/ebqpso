## Quantum-behaved Particle Swarm Optimization with Eltist Breeding
This is a Python implementation of quantum-behaved particle swarm optimization with elitist breeding. The project is at its earliest stage but can be used for optimization tasks.

### Description
QPSO introduces a quantum-like behavior into the PSO algorithm by allowing the particles to "tunnel" through potential barriers. 
This means that the particles can escape from local optima more easily, and they can also explore the search space more efficiently.  In PSO and QPSO,
the personla best of particle and the global best of the swam (collectively known as elitists) are simply stored in memory and used for solution comparison 
or go through basic computational steps to help update the exploration process. However, empirical evidence suggests that taking advantage of these elit partciles
by breeding them can result in significant improvement in the performance and global searching capability of the algorithm.

EB-QPSO is a more powerful and efficient optimization algorithm than QPSO. It is able to escape from local optima more easily, and it converges to the global optimum faster. 
As a result, EB-QPSO is a better choice for solving difficult optimization problems.

### Using the library
For usage simply clone using:
```
git clone https://github.com/ephrem-admasu/ebqpso.git
```

### Example
```python
from ebqpso import EBQPSO
import benchmark_functions as bf # install benchmark_functions using pip install benchmark_functions
import numpy as np

func = bf.Schwefel(n_dimensions=4)

# Initialize parameters
D = func.n_dimensions() # get num of dimensions from fun
nPop = 50
maxiter = 1000
lambda_ = 5
lb = func.suggested_bounds()[0]
ub = func.suggested_bounds()[1]

gbest, fbest, history = EBQPSO(func, D, nPop, lb, ub, maxiter, lambda_)

print("Optimum paramters: ", gbest)
print("Minimum value: ", fbest)

import matplotlib.pyplot as plt
plt.plot(np.arange(len(history))+1, history)
plt.ylabel("Fitness")
plt.xlabel("Generation")
plt.show()
```
Optimum paramters:  [420.65544333 420.65544333 420.65544333 420.65544333]  
Minimum value:  0.04958878158049629

![FitnessPlot](https://github.com/ephrem-admasu/ebqpso/blob/main/fitness_plot.png)

