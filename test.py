from PySSM import RBFKernel
from PySSM import IVM
from PySSM import Greedy
import numpy as np

X = [
    [0, 0],
    [1, 1],
    [0.5, 1.0],
    [1.0, 0.5],
    [0, 0.5],
    [0.5, 1],
    [0.0, 1.0],
    [1.0, 0.]
]    

K = 3
kernel = RBFKernel(sigma=1,scale=1)
ivm = IVM(kernel = kernel, sigma = 1.0)
greedy = Greedy(K, ivm)

greedy.fit(X)

# Alternativley, you can use the streaming interface. 
#for x in X:
#    opt.next(x)

fval = greedy.get_fval()
solution = np.array(greedy.get_solution())

print("Found a solution with fval = {}".format(fval))
print(solution)
