from numpy.random import standard_normal
from numpy import exp, matmul
from numpy.linalg import cholesky, solve
from mlmc import mlmc
from mlmc_fn import mlmc_fn
from mlmc_test import mlmc_test
from mlmc_plot import mlmc_plot
from matplotlib import pyplot as plt
import numpy as np

# Generate suitable RVs
def gen_rvs(n):
    normals = np.random.normal(size = n, scale = 0.1)
    covmat = np.zeros((n,n))
    for x in range(n):
        for y in range(n):
            covmat[x][y] = np.exp(-np.abs((1/(n-1))*(x-y)))
    return np.matmul(np.linalg.cholesky(covmat), normals)

# Integral estimator of kdp where dp is unknown and estimated by centred difference
# Can cope with there being some integer multiple amount of datapoints for k relative to p
def est_int(p, r):
    cf = int(len(r)/len(p))
    total = 0
    for point in range(len(p)-2):
        total += r[cf*point+1]*(p[point+2]- p[point])*(0.5*((len(p)-1)))
    return total/(len(p)-2)
    
# Input function for mlmc
def testup(l, n_samp):
    n = 10*(2**l)   
    
    estimates = []
    
    for samp in range(n_samp):

        samples = gen_rvs(n)
        k = np.exp(samples)
        dk = [(n-1)*(samples[x+1]-samples[x])*k[x] for x in range(n-1)]
        
        this_est = []
        
        # Fine samples first then coarse
        for cf in range(1,3):          
            mesh = int(n/cf)
            
            # Initialize p subject to boundary conditions
            p = np.zeros(mesh)
            p[-1] = 1
    
            # Implement finite difference method
            syst = []
            s1 = [0 for x in range(mesh-2)]
            s1[0] = -2*k[cf]*(mesh-1)
            s1[1] = 0.5*(dk[cf]) + k[cf]*(mesh-1)
            syst.append(s1)
            for x in range(mesh-4):
                sx = [0 for x in range(mesh-2)]
                sx[x] = -0.5*(dk[cf*(2+x)]) + k[cf*(2+x)]*(mesh-1)
                sx[x+1] = -2*k[cf*(2+x)]*(mesh-1)
                sx[x+2] = 0.5*(dk[cf*(2+x)]) + k[cf*(2+x)]*(mesh-1)
                syst.append(sx)
            sn = [0 for x in range(mesh-2)]
            sn[mesh-3] = -2*k[n-2]*(mesh-1)
            sn[mesh-4] = -0.5*(dk[n-2]) + k[n-2]*(mesh-1)
            syst.append(sn)     

            rhs = [0 for x in range(mesh-2)]
            rhs[-1] = -0.5*(dk[n-2]) - k[n-2]*(mesh-1)

            sol = np.linalg.solve(syst, rhs)

            for x in range(len(sol)):
                p[x + 1] = sol[x]    
            this_est.append(est_int(p, k))
        
        estimates.append(this_est)

    sum1, sum2, sum3, sum4, sum5, sum6, cost = 0, 0, 0, 0, 0, 0, 0
    if l > 0:
        for estimate in estimates:
            sum1 += estimate[0] - estimate[1]
            sum2 += (estimate[0] - estimate[1])**2
            sum3 += (estimate[0] - estimate[1])**3
            sum4 += (estimate[0] - estimate[1])**4
            sum5 += (estimate[0])
            sum6 += (estimate[0])**2
            cost += 2**l
    else:
        for estimate in estimates:
            sum1 += estimate[0] 
            sum2 += estimate[0]**2
            sum3 += estimate[0]**3
            sum4 += estimate[0]**4
            sum5 += (estimate[0])
            sum6 += (estimate[0])**2
            cost += 2**l
        
    # Slide 13 of Lecture 12 indicates we can choose C_l = 2**l
    return [np.array([sum1, sum2, sum3, sum4, sum5, sum6]), cost]

N0 = 10 # initial samples on coarse levels
Lmin = 2  # minimum refinement level
Lmax = 10 # maximum refinement level
M = 2 # refinement factor
N = 100 # samples for convergence tests
L = 5 # levels for convergence tests
Eps = [0.0005, 0.001, 0.002, 0.005]

name = "Question 1"
filename = "Question_1.txt"
logfile = open(filename, "w")
print('\n ---- ' + name + ' ---- \n')
mlmc_test(testup, N, L, N0, Eps, Lmin, Lmax, logfile)
del logfile
mlmc_plot(filename, nvert=3)
plt.savefig(filename.replace('.txt', '.eps'))
