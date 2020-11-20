from scipy.optimize import minimize,LinearConstraint,Bounds
import numpy as np
import numdifftools as nd
import matplotlib as mpl
import matplotlib.pyplot as plt
from FrankWolfe import frank_wolfe


def min_fun(x):
    """The function to minimize"""
    return 15*(x[0]-1)**2 + x[1]**2 + 0.5*x[2]**2 + 2*x[0]*x[1] - x[0]*x[2] - 2*x[0] + 6*x[1]


# number of constraints and variables
constr_num = 4
var_num = 3

# Ax <= ub
A = np.random.randint(-10,10, (constr_num,var_num))
ub = 10*np.ones(constr_num)

# 0 <= x_i <= 10
bounds=[(0,10) for i in range(0,var_num)]
# Initial point x0
x0 = np.random.randint(0,10,(var_num))


# Use Frank-Wolfe
iterations = 200
fw = frank_wolfe(min_fun,A,ub,bounds,x0,iterations)
results = fw.optimize()


# Use a convex solver
bounds = Bounds(np.zeros(var_num), 10*np.ones(var_num))
lb = -np.inf*np.ones(constr_num)
linear_constraint = LinearConstraint(A, lb, ub)

res = minimize(min_fun, x0, method='trust-constr', jac=nd.Gradient(min_fun),
                constraints=linear_constraint, bounds=bounds,
                options={'verbose': 0,'gtol': 1e-8, 'disp': True})


# Optimality gap
f_star = [res.fun for i in range(0,len(results.f_t))]

fig = plt.figure()
mpl.rc('xtick', labelsize=16) 
mpl.rc('ytick', labelsize=16) 
plt.yscale('log')
plt.grid()
plt.plot(range(0,iterations), np.abs(np.array(f_star) - np.array(results.f_t)),'bo-',lw=2)
plt.ylabel('Optimality Gap',fontsize=20)
plt.xlabel('Iteration',fontsize=20)
plt.tight_layout()
plt.show()
#fig.savefig('opt_gap.jpeg', format='jpg')
    

