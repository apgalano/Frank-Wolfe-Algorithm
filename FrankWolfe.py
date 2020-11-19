from scipy.optimize import linprog
import numdifftools as nd
import numpy as np


class frank_wolfe():
    
    def __init__(self,min_fun,A,b,bounds,x0,iterations=100):
        self.min_fun = min_fun
        self.A = A
        self.b = b
        self.bounds = bounds
        self.x0 = x0
        self.iterations = iterations
        self.x_min = []
        self.f_min = 0
        self.x_t = []
        self.s_t = []
        self.f_t = []
        self.violation = 0
        
    def __repr__(self):
        out = 'f_min: '+str(self.f_min)+'\n' + \
                'x_min: '+str(self.x_min)+'\n' + \
                'violation: '+str(self.violation)
        return out

    def optimize(self):
        x = self.x0
        for i in range(0,self.iterations):
            gamma = 2 / (i+2)
            grad_def = nd.Gradient(self.min_fun)
            grad = grad_def(x)
            update = linprog(grad, A_ub=self.A, b_ub=self.b, A_eq=None, b_eq=None, bounds=self.bounds, 
                 method='interior-point', callback=None, x0=None,
                 options={'sym_pos':False,'lstsq':True})

            s = update.x
            self.s_t.append(s)
            x = x + gamma*(s-x)
            self.f_t.append(self.min_fun(x))
            self.x_t.append(x)
            
        constraints = np.dot(self.A,x) - self.b
        self.violation = np.sum([i for i in constraints if i > 0])
        self.x_min = x
        self.f_min = self.f_t[-1]
        return self
        
            
        

