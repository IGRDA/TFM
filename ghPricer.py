import numpy as np
import scipy.stats as st
import scipy.special as scps
from scipy.optimize import minimize
from functools import partial
import utils


class GhPricer():
    """
    European option Generalyzed Hyperbolic pricer
    """
    def __init__(self):

        #Parameters
        self.delta = None
        self.mu = None
        self.lam = None
        self.alpha = None
        self.betta = None
        

        #Mean correcting martingale
        self.mcm = None

        #AIC
        self.aic = None

    def GH_density(self,x,delta,mu,lam,alpha,betta):
    
        C = (alpha**2-betta**2)**(0.5*lam) / \
            (np.sqrt(2*np.pi)*(alpha**(lam-0.5))*(delta**lam)*\
            scps.kv(lam,delta*np.sqrt(alpha**2-betta**2)))
        
        px = C*((delta**2+(x-mu)**2)**(0.5*lam-0.25))*\
            scps.kv(lam-0.5,alpha*(np.sqrt(delta**2+(x-mu)**2)))*np.exp(betta*(x-mu))
        
        return px

    def cf_GH(self,u,t,delta,mu,lam,alpha,betta):
    
        A = ( ( alpha**2-betta**2 )/ (alpha**2- (betta+1j*u)**2) )**(0.5*lam)
        
        B = ( scps.kv(lam,delta*np.sqrt(lam**2-( betta + 1j*u)**2)) / \
            scps.kv(lam,delta*np.sqrt(alpha**2-betta**2)) )
        
        return np.exp(1j*mu*u)*A*B

    def log_likely_GH(self,x, data):
        return (-1) * np.sum( np.log( self.GH_density(data, x[0], x[1], x[2], x[3], x[4]) ))


    def fit(self,data,N=100):
        cons = [{'type':'ineq', 'fun': lambda x: x[3]-x[4]+1/2},
                {'type':'ineq', 'fun': lambda x: x[4]+x[3]-1/2}]

        #Best parameters with random inicialization, triying to avoid local minima
        x0=[]
        for i in range(N):
            x0.append([np.random.exponential(0.1),
                       np.random.exponential(0.1),
                       np.random.exponential(0.1),
                       0,
                       0])

        a=[]
        for x in x0:
            a.append(minimize(self.log_likely_GH, x0=x,
                     method='Nelder-Mead', args=(data), constraints=cons))

        a_best = min(a, key=lambda x:x['fun'])["x"]
        

        self.delta, self.mu, self.lam, self.alpha, self.betta = a_best

        self.mcm = np.log(self.cf_GH(
                                    u=-1j,
                                    t=1,
                                    delta=self.delta,
                                    mu=self.mu,
                                    lam=self.lam,
                                    alpha=self.alpha,
                                    betta=self.betta))

        self.aic = 2*5+2*self.log_likely_GH(x=a_best,data=data)

    

