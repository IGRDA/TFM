import numpy as np
import scipy.stats as st
import scipy.special as scps
from scipy.optimize import minimize
from functools import partial
import utils
import random
from itsample import sample


class GhPricer():
    """
    European Generalyzed Hyperbolic
    Descripcioon de la clase
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
        
        B = scps.kv(lam,delta*np.sqrt(lam**2-( betta + 1j*u)**2)) / \
            scps.kv(lam,delta*np.sqrt(alpha**2-betta**2))
        
        return np.exp(1j*mu*u*t)*(A**t)*(B**t)

    def log_likely_GH(self,x, data):
        return (-1) * np.sum( np.log( self.GH_density(data, x[0], x[1], x[2], x[3], x[4]) ))


    def fit(self,data,N=1000):
        cons = [{'type':'ineq', 'fun': lambda x: x[4]-x[3]-1/2},
                {'type':'ineq', 'fun': lambda x: x[4]+x[3]+1/2}]

        #Montecarlo median of maximum likelihood stimation
        x0=[]
        for i in range(N):
            x0.append([random.uniform(0,1),
                       random.uniform(-0.1,0.1),
                       random.uniform(-1,1),
                       random.uniform(-1,1),
                       random.uniform(-1,1)])

        a=[]
        for x in x0:
            a.append(minimize(self.log_likely_GH, x0=x,
                     method='Nelder-Mead', args=(data), constraints=cons))

        a_best = np.median(np.array([i["x"] for i in a if i["fun"]!=-0.0]),axis=0)

        self.delta, self.mu, self.lam, self.alpha, self.betta = a_best

        self.mcm = np.log(self.cf_GH(-1j,
                                    t=1,
                                    delta=self.delta,
                                    mu=self.mu,
                                    lam=self.lam,
                                    alpha=self.alpha,
                                    betta=self.betta))

        self.aic = 2*5+2*self.log_likely_GH(x=a_best,data=data)

    def mcPricer(self,K,r,T,S0,payoff,N=10000):
        pdf_gh = partial(self.GH_density,delta=self.delta,
                                     mu=self.mu,
                                     lam=self.lam,
                                     alpha=self.alpha,
                                     betta=self.betta)
        X = sample(pdf_gh,N)

        S_T = S0 * np.exp( (r-self.mcm.real)*T + X )

        
        option_payoff = utils.payoff(S=S_T,K=K,payoff=payoff)
        option = np.exp(-r*T) * np.mean( option_payoff ) # Mean
        option_error = np.exp(-r*T) * st.sem( option_payoff ) # Standar error of mean

        return option.real, option_error

    

