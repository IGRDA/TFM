import numpy as np
import scipy.stats as st
import scipy.special as scps
from scipy.optimize import minimize
from functools import partial
import utils
import random
from math import factorial


class MertonPricer():
    """
    European option Merton pricer
    """
    def __init__(self):

        #Parameters
        self.mu = None
        self.sig = None
        self.lam = None
        self.muJ = None
        self.sigJ = None
        

        #Mean correcting martingale
        self.mcm = None

        #AIC
        self.aic = None
    
    def Merton_density(self,x, T, mu, sig, lam, muJ, sigJ,nterms=150):
        serie = 0
        for k in range(nterms):
            serie += (lam*T)**k * np.exp(-(x-mu*T-k*muJ)**2/( 2*(T*sig**2+k*sigJ**2) ) ) \
                    / (factorial(k) * np.sqrt(2*np.pi * (sig**2*T+k*sigJ**2) ) )  
        return np.exp(-lam*T) * serie

    
    def cf_mert(self,u, t, mu, sig, lam, muJ, sigJ):

        return np.exp( t * ( 1j * u * mu - 0.5 * u**2 * sig**2 \
                    + lam*( np.exp(1j*u*muJ - 0.5 * u**2 * sigJ**2) -1 ) ) )
    
    def log_likely_Merton(self,x, data, T):
        return (-1) * np.sum( np.log(self.Merton_density(data, T, x[0], x[1], x[2], x[3], x[4]) ))

    
    def fit(self,data,T):

        cons = [{'type':'ineq', 'fun': lambda x: x[1]},
                {'type':'ineq', 'fun': lambda x: x[4]}]

        a =minimize(self.log_likely_Merton, x0=[data.mean(),data.std(),2,data.mean(),data.std()], 
                    method='Nelder-Mead', args=(data,T) , constraints=cons)

        self.mu, self.sig, self.lam, self.muJ, self.sigJ = a["x"]


        self.mcm = 0.5*self.sig**2 + self.lam * (np.exp(self.muJ + (self.sigJ**2)/2) -1)

        
        self.mcm = np.log(self.cf_mert( u = -1j,
                                        t = 1,
                                        mu = self.mu,
                                        sig = self.sig,
                                        lam = self.lam,
                                        muJ = self.muJ,
                                        sigJ = self.sigJ))
        self.aic =  2*5+ 2*a["fun"]
    
    def mcPricer(self,K,r,S0,payoff,N,T):

        W = st.norm.rvs(0, 1, N)                  #Gaussian part  
        P = st.poisson.rvs(self.lam*T, size=N)    #Poisson number of arrivals
        Jumps = np.asarray([st.norm.rvs(self.muJ, self.sigJ, ind).sum() for ind in P ]) # Jumps
        S_T = S0 * np.exp( (r - self.mcm )*T + np.sqrt(T)*self.sig*W + Jumps )     # Martingale exponential
        S_T= S_T.reshape((N,1))
        
        option = np.mean( np.exp(-r*T) * utils.payoff(S=S_T,K=K,payoff=payoff), axis=0 )[0]  # Mean
        option_error =  st.sem( np.exp(-r*T) * utils.payoff(S=S_T,K=K,payoff=payoff), axis=0 )[0] # Standar error of mean

        return option.real, option_error


                                               