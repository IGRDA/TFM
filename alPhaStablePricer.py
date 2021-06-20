import numpy as np
import scipy.stats as st
from scipy.optimize import fmin_bfgs
from functools import partial
import utils

class AlPhaStablePricer():
    """
    European Alpha Stable
    Descripcioon de la clase
    """
    def __init__(self):

        #st.levy_stable.pdf_default_method = 'quadrature' #More acurate but slow

        #Parameters
        self.alpha = None
        self.betta = None
        self.mu = None
        self.c = None

        #Mean correcting martingale
        self.mcm = None

        #AIC
        self.aic = None

    def cf_stable(self,t,alpha,betta,c,mu):
        if alpha!=1:
            psi=np.tan(np.pi*alpha/2)
        if alpha==1:
            psi=-2/np.pi * np.log(np.abs(t))
        return np.exp(1j*t*mu-np.abs(c*t)**alpha * ( (1-1j*betta*np.sign(t)*psi)))


    def fit(self,data):
        
        
        self.alpha, self.betta, self.mu, self.c =st.levy_stable.fit(data=data,
                                                                    optimizer=fmin_bfgs)

        self.mcm = np.log(self.cf_stable(t=-1j,
                                        alpha=self.alpha, 
                                        betta=self.betta,
                                        mu=self.mu,
                                        c=self.c))

        self.aic = 2*4 - 2*np.sum( st.levy_stable.logpdf(
                                                 data,
                                                 alpha=self.alpha, 
                                                 beta=self.betta,
                                                 loc=self.mu,
                                                 scale=self.c) )

            
    def mcPricer(self,K,r,T,S0,payoff,N):

        X=st.levy_stable.rvs(
                        alpha=self.alpha, 
                        beta=self.betta,
                        loc=self.mu,
                        scale=self.c,
                        size=N)*T
        
        

        X=np.sum(st.levy_stable.rvs(
                        alpha=self.alpha, 
                        beta=self.betta,
                        loc=self.mu,
                        scale=self.c,
                        size=(N,T)),axis=1)

        """
        W = st.norm.rvs(0, 1, N)                  #Gaussian part  
        P = st.poisson.rvs(self.lam*T, size=N)    #Poisson number of arrivals
        Jumps = np.asarray([st.norm.rvs(self.muJ, self.sigJ, ind).sum() for ind in P ]) # Jumps
        S_T = S0 * np.exp( (r - self.mcm )*T + np.sqrt(T)*self.sig*W + Jumps )     # Martingale exponential
        S_T= S_T.reshape((N,1))
        """

        X = X[abs(X-np.mean(X))<np.std(X)]

        S_T = S0 * np.exp((r-self.mcm)*T+  X )     # Martingale exponential Merton
        S_T= S_T.reshape((len(S_T),1))
        
        option = np.mean( np.exp(-(r)*T) * utils.payoff(S=S_T,K=K,payoff=payoff), axis=0 )[0]

        option_error =  st.sem( np.exp(-(r)*T) * utils.payoff(S=S_T,K=K,payoff=payoff), axis=0 )[0]
        return option.real, option_error
    
