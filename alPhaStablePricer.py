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
        if alpha==1:
            psi=np.tan(np.pi*np.alpha/2)
        if alpha!=1:
            psi=-2/np.pi * np.log(np.abs(t))
        return np.exp(1j*t*mu-np.abs(c*t)**alpha * ( (1-1j*betta*np.sign(t)*psi)))


    def fit(self,data):
        
        
        self.alpha, self.betta, self.mu, self.c =st.levy_stable.fit(data=data,
                                                                    optimizer=fmin_bfgs)

        self.mcm = np.log(self.cf_stable(t=1,
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


        
        X=np.sum(st.levy_stable.rvs(
                        alpha=self.alpha, 
                        beta=self.betta,
                        loc=self.mu,
                        scale=self.c,
                        size=(T,N)),axis=0)


        X_clean =X[abs(X-X.mean())<10*np.std(X)]



        S_T = S0 * np.exp( (r-self.mcm)*T + X_clean )

        
        option_payoff = utils.payoff(S=S_T,K=K,payoff=payoff)
        option = np.exp(-r*T) * np.mean( option_payoff ) # Mean
        option_error = np.exp(-r*T) * st.sem( option_payoff ) # Standar error of mean

        return option.real, option_error

    
