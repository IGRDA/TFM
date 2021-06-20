import numpy as np
from scipy.integrate import quad

def payoff(payoff, S,K):
    """
    Payoff function for European options
    """
    if payoff == "call":
        return np.maximum( S - K, 0 )
    elif payoff == "put":    
        return np.maximum( K - S, 0 ) 
    
def fourierPricer(K,S0,payoff,r,T,cf,udep,mcm,xmax=np.inf,limit=1000):

        """ 
        Lewis integral for European options
        K = strike
        S0 = initial value
        payoff = call/put
        cf = characteristic function
        """
        k = np.log(S0/K)+r*T

        if udep==True:
            integrand = lambda u: np.real( np.exp(u*k*1j) * cf(u - 0.5j) ) * 1/(u**2 + 0.25)
        if udep==False:
            integrand = lambda u: np.real( np.exp(u*k*1j) * cf ) * 1/(u**2 + 0.25)

        call = S0 - np.sqrt(S0 * K) * np.exp(-(r)*T)/np.pi * \
               quad(integrand, 0, xmax, limit=limit)[0]

        if payoff=="call":
            return call

        if payoff=="put":
            return call - S0 + K*np.exp(-(r)*T) # Call-put parity