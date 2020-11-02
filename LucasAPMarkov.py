# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.2'
#       jupytext_version: 1.2.4
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# \begin{equation*}
# P_t = E_{t}\left[ \frac{u(d_{t+1})}{u(d_t)} (P_{t+1} + d_{t+1}) \right]
# \end{equation*}

# %% Preamble
import numpy as np
from HARK.utilities import CRRAutilityP

# %% Definitions

class MarkovDstn:
    
    def __init__(self, Values, TransMat):
        
        self.Values = Values
        self.TransMat = TransMat

class LucasEconomy:
    
    def __init__(self, CRRA, DiscFac, DivDist):
        
        self.CRRA = CRRA
        self.DiscFac = DiscFac
        self.DivDist = DivDist
        self.uP = lambda c: CRRAutilityP(c, self.CRRA)
        
    def priceOnePeriod(self, price_next):
        
        # Create 'tiled arrays' rows are state today, columns are state
        # tomorrow.
        n_states = len(self.DivDist.Values)
        
        # Dividends now
        d_now = np.tile(np.reshape(self.DivDist.Values, (n_states,1)),
                        (1, n_states))
        # Dividends tomorrow
        d_next = np.tile(np.reshape(self.DivDist.Values, (1, n_states)),
                        (n_states,1))
        # Prices tomorrow
        p_next = np.tile(np.reshape(price_next, (1, n_states)),
                        (n_states,1))
        
        # Compute the RHS of the pricing equation, pre-expectation
        Mpd = self.uP(d_next)/self.uP(d_now) * (p_next + d_next)

        # Take expectation and discount
        P_now = self.DiscFac * np.sum(Mpd*self.DivDist.TransMat, axis = 1, keepdims=True)
        
        return(P_now)
        
    def solve(self, P_0= None, tol = 1e-5, disp = False):
        
        # Define an initial price vector if not given
        if P_0 is None:
            P_0 = np.ones((len(self.DivDist.Values),1))
        
        # Initialize the norm
        norm = tol + 1
        
        it = 0
        while norm > tol:
            
            # Apply the pricing equation
            P_next = self.priceOnePeriod(P_0)
            # Measure the change between price vectors
            norm = np.linalg.norm(P_0 - P_next)
            # Update price vector
            P_0 = P_next
            it = it + 1
            # Print iteration information
            if disp:
                print('Iter:' + str(it) + '   P = '+ np.array2string(np.transpose(P_0)))
        
        self.EqPrice = P_0
    
        
# %% Example

# Create a Markov process for dividends. A basic high-mid-low with 
# persistence
divs = np.array([0.5, 1, 1.5])
Trans = np.array([[0.5, 0.5, 0.0],
                  [0.2, 0.6, 0.2],
                  [0.0, 0.5, 0.5]])

DivDist = MarkovDstn(divs, Trans)


economy = LucasEconomy(CRRA = 2, DiscFac = 0.9, DivDist = DivDist)

# Some initial guess for the price function
p0 = np.array([[1],[2],[3]])
# 
print(economy.priceOnePeriod(p0))
economy.solve(disp = True)