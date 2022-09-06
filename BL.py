# -*- coding: utf-8 -*-
"""
Created on Thu Aug 25 11:14:57 2022

@author: Wu
"""
import numpy as np
import pandas as pd
import scipy.optimize as spo

def market_cap_scalar(Pw,w):
    for x,y in enumerate(Pw):
        temp = np.multiply(np.sign(y),w)
        pos = np.where(temp>0,temp,0)
        neg = np.where(temp<0,temp,0)
        
        if not np.any(neg):
            Pw[x] = pos/pos.sum()
        elif not np.any(pos):
            Pw[x] = -neg/neg.sum()
        else:
            Pw[x] = pos/pos.sum() + -neg/neg.sum()

    return Pw

def objective(X, data):
    cov_matrix_d_sample = data.cov()

    port_volatility = np.sqrt(np.dot(X.T, np.dot(cov_matrix_d_sample, X)))
    
    port_return = data.mean().dot(X)

    sample_size = np.sqrt(252)
    
    sharpe = sample_size * port_return / port_volatility

    return -sharpe

def optimizer(weights, data):
    cons = ({'type': 'eq', 'fun': lambda weights: np.sum(weights) -1  })
    bnds = [(0, 1)] * len(weights)    # https://stackoverflow.com/questions/29150064/python-scipy-indexerror-the-length-of-bounds-is-not-compatible-with-that-of-x0
    best_mix = spo.minimize(objective, weights, args=(data), method='SLSQP', bounds = bnds, constraints = cons, options={'disp':False})

    return best_mix.x

# K = number of views
# N = number of assets

np.random.seed(5)

# Generate synthetic data
stocks = ['A','B','C']
ret = [0.02,0.03,0.07]
sigma = [0.15,0.2,0.3]

n = 1000
# Generated daily historical returns; assume normally distributed
data = pd.DataFrame(data={'A':np.random.normal(ret[0],sigma[0],n),
                         'B':np.random.normal(ret[1],sigma[1],n),
                         'C':np.random.normal(ret[2],sigma[2],n)})
cov = data.cov()

# Market cap weights
w = np.array([1/len(stocks)]*3)  # assume equal weights

# Scalar tau varies from 0 to 1; we fall back to 0.025 as in origianl BL paper
tau = 0.025

# Inverse of tau (scalar) * sigma
scaled_cov = np.linalg.inv(tau*cov) # NxN matrix of excess return; assume risk free is 0 for simplicity

# Idzorek uses a market cap weighting scheme, e.g. if the P entry is
# [0.5,-1,0.5] then a market cap weight may look like [0.7,-1,0.3].
P = np.array([[0,0,1],[1,-1,0]])  # KxN matrix that identifies the asset involved in the view
Pw = market_cap_scalar(P,w)
print('We assume asset C outperform all other assets by 5%, and asset A will outperform B by 2.5%')

# Q = expected return of the portfolios from the views described in P (Kx1)
# C to outperform all other assets by 5%; A to outpeform B by 2.5%
Q = np.array([0.05,0.025])
Q = Q.reshape(len(Q),1)

# Now calculate implied excess equilibrium return vector (Nx1)
# First, calculate risk aversion coefficient lambda given by (E(r)-rf)/sigma^2
# Assume 5% excess benchmark return & ~2% variance; sidenote BBG calculates it from aggregate corporate earnings + div yield
lam = 0.05/0.01667
Pi = lam * cov.dot(w)

# Calculate diagonal covariance matrix Omega
omega = tau * Pw.dot(cov).dot(Pw.T)
omega = np.diag(np.diag(omega))

# Finally calculate E(R), a Nx1 vector of assets under Black-Litterman model
ER1 = scaled_cov + Pw.T.dot(np.linalg.inv(omega)).dot(Pw)
ER1 = np.linalg.inv(ER1) 

ER2 = scaled_cov.dot(Pi) + np.hstack(Pw.T.dot(np.linalg.inv(omega)).dot(Q))  

ER = ER1.dot(ER2)


# Aggregate data
agg_ret = pd.DataFrame([ret, Pi, ER], columns=stocks,index=['Historical returns (synthetic)',\
                        'Implied equilibrium returns','BL new combined returns'])


cov_est = cov + ER1
w_est = np.linalg.inv(cov_est * lam).dot(Pi)
w_est = w_est/w_est.sum()
w_mvo = optimizer(w, data)



agg_weights = pd.DataFrame([w,w_est,w_mvo],columns=stocks,\
                   index=['Market cap weights','Black Litterman weights',\
                          'Traditional MVO'])

print(agg_ret, '\n')
print(agg_weights)