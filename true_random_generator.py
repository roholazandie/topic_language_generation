import quantumrandom as qr
import numpy as np
import torch
from scipy.stats import uniform, expon

def real_uniform(N):
    rands = []
    i = 0
    while i < N:
        rands.append(qr.randint(0, 1))
        i+=1
    return np.array(rands)


def exponential_inverse_trans(n=1, mean=1):
    #U = uniform.rvs(size=n)
    U = real_uniform(n)
    X = -mean*np.log(1-U)
    return X



def multinomial_log(N, p):
    logp = np.log(p)
    #log_rand = -np.random.exponential(size=N)
    log_rand = -exponential_inverse_trans(N)
    logp_cuml = np.logaddexp.accumulate(np.hstack([[-np.inf], logp]))
    logp_cuml -= logp_cuml[-1]
    return [int(np.argwhere(logp_cuml-r>0)[0][0]) for r in log_rand]


def real_multinomial(N, p):
    #rand = [r for i, r in enumerate() if i < N]#np.random.uniform(size=N)
    rands = real_uniform(N)
    p_cuml = np.cumsum(p)
    p_cuml /= p_cuml[-1]
    return np.array([int(np.argwhere(p_cuml-r>0)[0][0]) for r in rands])


# #x = real_multinomial(1, [0.3, 0.3, 0.3])
# while True:
#     x = multinomial_log(1, [0.3, 0.3, 0.3])
#     print(x)
