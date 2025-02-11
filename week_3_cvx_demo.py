
import numpy as np
import scipy as sp
import cvxpy as cvx
from tqdm import trange
from quality_of_life.my_numpy_utils import generate_random_1d_data

#
# ~~~ Data from the main demo
f = lambda x: np.abs(x) # ~~~ the so called "ground truth" by which x causes y
np.random.seed(680)     # ~~~ for reproducibility
x_train,y_train,_,_ = generate_random_1d_data( f, n_train=100, noise=.15 )
np.random.seed(680)
y_train += .1*np.random.random( size=y_train.shape )
D = 20

#
# ~~~ Build the basic data
A = np.column_stack([ x_train**j for j in range(D+1) ])
b = y_train.reshape((-1,1))
lamb = 0.1

#
# ~~~ Solve eq'n (6.7) using scipy
for _ in trange(1000):
    big_A = np.vstack((
            np.hstack([ A, -A ]),
            lamb*np.ones( shape=(1,2*(D+1)) )
        ))
    big_b = np.stack([ *b, [0.] ])
    big_w, res = sp.optimize.nnls( big_A, big_b.flatten() )
    w = big_w[:21] - big_w[21:]


#
# ~~~ Solve eq'n (6.7) using cvxpy
w_plus = cvx.Variable( (D+1,1) )
w_minus = cvx.Variable( (D+1,1) )
big_A = cvx.vstack((
        cvx.hstack([ A, -A ]),
        lamb*np.ones( shape=(1,2*(D+1)) )
    ))
big_b = np.stack([ *b, [0.] ])
big_w = cvx.vstack(( w_plus, w_minus ))
objective = cvx.Minimize( cvx.norm(big_A@big_w - big_b) )
constraints = [ w_plus>=0, w_minus>=0 ]
problem = cvx.Problem( objective )
_ = problem.solve()

#
# ~~~ Solve eq'n (6.6) directly
for _ in trange(1000):
    w = cvx.Variable( (D+1,1) )
    objective = cvx.Minimize( cvx.norm(A@w-b)**2 + lamb**2*cvx.norm1(w)**2 )
    problem = cvx.Problem(objective)
    res_cvx = problem.solve()
