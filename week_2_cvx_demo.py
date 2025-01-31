
import cvxpy as cvx

def hard_svm(X,y):
    #
    # ~~~ Define the optimization problem
    m,n = X.shape
    w = cvx.Variable((n,1))
    b = cvx.Variable()
    objective = cvx.Minimize(cvx.norm(w,2))
    constraints = [ y[i]*(X[i]@w - b) >= 1 for i in range(m) ]
    prob = cvx.Problem(objective, constraints)
    #
    # ~~~ Solve the optimization problem
    prob.solve(solver=cvx.ECOS)
    #
    # ~~~ Return the solution
    return w.value, b.value
