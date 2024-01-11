### ~~~
## ~~~ Dependencies
### ~~~

import numpy as np
from quality_of_life.my_numpy_utils         import generate_random_1d_data, my_min, my_max

#
# ~~~ A helper function that prepares data identical to Fouract's in https://github.com/foucart/Mathematical_Pictures_at_a_Data_Science_Exhibition/blob/master/Python/Chapter01.ipynb
def Foucarts_training_data():
    # ~~~ equivalent to:
        # np.random.seed(12)
        # m = 15
        # x_train = np.random.uniform(-1,1,m)
        # x_train.sort()
        # x_train[0] = -1
        # x_train[-1] = 1
        # y_train = abs(x_train) + np.random.normal(0,1,m)
    x_train = np.array([-1.        , -0.97085008, -0.93315714, -0.72558136, -0.69167432,
                       -0.47336997, -0.43234329,  0.06747879,  0.21216637,  0.48009939,
                        0.70547108,  0.80142971,  0.83749402,  0.88845027,  1.        ])
    y_train = np.array([ 3.73781428,  2.08759803,  2.50769528,  0.63971456,  1.16841094,
                        -0.13801677,  0.08287235, -0.63793798, -0.12801989,  2.5073981 ,
                         0.12439097,  1.67456455,  1.7480593 ,  1.93609588, -0.18963857])
    return x_train, y_train



### ~~~
## ~~~ Possible answers to the first exercise
### ~~~

#
# ~~~ **When the data is not too noisy,** a complicated/aggressive/expressive model (i.e., higher degree of polynomial regression) might be appropriate
f_a = lambda x: np.abs(x)                   # ~~~ the so called "ground truth" by which x causes y
x_train_a, _ = Foucarts_training_data()     # ~~~ take only the x data
np.random.seed(123)                         # ~~~ for improved reproducibility
y_train_a = f_a(x_train_a) + (1/20)*np.random.random(size=x_train_a.size)   # ~~~ a less noisy version of Foucart's data
explanation_a_4 = "A Degree 4 Polynomial Can't Approximate the Point of Discontinuity Very Well"
explanation_a_10 = "A Degree 10 Polynomial May Fit Well when the Data is not Too Noisy"

#
# ~~~ **When the ground truth is complicated,** even if the data is fairly noisy, a more complicated/aggressive model might be appropriate
f_b = lambda x: np.exp(x)*np.cos(3*x*np.exp(x))     # ~~~ a more complicated choice of ground truth
np.random.seed(680)                                 # ~~~ for improved reproducibility
x_train_b, y_train_b, _, _ = generate_random_1d_data( ground_truth=f_b, n_train=50, noise=0.3 )
explanation_b_4 = "A Degree 4 Polynomial Can't Capture the Behavior of a Complex Ground Truth"
explanation_b_10 = "A Degree 10 Polynomial May Fit Well when the Ground Truth is Complicated"



### ~~~
## ~~~ Possible answers to the second exercise
### ~~~

#
# ~~~ Perform ERM for any vector space of functions H that the user supplies (recall: degree d polynomial regression is ERM with H={1,x,...,x^d})
def empirical_risk_minimization_with_linear_H( x_train, y_train, list_of_functions_that_span_H ):
    #
    # ~~~ If list_of_functions_that_span_H = \{ \phi_1, \ldots, \phi_d \}, then the model matrix is the matrix with j-th column \phi_j(x_train), i.e., (i,j)-th entry \phi_j(x^{(i)})
    model_matrix = np.vstack([ phi(x_train) for phi in list_of_functions_that_span_H ]).T   # ~~~ for list_of_functions_that_span_H==[1,x] this coincides with ordinary least squares
    #
    #~~~ An optional sanity check
    m,n = model_matrix.shape
    assert m==len(x_train)==len(y_train)
    assert n==len(list_of_functions_that_span_H)
    #
    #~~~ Find some coefficients which minimize MSE
    coeffs = np.linalg.lstsq( model_matrix, y_train, rcond=None )[0]
    #
    #~~~ A vectorized implementation of the MSE-minimizing function \widehat{\phi}(x) = \sum_j c_j\phi_j(x)
    empirical_risk_minimizer = lambda x, c=coeffs: np.vstack([ phi(x) for phi in list_of_functions_that_span_H ]).T @ c
    return empirical_risk_minimizer, coeffs

#
# ~~~ Wrap polynomial regression for convenience
def my_univar_poly_fit( x_train, y_train, degree ):
    H = [ (lambda x,j=j: x**j) for j in range(degree+1) ]   # ~~~ define a list of functions which span the hypothesis class, in this case [ 1, x , x^2, ..., x^degree ]
    H = H[::-1]     # ~~~ reverse the order in which they're listed (merely a convention adopted to be consistent with the convention used by np.polyfit)
    return empirical_risk_minimization_with_linear_H( x_train, y_train, H )   # ~~~ that's it!

#
# ~~~ Define a routint that creates the list "H = [(j-th hat function) for j in range(n)]" where n is the length of a sequence of knots
def list_all_the_hat_functions(knots):
    knots = np.sort(knots)
    n = len(knots)
    hat_functions = []
    for j in range(n):
        midpoint = knots[j]
        if j==0:
            next_point = knots[j+1]
            hat_functions.append( 
                    lambda x, b=midpoint, c=next_point: my_max( 0, 1-(x-b)/(c-b) )
                )   # ~~~ the positive part of the the line with value 1 at b going down to value 0 at c
        if j==(n-1):
            prior_point = knots[j-1]
            hat_functions.append(
                    lambda x, a=prior_point, b=midpoint: my_max( 0, (x-a)/(b-a) )
                )   # ~~~ the positive part of the the line with value 0 at a going up to value 1 at b
        else:
            prior_point = knots[j-1]
            next_point = knots[j+1]
            hat_functions.append(
                    lambda x, a=prior_point, b=midpoint, c=next_point: my_max( 0, my_min(
                            (x-a) / (b-a),
                        1 - (x-b) / (c-b)
                        ))
                )
    return hat_functions

#
# ~~~ A wrapper for globally continuous linear spline regression
def univar_spline_fit( x_train, y_train, knots ):
    return empirical_risk_minimization_with_linear_H( x_train, y_train, list_all_the_hat_functions(knots) )
