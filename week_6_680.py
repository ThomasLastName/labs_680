
# ~~~ Tom Winckelman wrote this; maintained at: https://github.com/ThomasLastName/labs_680

exercise_mode = False   # ~~~ see https://github.com/ThomasLastName/labs_680?tab=readme-ov-file#usage
install_assist = False  # ~~~ see https://github.com/ThomasLastName/labs_680/blob/main/README.md#assisted-installation-for-environments-other-than-colab-recommended


### ~~~
## ~~~ Boiler plate stuff; basically just loading packages
### ~~~

#
# ~~~ Standard python libraries
import torch    # ~~~ pytorch is simply required for this lab (actually a lot of it could be done in numpy, but I wrote it in pytorch and can't be bothered to change it)
torch.set_default_dtype(torch.double)   # ~~~ use high precision computer arithmetic

#
# ~~~ see https://github.com/ThomasLastName/labs_680/blob/main/README.md#assisted-installation-for-environments-other-than-colab-recommended
import os
this_is_running_in_colab = os.getenv("COLAB_RELEASE_TAG")   # ~~~ see https://stackoverflow.com/a/74930276
if install_assist or this_is_running_in_colab:              # override necessary permissions if this is running in Colab
    confirm_permission_to_modify_files = not install_assist
    if (install_assist and confirm_permission_to_modify_files) or this_is_running_in_colab:
        #
        # ~~~ Base package for downloading files
        from urllib.request import urlretrieve
        #
        # ~~~ Define a routine that downloads a raw file from GitHub and locates it at a specified path
        def download_dotpy_from_GitHub_raw( url_to_raw, file_name, folder_name, deisred_parent_directory=None, verbose=True ):
            #
            # ~~~ Put together the appropriate path
            this_is_running_in_colab = os.getenv("COLAB_RELEASE_TAG")   # ~~~ see https://stackoverflow.com/a/74930276
            parent_directory = os.path.dirname(os.path.dirname(torch.__file__)) if (deisred_parent_directory is None) else deisred_parent_directory
            parent_directory = "" if this_is_running_in_colab else parent_directory
            folder_path = os.path.join( parent_directory, folder_name )
            file_path = os.path.join( folder_path, file_name )
            print_path = os.path.join("/content",folder_name,file_name) if this_is_running_in_colab else file_path
            #
            # ~~~ Create the folder if it doesn't already exist
            if not os.path.exists(folder_path):
                os.makedirs(folder_path)
                if verbose:
                    print("")
                    print(f"Folder {folder_name} created at {os.path.dirname(print_path)}")
                    print("")
            #
            # ~~~ Download that file and place it at the path `file_path`, overwritting a file of the same name in the same location, if one exists
            prefix = "Updated" if os.path.exists(file_path) else "Created"
            urlretrieve( url_to_raw, file_path )
            if verbose:
                suffix = " (click the folder on the left)" if this_is_running_in_colab else ""
                print( f"{prefix} file {file_name} at {print_path}{suffix}" )
        #
        # ~~~ A routine that downloads from Tom's GitHub repos
        def intstall_Toms_code( folder_name, files, repo_name=None, verbose=True ):
            repo_name = folder_name if repo_name is None else repo_name
            base_url = f"https://raw.githubusercontent.com/ThomasLastName/{repo_name}/main/"
            for file_name in files:
                download_dotpy_from_GitHub_raw( url_to_raw=base_url+file_name, file_name=file_name, folder_name=folder_name, verbose=verbose )
        #
        # ~~~ "Install/update" quality_of_life
        folder = "quality_of_life"
        files = [ "ansi.py", "my_base_utils.py", "my_visualization_utils.py" ]
        intstall_Toms_code( folder, files )
        #
        # ~~~ "Install/update" answers_680
        folder = "answers_680"
        files = [ "answers_week_6.py" ]
        intstall_Toms_code( folder, files )

#
# ~~~ Tom's helper routines (which the above block of code installs for you); maintained at https://github.com/ThomasLastName/quality_of_life
from quality_of_life.my_visualization_utils import points_with_curves



### ~~~
## ~~~ DEMONSTRATION 1 of 3: Use gradient descent to minimize a univariate quadratic (this also demonstrates what the next exercises should kind of look like)
### ~~~

#
# ~~~ Analogously to exercise 1 below, define a function that, itself, defines a quadratic function that we'd like to minimize based on some user-supplied info
def build_objective( quadratic_term, slope, intercept=0 ):
    #
    # ~~~ Safety feature
    assert quadratic_term>=0     # ~~~ for only then is the quadratic function convex and even bounded from below
    #
    # ~~~ Define a quadratic function
    def q(number):
        return quadratic_term*number**2 + slope*number + intercept
    #
    # ~~~ Return the function
    return q

#
# ~~~ Analogously to exercise 2 below, define a function that, itself, defines a function which applies an exact formula to compute the derivative of the function we want to minimize
def formula_for_the_derivative( quadratic_term, slope, intercept=0 ):
    def derivative(number):
        return 2*quadratic_term*number + slope
    return derivative

#
# ~~~ Analogously to exercise 3 below, define a function that applies a formula to compute the lambda-smoothness parameter of a quadratic function
def compute_lambda( quadratic_term, slope, intercept=0 ):
    return 2*abs(quadratic_term)

#
# ~~~ Test it out
q = build_objective( quadratic_term=1, slope=2, intercept=5.3 ) # ~~~ q(x) == x**2 + 2*x + 5.3
assert q(-2.4)==6.26

#
# ~~~ Intended usage
a, b, c = 1, 2, 5.3
x = 0
q_prime = formula_for_the_derivative( a, b, c )
learning_rate = 1/compute_lambda( a, b, c )
for _ in range(100):
    x -= learning_rate * q_prime(x)

assert x == -b/(2*a)    # ~~~ which is the true minimizer
print(f"Gradient descent found x={x:.4} which matches the true minimizer {-b/(2*a)}")



### ~~~
## ~~~ EXERCISE 1 of 3 (medium -- defining a quadratic function): Write a function that, based on given data, returns the function that one would like to minimize when fitting a polynomial to that data
### ~~~

#
# ~~~ An example loss function
def mse( vector_of_target_values, vector_of_predicted_values ):
    return torch.mean(( vector_of_target_values - vector_of_predicted_values )**2)

if exercise_mode:
    #
    # ~~~ In analogy to the above function `build_objective`, write a function that (when loss_function=mse) returns the function f(w)=mse(y,p_w(x))
    def build_objective_from_data( x_train, y_train, deg=1, loss_function=mse ):
        #
        # ~~~ Define the objective function as a function of the polynomial coefficients only
        def function_that_we_need_to_minimize_in_order_to_fit_a_polynomial_to_the_given_data(coefficients):
            # YOUR CODE HERE
            return value_given_these_coefficients
        #
        # ~~~ This function returns the function that we want to minimize when fitting a polynomial to our data
        return function_that_we_need_to_minimize_in_order_to_fit_a_polynomial_to_the_given_data
else:
    from answers_680.answers_week_6 import build_objective_from_data

#
# ~~~ Test intended functionality
w = torch.tensor([0.7133, 0.5497, 1.1640])  # ~~~ some random coefficients
x = torch.arange(100)                       # ~~~ some arbitrary data
y = torch.arange(100)                       # ~~~ some arbitrary data
your_objective_function = build_objective_from_data(x,y,deg=2)
p_w = lambda x: w[0] + w[1]*x + w[2]*x**2   # ~~~ the polynomial with coefficients w
assert your_objective_function(w) == mse( y, p_w(x) )   # ~~~ this is mse of the residual vector y-p_w(x)



### ~~~
## ~~~ EXERCISE 2 of 3 (medium -- computing the gradient of a quadratic function): Write a function that, based on given data, returns the callable gradient function R^p \to R^p of the function that one would like to minimize when fitting a polynomial to that data
### ~~~

if exercise_mode:
    #
    # ~~~ In analogy to the above function `formula_for_the_derivative`, write a function that, itself, defines a function which applies an exact formula to compute the gradient of the function we want to minimize
    def formula_for_the_gradient( x_train, y_train, deg=1 ):
        def gradient(w): # of the function f(w)=mse(y,p_w(x)) returned by `build_objective_from_data(x_train,y_train,deg=deg)`
            # YOUR CODE HERE
            return the_gradient_at_w
        return gradient
else:
    from answers_680.answers_week_6 import formula_for_the_gradient

#
# ~~~ Validate that the code gives the expected output
w = torch.tensor([0.7133, 0.5497, 1.1640])  # ~~~ some random coefficients
x = torch.arange(100)                       # ~~~ some arbitrary data
y = torch.arange(100)                       # ~~~ some arbitrary data
grad = formula_for_the_gradient(x,y,deg=2)
g = grad(w)                                 # ~~~ should be the gradient of in w of the funcion f(w)=mse(y,p_w(x)) returned by `build_objective_from_data(x,y,deg=2)`
assert max(abs(torch.tensor([ 7600.8349, 567531.6966, 45187774.6485 ])-g)) < 1e-8   # ~~~ the gradient should be torch.tensor([ 760083.49, 56753169.66, 4518777464.85 ]) up to computer arithmetic error



### ~~~
## ~~~ EXERCISE 3 of 3 (hard -- compute the lambda-smoothness parameter): Write a function that, based on the given data, computes the lambd-smoothness parameter of the function that one would like to minimize when fitting a polynomial to that data
### ~~~

if exercise_mode:
    #
    # ~~~ In analogy to the function of the same name above, define a function that applies a formula to compute the lambda-smoothness parameter of the quadratic function we want to minimize when fitting a polynomial to our data
    def compute_lambda( x_train, y_train, deg=1 ):
        # YOUR CODE HERE
        return the_lambda_smoothness_parameter_of_the_objective_function_in_the_polynomial_regression_problem
else:
    from answers_680.answers_week_6 import compute_lambda

#
# ~~~ Validate that the code gives the expected output
x = torch.arange(100)   # ~~~ some made-up data
y = torch.arange(100)   # ~~~ some made-up data
assert compute_lambda(x,y,deg=2)==39012824.38475482     # ~~~ admittedly, this == assertion could give false negatives due to computer arithmetic error



### ~~~
## ~~~ DEMONSTRATION 2 of 3: Use gradient descent to fit a polynomial to some made-up data
### ~~~

#
# ~~~ Function that uses GD with constant learning rate to fit a polynomial to user-supplied data by minimizing the appropriate function
def fit_polynomial_by_gd( x_train, y_train, deg=1, learning_rate=None, initial_guess=None, max_iter=10000 ):
    #
    # ~~~ Collect the ingredients needed to perform gradient descent
    grad = formula_for_the_gradient( x_train, y_train, deg )
    learning_rate = 1/compute_lambda( x_train, y_train, deg ) if learning_rate is None else learning_rate
    w = torch.zeros(deg+1) if initial_guess is None else initial_guess  # ~~~ initialize w to all zeros if no initial guess is provided
    assert w.shape == (deg+1,)
    #
    # ~~~ Iterate the gradient descent algorithm as many times as desired
    for _ in range(max_iter):
        w -= learning_rate * grad(w)
    #
    # ~~~ Whatever `w`` is by this point, define fitted_model(x) = w[0] + w[1]*x + w[2]*x**2 +...+ w[deg]*x**deg
    fitted_model = lambda x, coefficients=w: torch.column_stack([ x**j for j in range(deg+1) ]) @ coefficients
    return fitted_model, w

#
# ~~~ Run it and see
torch.manual_seed(680)                      # ~~~ set the random seed for reproducibility
x_train = 2*torch.rand(100)-1               # ~~~ make up some random data
f = lambda x: torch.abs(x)                  # ~~~ make up some ground truth
y_train = f(x_train) + 0.1*torch.randn(100) # ~~~ produce noisy measurements
poly,coeffs = fit_polynomial_by_gd( x_train, y_train, deg=4 )
points_with_curves( x_train, y_train, (poly,f) )    # ~~~ plot it and see that it looks good



### ~~~
## ~~~ DEMONSTRATION 3 of 3: Observe that there is a bifurcation depending whether the learning rate is more or less than 2/\lambda
### ~~~

L = compute_lambda( x_train, y_train, deg=4 )   # ~~~ the lambda-smoothness parameter (i.e., the Lipschitz parameter of the gradient) of the function you gotta minimize for polynomial regression
_, coeffs = fit_polynomial_by_gd( x_train, y_train, deg=4, learning_rate=2/L+0.0005 )
print(f"As soon as eta>2/L, gradient descent suddenly diverges. In this case, the coefficient vector ended up with norm {coeffs.norm().item():.6}")   


