
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
        files = [ "ansi.py", "my_base_utils.py", "my_torch_utils.py", "my_visualization_utils.py" ]
        intstall_Toms_code( folder, files )
        #
        # ~~~ "Install/update" answers_680
        folder = "answers_680"
        files = [ "answers_week_6.py" ]
        intstall_Toms_code( folder, files )

#
# ~~~ Tom's helper routines (which the above block of code installs for you); maintained at https://github.com/ThomasLastName/quality_of_life
from quality_of_life.ansi import bcolors
from quality_of_life.my_visualization_utils import points_with_curves



### ~~~
## ~~~ DEMONSTRATION 1 of 7: Use gradient descent to minimize a univariate quadratic (this also demonstrates what the next exercises should kind of look like)
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
## ~~~ EXERCISE 1 of 5 (medium -- defining a quadratic function): Write a function that, based on given data, returns the function that one would like to minimize when fitting a polynomial to that data
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
## ~~~ EXERCISE 2 of 5 (medium -- computing the gradient of a quadratic function): Write a function that, based on given data, returns the callable gradient function R^p \to R^p of the function that one would like to minimize when fitting a polynomial to that data
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
assert max(abs(torch.tensor([ 760083.49, 56753169.66, 4518777464.85 ])-g)) < 1e-6   # ~~~ the gradient should be torch.tensor([ 760083.49, 56753169.66, 4518777464.85 ]) up to computer arithmetic error



### ~~~
## ~~~ EXERCISE 3 of 5 (hard -- compute the lambda-smoothness parameter): Write a function that, based on the given data, computes the lambd-smoothness parameter of the function that one would like to minimize when fitting a polynomial to that data
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
assert compute_lambda(x,y,deg=2)==3901282438.475482     # ~~~ admittedly, this == assertion could give false negatives due to computer arithmetic error



### ~~~
## ~~~ DEMONSTRATION 2 of 7: Use gradient descent to fit a polynomial to some made-up data
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
## ~~~ DEMONSTRATION 3 of 7: Observe that there is a bifurcation depending whether the learning rate is more or less than 2/\lambda
### ~~~

L = compute_lambda( x_train, y_train, deg=4 )   # ~~~ the lambda-smoothness parameter (i.e., the Lipschitz parameter of the gradient) of the function you gotta minimize for polynomial regression
_, coeffs = fit_polynomial_by_gd( x_train, y_train, deg=4, learning_rate=2/L+0.000005 )
print(f"As soon as eta>2/L, gradient descent suddenly diverges. In this case, the coefficient vector ended up with norm {coeffs.norm().item():.6}")   



### ~~~
## ~~~ DEMONSTRATION 4 of 7: Let's see how you can use pytorch to compute derivatives
### ~~~

#
# ~~~ Define several things, as well as some function z of those things
torch.manual_seed(680)  # ~~~ set the random seed for reproducibility
x = torch.randn(3)      # ~~~ x should be tensor([0.7133, 0.5497, 1.1640])
w = torch.randn(3,requires_grad=True)   # ~~~ set `requires_grad=True` to enable computing derivatives with respect to w
b = torch.randn(1,requires_grad=True)   # ~~~ also enable computing derivatives with respect to b
z = torch.inner(x,w)+b  # ~~~ compute z = \langle w,x \rangle + b

#
# ~~~ Derivatives with respect to a tensor are stored in a tensor's `grad` attribute
assert x.grad is None   # ~~~ this is None because no partial derivative with respect to x has been computed
assert w.grad is None   # ~~~ this is None because no partial derivative with respect to w has been computed
assert b.grad is None   # ~~~ this is None because no partial derivative with respect to b has been computed

#
# ~~~ Genearally, to compute \nabla f at a particular value `v`, set `z=f(v)` and then call `z.backward()`; the gradient of f at `v` is then stored in `v.grad`
z.backward()            # ~~~ compute z's parital derivatives with respect to all those variables for which `requires_grad==True`
assert x.grad is None   # ~~~ this is still None because no partial derivative with respect to x was computed
assert min(w.grad == x) # ~~~ a gradient with respect to w was computed and stored here: namely, that of z with respect to w; it's value is x
assert b.grad == 1.     # ~~~ similarly, here is stored the gradient with respect to b which was computed when we called z.backward()



### ~~~
## ~~~ DEMONSTRATION 5 of 7: Pytorch allows you to compute a derivative only with respect to "leaf variables," which can be thought of as "independent" quantities
### ~~~

#
# ~~~ Define several things, as well as some function f using those things
torch.manual_seed(680)  # ~~~ set the random seed for reproducibility
x = torch.randn(3)      # ~~~ x should be tensor([0.7133, 0.5497, 1.1640])
w = torch.randn(3,requires_grad=True)   # ~~~ set `requires_grad=True` to enable computing derivatives with respect to w
b = torch.randn(1,requires_grad=True)   # ~~~ also enable computing derivatives with respect to b
f = lambda x: torch.inner(x,w)+b    # ~~~ an affine functional

#
# ~~~ Correct usage
y = f(x)        # ~~~ machine learner like to call this "the forward step"
y.backward()    # ~~~ P.S. the function that computes derivatives is calleed `backward` because machine learners like to call it "the backward step" when you compute derivatives
print(f"The gradient is {w.grad}")

#
# ~~~ Incorrect usage
w = f(x)        # ~~~ now that we redefine w to be a function of some other variable, it ceases to be a "leaf variable" (i.e., an independent variable)
w.backward()    # ~~~ this will fail because w is no longer a leaf variable
print(f"The gradient is {w.grad}")



### ~~~
## ~~~ DEMONSTRATION 6 of 7: pytorch wants you to zero out the gradients as well as re-define z between calls to z.backward()
### ~~~

#
# ~~~ This block of code is included only for the sake of reproducibility
torch.manual_seed(680)  # ~~~ set the random seed for reproducibility
x = torch.randn(3)      # ~~~ x should be tensor([0.7133, 0.5497, 1.1640])
w = torch.randn(3,requires_grad=True)   # ~~~ set `requires_grad=True` to enable computing derivatives with respect to w
b = torch.randn(1,requires_grad=True)   # ~~~ also enable computing derivatives with respect to b
z = torch.inner(x,w)+b  # ~~~ compute z = \langle w,x \rangle + b
z.backward()            # ~~~ compute the gradient of z with respect to the variables w and b for which `requires_grad==True`

#
# ~~~ Internally, pytorch likes to conserve RAM by "forgetting" the derivative computations ASAP; as a result, you can't call `z.backward()` twice
try:
    z.backward()
    print("This worked: z.backward() was called successfully!")
except RuntimeError as e:
    print("This didn't work: calling z.backward() provoked a RuntimeError with the following message:")
    error_message = str(e)
    red_error_message = bcolors.FAIL + error_message
    print(red_error_message)    # ~~~ The error message isn't great.... it says we must specify `retain_graph=True` but doesn't elaborate where

#
# ~~~ The error message neglects to mention that it's already too late: this can't be done retroactively
try:
    z.backward(retain_graph=True)
    print("This worked: z.backward() was called successfully!")
except RuntimeError as e:
    print("Again, this didn't work: calling z.backward() provoked a RuntimeError with the following message:")
    error_message = str(e)
    red_error_message = bcolors.FAIL + error_message
    print(red_error_message)

#
# ~~~ By default, pytorch already forgot how it computed the gradient of z and you can't undo the "forgetting" unless you redo the computations
try:
    z = torch.inner(x,w)+b          # ~~~ start over
    z.backward(retain_graph=True)   # ~~~ tell the computer *in advance* to remember everything it does while computing gradient of z
    z.backward()                    # ~~~ try to compute the gradient of z, again
    print("This worked: z.backward() was called successfully!")
except RuntimeError as e:
    print("This didn't work: calling z.backward() provoked a RuntimeError with the following message:")
    error_message = str(e)
    red_error_message = bcolors.FAIL + error_message
    print(red_error_message)

#
# ~~~ However, pytorch fails to warn us that there is another issue entirely...
try:
    assert x.grad is None   # ~~~ this is still None because no partial derivative with respect to x was computed
    assert min(w.grad == x) # ~~~ a gradient with respect to w was computed and stored here: namely, that of z with respect to w; it's value is x
    assert b.grad == 1.     # ~~~ similarly, here is stored the gradient with respect to b which was computed when we called z.backward()
    print("Pytorch agrees with math.")
except AssertionError:
    print("The value of the gradient in pytorch does not allign with the mathematical value of the gradient.")

#
# ~~~ For reasons not entirely clear to me, pytorch chooses to let gradients accumulate unless you explicitly zero them out
multiple = round((w.grad/x)[0].item())  # ~~~ convert to integer the first element of the array w.grad/x (division is carried out elementwise)
assert min(w.grad == multiple*x)        # ~~~ observe that w.grad is, exactly, `multiple` times x
print(f"Pytroch thinks the value of w.grad is {multiple} times what it should be. This happens because z.backward() has been called successfuly {multiple} times." )

#
# ~~~ However, even this doesn't appear to behave consistently
multiple = int(b.grad.item())   # ~~~ convert b.grad to integer
assert b.grad>1                 # ~~~ assert that b.grad is not the mathematically expected value of 1
print(f"Pytroch thinks the value of b.grad is {multiple} times what it should be. This happens because z.backward() has been called {multiple} times, including the times that resulted in a RuntimeError (presumably, some but not all computations were completed before the errors occured)" )

#
# ~~~
if True:
    print("")
    print("-----------------------------------------------------------------------")
    print("MORAL OF THE STORY: in z=f(v), pytorch really wants you to manually zero-out v.grad between calls to .backward(), and does not like you calling z.backward() on the same item z more than once.")
    print("-----------------------------------------------------------------------")



### ~~~
## ~~~ EXERCISE 4 of 5 (hard -- demands an understanding of torch.autograd): debug the following code (as presented, the graident computation is correct, but the update does not work)
### ~~~

if exercise_mode:
    #
    # ~~~ Debug the following example of gradient descent; the gradient is computated correctly, but THE UPDATE IS NOT CORRECTLY REALIZED
    f = lambda x: x**4 + x**3 - 2*x**2 - 2*x    # ~~~ some function that I want to minimize via graident descent
    x = torch.tensor( 1., requires_grad=True )  # ~~~ some initial guess
    eta = 0.01                                  # ~~~ some learning rate
    tol = 0.000001                              # ~~~ some tolerance
    progress = tol+1
    while progress > tol:                       # ~~~ while progress continues...
        #
        # ~~~ Use pytorch to compute f'(x)
        y = f(x)
        y.backward()  # retain_grad=True?                      # ~~~ compute f'(x) and store its value in x.grad
        y_prime = torch.clone(x.grad)       # ~~~ record the value before zeroing out the gradient
        _ = x.grad.zero_()                  # ~~~ zero out the gradient ASAP to hopefully appease pytorch
        #
        # ~~~ Update x according to the gradient descent algorithm (BUG IS HERE)
        update = eta * y/y_prime
        x = x - update              # ~~~ we won't be able to call x.grad anymore after this, for the same reason why we can't call y.grad: it's not a "leaf variable" (basically, a primitative variable that depends on nothing else)
        progress = eta*abs(update)  # ~~~ the absolute value of the difference between x's values before and after the update
    #
    # ~~~ If correctly implemented, this assertion should pass
    assert abs(x.item()-0.922231674) < 1e-8 # ~~~ for the expected output is x==0.9222316741943359



### ~~~
## ~~~ DEMONSTRATION 7 of 7: Instead of computing derivatives by hand, let pytorch do it for us
### ~~~

def gradient_descent( f, learning_rage, initial_guess, max_iter=10000 ):
    x = initial_guess
    x.requires_grad = True
    for _ in range(max_iter):
        #
        # ~~~ Compute the gradient
        y = f(x)
        y.backward()                    # ~~~ compute f'(x) and store its value in x.grad
        grad_x = torch.clone(x.grad)    # ~~~ record the value before zeroing out the gradient
        _ = x.grad.zero_()              # ~~~ zero out the gradient ASAP to hopefully appease pytorch
        #
        # ~~~ Use the gradient
        with torch.no_grad():
            x -= learning_rage * grad_x
    return x

#
# ~~~ Run it and see
torch.manual_seed(680)          # ~~~ set the random seed for reproducibility
x_train = 2*torch.rand(100)-1   # ~~~ make up some random data
f = lambda x: torch.abs(x)      # ~~~ make up some ground truth
y_train = f(x_train) + 0.1*torch.randn(100)   # ~~~ produce noisy measurements
degree = 4
overall_loss = build_objective_from_data( x_train, y_train, deg=degree )
L = compute_lambda( x_train, y_train, deg=degree )
coefficients = gradient_descent( overall_loss, learning_rage=len(x_train)/L, initial_guess=torch.zeros(degree+1) )
poly, coeffs = fit_polynomial_by_gd( x_train, y_train, deg=degree )
assert abs(coeffs-coefficients).max() < 1e-14    # ~~~ if this assertion passes, it demonstrates that the two implementations are equivalent


### ~~~
## ~~~ EXERCISE 5 of 5 (hard -- involves understanding a lot of context): determine for yourself why, in the preceding DEMO, I up-scale the learning rate by len(x_train); i.e., why do I take learning_rage=len(x_train)/L instead of 1/L?
### ~~~
