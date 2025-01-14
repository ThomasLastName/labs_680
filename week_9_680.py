
# ~~~ Tom Winckelman wrote this; maintained at: https://github.com/ThomasLastName/labs_680

exercise_mode = False   # ~~~ see https://github.com/ThomasLastName/labs_680?tab=readme-ov-file#usage
install_assist = False  # ~~~ see https://github.com/ThomasLastName/labs_680?tab=readme-ov-file#assisted-installation-for-environments-other-than-colab-deprecated


### ~~~
## ~~~ Boiler plate stuff; basically just loading packages
### ~~~

import numpy as np
import torch
from tqdm import trange

#
# ~~~ see https://github.com/ThomasLastName/labs_680?tab=readme-ov-file#assisted-installation-for-environments-other-than-colab-deprecated
import os
this_is_running_in_colab = os.getenv("COLAB_RELEASE_TAG")   # ~~~ see https://stackoverflow.com/a/74930276
if install_assist or this_is_running_in_colab:              # ~~~ override necessary permissions if this is running in Colab
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
            parent_directory = os.path.dirname(os.path.dirname(np.__file__)) if (deisred_parent_directory is None) else deisred_parent_directory
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
        files = [ "ansi.py", "my_base_utils.py", "my_numpy_utils.py", "my_visualization_utils.py" ]
        intstall_Toms_code( folder, files )
        #
        # ~~~ "Install/update" answers_680
        folder = "answers_680"
        files = [ "answers_week_9.py" ]
        intstall_Toms_code( folder, files )

#
# ~~~ Tom's helper routines (which the above block of code installs for you); maintained at https://github.com/ThomasLastName/quality_of_life
from quality_of_life.my_base_utils import support_for_progress_bars
from quality_of_life.my_numpy_utils import generate_random_1d_data
from quality_of_life.my_visualization_utils import points_with_curves, GifMaker
# from answers_680.answers_week_8 import build_shallow_network

#
# ~~~ Define function that tests whether or not some permutation of a vector is equal to another vector, i.e., whether one is just a possibly rearranged version of the other
from itertools import permutations
def check_permuted_equality( vec, shuffled_vec, short_only=True ):
    if not len(vec)==len(shuffled_vec):
        return False
    if short_only and len(vec)>10:
        raise ValueError(f"The cost is n! where n=len(vec). For the supplied value of n={len(vec)} this may be slow. Specify short_only=False to run anyway.")
    for permutation_of_vec in permutations(vec):
        if np.allclose( permutation_of_vec, shuffled_vec ):
            return True
    return False



### ~~~
## ~~~ EXERCISE 1 of 3 (hard): Write a function using only numpy that, given x (a vector, not matrix) and sigma (the activation function), computes the gradient of the function f(a,W.flatten()) = a^T\sigma(Wx)
### ~~~

if exercise_mode:
    def par_grad_lacking_bias( a, W, x, sigma, sigma_prime ):
        H = len(a)              # ~~~ the number of hidden layers in the (shallow) neural network pred(x)=a^T\sigma(Wx)
        d = len(x)              # ~~~ the number of input dimensions to the neural network pred(x)=a^T\sigma(Wx)
        assert x.shape==(d,)    # ~~~ assume x is simply a vector of length x (only a single d-dimensional input at this time)
        assert W.shape==(H,d)
        assert a.shape==(H,)
        # NOTE: the intent is that you work out the derivative by hand and then hard-code the forumla for the derivative
        return #the gradient of the function f(theta) that sends parameters theta=(a,W) to the prediction f(theta)=a.T@sigma(W@x) at x of the shallow network
else:
    from answers_680.answers_week_9 import par_grad_lacking_bias

#
# ~~~ Some made up data and stuff
x = np.random.normal(size=(2,))
W = np.random.normal(size=(2,2))
a = np.random.normal(size=(2,))
ReLU_prime = lambda x: (x>=0) +1.-1.    # ~~~ the `+1.-1.` achieves a module-agnostic converstion to float, i.e., works whether x is a torch.tensor or numpy array
ReLU = lambda x: x*ReLU_prime(x)        # ~~~ equvalent to np.maximum(x,0) if x is a numpy array, or torch.clamp(x,min=0) if x is a torch.tensor

#
# ~~~ Collect my implementation
with torch.no_grad():   # ~~~ with no cheating ;)
    my_gradient = par_grad_lacking_bias( a, W, x, ReLU, ReLU_prime )

#
# ~~~ Collect the standard torch.autograd implementation for comparison
W = torch.from_numpy(W)
a = torch.from_numpy(a)
x = torch.from_numpy(x)
W.requires_grad = True
a.requires_grad = True
f = torch.inner(a,ReLU(W@x))
f.backward()
torch_gradient = np.concatenate(( a.grad.numpy(), W.grad.numpy().flatten() ))

#
# ~~~ The order of `my_gradient` depends on how you choose to enumerate the parameters, which is an arbitrary convention; we search for equality agnostic to that convention
assert check_permuted_equality( my_gradient, torch_gradient )



### ~~~
## ~~~ EXERCISE 2 of 3 (medium, given the prior exercise): Write a function using only numpy that, given x and sigma, computes the gradient of the function f(a,W,b,c) = c + a^T\sigma(Wx + b)
### ~~~

if exercise_mode:
    def par_grad_with_bias( a, W, b, c, x, sigma, sigma_prime ):
        H = len(a)              # ~~~ the number of hidden layers in the (shallow) neural network pred(x)=a^T\sigma(Wx)
        d = len(x)              # ~~~ the number of input dimensions to the neural network pred(x)=a^T\sigma(Wx)
        assert x.shape==(d,)    # ~~~ assume x is simply a vector of length x (only a single d-dimensional input at this time)
        assert W.shape==(H,d)
        assert a.shape==(H,)
        assert b.shape==(H,)
        assert isinstance(c,float)
        # HINT: You can use the function `par_grad_lacking_bias` that you wrote in the prior exercise
        return #the gradient of the function f(theta) that sends parameters theta=(a,W,b,c) to the prediction at x of the
else:
    from answers_680.answers_week_9 import par_grad_with_bias

#
# ~~~ Some made up data and stuff
x = np.random.normal(size=(2,))
W = np.random.normal(size=(2,2))
a = np.random.normal(size=(2,))
b = np.random.normal(size=(2,))
c = 0.4
ReLU_prime = lambda x: (x>=0) +1.-1.    # ~~~ the `+1.-1.` achieves a module-agnostic converstion to float, i.e., works whether x is a torch.tensor or numpy array
ReLU = lambda x: x*ReLU_prime(x)        # ~~~ equvalent to np.maximum(x,0) if x is a numpy array, or torch.clamp(x,min=0) if x is a torch.tensor

#
# ~~~ Collect my implementation
with torch.no_grad():                   # ~~~ With no cheating ;)
    my_gradient = par_grad_with_bias( a, W, b, c, x, ReLU, ReLU_prime )

#
# ~~~ Collect the standard torch.autograd implementation for comparison
W = torch.from_numpy(W)
a = torch.from_numpy(a)
b = torch.from_numpy(b)
x = torch.from_numpy(x)
W.requires_grad = True
a.requires_grad = True
b.requires_grad = True
f = torch.inner(a,ReLU(W@x+b)) + c
f.backward()
torch_gradient = np.concatenate(( [1.], b.grad.numpy(), a.grad.numpy(), W.grad.numpy().flatten() ))

#
# ~~~ The order of `my_gradient` depends on how you choose to enumerate the parameters, which is an arbitrary convention; we search for equality agnostic to that convention
assert check_permuted_equality( my_gradient, torch_gradient )



### ~~~
## ~~~ EXERCISE 3 of 3 (easy, given the prior exercise): Write a function using only numpy that, given an input x (a vector, not matrix) and response y (scalar, not vector), computes the gradient of the function \ell(a,W,b,c) = \ell( y, c + a^T\sigma(Wx+b) ) using info about \ell and \sigma
### ~~~

if exercise_mode:
    def grad_of_item_loss( a, W, b, c, x, y, sigma, sigma_prime, ell_prime ):
        H = len(a)              # ~~~ the number of hidden layers in the (shallow) neural network pred(x)=a^T\sigma(Wx)
        d = len(x)              # ~~~ the number of input dimensions to the neural network pred(x)=a^T\sigma(Wx)
        assert x.shape==(d,)    # ~~~ assume x is simply a vector of length x (only a single d-dimensional input at this time)
        assert W.shape==(H,d)
        assert a.shape==(H,)
        assert b.shape==(H,)
        assert isinstance(c,float)
        assert isinstance(y,float)
        # NOTE: ell_prime should be the derivative of ell(y,p) with respect to the second argument p
        # HINT: You can use the function `par_grad_lacking_bias` that you wrote in the prior exercise
        return # the gradient of the function \ell(a,W,b,c) = \ell( y, c + a^T\sigma(Wx+b) )
else:
    from answers_680.answers_week_9 import grad_of_item_loss

#
# ~~~ Some made up data and stuff
x = np.random.normal(size=(2,))
W = np.random.normal(size=(2,2))
a = np.random.normal(size=(2,))
b = np.random.normal(size=(2,))
c = 0.4
y = np.pi
ReLU_prime = lambda x: (x>=0) +1.-1.    # ~~~ the `+1.-1.` achieves a module-agnostic converstion to float, i.e., works whether x is a torch.tensor or numpy array
ReLU = lambda x: x*ReLU_prime(x)        # ~~~ equvalent to np.maximum(x,0) if x is a numpy array, or torch.clamp(x,min=0) if x is a torch.tensor
ell = lambda y,p: (y-p)**2
ell_prime = lambda y,p: 2*(p-y)

#
# ~~~ Collect my implementation
with torch.no_grad():                   # ~~~ With no cheating ;)
    my_gradient = grad_of_item_loss( a, W, b, c, x, y, ReLU, ReLU_prime, ell_prime )

#
# ~~~ Collect the standard torch.autograd implementation for comparison
W = torch.from_numpy(W)
a = torch.from_numpy(a)
b = torch.from_numpy(b)
x = torch.from_numpy(x)
W.requires_grad = True
a.requires_grad = True
b.requires_grad = True
f = torch.inner(a,ReLU(W@x+b)) + c
loss = ell(y,f)
loss.backward()
torch_gradient = np.concatenate((
        [ell_prime(y,f).item()],    # ~~~ the partial derivative of loss=ell(y,a^T\sigma(Wx+b)+c) with respect to (w.r.t.) c
        b.grad.numpy(),             # ~~~ the partial derivatives of loss=ell(y,a^T\sigma(Wx+b)+c) w.r.t. the entries of b
        a.grad.numpy(),             # ~~~ the partial derivatives of loss=ell(y,a^T\sigma(Wx+b)+c) w.r.t. the entries of a
        W.grad.numpy().flatten()    # ~~~ .... w.r.t. the entries of W
    ))

#
# ~~~ The order of `my_gradient` depends on how you choose to enumerate the parameters, which is an arbitrary convention; we search for equality agnostic to that convention
assert check_permuted_equality( my_gradient, torch_gradient )



### ~~~
## ~~~ DEMONSTRATION 1 of 1: Visualize the difference between GD and SGD
### ~~~

# def xavier_uniform(d,n):
#     a = np.random.uniform( low=-1, high=1, size=(n,) ) * np.sqrt(6/(n+1))
#     W = np.random.uniform( low=-1, high=1, size=(n,d) ) * np.sqrt(6/(n+d))
#     return a, W, np.zeros(n), 0


# def visualize_GD( eta, width, x_train, y_train, ground_truth, max_iter=120, batch_size=1, ell_prime=ell_prime, gif_name="GDSGD", **kwargs ):
#     #
#     # ~~~ Generate some initial parameters for the desired architecture (in this case, the desired number of hidden units)
#     a, W, b, c = xavier_uniform(1,width)
#     initial_model = build_shallow_network(a,W,b,c,ReLU)
#     #
#     # ~~~ Establish some plotting parameters
#     alphas = [.05,.1,.15,.2,1]
#     delay = len(alphas)
#     linger = delay
#     past_few = [initial_model]*delay
#     gif = GifMaker()
#     #
#     # ~~~ Abbreviate the creation of a plot by defining it to be `snap()`
#     def snap():
#         _ = points_with_curves(
#             x = x_train,
#             y = y_train,
#             curves = past_few + [ground_truth,],
#             curve_colors = delay*["midnightblue"] + ["green"],
#             curve_alphas = alphas + [1,],
#             curve_marks = delay*["-"] + ["--"],
#             title = gif_name,
#             show = False,
#             legend = False,
#             model_fit = False   # ~~~ surpresses a warning that we are not using defaults
#             )
#         gif.capture()
#     #
#     # ~~~ Take several (identical) snapshots of the initial model
#     for _ in range(linger):
#         snap()
#     #
#     # ~~~ Now, do GD or SGD
#     batch_size = len(x_train) if batch_size=="full" else max(batch_size,len(x_train))
#     with support_for_progress_bars():
#         for _ in trange(max_iter):
#             #
#             # ~~~ Compute the gradients and update the parameters accordingly
#             indices_to_use_for_update = np.random.choice( len(x_train), size=batch_size, replace=False )
#             for i in indices_to_use_for_update:
#                 grad_a, grad_W, grad_b, grad_c = grad_of_item_loss( a=a, W=W, b=b, c=c, x=x_train[i], y=y_train[i], sigma=ReLU, sigma_prime=ReLU_prime, ell_prime=ell_prime, flatten_and_concatenate=False )
#                 a -= eta*grad_a
#                 W -= eta*grad_W
#                 b -= eta*grad_b
#                 c -= eta*grad_c
#             #
#             # ~~~ Take a snapshot including the new, updated model
#             _ = past_few.pop(0)
#             past_few.append( build_shallow_network(a=a,W=W,b=b,c=c,sigma=ReLU) )
#             snap()
#     #
#     # ~~~ Fially, take several
#     gif.develop( destination=gif_name, **kwargs )
#     return a,W,b,c, build_shallow_network(a=a,W=W,b=b,c=c,sigma=ReLU)



# f = lambda x: abs(x) + np.exp(-1.5*x**2) # ~~~ a quite difficult function to approximate
# f = lambda x: 2*np.cos(np.pi*((x+0.2))) + np.exp(2.5*(x+0.2))/2.5   # ~~~ a somewhat easy function to approximate

# np.random.seed(680)
# x_train, y_train, _, _ = generate_random_1d_data( ground_truth=f, n_train=30, n_test=1001, noise=.1 )
# lr = 0.005
# for n in (50,500):
#     np.random.seed(680)
#     a,W,b,c, model = visualize_GD( eta=lr, max_iter=485, width=n, x_train=x_train, y_train=y_train, ground_truth=f, batch_size="full", gif_name=f"Full GD on a Complex Ground Truth, Width {n}, lr={lr}", fps=60 )


# for i in range(len(x_train)):
#     grad_a, grad_W, grad_b, grad_c = grad_of_item_loss( a=a, W=W, b=b, c=c, x=x_train[i], y=y_train[i], sigma=ReLU, sigma_prime=ReLU_prime, ell_prime=ell_prime, flatten_and_concatenate=False )
#     a -= eta*grad_a
#     W -= eta*grad_W
#     b -= eta*grad_b
#     c -= eta*grad_c