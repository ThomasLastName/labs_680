
# ~~~ Tom Winckelman wrote this; maintained at: https://github.com/ThomasLastName/labs_680

exercise_mode = False   # ~~~ see https://github.com/ThomasLastName/labs_680?tab=readme-ov-file#usage
install_assist = False  # ~~~ see https://github.com/ThomasLastName/labs_680/blob/main/README.md#assisted-installation-for-environments-other-than-colab-recommended


### ~~~
## ~~~ Boiler plate stuff; basically just loading packages
### ~~~

import sys
import numpy as np
import matplotlib.pyplot as plt

#
# ~~~ see https://github.com/ThomasLastName/labs_680/blob/main/README.md#assisted-installation-for-environments-other-than-colab-recommended
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
        files = [ "ansi.py", "my_base_utils.py", "my_visualization_utils.py" ]
        intstall_Toms_code( folder, files )
        #
        # ~~~ "Install/update" answers_680
        folder = "answers_680"
        files = [ "answers_week_8.py" ]
        intstall_Toms_code( folder, files )

#
# ~~~ Tom's helper routines (which the above block of code installs for you); maintained at https://github.com/ThomasLastName/quality_of_life
from quality_of_life.my_visualization_utils import func_surf
from answers_680.answers_week_8 import build_shallow_network

#
# ~~~ A helper routine that checks whether or not any of the major ML packages have been imported
def numpy_only():
    return ("torch" not in sys.modules.keys()) and ("tensorflow" not in sys.modules.keys())  and ("sklearn" not in sys.modules.keys())



### ~~~
## ~~~ DEMONSTRATION 1 of 1: Plot some neural networks to see what they look like
### ~~~

#
# ~~~ Initialize biases to zero and weights according to the 'Xavier normal' rule: std=np.sqrt(2/(prev_dim+nex_dim))
def xavier_normal(d,n):
    a = np.random.normal( size=(n,) ) * np.sqrt(2/(n+1))
    W = np.random.normal( size=(n,d) ) * np.sqrt(2/(n+d))
    return a, W, np.zeros(n), 0

#
# ~~~ Initialize biases to zero and weights according to the 'Xavier uniform' rule: bound=np.sqrt(6/(prev_dim+nex_dim))
def xavier_uniform(d,n):
    a = np.random.uniform( low=-1, high=1, size=(n,) ) * np.sqrt(6/(n+1))
    W = np.random.uniform( low=-1, high=1, size=(n,d) ) * np.sqrt(6/(n+d))
    return a, W, np.zeros(n), 0

#
# ~~~ Initialize biases to zero and weights according to the 'Kaiming' rule: std=1/sqrt(prev_dim)
def He(d,n):
    a = np.random.normal( size=(n,) )/np.sqrt(n)
    W = np.random.normal( size=(n,d) )/np.sqrt(d)
    return a, W, np.zeros(n), 0

#
# ~~~ Graph several initializations of univariate networks
initialization = He
from answers_680.answers_week_8 import ReLU
n = 10  # ~~~ The width of the networks that we will graph
N = 5   # ~~~ How many such networks that we will graph
x = np.linspace(-3,3,501)
np.random.seed(680)
_ = plt.figure(figsize=(10,5))
for _ in range(N):
    model = build_shallow_network(*initialization(1,n),ReLU)
    plt.plot( x, model(x) )

plt.title(f"{N} Shallow Univariate Network(s) with ReLU Activation and {n} Hidden Units")
plt.grid()
plt.tight_layout()
plt.show()

#
# ~~~ Graph several initializations of bivariate networks
initialization = He
from answers_680.answers_week_8 import ReLU
n = 10  # ~~~ The width of the networks that we will graph
N = 2   # ~~~ How many such networks that we will graph
x = np.linspace(-3,3,501)
y = np.linspace(-3,3,501)
np.random.seed(680)
for _ in range(N):
    model = build_shallow_network(*initialization(2,n),ReLU)
    func_surf(x,y,model)

#
# ~~~ Delete the ReLU function from the namespace (if present); leave the user to define it themselves in the following exercises
try:
    del ReLU
except NameError:   # ~~~ except if there's nothing called `ReLU` in the name space
    pass            # ~~~ in that case, do nothing



### ~~~
## ~~~ EXERCISE 1 of 2: Using numpy and the supplied parameters, define a univariate shallow neural network with ReLU activation and 10 hidden units that works on both scalars and vectors
### ~~~

#
# ~~~ Here are some parameters; define (the forward pass of) the neural network with these parameters
np.random.seed(680)                 # ~~~ for reproducibility
n = 10                              # ~~~ number of hidden units
w = np.random.normal(size=(n,))     # ~~~ the inner weights
a = np.random.normal(size=(n,))     # ~~~ the outer weights
b = np.random.normal(size=(n,))     # ~~~ the inner bias
c = 0.4                             # ~~~ the outer bias

if exercise_mode:
    def ReLU(x):
        # YOUR CODE HERE
        return #the output at x of a ReLU activation function
    def model(x):
        # YOUR CODE HERE
        # NOTE: please allow x to be *either* a scalar like 1.2 or a vector like x=np.linspace(-1,1,100) (see `Check that it "works on both scalars and vectors"` below)
        return #the output at x of a shallow univariate network with ReLU activation, 10 hidden units
else:
    from answers_680.answers_week_8 import ReLU
    model = build_shallow_network(a,w,b,c,ReLU)
    del ReLU

#
# ~~~ `model` should be a univariate function; let's graph it!
x = np.linspace(-3,3,1001)
_ = plt.figure(figsize=(10,5))
plt.plot( x, model(x) )
plt.title("A Shallow Univariate Network with ReLU Activation and 10 Hidden Units")
plt.grid()
plt.tight_layout()
plt.show()

#
# ~~~ Check that it matches Tom's implementation
from answers_680.answers_week_8 import univar_model as Toms_model
assert np.allclose( model(x), Toms_model(x) ) and numpy_only()

#
# ~~~ Check that it "works on both scalars and vectors" that is, works when *either* x=1.2 or x=[1.2,2.2]
model_at_one_point = model(1.2)
model_at_another_point = model(2.2)
assert np.all( model([1.2,2.2])==[model_at_one_point,model_at_another_point] ) and numpy_only()



### ~~~
## ~~~ EXERCISE 2 of 2: Using numpy define a bivariate shallow neural network with ReLU activation and 10 hidden units that works on pairs and lists of pairs
### ~~~

#
# ~~~ Use these parameters
np.random.seed(680)                 # ~~~ for reproducibility
n = 10                              # ~~~ number of hidden units
W = np.random.normal(size=(n,2))    # ~~~ the inner weights
a = np.random.normal(size=(n,))     # ~~~ the outer weights
b = np.random.normal(size=(n,))     # ~~~ the inner bias
c = 0.4                             # ~~~ the outer bias

if exercise_mode:
    def model(x):
        # NOTE: allow x to be *either* a pair like [1,2] or a list of such pairs like [ [1,2], [3,4], [5,6] ]
        return # the output at x of a shallow univariate network with ReLU activation, 10 hidden units
else:
    from answers_680.answers_week_8 import ReLU
    model = build_shallow_network(a,W,b,c,ReLU)
    del ReLU

#
# ~~~ `model` should be a bivariate function; let's graph it!
x = np.linspace(-3,3,501)
y = np.linspace(-3,3,501)
func_surf(x,y,model)

#
# ~~~ Check that it matches Tom's implementation
random_X = np.random.normal(size=(1000,2))
from answers_680.answers_week_8 import bivar_model as Toms_model
assert np.allclose(model(random_X), Toms_model(random_X) ) and numpy_only()

#
# ~~~ Check that it works when passed *either* a pair or a list of pairs: that is, *either* x=[1,2] or x=[[1,2],[3,4],[5,6]]
one_point = [0,1]
another_point = [1,2]
model_at_one_point = model(one_point)
model_at_another_point = model(another_point)
two_points = [[0,1], [1,2]]
assert np.all( model(two_points)==np.array([model_at_one_point,model_at_another_point]) ) and numpy_only()
