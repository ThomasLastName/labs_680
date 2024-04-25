
# ~~~ Tom Winckelman wrote this; maintained at: https://github.com/ThomasLastName/labs_680

exercise_mode = False   # ~~~ see https://github.com/ThomasLastName/labs_680?tab=readme-ov-file#usage
install_assist = False  # ~~~ see https://github.com/ThomasLastName/labs_680/blob/main/README.md#assisted-installation-for-environments-other-than-colab-recommended


### ~~~
## ~~~ Boiler plate stuff; basically just loading packages
### ~~~

#
# ~~~ Standard python libraries
from math import log, sqrt
from warnings import warn
from tqdm.auto import trange
from matplotlib import pyplot as plt
import numpy as np

#
# ~~~ Extra feature if you have pytorch installed
try:
    import torch
    from torch import nn
    torch.set_default_dtype(torch.double)
    pytorch_is_available = True
except ModuleNotFoundError:
    pytorch_is_available

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
        files = [ "ansi.py", "my_base_utils.py", "my_numpy_utils.py", "my_visualization_utils.py" ]
        intstall_Toms_code( folder, files )
        #
        # ~~~ "Install/update" answers_680
        folder = "answers_680"
        files = [ "answers_week_1.py" ]
        intstall_Toms_code( folder, files )

#
# ~~~ Tom's helper routines (which the above block of code installs for you); maintained at https://github.com/ThomasLastName/quality_of_life
from quality_of_life.my_numpy_utils         import generate_random_1d_data, list_all_the_hat_functions
from quality_of_life.my_visualization_utils import points_with_curves, side_by_side_prediction_plots
from answers_680.answers_week_1             import univar_spline_fit



### ~~~
## ~~~ DEMONSTRATION 1 of 1: Proposition 10.3 in the special case when F=H^1([0,1]) and V=span({"hat functions at evenly spaced knots"})
### ~~~

#
# ~~~ A function that applies Proposition 10.3 with dim(V)==deg
def H1_with_V_hat( x_train, y_train, deg, tol=1e-13 ):
    #
    # ~~~ Transform the data to the interior of the unit interval [0,1]
    lo, hi = min(x_train), max(x_train)
    x_train = (x_train-lo)/(hi-lo)
    #
    # ~~~ Define the matrices G_u, G_v, and C from the Proposition
    V = list_all_the_hat_functions(knots=np.linspace(0,1,deg))
    kernel = lambda x,y: 1 + np.minimum.outer(x,y)      # ~~~ according to exercise 5.4, this is the kernel for the RKHS H_1[0,1]
    G_u = kernel(x_train,x_train)                       # ~~~ for point evaluations, we can compute this easily using the kernel
    C = np.column_stack([ v_j(x_train) for v_j in V ])  # ~~~ for point-evaluations, C_{i,j} is sipmly v_j(x_i)
    #
    # ~~~ This is how you would compute G_v (I just computed the integrals by hand as one would in a FEM class)
    # h = 1/(deg-1)
    # G_v = h*(2*np.eye(deg) - np.eye(deg,k=1) - np.eye(deg,k=-1))
    # G_v[0][0] += 1    # ~~~ the inner product in question is that from exercise 5.4
    #
    # ~~~ Apply the formulas 10.12 in the text
    try:
        b = np.linalg.lstsq(C,y_train,rcond=None)[0]    # ~~~ notice that if Cb=y then b satisfies the first equation 10.12
        assert (y_train-C@b).mean() < tol
        print("Nice! y_train is in the range of C!")
        a = np.linalg.solve(G_u,y_train-C@b)            # ~~~ and, in that case, there is no need to invert G_u, at all
    except AssertionError:
        try:
            G_u_inv = np.linalg.inv(G_u)
        except np.linalg.LinAlgError:
            warn("G_u matrix is singular. This is probably due to two data points being too close together.")
            raise
        try:
            b = np.linalg.solve( C.T@G_u_inv@C, C.T@G_u_inv@y_train )
        except np.linalg.LinAlgError:
            warn("The equation for b is singular. This may be due to dim(V) being too large.")
            raise
        a = G_u_inv@y_train - G_u_inv@C@b
    #
    # ~~~ Apply formula 10.11 in the text
    return lambda x, a=a, b=b: kernel((x-lo)/(hi-lo),x_train)@a + np.column_stack([ phi((x-lo)/(hi-lo)) for phi in V ])@b

#
# ~~~ Make up some fake data
np.random.seed(680)
f = lambda x: np.sin(2*np.pi*x)
n_train = 40
x_train, y_train, x_test, y_test = generate_random_1d_data( f, n_train=n_train )

#
# ~~~ Apply the fitting method of Proposition 10.3 with this choice of V and ambeint space H
dim_of_V = 10
Delta = H1_with_V_hat( x_train, y_train, deg=dim_of_V )

#
# ~~~ Do spline regression, for comparison
spline,_ = univar_spline_fit( x_train, y_train, knots=np.linspace(-1,1,dim_of_V) )

#
# ~~~ Look at the two
side_by_side_prediction_plots( x_train, y_train, f, Delta, spline, "Optimal Recovery (Prop 10.3 with V = 'certain linear splines')", "Ordinary Regression on the same V" )

#
# ~~~ A relu network, for comparison
if pytorch_is_available:
    torch.manual_seed(680)  # ~~~ try playing with this seed to see how the initialization affects the outcome
    #
    # ~~~ Instantiate an untrained neural network
    model = nn.Sequential(
            nn.Unflatten( dim=-1, unflattened_size=(-1,1) ),
            nn.Linear(1, 20),
            nn.ReLU(),
            nn.Linear(20, 20),
            nn.ReLU(),
            nn.Linear(20, 1),
            nn.Flatten()
        )
    #
    # ~~~ Prepare for training
    loss_fn = nn.MSELoss()
    eta = 1e-1
    x_t = torch.from_numpy(x_train)
    y_t = torch.from_numpy(y_train).reshape(-1,1)
    optimizer = torch.optim.Adam( model.parameters(), lr=eta )
    #
    for iter in trange(1000):
        loss = loss_fn( model(x_t), y_t ) 
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    #
    # ~~~ Look at the result
    f_t = lambda x: f(x.numpy())
    points_with_curves( x_train, y_train, (model,f_t), title="A ReLU Network" )


#