
# ~~~ Tom Winckelman wrote this; maintained at: https://github.com/ThomasLastName/labs_680

exercise_mode = False   # ~~~ see https://github.com/ThomasLastName/labs_680?tab=readme-ov-file#usage
install_assist = False  # ~~~ see https://github.com/ThomasLastName/labs_680/blob/main/README.md#assisted-installation-for-environments-other-than-colab-recommended


### ~~~
## ~~~ Boiler plate stuff; basically just loading packages
### ~~~

#
# ~~~ Standard python libraries
import os
import numpy as np
from matplotlib import pyplot as plt

#
# ~~~ see https://github.com/ThomasLastName/labs_680/blob/main/README.md#assisted-installation-for-environments-other-than-colab-recommended
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
from quality_of_life.my_visualization_utils import points_with_curves, side_by_side_prediction_plots, GifMaker
from quality_of_life.my_numpy_utils         import generate_random_1d_data



### ~~~
## ~~~ EXERCISE 1 of 1 (medium): define a function that fits a polyonomial to the data with a user-specified regularization parameter called `penalty`
### ~~~

if exercise_mode:
    def my_univar_poly_fit( x_train, y_train, degree, penalty=0 ):
        # YOUR CODE HERE; Hint: you should just set up the appropriate matrix and call a least squares solver
        return poly, coeffs
else:
    from answers_680.answers_week_1 import my_univar_poly_fit

#
# ~~~ A helper function that prepares data identical to Fouract's in https://github.com/foucart/Mathematical_Pictures_at_a_Data_Science_Exhibition/blob/master/Python/Chapter06.ipynb
def Foucarts_training_data( m=15, sigma=1, seed=12 ):
    if seed is not None:
        np.random.seed(seed)
    x = np.random.uniform(-1,1,m)
    x.sort()
    x[0] = -1
    x[-1] = 1
    y = x**3 - 4*x**2 + x + sigma*np.random.normal(0,1,m)
    return x, y

#
# ~~~ A helper function that wraps numpy's implementation of polynomial regression
def univar_poly_fit( x, y, degree=1 ):
    coeffs = np.polyfit( x, y, degree )
    poly = np.poly1d(coeffs)
    return poly, coeffs

#
# ~~~ Validate our code by checking that our routine implements polynomial regression correctly (compare to the numpy implementation of polynmoial regression)
x_train, y_train = Foucarts_training_data()
poly, coeffs = univar_poly_fit( x_train, y_train, degree=2 )
my_poly, my_coeffs = my_univar_poly_fit( x_train, y_train, degree=2 )
x = np.linspace(-1,1,1001)
assert abs(coeffs-my_coeffs).max() + abs( poly(x)-my_poly(x) ).max() < 1e-14    # ~~~ if this passes, it means that our implementation is equivalent to numpy's

#
# ~~~ Next, validate the implementation when the regularization parameter is positive and the degree is higher
from answers_680.answers_week_1 import my_univar_poly_fit as toms_univar_poly_fit
x_train, y_train = Foucarts_training_data()
poly, coeffs = toms_univar_poly_fit( x_train, y_train, degree=7, penalty=1.1 )
my_poly, my_coeffs = my_univar_poly_fit( x_train, y_train, degree=7, penalty=1.1 )
x = np.linspace(-1,1,1001)
assert abs(coeffs-my_coeffs).max() + abs( poly(x)-my_poly(x) ).max() < 1e-14    # ~~~ if this passes, it means that our implementation is equivalent to numpy's



### ~~~
## ~~~ DEMONSTRATION 1 of 1: The problem that regularization hopes to solve
### ~~~

#
# ~~~ A helper function for polynomial regression
def univar_poly_fit( x, y, degree=1 ):
    coeffs = np.polyfit( x, y, degree )
    poly = np.poly1d(coeffs)
    return poly, coeffs

#
# ~~~ Make some data and then fit two polynomials to that data
f = lambda x: np.abs(x) # ~~~ the so called "ground truth" by which x causes y
np.random.seed(680)     # ~~~ for reproducibility
x_train,y_train,_,_ = generate_random_1d_data(f, n_train=100, noise=.15)
d,D = 2,20
simple_fit,_ = univar_poly_fit( x_train, y_train, degree=d )    # ~~~ lo degree polynomial regression
complex_fit,_ = univar_poly_fit( x_train, y_train, degree=D )   # ~~~ hi degree polynomial regression
side_by_side_prediction_plots( x_train, y_train, f, simple_fit, complex_fit, f"A Degree {d} Polynomial is Not Expressive Enough", f"A Degree {D} Polynomial is Too Wiggly" )

#
# ~~~ Now, regularize
lam = 1.75/np.sqrt(D)
regularized_fit,_ = my_univar_poly_fit( x_train, y_train, degree=D, penalty=lam )
side_by_side_prediction_plots( x_train, y_train, f, simple_fit, regularized_fit, f"A Degree {d} Polynomial is Still Not Expressive Enough", f"This Regularized Degree {D} Polynomial has 40% as Much Test Error", grid=np.linspace(-1,1,501) )

#
# ~~~ Visualize how the fitted polynomial evolves when \lambda increasees (may take a minute to run)
np.random.seed(680)
y_train += .1*np.random.random( size=y_train.shape )    # ~~~ make the data a little noiser to keep things interesting
gif = GifMaker()
lambs = np.linspace(0,1.7,150)**2
for l in lambs:     # ~~~ fit the polynomial, graph it, take a picture, erase the graph
    regularized_fit,_ = my_univar_poly_fit( x_train, y_train, degree=D, penalty=l )
    _,_ = points_with_curves( x_train, y_train, (regularized_fit,f), show=False, title=r"Progressively Increasing the Regularization Parameter $\lambda$" )
    gif.capture()
    plt.close()

gif.develop( "Regularized Polynomial Regression 680", fps=15 )



# ### ~~~
# ## ~~~ DEMONSTRATION 2 of 2: And this is what it looks like if you use splines instead of polynomials
# ### ~~~

# from answers_680.answers_week_1 import univar_spline_fit

# #
# # ~~~ Wraper that implies a notion of "degree"
# def spline_with_equidistant_knots( *args, degree, **kwargs):
#     return univar_spline_fit( *args, knots=np.linspace(-1,1,degree+1), **kwargs )

# #
# # ~~~ Make some data and then fit two polynomials to that data
# f = lambda x: np.abs(x) # ~~~ the so called "ground truth" by which x causes y
# np.random.seed(680)     # ~~~ for reproducibility
# x_train,y_train,_,_ = generate_random_1d_data(f, n_train=100, noise=.15)
# d,D = 2,20
# simple_fit,_ = spline_with_equidistant_knots( x_train, y_train, degree=d )    # ~~~ lo degree polynomial regression
# complex_fit,_ = spline_with_equidistant_knots( x_train, y_train, degree=D )   # ~~~ hi degree polynomial regression
# side_by_side_prediction_plots( x_train, y_train, f, simple_fit, complex_fit, f"A Spline with {d} Intervals of Linearity is not Expressive Enough", f"A Spline with {D} Intervals of Linearity is Too 'Wiggly'" )

# #
# # ~~~ Now, regularize
# lam = 1.75/np.sqrt(D)
# regularized_fit,_ = spline_with_equidistant_knots( x_train, y_train, degree=D, penalty=lam )
# side_by_side_prediction_plots( x_train, y_train, f, simple_fit, regularized_fit, f"A Spline with {d} Intervals of Linearity Still is not Expressive Enough", f"This Regularized Splinewith {D} Intervals of Linearity Does Well", grid=np.linspace(-1,1,501) )

# #
# # ~~~ Visualize how the fitted polynomial evolves when \lambda increasees (may take a minute to run)
# np.random.seed(680)
# y_train += .1*np.random.random( size=y_train.shape )    # ~~~ make the data a little noiser to keep things interesting
# gif = GifMaker()
# lambs = np.linspace(0,5,150)**2
# for l in lambs:     # ~~~ fit the polynomial, graph it, take a picture, erase the graph
#     regularized_fit,_ = spline_with_equidistant_knots( x_train, y_train, degree=D, penalty=l )
#     _,_ = points_with_curves( x_train, y_train, (regularized_fit,f), show=False, title=r"Progressively Increasing the Regularization Parameter $\lambda$" )
#     gif.capture()
#     plt.close()

# gif.develop( "Regularized Spline Regression 680", fps=15 )
