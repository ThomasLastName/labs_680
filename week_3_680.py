
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
from quality_of_life.my_visualization_utils import points_with_curves, side_by_side_prediction_plots, GifMaker, buffer
from quality_of_life.my_numpy_utils         import generate_random_1d_data



### ~~~
## ~~~ EXERCISE 1 of 2 (medium): define a function that fits a polyonomial to the data with a user-specified regularization parameter called `penalty`
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
## ~~~ DEMONSTRATION 1 of 2: The problem that regularization hopes to solve
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
if False:
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



### ~~~
## ~~~ DEMONSTRATION 2 of 2: **Cross validation** -- a standard workflow for model selection (which in this case means selecting the appropriate polynomial degree and regularization parameter)
### ~~~

#
# ~~~ Measure how well the model does when certain subsets of the data are withheld from training (written to mimic the sklearn.model_selection function of the same name)
def cross_val_score( estimator, eventual_x_train, eventual_y_train, cv, scoring, shuffle=False, plot=False, ncol=None, nrow=None, f=None, grid=None ):
    #
    # ~~~ Boiler plate stuff, not important
    scores = []
    if plot:
        ncol = cv if ncol is None else ncol
        nrow = 1  if nrow is None else nrow
        fig,axs = plt.subplots(nrow,ncol)
        axs = axs.flatten()
        xlim = buffer(eventual_x_train)
        ylim = buffer(eventual_y_train)
        grid = np.linspace( min(xlim), max(xlim), 1001 )
    #
    # ~~~ Partition the training data
    if shuffle: # ~~~ shuffle the data before partitionint it
        reordered_indices = np.random.permutation( len(eventual_y_train) )
        eventual_x_train = eventual_x_train[reordered_indices]
        eventual_y_train = eventual_y_train[reordered_indices]
    x_val_sets = np.array_split( eventual_x_train, cv )     # ~~~ split `eventual_x_train` into `cv` different pieces
    y_val_sets = np.array_split( eventual_y_train, cv )     # ~~~ split `eventual_y_train` into `cv` different pieces
    #
    # ~~~ For each one of the pieces (say, the i-th piece) into which we split our data set...
    for i in range(cv):
        #
        # ~~~ Use the i-th subset of our data (which is 1/cv percent of our data) to train a model
        x_train = x_val_sets[i]
        y_train = y_val_sets[i]
        model = estimator( x_train, y_train )
        #
        # ~~~ Use the remaining cv-1 parts of our data (i.e., (cv-1)/cv percent of our data) to test the fit
        x_test = np.concatenate( x_val_sets[:i] + x_val_sets[(i+1):] )  # ~~~ all the data we didn't train on
        y_test = np.concatenate( y_val_sets[:i] + y_val_sets[(i+1):] )  # ~~~ all the data we didn't train on
        scores.append(scoring( y_test, model(x_test) ))
        #
        # ~~~ Plot the model that was trained on this piece of the data, if desired (this is mostly useful for building intuition)
        if plot:
            axs[i].plot( x_train, y_train, "o", color="blue", label="Training Data" )
            axs[i].plot( x_test, y_test, "o", color="green", label="Test Data" )
            axs[i].plot( grid, model(grid), "-", color="blue", label="Predictions" )
            if (f is not None and grid is not None):
                axs[i].plot( grid, f(grid), "--", color="green", label="Ground Truth" )
            axs[i].set_xlim(xlim)
            axs[i].set_ylim(ylim)
            axs[i].grid()
            axs[i].legend()
    if plot:    # ~~~ after the loop is over, perform the final configuration of the plot, if applicable, and then render it
        fig.tight_layout()
        plt.show()
    return scores

#
# ~~~ Define the metric by which we will assess accurcay: mean squared error
def mean_squared_error( true, predicted ):
    return np.mean( (true-predicted)**2 )
    # ~~~ usually you'd load this or one of the other options from sklearn.meatrics (https://scikit-learn.org/stable/modules/model_evaluation.html#common-cases-predefined-values)
    # ~~~ we have defined it explicitly for transparency and simplicity

#
# ~~~ A simple wrapper
def poly_cv_scores( degree, x, y, penalty=None, **kwargs ):
    polynomial_regression = lambda x_train,y_train: my_univar_poly_fit( x_train, y_train, degree=degree, penalty=penalty )[0]       # ~~~ define the modeling technique    
    return cross_val_score( estimator=polynomial_regression, eventual_x_train=x, eventual_y_train=y, **kwargs ) # ~~~ do cv with this modeling technique and data

#
# ~~~ Make a slightly smaller, noisier data set to keep things interesting
np.random.seed(680)     # ~~~ for reproducibility
f = lambda x: np.abs(x) # ~~~ the so called "ground truth" by which x causes y
x_train,y_train,x_test,y_test = generate_random_1d_data(f, n_train=80, noise=.2)

#
# ~~~ Hyperparameters for the example
d,D = 5,10
lamb = 1/D

#
# ~~~ A printing routine
def print_scores(vector_of_scores,header=None):
    N = len(vector_of_scores)
    if header is not None:
        print(header)
    print( "\n".join([f"Test error {score:.4f} when trained on subset {j+1} of {N}" for j,score in enumerate(vector_of_scores)]) )

#
# ~~~ Cross validation with a high degree of polynomial regression
high_degree_scores = poly_cv_scores( degree=D, x=x_train, y=y_train, cv=3, penalty=0, scoring=mean_squared_error, plot=True )
print_scores( high_degree_scores, header= 10*"-" + f" Results of Cross Validation: degree={D}, lambda=0 " + 10*"-" )

#
# ~~~ Cross validation with a low degree of polynomial regression
low_degree_scores = poly_cv_scores( degree=d, x=x_train, y=y_train, cv=3, penalty=0, scoring=mean_squared_error, plot=True )
print_scores( low_degree_scores, header= 10*"-" + f" Results of Cross Validation: degree={d}, lambda=0 " + 10*"-" )

#
# ~~~ Cross validation with a low degree of polynomial regression
regularized_scores = poly_cv_scores( degree=D, x=x_train, y=y_train, cv=3, penalty=lamb, scoring=mean_squared_error, plot=True )
print_scores( regularized_scores, header= 10*"-" + f" Results of Cross Validation: degree={D}, lambda={lamb} " + 10*"-" )

#
# ~~~ Does a regularized degree D polynomial ever perform better than a degree 5 polynomial on this data? Use CV to find out
scores = []
possible_hyper_parameters = np.linspace(0,1.5,101)**2
for lamb in possible_hyper_parameters:
    scores.append(poly_cv_scores( degree=D, x=x_train, y=y_train, cv=3, penalty=lamb, scoring=mean_squared_error ))

#
# ~~~ Collect the results
best = np.median(scores,axis=1).argmin()    # ~~~ index for which the median error is smallest
lamb = possible_hyper_parameters[best]      # ~~~ the lambda value corresponding to that index

#
# ~~~ See the improvement? (or not!)
low_degree_fit,_ = my_univar_poly_fit( x_train, y_train, degree=d )
hi_regular_fit,_ = my_univar_poly_fit( x_train, y_train, degree=D, penalty=lamb )
side_by_side_prediction_plots( x_train, y_train, f, low_degree_fit, hi_regular_fit, f"This Simple Model has Test Error {mean_squared_error(low_degree_fit(x_test),y_test):.2}", f"This Regularized Complex Model has Test Error {mean_squared_error(hi_regular_fit(x_test),y_test):.2}" )



### ~~~
## ~~~ EXERCISE 2 of 2 (medium): Implement a full CV work flow in order to choose the best combination of degree and hyperparameter with this data. In particular, how much is \lambda in this best case? Finally, regress on the full data set with the degree and \lambda chosen by cross-validation and test the result
### ~~~

#
# ~~~ First, define our hyper-hyper-parameters: namely, the grid of hyper-parameters to be considered
possible_degree = [ j+1 for j in range(20) ]    # ~~~ our first hyper-parameter: the degree of polynomial regression
possible_lambda = np.linspace(0,1.5,101)**2     # ~~~ our second hyper-parameter: the penalization coefficient

#
# ~~~ Do CV to find the combination of degree and \lambda with smallest median error
# YOUR CODE HERE
# with cv=3, I got lamb=0.245025 and degree=4, but I found that the test error when I regressed on the whole data set still exceeds that of a simple degree 5 polynomial regression