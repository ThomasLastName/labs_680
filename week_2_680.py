
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
# ~~~ Extra featuers if you have tqdm
try:
    from tqdm import tqdm
    use_tqdm = True
except Exception as probably_ModuleNotFoundError:
    if type(probably_ModuleNotFoundError) is ModuleNotFoundError:
        use_tqdm = False
    else:
        raise

#
# ~~~ Extra featuers if you have cvxpy
try:
    import cvxpy as cvx
    use_cvx = True
except Exception as probably_ModuleNotFoundError:
    if type(probably_ModuleNotFoundError) is ModuleNotFoundError:
        use_cvx = False
    else:
        raise

#
# ~~~ Extra features if you have sklearn installed
try:
    from sklearn.datasets import load_digits as scikit_NIST_data
    use_sklearn = True
except Exception as probably_ModuleNotFoundError:
    if type(probably_ModuleNotFoundError) is ModuleNotFoundError:
        use_sklearn = False
    else:
        raise

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
        files = [ "answers_week_2.py" ]
        intstall_Toms_code( folder, files )

#
# ~~~ Tom's helper routines (which the above block of code installs for you); maintained at https://github.com/ThomasLastName/quality_of_life
from answers_680.answers_week_2             import Foucarts_training_data
from quality_of_life.my_base_utils          import my_warn, support_for_progress_bars
from quality_of_life.my_numpy_utils         import augment
from quality_of_life.my_visualization_utils import GifMaker, abline



### ~~~
## ~~~ EXERCISE 1 of 5 (easy): Write a helper function that attaches a column of all 1's to a matrix, if there isn't a column of all 1's alreay
### ~~~

if exercise_mode:
    def augment(X):
        dimension = X.shape[0]
        return np.concatenate([ X, np.ones((dimension,1)) ], axis=1 )
else:
    pass

from quality_of_life.my_numpy_utils import augment as my_augment
X,y = Foucarts_training_data(plot=False)
assert np.isclose( augment(X), my_augment(X) ).min()
assert np.isclose( augment(X), augment(augment(X)) ).min()



### ~~~ 
## ~~~ EXERCISE 2 of 5 (easy): Compute the slope and intercept of the boundary between classification regions (that line you see plots of) for a classifier with weight vector w and bias b
### ~~~ 

if exercise_mode:
    def compute_slope_and_intercept(w,b):
        # YOUR CODE HERE
        return slope, intercept
else:
    from answers_680.answers_week_2 import compute_slope_and_intercept

w = np.array([ 3.1, -6.5 ])
b = 8.4
slope, intercept = compute_slope_and_intercept(w,b)
assert abs(slope-0.47692307692307695)<1e-10 and abs(intercept-1.2923076923076924)<1e-10

#
# ~~~ Now, we can make a helper routine that plots the data along with the boundary between classification regions
def points_with_binary_classifier_line( w, b, X_train, y_train ):
    plt.scatter( *X_train.T, c=np.where( y_train>0, "r", "b" ) )
    abline( *compute_slope_and_intercept(w,b) )
    plt.xlabel('One Feature')
    plt.ylabel('Another Feature')
    plt.grid()
    plt.tight_layout()
    plt.show()



### ~~~
## ~~~ EXERCISE 3 of 5 (hard): Implement the perceptron algorithm (without bias) in a manner that raises a `StopIteration` exception if the supplied w meets the stopping condition already
### ~~~

if exercise_mode:
    def preceptron_update_without_bias( X, y, w, random_update=False ):
        if the_algorithms_stopping_condition_is_met:
            pass # raise a StopIteration exception
        else:
            if random_update:
                pass # update based on a randomly selected misclassified index
            else:
                pass # update based on the first misclassified index (this is the most vanilla version of the algorithm)
        return # both the updated vector w^{(t+1)} and the index i used to compute it
else:
    from answers_680.answers_week_2 import preceptron_update_without_bias


X,y = Foucarts_training_data(plot=False)
w,i = preceptron_update_without_bias( X, y, np.array([1.1,-.6,]) )
correct = np.array([ 0.72127109, -1.21833634])
assert np.all(np.isclose(w,correct))

np.random.seed(680)
w,i = preceptron_update_without_bias( X, y, np.array([1.1,-.6,]), True )
correct = np.array([ 0.24517112, -1.91427784])
assert i==77 and np.all(np.isclose(w,correct))



### ~~~ 
## ~~~ DEMONSTRATION 1 of 2: The perceptron algorithm in action
### ~~~ 

#
# ~~~ The perceptron algorithm (basic demonstration)
np.random.seed(680)
X,y = Foucarts_training_data(plot=False)    # ~~~ the data
X_aug = augment(X)
w,t = np.zeros(3),0     # ~~~ initialization
while True:
    try:
        w,index = preceptron_update_without_bias(X_aug,y,w) # ~~~ perform an update
        t += 1
    except StopIteration:
        print(f"Perceptron algorithm converged after {t} iterations.")
        break

#
# ~~~ Plot the results
w,b = np.array_split(w,2)
points_with_binary_classifier_line(w,b,X,y) # ~~~ sure enough, the classification appears to be exact

#
# ~~~ The perceptron algorithm (with programming bells and whistles)
def my_perceptron( X_train, y_train, bias=True, initialization="random", random_update=False, max_iter="auto", verbose=True, progress=False, plot=False, gif_destination="perceptron_680", **kwargs ):
    #
    # ~~~ Set m to be the number of data points
    y = y_train.squeeze()
    assert y.ndim==1
    assert set(y_train)=={-1,1}
    m = len(y)
    if not X_train.shape[0]==m:
        X_train = X_train.T
    assert X_train.shape[0]==m
    #
    # ~~~ Augment the data if a bias is desired (if it's not already augmented) and set d to be the number of features
    X = augment(X_train) if bias else X_train
    d = X.shape[1]
    if plot and not (d==3 and bias):
        my_warn("The argument `plot=True` is currently only supported when `bias=True` and the number of features is 2.")
    #
    # ~~~ Deafult to more max iterations if we're not spending our time on plotting
    if max_iter=="auto":
        max_iter = 150**(2-plot)
    elif max_iter is None:
        pass
    else:
        max_iter = int(max_iter)+1  # ~~~ round up to the nearest integer
    #
    # ~~~ Do not support progress bars if tqdm is unavailable
    if not use_tqdm:
        if progress:
            my_warn("You specified `progress=True`, however this feature requires `tqdm` which was not able to be loaded. User input will be overriden to instead `progress=False`")
        progress = False
    #
    # ~~~ Handle the initialization
    if isinstance(initialization,(list,np.ndarray)):
        w = initialization
    elif initialization=="random":
        w = np.random.normal(size=(d,))/np.sqrt(d)
    elif initialization=="zeros" or initialization==0:
        w = np.zeros(shape=(d,))
    else:
        raise ValueError("Failed to convert initialization.")        
    assert w.shape == (d,)
    #
    # ~~~ Introduce plotting parameters
    if plot:
        alpha = [.05,.1,.15,.2,1]
        delay = len(alpha)
        linger = delay
        past_few = [initialization]*delay
        gif = GifMaker()
        def make_and_take_pic(i,past_few):
            #
            # ~~~ Make pic
            plt.scatter( *X_train.T, c=np.where( y_train>0, "r", "b" ) )    # ~~~ scatter plot of the data
            if i is not None:   # ~~~ mark with an "X" on the scatter plot the index i that was used to update the model (if applicable)
                plt.scatter( *X_train[i], color=("r" if y[i]>0 else "b"), marker="x", s=200 )
            for j,params in enumerate(past_few):    # ~~~ plot the classification boundary regions for the previous few iterations
                abline( *compute_slope_and_intercept( *np.array_split(params,2)), color="black", alpha=alpha[j] )
            plt.grid()
            plt.tight_layout()
            #
            # ~~~ Take pic
            gif.capture()
    #
    # ~~~ Do the iterations
    t = 0
    i = None
    early = False
    with support_for_progress_bars():
        if progress:
            pbar = tqdm( desc="Percptron algorithm", total=max_iter )
        while (max_iter is None) or t<=max_iter:
            #
            # ~~~ Capture the image of the state before updating
            if plot:
                make_and_take_pic(i,past_few)
            #
            # ~~~ Update the model parameters
            try:
                w,i = preceptron_update_without_bias(X,y,w,random_update)
                t += 1
            except Exception as error:
                if type(error) is StopIteration:
                    early = True
                    break
                else:
                    raise error
            #
            # ~~~ Update the lines that will be shown in the image
            if plot:
                _ = past_few.pop(0)
                past_few.append(w)
            #
            # ~~~ Update the progress bar, if applicable
            if progress:
                n_correct = np.sum( np.sign(X@w)==np.sign(y) )
                pbar.set_description("Accuracy is " +f"{(n_correct/m):.3}".rjust(5) + f" after {t} iterations.")
                pbar.update()
        if progress:
            pbar.close()
        if t>max_iter:
            my_warn(f"Perceptron algorithm did not converge in {max_iter} iterations")
    #
    # ~~~ Conclude
    if verbose and early:
        print(f"Converged after {t} iterations.")
    if plot:
        i = None
        alpha = [1]
        past_few = [w]
        for _ in range(linger):
            make_and_take_pic(i,past_few)
        gif.develop( destination=gif_destination, verbose=verbose, **kwargs )
    return ( w[:-1], w[-1], t ) if bias else (w,t)

#
# ~~~ Deterministic perceptron with a (bad) random initialization and linearly separable data
np.random.seed(680)
initial_w_and_b = -np.random.normal(size=(3,))/np.sqrt(3)
X,y = Foucarts_training_data( plot=False )
w,b,T = my_perceptron( X, y, initialization=initial_w_and_b, fps=3, plot=True, gif_destination="680 perceptron det good" )
points_with_binary_classifier_line(w,b,X,y)

#
# ~~~ Stochastic perceptron with the same (bad) random initialization and linearly separable data
np.random.seed(680)
initial_w_and_b = -np.random.normal(size=(3,))/np.sqrt(3)
X,y = Foucarts_training_data( plot=False )
w,b,T = my_perceptron( X, y, initialization=initial_w_and_b, fps=3, random_update=True, plot=True, gif_destination="680 stochastic stoch good" )
points_with_binary_classifier_line(w,b,X,y)

#
# ~~~ Deterministic perceptron with the (bad) random initialization and __non__-separable data
np.random.seed(680)
initial_w_and_b = -np.random.normal(size=(3,))/np.sqrt(3)
X,y = Foucarts_training_data( plot=False, tag="nonseparable" )
w,b,T = my_perceptron( X, y, initialization=initial_w_and_b, fps=15, plot=True, progress=True, gif_destination="680 perceptron det bad" )
points_with_binary_classifier_line(w,b,X,y)

#
# ~~~ Stochastic perceptron with the same (bad) random initialization and __non__-separable data
np.random.seed(680)
initial_w_and_b = -np.random.normal(size=(3,))/np.sqrt(3)
X,y = Foucarts_training_data( plot=False, tag="nonseparable" )
w,b,T = my_perceptron( X, y, initialization=initial_w_and_b, fps=15, random_update=True, plot=True,  progress=True, gif_destination="680 perceptron stoch bad" )
points_with_binary_classifier_line(w,b,X,y)



### ~~~
## ~~~ EXERCISE 4 of 5 (easy): A numerical test for linear separability
### ~~~

#
# ~~~ Given data X and y, build the matrix A and vector b for which linear separability is equivalent to "\exists w : Aw \geq b"
if exercise_mode:
    def training_data_to_feasibility_parameters(X_train,y_train):
        # YOUR CODE HERE; Hint: see equation (4.2) in the text
        return A,b  # the things for which we want to run the linear feasibility program "find x with Ax \geq b"
else:
    from answers_680.answers_week_2 import training_data_to_feasibility_parameters

#
# ~~~ Try to find a vector x satistfying Ax \geq b
def linear_feasibility_program( A, b, solver=cvx.ECOS ):
    m,n = A.shape               # ~~~ get the number of rows and columns of A
    assert b.shape==(m,1)       # ~~~ safety feature
    x = cvx.Variable((n,1))     # ~~~ define the optimization variable x
    constraints = [A @ x >= b]  # ~~~ define the constraints of the problem
    objective = cvx.Minimize(cvx.norm(x,2))         # ~~~ objective function can be anything for linear feasibility problem
    problem = cvx.Problem(objective, constraints)   # ~~~ put it all together into a complete minimization program
    problem.solve(solver=solver)              # ~~~ try to solve it
    return x.value if problem.status==cvx.OPTIMAL else None

#
# ~~~ Given data X and y, build the matrix A and vector b for which linear separability is equivalent to Aw \geq b, and test the latter
def test_for_linear_separability( X_train, y_train, verbose=True, **kwargs ):
    assert set(y_train)=={-1,1}
    A,b = training_data_to_feasibility_parameters(X_train,y_train)
    w_tilde = linear_feasibility_program( A, b, **kwargs )
    if w_tilde is None:
        if verbose:
            print("The data is not linearly separable.")
        return None, None
    else:
        if verbose:
            print("The data is linearly separable.")
        w_tilde = w_tilde.flatten()
        w,b = w_tilde[:-1], w_tilde[-1]
    return w,b

#
# ~~~ Our linear feasibility program was in fact using the objective function for hard svm; see equation (4.3) in the text
def hard_svm(*args, verbose=True, **kwargs):
    return test_for_linear_separability( *args, verbose=verbose, **kwargs )

#
# ~~~ Test whether or not the exercise was completed successfully
if use_cvx:
    #
    # ~~~ Test it on linearly separable data
    X,y = Foucarts_training_data(plot=False)
    w,b = test_for_linear_separability(X,y)
    points_with_binary_classifier_line(w,b,X,y)
    assert abs(b+1.1512453137620453) + abs(w-np.array([4.06811412,-2.36310124])).max() < 1e-8
    #
    # ~~~ Test it on data that is not linearly seprable 
    X,y = Foucarts_training_data(plot=False,tag="nonseparable")
    w,b = test_for_linear_separability(X,y)
    assert w is None and b is None



### ~~~
## ~~~ EXERCISE 5 of 5 (medium): Define the class  HalfSpaceClassifier that has an __init__ and __call__ method, as well as the attributes w, b, and n_features 
### ~~~

if exercise_mode:
    class HalfSpaceClassifier:
        def __init__(arguments):
            # YOUR CODE HERE: define w, b, and n_features as attrebutes
            pass
        def __call__(arguments):
            # YOUR CODE HERE: define how w and b are used to clasify new data
            pass
else:
    from answers_680.answers_week_2 import HalfSpaceClassifier

#
# ~~~ Measure the accuracy of our classifier on test data
def score( classifier, X_test, y_test ):
    return np.sum( classifier(X_test)==np.sign(y_test) ) / X_test.shape[0]

#
# ~~~ Check that the class works as intended
X,y = Foucarts_training_data(plot=False)
w,b = hard_svm(X,y,verbose=False)
my_classifier = HalfSpaceClassifier(w,b)    # ~~~ callable; classifier(X) should return a vector with i-th entry to be the binary class predicted of X[i,:]
assert score(my_classifier,X,y)==1          # ~~~ assert that our classifier is 100% accurate on the training data
assert hasattr(my_classifier,"n_features")  # ~~~ check that my_classifier indeed has an n_feaures attribute



### ~~~
## ~~~ DEMONSTRATION 2 of 2: Real world data
### ~~~

#
# ~~~ Look at images of 0's and 1's
if use_sklearn:
    plt.gray()
    plt.close()
    X,y = scikit_NIST_data(n_class=2,return_X_y=True)  # https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_digits.html
    assert set(y)=={0, 1}
    y = 2*y-1
    assert set(y)=={-1, 1}
    #
    # ~~~ A little plotting routine
    def show(image,flat=True):
        if flat:
            d = int(np.sqrt(len(image)))
            image = image.reshape(d,d)
        plt.matshow(image)
        plt.show()
    #
    # ~~~ This data is so big that we can only plot one data point at a time!
    show(X[0])
    show(X[1])
    show(X[2])

#
# ~~~ Test whether or not images of 0's and 1's are linearly separable
if use_cvx:
    #
    # ~~~ Check whether or not this data is linearly separable (it is!)
    X,y = scikit_NIST_data(n_class=2,return_X_y=True)  # https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_digits.html
    assert set(y)=={0, 1}
    y = 2*y-1
    assert set(y)=={-1, 1}
    w_svm,b_svm = test_for_linear_separability(X,y)
    #
    # ~~~ Observe that perceptron algorithm converges faster than the worst case scenario with an initialization of 0
    m,d = X.shape
    upper_bound_with_w_chosen_by_hard_svm = (np.sum(w_svm**2)+b_svm**2) * np.sum( augment(X)**2, axis=1 ).max().item()
    w,b,T = my_perceptron( X, y, max_iter=upper_bound_with_w_chosen_by_hard_svm, initialization=0, verbose=False )
    upper_bound_with_w_chosen_by_perceptron = (np.sum(w**2)+b**2) * np.sum( augment(X)**2, axis=1 ).max().item()
    print(f"The perception with initialization zero converged in {T} iterations, which is fewer than the upper bounds {int(upper_bound_with_w_chosen_by_hard_svm)+1} and {int(upper_bound_with_w_chosen_by_perceptron)+1}.")
    #
    # ~~~ For non-zero initializations, the upper bound needs to be modified
    np.random.seed(680)
    w0 = np.random.normal(size=(d+1,))/(d+1)
    ub_svm = np.linalg.norm( np.append(w_svm,b_svm) - w0 )**2 * np.sum( augment(X)**2, axis=1 ).max().item()
    ub_perceptron  = np.linalg.norm( np.append(w,b) - w0 )**2 * np.sum( augment(X)**2, axis=1 ).max().item()
    w,b,T = my_perceptron( X, y, max_iter=upper_bound_with_w_chosen_by_hard_svm, initialization=w0, verbose=False )
    print(f"The perception with random initialization converged in {T} iterations, which is fewer than the modified upper bounds {int(ub_svm)+1} and {int(ub_perceptron)+1}.")

#
# ~~~ On the other hand, images multiples of 3 fail to be linearly separable from images of digits that are not multiples of 3
if use_cvx:
    X,y = scikit_NIST_data(n_class=10,return_X_y=True)  # https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_digits.html
    assert set(y)=={0,1,2,3,4,5,6,7,8,9}
    is_multiple_of_3 = y%3==0
    y = 2*is_multiple_of_3-1
    w,b = test_for_linear_separability(X,y)


# todo: show a linear combination of images
# todo: test whether or not other digits like 8's vs. multiples of three are linearly separable
# todo: perceptron on MNIST 0's and 1's
# todo experiment with different initializations in the perceptron algorithm
# experiment with how sharp empirically is our estimate T < R^2\|\tilde{w}^\ast\|^2; try scaling the data by a factor R>0 and see how/whether this effects the number of iterations
# are the 0's and 1's of MNIST linearly separable?
# what happens when you run the perceptron algorithm on some data that is not linearly separable?
# todo: what is the effect on the algorithm if we permute the positions of the classes?

