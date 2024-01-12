
# ~~~ Tom Winckelman wrote this; maintained at: https://github.com/ThomasLastName/labs_680

exercise_mode = False   # ~~~ when `False`, my solutions to exercises will be imported; otherwise, you'll need to add code to this file, yourself (Ctrl+F: EXERCISE)
install_assist = False
confirm_permission_to_modify_files = not install_assist


### ~~~
## ~~~ Boiler plate stuff
### ~~~

#
# ~~~ Standard python libraries
import warnings
import numpy as np
from matplotlib import pyplot as plt

#
# ~~~ Extra features if you have tensorflow installed
try:
    import os               # ~~~ used once for saving/ loading data in the example of CV for NN architecture selection
    import sys                  # ~~~ used in conjunction with os that one time
    from itertools import product   # ~~~ used for conviently assembling a list of candidate NN architectures in the example of CV for NN's
    import tensorflow as tf             # ~~~ one of the major python machine learning libraries
    use_tensorflow = True
except Exception as probably_ModuleNotFoundError:
    if type(probably_ModuleNotFoundError) is ModuleNotFoundError:
        use_tensorflow = False

#
# ~~~ In order to reproduce the neural network CV that I ran before lab, you'll "need" the package alive_progres (otherwise, you'll just need to sligly edit my code)
try:
    from alive_prgress import alive_bar    # ~~~ sorry, I'm gonna make you install this package lol
    use_progress_bar = True
except Exception as probably_ModuleNotFoundError:
    if type(probably_ModuleNotFoundError) is ModuleNotFoundError:
        use_progress_bar = False

#
# ~~~ An automation of th process "Installation Using the Graphical Interface" described at https://github.com/ThomasLastName/labs_680?tab=readme-ov-file#installation-using-the-graphical-interface-not-recommended
if install_assist and confirm_permission_to_modify_files:
    import os
    import sys
    from urllib.request import urlretrieve
    #
    # ~~~ Define a routine that downloads a raw file from GitHub and locates it at a specified path
    def download_dotpy_from_GitHub_raw( url_to_raw, file_name, name_of_desired_folder, desired_folder_in_Lib=True, verbose=True ):
        #
        # ~~~ Put together the appropriate path
        python_directory = os.path.dirname(sys.executable)  # ~~~ as far as I understand, this is basically where python is installed on your computer
        if desired_folder_in_Lib:
            folder_path = os.path.join( "Lib", name_of_desired_folder )
            folder_path = os.path.join( python_directory, folder_path )
        else:
            folder_path = os.path.join( python_directory, name_of_desired_folder )
        file_path = os.path.join( folder_path, file_name )
        #
        # ~~~ Create the folder if it doesn't already exist
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
            if verbose:
                print("")
                print(f"Folder '{name_of_desired_folder}' created at '{folder_path}'.")
                print("")
        #
        # ~~~ Download that file and place it at the path `file_path`, overwritting a file of the same name in the same location, if one exists
        preffix = "Updated" if os.path.exists(file_path) else "Created"
        urlretrieve( url_to_raw, file_path )
        if verbose:
            print( preffix + f" file {file_name} at {file_path}" )
    #
    # ~~~ A routine that downloads from Tom's GitHub repos
    def intstall_Toms_code( folder_name, files, repo_name=None ):
        if repo_name is None:
            repo_name = folder_name
        base_url = "https://raw.githubusercontent.com/ThomasLastName/" + repo_name + "/main/"
        for file_name in files:
            download_dotpy_from_GitHub_raw( url_to_raw=base_url+file_name, file_name=file_name, name_of_desired_folder=folder )
    #
    # ~~~ "Install/update" quality_of_life
    folder = "quality_of_life"
    files = [ "ansi.py", "my_base_utils.py", "my_keras_utils.py", "my_numpy_utils.py", "my_visualization_utils.py" ]
    intstall_Toms_code( folder, files )
    #
    # ~~~ "Install/update" answers_680
    folder = "answers_680"
    files = [ "answers_week_1.py" ]
    intstall_Toms_code( folder, files )

#
# ~~~ Toms helper routines; maintained at https://github.com/ThomasLastName/quality_of_life
from quality_of_life.my_visualization_utils import points_with_curves, buffer
from quality_of_life.my_numpy_utils         import generate_random_1d_data, my_min, my_max
from quality_of_life.my_base_utils          import colored_console_output, support_for_progress_bars    # ~~~ optional: print outputs in green
colored_console_output(warn=False)
if use_tensorflow:
    from quality_of_life.my_keras_utils     import keras_seed, make_keras_network   # ~~~ optional: only necessary for the examples involving neural networks



### ~~~ 
## ~~~ DEMONSTRATION 1 of 4: What Underfitting and Overfitting Look Like (reproduce https://github.com/foucart/Mathematical_Pictures_at_a_Data_Science_Exhibition/blob/master/Python/Chapter01.ipynb)
### ~~~

#
# ~~~ A helper function for polynomial regression
def univar_poly_fit( x, y, degree=1 ):
    coeffs = np.polyfit( x, y, degree )
    poly = np.poly1d(coeffs)
    return poly, coeffs

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

#
# ~~~ A helper routine for plotting (and thus comparing) results
def side_by_side_prediction_plots( x, y, true_fun, pred_a, pred_b, title_a="One Model", title_b="Another Model", like_Foucart=False ):
    fig,(ax2,ax20) = plt.subplots(1,2)
    #
    # ~~~ `None` reverts to default behavior of `points_with_curves, otherwise use what Fouract did in https://github.com/foucart/Mathematical_Pictures_at_a_Data_Science_Exhibition/blob/master/Python/Chapter01.ipynb
    fig,ax2 = points_with_curves(
            x = x, 
            y = y, 
            grid = np.linspace(-1,1,1001) if like_Foucart else None, 
            curves = (pred_a,true_fun), 
            title = title_a, 
            xlim = [-1.1,1.1] if like_Foucart else None, 
            ylim = [-1.3,5.3] if like_Foucart else None, 
            show = False, 
            fig = fig, 
            ax = ax2
        )
    fig,ax20 = points_with_curves(
            x = x, 
            y = y, 
            grid = np.linspace(-1,1,1001) if like_Foucart else None, 
            curves = (pred_b,true_fun), 
            title = title_b, 
            xlim = [-1.1,1.1] if like_Foucart else None, 
            ylim = [-1.3,5.3] if like_Foucart else None, 
            show = False, 
            fig = fig, 
            ax = ax20
        )
    fig.tight_layout()
    plt.show()

#
# ~~~ Retrieve our data and train our two models, then plot the results (reproduces https://github.com/foucart/Mathematical_Pictures_at_a_Data_Science_Exhibition/blob/master/Python/Chapter01.ipynb)
x_train, y_train = Foucarts_training_data()
d,D = 2,20
quadratic_fit,_ = univar_poly_fit( x_train, y_train, degree=d )     # ~~~ degree 2 polynomial regression
dodeca_fit,_ = univar_poly_fit( x_train, y_train, degree=D )        # ~~~ degree 20 polynomial regression
f = lambda x: np.abs(x)                                             # ~~~ the so called "ground truth" by which x causes y
side_by_side_prediction_plots( x_train, y_train, f, quadratic_fit, dodeca_fit, f"Underfitting with a Degree {d} Polynomial", f"Overfitting with a Degree {D} Polynomial", like_Foucart=True )



### ~~~
## ~~~ EXERCISE 1 of 2 (easy): A Simpler Model is *not* Always Best
### ~~~

if exercise_mode:
    #
    # ~~~ Create some data with a non-zero level of noise in the y_train such that a degree 10 polynomial regression outperforms degree 4 polynomial regression
    my_x_train = None           # ~~~ 1d numpy array
    my_y_train = None           # ~~~ 1d numpy array
    my_ground_truth = None      # ~~~ function that acts on 1d numpy arrays
    my_explanation_4 = None     # ~~~ text string
    my_explanation_10 = None    # ~~~ text string
else:
    #
    # ~~~ Load my values for these variables from the file `ch1.py`` in the folder `answers_680` (this requires said folder to be within your python's `Lib` folder)
    from answers_680.answers_week_1 import x_train_a        as my_x_train
    from answers_680.answers_week_1 import y_train_a        as my_y_train
    from answers_680.answers_week_1 import f_a              as my_ground_truth
    from answers_680.answers_week_1 import explanation_a_4  as my_explanation_4
    from answers_680.answers_week_1 import explanation_a_10 as my_explanation_10


quartic_fit,_ = univar_poly_fit( my_x_train, my_y_train, degree=4 )     # ~~~ degree 4 polynomial regression
dodeca_fit,_ = univar_poly_fit( my_x_train, my_y_train, degree=10 )     # ~~~ degree 10 polynomial regression
side_by_side_prediction_plots( my_x_train, my_y_train, my_ground_truth, quartic_fit, dodeca_fit, my_explanation_4, my_explanation_10 )

if exercise_mode:
    #
    # ~~~ Create some data with a non-zero level of noise in the y_train such that a degree 10 polynomial regression outperforms degree 4 polynomial regression
    my_x_train = None           # ~~~ 1d numpy array
    my_y_train = None           # ~~~ 1d numpy array
    my_ground_truth = None      # ~~~ function that acts on 1d numpy arrays
    my_explanation_4 = None     # ~~~ text string
    my_explanation_10 = None    # ~~~ text string
else:
    #
    # ~~~ Load my values for these variables from the file `ch1.py`` in the folder `answers_680` (this requires said folder to be within your python's `Lib` folder)
    from answers_680.answers_week_1 import x_train_b        as my_x_train
    from answers_680.answers_week_1 import y_train_b        as my_y_train
    from answers_680.answers_week_1 import f_b              as my_ground_truth
    from answers_680.answers_week_1 import explanation_b_4  as my_explanation_4
    from answers_680.answers_week_1 import explanation_b_10 as my_explanation_10

quartic_fit,_ = univar_poly_fit( my_x_train, my_y_train, degree=4 )     # ~~~ degree 4 polynomial regression
dodeca_fit,_ = univar_poly_fit( my_x_train, my_y_train, degree=10 )     # ~~~ degree 10 polynomial regression
side_by_side_prediction_plots( my_x_train, my_y_train, my_ground_truth, quartic_fit, dodeca_fit, my_explanation_4, my_explanation_10 )



### ~~~
## ~~~ DEMONSTRATION 2 of 4: The Present Discussion Applies to Neural Networks, too, as they are Also an Instance of Empirical Risk Minimization
### ~~~

#
# ~~~ An analog to poly_fit for neural networks
def make_and_train_1d_network( x_train, y_train, hidden_layers, epochs=20, verbose=0, reproducible=True, activations="tanh" ):
    if reproducible:
        keras_seed(680)
    model = make_keras_network( 1, 1, hidden_layers, activations=activations )
    model.compile(
        optimizer = "adam",
        loss = tf.keras.losses.MeanSquaredError()
    )
    _ = model.fit(
            tf.convert_to_tensor(np.expand_dims(x_train,axis=-1)),
            tf.convert_to_tensor(y_train),
            verbose=verbose,
            callbacks=[],
            epochs=epochs
        )
    return model, model.get_weights()

#
# ~~~ Reproduce with neural networks some demonstrations analogous to the ones above for polynomial regression
if use_tensorflow:  # ~~~ Note: this block takes a minute or two because the code isn't optimized
    #
    # ~~~ Example 1/4: Underfitting and Overfitting due to Training Decisions
    x_train, y_train = Foucarts_training_data()     # ~~~ for improved reproducibility
    e,E = 1000,15000
    width = 6
    under_trained,_ = make_and_train_1d_network( x_train, y_train, hidden_layers=(width,), epochs=e )  # ~~~ too few epochs for this architecture
    over_trained,_ = make_and_train_1d_network( x_train, y_train, hidden_layers=(width,), epochs=E )   # ~~~ too many epochs for this architecture
    f = lambda x: np.abs(x)     # ~~~ the so called "ground truth" by which x causes y
    side_by_side_prediction_plots( x_train, y_train, f, under_trained, over_trained, f"Stopping at {e} Epochs Underfits a Shallow Width {width} Network", f"{E} Training Epochs Overfits the Same Shallow Width {width} Network" )
    #
    # ~~~ Example 2/4: A complicated/aggressive/expressive model (in this case, more training epochs) might be appropriate when the data is not too noisy
    f = lambda x: np.abs(x)                                             # ~~~ the so called "ground truth" by which x causes y
    x_train, _ = Foucarts_training_data()                               # ~~~ take only the x data
    np.random.seed(123)                                                 # ~~~ for improved reproducibility
    y_train = f(x_train) + (1/20)*np.random.random(size=x_train.size)   # ~~~ a less noisy version of the data
    e,E = 2000,20000
    width = 6
    under_trained,_ = make_and_train_1d_network( x_train, y_train, hidden_layers=(width,), epochs=e )  # ~~~ too few epochs for this architecture
    over_trained,_ = make_and_train_1d_network( x_train, y_train, hidden_layers=(width,), epochs=E )   # ~~~ too many epochs for this architecture
    side_by_side_prediction_plots( x_train, y_train, f, under_trained, over_trained, f"Even with Simple Data, {e} Epochs of Training May Underfit", f"{E-e} Additional Epochs May Help when the Data isn't Too Noisy" )
    #
    # ~~~ Example 3/4: Underfitting and Overfitting due to Architecture Decisions
    f = lambda x: np.abs(x)                                             # ~~~ the so called "ground truth" by which x causes y
    x_train, _ = Foucarts_training_data()                               # ~~~ take only the x data
    np.random.seed(123)                                                 # ~~~ for improved reproducibility
    y_train = f(x_train) + (1/20)*np.random.random(size=x_train.size)   # ~~~ a less noisy version of the data
    epochs = 2500
    w = 6
    d = 8
    shallow,_ = make_and_train_1d_network( x_train, y_train, hidden_layers=(w,), epochs=epochs )   # ~~~ too few epochs for this architecture
    deep,_ = make_and_train_1d_network( x_train, y_train, hidden_layers=[w]*d, epochs=epochs )     # ~~~ too many epochs for this architecture
    side_by_side_prediction_plots( x_train, y_train, f, shallow, deep, f"This Shallow Network Isn't Expressive Enough to Capture the Sharp Turn", f"{d} Hidden Layers (Same Training) may Overfit Even Fairly Clean Data" )
    #
    # ~~~ Example 4/4: A complicated/aggressive/expressive model (in this case, a bigger network architecture) might be appropriate when the ground truth is complex
    f = lambda x: np.exp(x)*np.cos(3*x*np.exp(x))                       # ~~~ a more complicated choice of ground truth
    np.random.seed(680)                                                 # ~~~ for improved reproducibility
    x_train, y_train, x_test, y_test = generate_random_1d_data( ground_truth=f, n_train=50, noise=0.3 )
    epochs = 2500
    w = 4
    d = 4
    shallow,_ = make_and_train_1d_network( x_train, y_train, hidden_layers=(w,), epochs=epochs )   # ~~~ too few epochs for this architecture
    deep,_ = make_and_train_1d_network( x_train, y_train, hidden_layers=[w]*d, epochs=epochs )     # ~~~ too many epochs for this architecture
    side_by_side_prediction_plots( x_train, y_train, f, shallow, deep, f"A Shallow Network May Fail to Approximate a Complex Ground Truth", f"{d} Hidden Layers (Same Training) Offers Better Approximation Power" )
    


### ~~~
## ~~~ DEMONSTRATION 3 of 4: We can Change the Hypothesis Class in More Radical Ways, too
### ~~~

#
# ~~~ ReLU networks are always piecewise linear; since the ground truth is piecewise linear, too, we expect the result to be better
if use_tensorflow:
    f = lambda x: np.abs(x)     # ~~~ the so called "ground truth" by which x causes y
    x_train, y_train, x_test, y_test = generate_random_1d_data( ground_truth=f, n_train=50, noise=0.1 )     # ~~~ increased sample size and decreased noise for a more meaningful example
    tanh_network,_ = make_and_train_1d_network( x_train, y_train, hidden_layers=(5,5), epochs=1500 )
    relu_network,_ = make_and_train_1d_network( x_train, y_train, hidden_layers=(5,5), epochs=1500, activations="relu" )
    side_by_side_prediction_plots( x_train, y_train, f, tanh_network, relu_network, r"$\tanh$ Networks are Smooth, Implying $f \notin \mathcal{H}$", r"ReLU Networks are Piecewise Linear, in Fact $f \in \mathcal{H}$" )



### ~~~
## ~~~ EXERCISE 2 of 2 (hard): Just as we can get better results using *piecewise linear neural networks*, we can linkewise get better results by using *piecewise linear polynomial regression*: implement it and see for yourself!
### ~~~

if exercise_mode:
    #
    # ~~~ Perform empirical risk minimization when H is the space of polynomials with degree at most `degree`
    def my_univar_poly_fit( x_train, y_train, degree ):
        # YOUR CODE HERE
        return fitted_polynomial, coefficients
else:
    #
    # ~~~ Load my answer
    from answers_680.answers_week_1 import my_univar_poly_fit

#
# ~~~ Validate our code by checking that our routine implements polynomial regression correctly (compare to the numpy implementation of polynmoial regression)
poly, coeffs = univar_poly_fit( x_train, y_train, degree=2 )
my_poly, my_coeffs = my_univar_poly_fit( x_train, y_train, degree=2 )
x = np.linspace(-1,1,1001)
assert abs(coeffs-my_coeffs).max() + abs( poly(x)-my_poly(x) ).max() < 1e-14    # ~~~ if this passes, it means that our implementation is equivalent to numpy's


if exercise_mode:
    #
    # ~~~ Perform empirical risk minimization when H is the space of globally continuous piecewise linear functions with break points occuring only at `knots` (a numpy array)    
    def univar_spline_fit( x_train, y_train, knots ):
        # YOUR CODE HERE
        return fitted_spline, coefficients
else:
    #
    # ~~~ Load my answer
    from answers_680.answers_week_1 import univar_spline_fit

#
# ~~~ A demonstration of both underfitting and overfitting with spline regression
f = lambda x: np.exp(x)*np.cos(3*x*np.exp(x))   # ~~~ a more complicated choice of ground truth
np.random.seed(680)                             # ~~~ for improved reproducibility
x_train, y_train, x_test, y_test = generate_random_1d_data( ground_truth=f, n_train=50, noise=0.3 )
n,N = 5,15
simple_spline,_ = univar_spline_fit( x_train, y_train, np.linspace(-1,1,n) )
complex_spline,_ = univar_spline_fit( x_train, y_train, np.linspace(-1,1,N) )
side_by_side_prediction_plots( x_train, y_train, f, simple_spline, complex_spline, r"With Splines, as Always, if $\mathcal{H}$ is too Small, we Get Underfitting", r"Likewise, if $\mathcal{H}$ is too Big (e.g., $\mathrm{dim}(\mathcal{H})$ is Large), we Get Overfitting"  )


#
# ~~~ As with ReLU networks; since the ground truth is a spline but not a polynomial, we expect better results from spline regression than polynomial regression
f = lambda x: np.abs(x)     # ~~~ the so called "ground truth" by which x causes y
np.random.seed(680)         # ~~~ for improved reproducibility
x_train, y_train, x_test, y_test = generate_random_1d_data( ground_truth=f, n_train=50, noise=0.1 )     # ~~~ increased sample size and decreased noise for a more meaningful example
poly,_ = univar_poly_fit( x_train, y_train, degree=4 )                  # ~~~ dim(H)==5 (the length of the returned `coeffs` vector)
spline,_ = univar_spline_fit( x_train, y_train, np.linspace(-1,1,3) )   # ~~~ dim(H)==4 (the length of the returned `coeffs` vector)
side_by_side_prediction_plots( x_train, y_train, f, poly, spline, r"Polynomials are Smooth, Implying $f \notin \mathcal{H}$", r"Continuous Linear Splines are Piecewise Linear, in Fact $f \in \mathcal{H}$" )



### ~~~
## ~~~ DEMONSTRATION 4 of 4: **Cross validation** -- a standard workflow for model selection (which in this case means selecting the appropriate polynomial degree)
### ~~~

warnings.filterwarnings( "ignore", message="Polyfit may be poorly conditioned" )    # ~~~ otherwise crossvalidation will spit out th[is warning a million times

#
# ~~~ Define the metric by which we will assess accurcay
def mean_squared_error( true, predicted ):
    return np.mean( (true-predicted)**2 )
    # ~~~ usually you'd load this or one of the other options from sklearn.meatrics (https://scikit-learn.org/stable/modules/model_evaluation.html#common-cases-predefined-values)
    # ~~~ we have defined it explicitly for transparency and simplicity

#
# ~~~ Measure how well the model does when certain subsets of the data are withheld from training (written to mimic the sklearn.model_selection function of the same name)
def cross_val_score( estimator, eventual_x_train, eventual_y_train, cv, scoring, shuffle=False, plot=False, ncol=None, nrow=None, f=None, grid=None ):
    #
    # ~~~ Boiler plate stuff, not important
    scores = []
    # models = []
    if plot:
        ncol = 1 if ncol is None else ncol
        nrow = cv if nrow is None else nrow
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
# ~~~ Example usage of the big bad function we just defined
x_train, y_train = Foucarts_training_data() # ~~~ for improved reproducibility
polynomial_regression = lambda x_train,y_train: univar_poly_fit( x_train, y_train, degree=2 )[0]        # ~~~ define the modeling technique
scores = cross_val_score( polynomial_regression, x_train, y_train, cv=2, scoring=mean_squared_error )   # ~~~ do cv with this modeling technique and data
print(scores)           # ~~~ returns [5.659962939942039, 434.55267825346857], meaning that the model trained on the second half of the data performed terribly

#
# ~~~ Investigate further by specifying `plot=True`
_ = cross_val_score( polynomial_regression, x_train, y_train, cv=2, scoring=mean_squared_error, plot=True )

#
# ~~~ And that's why people shuffle their data...
np.random.seed(680)     # ~~~ fix seed for improved reproducibility
scores = cross_val_score( polynomial_regression, x_train, y_train, cv=2, scoring=mean_squared_error, shuffle=True, plot=True )
print(scores)           # ~~~ returns [1.408731127467586, 1.1814320746081544]; both models fit about as well as they can given how noisy the data is



### ~~~
## ~~~ Simulate the full cross validation workflow on simulated data
### ~~~

#
# ~~~ Make enough executive decisions that all that remains is for someone to supply us with the data
def an_example_of_the_cv_workflow( x_train, y_train, x_test, y_test, plot_like_Foucart=False ):
    #
    # ~~~ Set hyperhyperparameters: those which will be used when determining the hyperparameters
    max_degree = 20
    possible_hyperparameters = np.arange(max_degree)+1      # ~~~ i.e., np.array([1,2,...,max_degree])
    n_splits = 2    # ~~~ when this is higher, the models are trained on less data but tested on more data; this will tend to make generalization more challenging and, then, simpler models will beprefered
    scores = []     # ~~~ an object in which to record the results
    #
    # ~~~ For each possible degree that we're considering for polynomial regression
    for deg in possible_hyperparameters:
        #
        # ~~~ Do cross validation
        estimator = lambda x_train,y_train: univar_poly_fit( x_train, y_train, degree=deg )[0]  # ~~~ wrapper that fits a polynomial of degree `deg` to the data y_train ~ x_train
        current_scores = cross_val_score( estimator, x_train, y_train, cv=n_splits, scoring=mean_squared_error )
        scores.append(current_scores)
    #
    # ~~~ Take the best hyperparameter and train using the full data
    best_degree = possible_hyperparameters[ np.median(scores,axis=1).argmin() ] # ~~~ lowest median "generalization" error of trianing on only a subset of the data
    best_poly,_ = univar_poly_fit( x_train, y_train, degree=best_degree )
    points_with_curves(
            x = x_train,
            y = y_train,
            curves = (best_poly,f),
            title = f"Bassed on CV, Choose Degree {best_degree} Polynomial Regression, Resulting in a Test MSE of {mean_squared_error(best_poly(x_test),y_test):.6}",
            xlim = [-1,1] if plot_like_Foucart else None,       # ~~~ `None` reverts to default settings of `points_with_curves`
            ylim = [-1.3,5.3] if plot_like_Foucart else None    # ~~~ `None` reverts to default settings of `points_with_curves`
        )
    return scores


#
# ~~~ Collect the data; IRL these 2 lines of code would not exist; instead, our company or whatever would just hand us x_train, y_train, x_test, y_test
np.random.seed(680)
x_train, y_train, x_test, y_test = generate_random_1d_data( f, n_train=50, noise=0.1 ) 
scores = an_example_of_the_cv_workflow( x_train, y_train, x_test, y_test )

#
# ~~~ As for the data we started out with...
f = lambda x: abs(x)
x_train, y_train = Foucarts_training_data()
x_test = np.linspace(-1,1,101)
y_test = np.abs(x_test)
scores = an_example_of_the_cv_workflow( x_train, y_train, x_test, y_test, plot_like_Foucart=True )

#
# ~~~ In fact, this particular instance of cross validation failed to find to the best model: the quadratic model is better
x_train, y_train = Foucarts_training_data()     # ~~~ same data as above
affine_fit,_ = univar_poly_fit( x_train, y_train, degree=1 )     # ~~~ degree 1 polynomial regression, also known as ordinary least squares
quadratic_fit,_ = univar_poly_fit( x_train, y_train, degree=2 )     # ~~~ degree 2 polynomial regression
f = lambda x: abs(x)                            # ~~~ same data as above
x_test = np.linspace(-1,1,101)                  # ~~~ same data as above
y_test = np.abs(x_test)                         # ~~~ same data as above
points_with_curves(
            x = x_train,
            y = y_train,
            curves = (affine_fit,quadratic_fit,f),  # ~~~ `points_with_curves` makes the last curve green dashed by default; deactivate this by passing `model_fit=False`
            title = f"A Degree 2 Polynomial Regression (not Chosen by CV) Results in a Test MSE of {mean_squared_error(quadratic_fit(x_test),y_test):.3} Instead of the Test MSE of {mean_squared_error(affine_fit(x_test),y_test):.3} Resulting from Degree 1 Polynomial Regression",
            xlim = [-1,1],
            ylim = [-1.3,5.3]
        )



### ~~~
## ~~~ We can do CV with neural networks, too
### ~~~

if use_tensorflow:
    #
    # ~~~ Use the same data as for polynomials
    np.random.seed(680)
    x_train, y_train, x_test, y_test = generate_random_1d_data( f, n_train=50, noise=0.1 )
    #
    # ~~~ Set (some*) hyperhyperparameters: those which will be used to determine the (some*) hyperparameters   *others are implicly fixed, e.g., the choice of activation funciton
    foo = [4,6,8,10,12,14]
    possible_architectures = foo + list(product(foo,foo)) + list(product(foo,foo,foo))
    foo = [4,8]
    possible_architectures += list(product(foo,foo,foo,foo)) + list(product(foo,foo,foo,foo,foo))  + list(product(foo,foo,foo,foo,foo,foo))
    possible_n_epochs = [300,500,1000,1500]
    possible_hyperparameters = list(product( possible_architectures, possible_n_epochs ))
    n_splits = 2
    #
    # ~~~ Add save+load functionality
    folder = os.path.join( os.path.dirname(sys.executable), 'Lib', 'answers_680' )      # ~~~ replace with your preferred path; e.g., "C:\\Users\\thoma\\Downloads" if I wanted to load/save a file from/to my Downloads folder
    file_name = 'results_of_cv_ch1.npy'
    assert os.path.exists(folder)
    file_path = os.path.join( folder, file_name )
    #
    # ~~~ Do cross validation
    i_am_ok_with_this_running_for_an_hour_or_two_because_tom_coded_it_inefficiently = False
    if use_progress_bar and i_am_ok_with_this_running_for_an_hour_or_two_because_tom_coded_it_inefficiently:
        scores = []
        current_scores = [np.nan,np.nan]
        with support_for_progress_bars():
            with alive_bar( len(possible_hyperparameters), bar="classic" ) as bar:
                #
                # ~~~ Do basically what we did for polynomials, except with a different estimator, except with extra baggage in order to suppport the progress bar
                for (architecture,n_epochs) in possible_hyperparameters:
                    bar.text(f"Now training with {architecture} for {n_epochs} epochs. Last round had {current_scores[0]:.3}, {current_scores[1]:.3}")      # ~~~ for the progress bar
                    #
                    # ~~~ Begin key functionality (c.f., `"Do cross validation" for a bunch of different polynomial degrees`)
                    estimator = lambda x_train,y_train: make_and_train_1d_network( x_train, y_train, hidden_layers=architecture, epochs=n_epochs )[0]  # ~~~ as before
                    current_scores = cross_val_score( estimator, x_train, y_train, cv=n_splits, scoring=mean_squared_error )                        # ~~~ as before
                    scores.append(current_scores)                                                                                                   # ~~~ as before
                    # ~~~ End key functionality
                    #
                    bar()                                                                                                                                   # ~~~ for the progress bar
        #
        # ~~~ Save the result of all that for future convenience
        np.save( file_path, np.array(scores) )
    else:
        #
        # ~~~ Load the results from a saved file (you must first place the saved file in the correct path)
        scores = np.load(file_path)
    #
    # ~~~ Take the best hyperparameter and train using the full data
    if use_tensorflow:
        best_hyperparameters = possible_hyperparameters[ np.median(scores,axis=1).argmin() ]    # ~~~ lowest median "generalization" error when trianed on only a subset of the data
        architecture, n_epochs = best_hyperparameters
        best_nn = make_and_train_1d_network( x_train, y_train, hidden_layers=architecture, epochs=n_epochs, verbose=0 )[0]
        points_with_curves(
                x=x_train,
                y=y_train,
                curves=(best_nn,f),
                title=f"Test MSE {mean_squared_error(best_nn(x_test),y_test):.6} | {len(architecture)} Hidden Layers | widths {architecture} | {n_epochs} epochs"
            )

# #
# # ~~~ Examine whether or not we experience double descent (we do not)
# architectures_300,  scores_300 =  [], []
# architectures_500,  scores_500 =  [], []
# architectures_1000, scores_1000 = [], []
# architectures_1500, scores_1500 = [], []

# #
# #~~~ Filter items based on the number of epochs for a more apples-to-apples comparisson
# assert len(scores)==len(possible_hyperparameters)
# for j in range(len(scores)):
#     architecture, epochs = possible_hyperparameters[j]
#     if epochs == 300:
#         scores_300.append(scores[j])
#         architectures_300.append(architecture)
#     elif epochs == 500:
#         scores_500.append(scores[j])
#         architectures_500.append(architecture)
#     elif epochs == 1000:
#         scores_1000.append(scores[j])
#         architectures_1000.append(architecture)
#     elif epochs == 1500:
#         scores_1500.append(scores[j])
#         architectures_1500.append(architecture)

# assert len(scores_300)==len(architectures_300)
# assert len(scores_500)==len(architectures_500)
# assert len(scores_1000)==len(architectures_1000)
# assert len(scores_1500)==len(architectures_1500)
# assert architectures_1500==architectures_1000==architectures_500==architectures_300
# arthitectures = [ list(architecture) for architecture in architectures_300 ]
# complexity = [ np.prod(architecture) for architecture in arthitectures ]
# # n_hidden_layers = [ len(architecture) for architecture in arthitectures ]
# # avg_width = [ np.mean(architecture) for architecture in arthitectures ]
# plt.scatter( np.log(complexity), np.median(scores_1500,axis=1) ); plt.show()