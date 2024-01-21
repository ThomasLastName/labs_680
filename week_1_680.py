
# ~~~ Tom Winckelman wrote this; maintained at: https://github.com/ThomasLastName/labs_680

exercise_mode = False   # ~~~ see https://github.com/ThomasLastName/labs_680?tab=readme-ov-file#usage
install_assist = False  # ~~~ see https://github.com/ThomasLastName/labs_680/blob/main/README.md#assisted-installation-for-environments-other-than-colab-recommended


### ~~~
## ~~~ Boiler plate stuff; basically just loading packages
### ~~~

#
# ~~~ Standard python libraries
import os
import warnings
import numpy as np
from matplotlib import pyplot as plt

#
# ~~~ Extra features if you have tensorflow installed
try:
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
        files = [ "ansi.py", "my_base_utils.py", "my_keras_utils.py", "my_numpy_utils.py", "my_visualization_utils.py" ]
        intstall_Toms_code( folder, files )
        #
        # ~~~ "Install/update" answers_680
        folder = "answers_680"
        files = [ "answers_week_1.py" ]
        intstall_Toms_code( folder, files )

#
# ~~~ Tom's helper routines (which the above block of code installs for you); maintained at https://github.com/ThomasLastName/quality_of_life
from quality_of_life.my_visualization_utils import side_by_side_prediction_plots, buffer
from quality_of_life.my_numpy_utils         import generate_random_1d_data, my_min, my_max
from quality_of_life.my_base_utils          import support_for_progress_bars 
if use_tensorflow:
    from quality_of_life.my_keras_utils     import keras_seed, make_keras_network   # ~~~ optional: only necessary for the examples involving neural networks



### ~~~ 
## ~~~ DEMONSTRATION 1 of 5: What Underfitting and Overfitting Look Like (reproduce https://github.com/foucart/Mathematical_Pictures_at_a_Data_Science_Exhibition/blob/master/Python/Chapter01.ipynb)
### ~~~

#
# ~~~ A helper function for polynomial regression
def univar_poly_fit( x, y, degree=1 ):
    coeffs = np.polyfit( x, y, degree )
    poly = np.poly1d(coeffs)
    return poly, coeffs

#
# ~~~ A helper function that prepares data identical to Fouract's in https://github.com/foucart/Mathematical_Pictures_at_a_Data_Science_Exhibition/blob/master/Python/Chapter01.ipynb
def Foucarts_training_data( m=15 ):
    np.random.seed(12)
    x_train = np.random.uniform(-1,1,m)
    x_train.sort()
    x_train[0] = -1
    x_train[-1] = 1
    y_train = abs(x_train) + np.random.normal(0,1,m)
    return x_train, y_train

#
# ~~~ Wrap the the function `side_by_side_prediction_plots` from https://github.com/ThomasLastName/quality_of_life with an on/off switch for reproducing the graphical settings of https://github.com/foucart/Mathematical_Pictures_at_a_Data_Science_Exhibition/blob/master/Python/Chapter01.ipynb
def compare_models_like_Foucart( *args, **kwargs ):
    side_by_side_prediction_plots( *args, grid=np.linspace(-1,1,1001), xlim=[-1.1,1.1], ylim=[-1.3,5.3], **kwargs )


#
# ~~~ Retrieve our data and train our two models, then plot the results (reproduces https://github.com/foucart/Mathematical_Pictures_at_a_Data_Science_Exhibition/blob/master/Python/Chapter01.ipynb)
x_train, y_train = Foucarts_training_data()
d,D = 2,20
quadratic_fit,_ = univar_poly_fit( x_train, y_train, degree=d )     # ~~~ degree 2 polynomial regression
dodeca_fit,_ = univar_poly_fit( x_train, y_train, degree=D )        # ~~~ degree 20 polynomial regression
f = lambda x: np.abs(x)                                             # ~~~ the so called "ground truth" by which x causes y
compare_models_like_Foucart( x_train, y_train, f, quadratic_fit, dodeca_fit, f"Underfitting with a Degree {d} Polynomial", f"Overfitting with a Degree {D} Polynomial" )


#
# ~~~ ERM with a bigger hypothesis class will be more data hungry, but will perform better if you can satisfy its appetite; in other words, with enough data both models do as well as possible, though how much is "enough" depends on the hypothesis class, as does how good is the outcome "as well as possible"
d,D = 2,20
md,mD = 1200,100000
x_train, y_train = Foucarts_training_data(m=md)
more_x_train, more_y_train = Foucarts_training_data(m=mD)
quadratic_fit,c = univar_poly_fit( x_train, y_train, degree=d )          # ~~~ degree 0 polynomial regression
dodeca_fit,_ = univar_poly_fit( more_x_train, more_y_train, degree=D )  # ~~~ degree 20 polynomial regression
f = lambda x: np.abs(x)                                                 # ~~~ the so called "ground truth" by which x causes y
side_by_side_prediction_plots( x_train, y_train, f, quadratic_fit, dodeca_fit, f"With m={md}, Degree {d} Regression Does About as Well as Possible", f"With m={mD}, Degree {D} Regression Does About as Well as Possible", other_x=more_x_train, other_y=more_y_train, grid=np.linspace(-1,1,1000), xlim=[-1,1], ylim=[-2,4] )


#
# ~~~ Degree 4 polynomial regression still gives plausibale results, while degree 10 does does not do much better than 20
x_train, y_train = Foucarts_training_data()
d,D = 4,10
quartic_fit,_ = univar_poly_fit( x_train, y_train, degree=4 )   # ~~~ degree 4 polynomial regression
deca_fit,_ = univar_poly_fit( x_train, y_train, degree=10 )     # ~~~ degree 10 polynomial regression
f = lambda x: np.abs(x)                                         # ~~~ the so called "ground truth" by which x causes y
side_by_side_prediction_plots( x_train, y_train, f, quartic_fit, deca_fit, f"A Degree {d} Polynomial Still Gives Relatively Palusible Predictions", f"A Degree {D} is Implausibly Wiggly" )



### ~~~
## ~~~ EXERCISE 1 of 4 (easy): A Simpler Model is *not* Always Best
### ~~~

if exercise_mode:
    #
    # ~~~ Create a data set with sample size <20 and an average level of noise >0.01 in the y_train such that degree 10 regression outperforms degree 4 regression
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

quartic_fit,_ = univar_poly_fit( my_x_train, my_y_train, degree=4 ) # ~~~ degree 4 polynomial regression
deca_fit,_ = univar_poly_fit( my_x_train, my_y_train, degree=10 )   # ~~~ degree 10 polynomial regression
x_test = np.linspace(-1,1,1000)
y_test = my_ground_truth(x_test)
error_10 = ( (deca_fit(x_test)-y_test)**2 ).mean()
error_4 = ( (quartic_fit(x_test)-y_test)**2 ).mean()
higher_degree_is_better = error_10<error_4
noise = abs(my_y_train-my_ground_truth(my_x_train)).mean() > 1e-2   # ~~~ some noise
small_sample_size = len(y_train)<20                                 # ~~~ sample size <20
if noise and small_sample_size and higher_degree_is_better:
    side_by_side_prediction_plots( my_x_train, my_y_train, my_ground_truth, quartic_fit, deca_fit, my_explanation_4, my_explanation_10 )
else:
    raise ValueError("Exercise instructions were not met.")

if exercise_mode:
    #
    # ~~~ Create a data set with sample size <100 and an average level of noise >0.1 in the y_train such that degree 10 regression outperforms degree 4 regression
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

quartic_fit,_ = univar_poly_fit( my_x_train, my_y_train, degree=4 ) # ~~~ degree 4 polynomial regression
deca_fit,_ = univar_poly_fit( my_x_train, my_y_train, degree=10 )   # ~~~ degree 10 polynomial regression
x_test = np.linspace(-1,1,1000)
y_test = my_ground_truth(x_test)
error_10 = ( (deca_fit(x_test)-y_test)**2 ).mean()
error_4 = ( (quartic_fit(x_test)-y_test)**2 ).mean()
higher_degree_is_better = error_10<error_4
noise = abs(my_y_train-my_ground_truth(my_x_train)).mean() > 1e-1   # ~~~ some noise
small_sample_size = len(y_train)<100                                # ~~~ sample size <100
if noise and small_sample_size and higher_degree_is_better:
    side_by_side_prediction_plots( my_x_train, my_y_train, my_ground_truth, quartic_fit, deca_fit, my_explanation_4, my_explanation_10 )
else:
    raise ValueError("Exercise instructions were not met.")



### ~~~
## ~~~ DEMONSTRATION 2 of 5: The Present Discussion Applies to Neural Networks, too, as they are Also an Instance of Empirical Risk Minimization
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
## ~~~ DEMONSTRATION 3 of 5: We can Change the Hypothesis Class in More Radical Ways, too
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
## ~~~ EXERCISE 2 of 4 (hard): Just as we can get better results using *piecewise linear neural networks*, we can linkewise get better results by using *piecewise linear polynomial regression*: implement it and see for yourself!
### ~~~

if exercise_mode:
    #
    # ~~~ Perform empirical risk minimization when H is the space of polynomials with degree at most `degree`
    def my_univar_poly_fit( x_train, y_train, degree ):
        # YOUR CODE HERE; Hint: you should just set up the appropriate matrix and call `np.linalg.lstsq`
        return fitted_polynomial, coefficients
else:
    #
    # ~~~ Load Tom's answer
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
        # YOUR CODE HERE; Hint: you should just set up the appropriate matrix and call `np.linalg.lstsq`
        return fitted_spline, coefficients
else:
    #
    # ~~~ Load Tom's answer
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
## ~~~ DEMONSTRATION 4 of 5: **Cross validation** -- a standard workflow for model selection (which in this case means selecting the appropriate polynomial degree)
### ~~~

warnings.filterwarnings( "ignore", message="Polyfit may be poorly conditioned" )    # ~~~ otherwise crossvalidation will spit out th[is warning a million times

#
# ~~~ Measure how well the model does when certain subsets of the data are withheld from training (written to mimic the sklearn.model_selection function of the same name)
def cross_val_score( estimator, eventual_x_train, eventual_y_train, cv, scoring, shuffle=False, plot=False, ncol=None, nrow=None, f=None, grid=None ):
    #
    # ~~~ Boiler plate stuff, not important
    scores = []
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
# ~~~ Define the metric by which we will assess accurcay: mean squared error
def mean_squared_error( true, predicted ):
    return np.mean( (true-predicted)**2 )
    # ~~~ usually you'd load this or one of the other options from sklearn.meatrics (https://scikit-learn.org/stable/modules/model_evaluation.html#common-cases-predefined-values)
    # ~~~ we have defined it explicitly for transparency and simplicity

#
# ~~~ A simple wrapper
def poly_cv( degree, x, y, **kwargs ):
    polynomial_regression = lambda x_train,y_train: univar_poly_fit( x_train, y_train, degree=degree )[0]       # ~~~ define the modeling technique    
    return cross_val_score( estimator=polynomial_regression, eventual_x_train=x, eventual_y_train=y, **kwargs ) # ~~~ do cv with this modeling technique and data

#
# ~~~ Example usage of the big bad function we just defined
x_train, y_train = Foucarts_training_data() # ~~~ for improved reproducibility
scores =  poly_cv( degree=2, x=x_train, y=y_train, cv=2, scoring=mean_squared_error )   # ~~~ returns [5.659962939942039, 434.55267825346857], meaning that the model trained on the second half of the data performed terribly
print( f"Test error {scores[0]:.4f} when trained on the 1st half of the data", f"Test error {scores[1]:.4f} when trained on the 2nd half of the data", sep="\n" )


#
# ~~~ Investigate further by specifying `plot=True`
_ = poly_cv( degree=2, x=x_train, y=y_train, cv=2, scoring=mean_squared_error, plot=True )

#
# ~~~ And that's why people shuffle their data...
np.random.seed(680)     # ~~~ fix seed for improved reproducibility
scores = poly_cv( degree=2, x=x_train, y=y_train, cv=2, scoring=mean_squared_error, shuffle=True, plot=True )   # ~~~ returns [1.408731127467586, 1.1814320746081544]; both models fit about as well as they can given how noisy the data is
print( f"Test error {scores[0]:.4f} when trained on the 1st half of the data", f"Test error {scores[1]:.4f} when trained on the 2nd half of the data", sep="\n" )




### ~~~
## ~~~ EXERCISE 3 of 4 (hard): Experiement with CV and implement a full CV work flow
### ~~~

#
# ~~~ Implement a full CV workflow for polynomial regressions of degree 1 through 20 **WITHOUT** shuffling the data (for reproducibility)
if exercise_mode:
    def full_CV_and_plot_the_result( x_train, y_train, n_bins, any_other_arguments_you_desire, ground_truth ):
        # YOUR CODE HERE FOR CHOOSING THE POLYNOMIAL DEGREE WITH MINIMAL MEDIAN ERROR (hint: loop over poly_cv)
        # ALSO, FIT A POLYNOMIAL OF THE SELECTED DEGREE AND PLOT IT ALONG WITH THE FUNCTION ground_truth
        return array_of_scores_of_all_degree_models_on_all_bins   # should have shape (20,n_bins)
else:
    from answers_680.answers_week_1 import Toms_example_of_the_cv_workflow as full_CV_and_plot_the_result

#
# ~~~ A simple helper routine for improved organization
def recall_data():
    f = lambda x: abs(x)
    x_train, y_train = Foucarts_training_data()
    np.random.seed(680)
    reordered_indices = np.random.permutation( len(y_train) )
    x_train = x_train[reordered_indices]
    y_train = y_train[reordered_indices]
    return x_train, y_train

#
# ~~~ Do CV for polynomial regression on the data we started out with
x_train, y_train = recall_data()
my_scores = full_CV_and_plot_the_result( x_train, y_train, n_bins=2, ground_truth=f )
true_scores = np.array([[2.31622333e+00, 1.75726951e+00],
                        [1.40873113e+00, 1.18143207e+00],
                        [2.53187114e+00, 3.46624062e+00],
                        [1.51372680e+00, 3.84275913e+01],
                        [1.39844359e+01, 4.71534761e+01],
                        [1.00247114e+01, 1.82661686e+02],
                        [1.53434202e+03, 1.39837248e+02],
                        [5.35387860e+02, 2.33165702e+02],
                        [6.25159890e+02, 2.61438676e+02],
                        [3.51934459e+02, 3.91230926e+02],
                        [4.29632292e+02, 4.64074269e+02],
                        [3.54558153e+02, 6.48744669e+02],
                        [4.49513970e+02, 7.58878078e+02],
                        [4.62513534e+02, 1.00320451e+03],
                        [5.94445044e+02, 1.13920810e+03],
                        [6.68594552e+02, 1.43462920e+03],
                        [8.53879297e+02, 1.57885767e+03],
                        [9.83273230e+02, 1.90664495e+03],
                        [1.23514684e+03, 2.03958356e+03],
                        [1.41839306e+03, 2.37728170e+03]])
assert abs(my_scores-true_scores).max() < 1e-4  # ~~~ the tolerance is high to account for numerical instability (recall the warning the we've surpressed)

#
# ~~~ Observe what happens when you toggle the number of subsets into which you split the training data
x_train, y_train = recall_data()
for n in [2,3,4,5,6,7,8]:
    _ = full_CV_and_plot_the_result( x_train, y_train, n_bins=n, ground_truth=f )

#
# ~~~ With a better data set the behavior is different
np.random.seed(680)
f = lambda x: abs(x)
x_train, y_train, x_test, y_test = generate_random_1d_data( f, n_train=500, noise=0.1 )
for n in [2,3,4,5,10,15,20,30,40]:
    _ = full_CV_and_plot_the_result( x_train, y_train, n_bins=n, ground_truth=f )


### ~~~
## ~~~ DEMONSTRATION 5 of 5: We can do CV with neural networks, too
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
    n_bins = 2
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
        #
        # ~~~ Run the computations locally
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
                    current_scores = cross_val_score( estimator, x_train, y_train, cv=n_bins, scoring=mean_squared_error )                        # ~~~ as before
                    scores.append(current_scores)                                                                                                   # ~~~ as before
                    # ~~~ End key functionality
                    #
                    bar()                                                                                                                                   # ~~~ for the progress bar
        #
        # ~~~ Save the result of all that for future convenience
        np.save( file_path, np.array(scores) )
    #
    # ~~~ Take the best hyperparameter and train using the full data
    if os.path.exists(file_path):
        #
        # ~~~ Load the results of the computations
        scores = np.load(file_path)     # ~~~ reaquires 'results_of_cv_ch1.npy' to be located in Lib\answers_680 along with the answer keys
        #
        # ~~~ Train on the full data a neural network with hyperparameters selected by CV
        best_hyperparameters = possible_hyperparameters[ np.median(scores,axis=1).argmin() ]    # ~~~ lowest median "generalization" error when trianed on only a subset of the data
        architecture, n_epochs = best_hyperparameters
        best_nn = make_and_train_1d_network( x_train, y_train, hidden_layers=architecture, epochs=n_epochs, verbose=0 )[0]
        #
        # ~~~ Compare with polynomial regression
        np.random.seed(680)
        x_train, y_train, x_test, y_test = generate_random_1d_data( f, n_train=50, noise=0.1 )
        from answers_680.answers_week_1 import Toms_example_of_the_cv_workflow
        scores = Toms_example_of_the_cv_workflow( x_train, y_train, plot=False )
        degrees_tested = np.arange(20)+1
        best_degree = degrees_tested[ np.median(scores,axis=1).argmin() ] # ~~~ lowest median "generalization" error of trianing on only a subset of the data
        best_poly,_ = univar_poly_fit( x_train, y_train, degree=best_degree )
        #
        # ~~~ Plot the results
        side_by_side_prediction_plots( x_train, y_train, f, best_poly, best_nn, f"The Polynomial Model Chosen by CV has Test MSE {mean_squared_error(best_poly(x_test),y_test):.4}", f"The NN model Chosen by CV has Test MSE {mean_squared_error(best_nn(x_test),y_test):.4}" )
 
 

### ~~~
## ~~~ EXERCISE 4 of 4 (hard): Using CV as above, select a good relu network and good spline regression, and compare the results
### ~~~

# (hint: to do CV on ReLU instsead of tanh networks, basically, just pass `activations="relu` to make_and_train_1d_network)
