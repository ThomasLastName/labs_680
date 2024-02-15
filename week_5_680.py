
# ~~~ Tom Winckelman wrote this; maintained at: https://github.com/ThomasLastName/labs_680

exercise_mode = False   # ~~~ see https://github.com/ThomasLastName/labs_680?tab=readme-ov-file#usage
install_assist = False  # ~~~ see https://github.com/ThomasLastName/labs_680/blob/main/README.md#assisted-installation-for-environments-other-than-colab-recommended

### ~~~
## ~~~ Boiler plate stuff; basically just loading packages
### ~~~

#
# ~~~ Standard python libraries
import sys
import numpy as np
from time import time as tick
tock = tick
import matplotlib.pyplot as plt

#
# ~~~ For this week, sklearn is simply a requirement
from sklearn.datasets import load_digits as scikit_NIST_data
from sklearn.datasets import fetch_lfw_people as scikit_faces
from sklearn.datasets import fetch_openml
from sklearn.decomposition import PCA


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
        # ~~~ "Install/update" answers_680
        folder = "answers_680"
        files = [ "answers_week_5.py" ]
        intstall_Toms_code( folder, files )



### ~~~
## ~~~ DEMONSTRATION 1 of 3: Do PCA on the images of handwritten digites
### ~~~

#
# ~~~ Download a simple data set to start with
X_lowres,y_lowres = scikit_NIST_data(return_X_y=True)

# ~~~ A simple plotting routine
def visualize_images( X, labeler, process_data=lambda x:x ):
    fig, axes = plt.subplots(3, 5, figsize=(12, 8), subplot_kw={'xticks': (), 'yticks': ()})    # ~~~ change this to change the number of subplots
    for i, ax in enumerate(axes.flat):
        ax.imshow( process_data(X[i]), cmap='gray' )
        ax.set_title(labeler(i))
    fig.tight_layout()
    plt.show()

#
# ~~~ Visualize the dataset
visualize_images( X_lowres, labeler=lambda i: f"Low-res Image of digit {y_lowres[i]}", process_data=lambda x:x.reshape((8,8)) )

#
# ~~~ Apply PCA on a matrix assuming that each *column* is a data point (as in the text)
def naive_column_PCA(X_Foucarts_format,k=None):
    U,s,Vt = np.linalg.svd(X_Foucarts_format)
    components = U if k is None else U[:,:k]    # ~~~ Theorem 8.1 tells us to take the first k columns of U (called the first k "left singular vectors")
    singular_values = s if s is None else s[:k] # ~~~ these are also nice to have
    return components, singular_values

#
# ~~~ Apply PCA on a matrix assuming that each *row* is a data point (as in most data science settings)
def naive_row_PCA(X_data_format,k=None):
    U,s,Vt = np.linalg.svd(X_data_format)
    V = Vt.T    # ~~~ the right singular vectors of a matrix's transpose are the same as the left singular vectors of the matrix itself
    components = V if k is None else Vt[:,:k]  
    singular_values = s if s is None else s[:k] # ~~~ these are also nice to have
    return components, singular_values

components, singular_values = naive_row_PCA(X_lowres)
visualize_images( -components.T, labeler=lambda i: f"'{i+1}-th' Principal Image", process_data=lambda x:x.reshape((8,8)) )



### ~~~
## ~~~  EXERCISE 1 of 6 (hard): Implement PCA efficiently in the manner advised by Remark 8.2 in the text
### ~~~

if exercise_mode:
    #
    # ~~~ Instead of implementing a full SVD (as in naive_row_PCA), implement a mathematically equivalent operation following Remark 8.2 of the text
    def row_PCA( X_data, k=None ):
        # YOUR CODE HERE
        return components, singular_values 
else:
    from answers_680.answers_week_5 import row_PCA

#
# ~~~ Attempt to validate the new implementation by comparing it to the old implementation
attempted_components, attempt_singular_values = row_PCA(X_lowres)   # ~~~ compare with what was computed in the prior demonstration
fig, (ax_attempted,ax_correct) = plt.subplots( 1, 2, figsize=(12, 6), subplot_kw={'xticks': (), 'yticks': ()})
ax_attempted.imshow( attempted_components[:,1].reshape((8,8)) )
ax_attempted.set_title("The new implementation")
ax_correct.imshow( components[:,1].reshape((8,8)) )
ax_correct.set_title("\nThe direct (naive) implementation")
fig.suptitle( "If these look similar (not necessarily identical) then the exercise was completed successfully.", fontsize=16 )
fig.tight_layout()
plt.show()

#
# ~~~ Speedup comparison
t = tick()
naively_computed_components, naively_computed_singular_values = naive_row_PCA(X_lowres)
time_svd = tock()-t
t = tick()
components, singular_values = row_PCA(X_lowres)
time_eigh = tock()-t
what = "Congradulations! Your implementation of PCA" if exercise_mode else "The implementation of PCA advised in remark 8.2 of the text"
print(f"{what} is more than {int(time_svd/time_eigh)} times as fast as a simple SVD!")



### ~~~
## ~~~ DEMONSTRATION 2 of 3: PCA on MNIST
### ~~~

#
# ~~~ Download the famous MNIST data set for higher resolution images of handwritten digits
X_MNIST, y_MNIST = fetch_openml( 'mnist_784', version=1, return_X_y=True )   # ~~~ takes a minute or two; triggers a deprecation warning that I choose not to surpress
X_MNIST = np.array(X_MNIST)
y_MNIST = np.array([ float(label) for label in y_MNIST ])

#
# ~~~ Inspect the data
visualize_images( X_MNIST, labeler=lambda i: f"Low-res Image of digit {y_MNIST[i]}", process_data=lambda x:x.reshape((28,28)) )

#
# ~~~ Perform PCA
components, singular_values = row_PCA(X_MNIST)

#
# ~~~ Inspect the singular components
visualize_images( components.T, labeler=lambda i: f"'{i+1}-th' Principal Image", process_data=lambda x:x.reshape((28,28)) )



### ~~~
## ~~~ DEMONSTRATION 3 of 3: PCA on images of faces
### ~~~

#
# ~~~ Download a dataset consting of images of faces
lfw_data = scikit_faces()
images = lfw_data.images
target_names = lfw_data.target_names
target = lfw_data.target

#
# ~~~ Inspect the data
visualize_images( images, labeler=lambda i:target_names[target[i]] )

#
# ~~~ Perform PCA
m,p1,p2 = images.shape
X_faces = images.reshape((m,p1*p2)) # ~~~ flatten each image (itself a p1-by-p2 matrix) into a vector of length p1*p2
components, singular_values = row_PCA(X_faces)

#
# ~~~ Inspect the singular components
visualize_images( -components.T, labeler=lambda i: f"'{i+1}-th' Principal Face", process_data=lambda x:x.reshape((p1,p2)) )



### ~~~
## ~~~ EXERCISE 2 of 6 (easy): Is every linear combination of the first left k singular faces, also, an image of a face?
### ~~~

# HINT: it's pretty simple; take a linear combination of the first left k singular vectors, plot it; is it an image of a face?



### ~~~
## ~~~ EXERCISE 3 of 6 (easy): figure out why sklearn's implementation of PCA is not equivalent to ours'
### ~~~

# Hint: this is one of the first things mentioned in the sklearn documentation for PCA: https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html

#
# ~~~ sklearn's implementation
def sklearn_PCA(X_data_format):
    m,d = X_data_format.shape
    pca = PCA(n_components=d)
    pca.fit(X_data_format)
    components = pca.components_.T
    singular_values = pca.singular_values_
    return components, singular_values



### ~~~
## ~~~ EXERCISE 4 of 6 (medium): add a pre-processing step such that if you pre-process and call row_PCA, it is equivalent to the sklearn implementation
### ~~~

# HINT: the key lies in the statistical interpretation of PCA; look at the empirical covariance matrix...



### ~~~
## ~~~ EXERCISE 5 of 6 (hard): Use PCA to project the MNIST data set into the plane (R^2) and make a scatter plot of this projected data, labeled with a key for what digit it is
### ~~~

# hint: simply pass c=y_MNIST to plt.scatter in order to color the points based on their label; does label=y_MNIST also work?



### ~~~
## ~~~ EXERCISE 6 of 6 (hard): Write a function that computes the term which Theorem 8.1 says that PCA minimizes; also, normalize it by dividing by m; I'd call this the "mean squared error" of the projection
### ~~~


