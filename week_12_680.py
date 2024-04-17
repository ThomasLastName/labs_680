
# ~~~ Tom Winckelman wrote this; maintained at: https://github.com/ThomasLastName/labs_680

exercise_mode = False   # ~~~ see https://github.com/ThomasLastName/labs_680?tab=readme-ov-file#usage
install_assist = False  # ~~~ see https://github.com/ThomasLastName/labs_680/blob/main/README.md#assisted-installation-for-environments-other-than-colab-recommended


### ~~~
## ~~~ Boiler plate stuff; basically just loading packages
### ~~~

#
# ~~~ Standard python libraries
import torchvision
import torch
from torch import nn
from tqdm.auto import tqdm

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
        files = [ "ansi.py", "my_base_utils.py", "my_torch_utils.py" ]
        intstall_Toms_code( folder, files )
        #
        # ~~~ "Install/update" answers_680
        folder = "answers_680"
        files = [ "answers_week_12.py" ]
        intstall_Toms_code( folder, files )

#
# ~~~ Tom's helper routines (which the above block of code installs for you); maintained at https://github.com/ThomasLastName/quality_of_life
from quality_of_life.my_torch_utils import convert_Dataset_to_Tensors



### ~~~
## ~~~ EXERCISE 1 of 3: Download MNIST
### ~~~

#
# ~~~ set the parent directory where you want to store data on your computer
my_data_directory = "C:\\Users\\thoma\\AppData\\Local\\Programs\\Python\\Python310\\pytorch_data"

MNIST_train = torchvision.datasets.MNIST(
        root = my_data_directory,   # where to look for the data on your computer (if the data isn't there and download=True, then it will download the data and put it there)
        train = True,               # specify whether you want the training data or the test data
        download = True,            # give permission to download the data if pytorch doesn't find it in `root`
        transform = torchvision.transforms.ToTensor()   # convert whatever weird format .jpeg data comes into an actual mathematical object
    )

MNIST_test = torchvision.datasets.MNIST(
        root = my_data_directory,   # where to look for the data on your computer (if the data isn't there and download=True, then it will download the data and put it there)
        train = False,              # specify whether you want the training data or the test data
        download = True,            # give permission to download the data if pytorch doesn't find it in `root`
        transform = torchvision.transforms.ToTensor()   # convert whatever weird format .jpeg data comes into an actual mathematical object
    )



### ~~~
## ~~~ DEMONSTRATION 1 of 3: Define a neural network with 3 hidden layers, of widths 500, 5000, and 500
### ~~~

torch.manual_seed(680)
model = nn.Sequential(
            nn.Flatten(),   # ~~~ flattens shit down to a 1D vector: in this case, shape (1,28,28)->(784,) and each batch of shape (n,1,28,28)->(n,784)
            nn.Linear(784, 500),    # ~~~ an affine map from R^{28*28} to R^500
            nn.ReLU(),
            nn.Linear(500, 5000),   # ~~~ an affine map from R^500 to R^5000
            nn.ReLU(),
            nn.Linear(5000, 500),   # ~~~ an affine map from R^5000 to R^500
            nn.ReLU(),
            nn.Linear(500, 10)      # ~~~ the outermost weights: an affine map from R^500 to R^10
        )

try:
    from torchinfo import summary
    summary(model)
except ModuleNotFoundError:
    print(model.parameters)



### ~~~
## ~~~ EXERCISE 2 of 3: Write a funciton which tests the model's accuracy on some data
### ~~~

if exercise_mode:
    def measure_accuracy(model,X_test,y_test):
        n_test,n_class = X_test.shape
        assert ( y_test.unique()==torch.arange(n_class) ).all()
        with torch.no_grad():
            predicted = model(X_test)   # ~~~ a matrix of shape (n_test,n_class) or, more generally, of shape
            target = y_test             # ~~~ a vector of shape (n_test,)
            n_correct = _____________________.item()
            percent_correct = n_correct/n_test
        return percent_correct.item()
else:
    from answers_680.answers_week_12 import measure_accuracy


X_test, y_test = convert_Dataset_to_Tensors(MNIST_test)   # ~~~ see the exercises from two weeks ago
torch.manual_seed(680)
model = nn.Sequential(      # ~~~ a shallow ReLU network of width 512
            nn.Flatten(), nn.Linear(28*28, 512), nn.ReLU(), nn.Linear(512, 10)
        )
assert measure_accuracy(model,X_test,y_test)==0.11599999666213989   # ~~~ the accuracy of the (randomly generated) untrained model is basically that of random guessing: 1/n_class==1/10



### ~~~
## ~~~ EXERCISE 3 of 3: Reproduce pytorch's nn.CrossEntropyLoss()
### ~~~

#
# ~~~ Example of what a neural network might spit out
predicted = torch.tensor([[-1.1933,  0.2766,  1.1250, -0.4440,  2.7818],
                          [-1.3816, -0.3540, -1.1034,  2.2402,  0.1682],
                          [-3.2324, -2.3961, -0.1085,  0.0205, -0.3228]])

#
# ~~~ Example of some class labels (from 0 to 4 inclusive)
targets = torch.tensor([3,1,0])

#
# ~~~ What `loss_fn(predicted,targets)` would do
loss_fn = nn.CrossEntropyLoss()
loss_fn(predicted,targets)

#
# ~~~ Based on the documentation for `nn.CrossEntropyLoss()` (which you must look up), write the loss function yourself
if exercise_mode:
    def my_cross_entropy(predicted,targets):
        return #the same value as loss_fn(predicted,targets)
else:
    from answers_680.answers_week_12 import my_cross_entropy

#
# ~~~ Do the two match?
print("These should be the same:")
print(loss_fn(predicted,targets))
print(my_cross_entropy(predicted,targets))



### ~~~
## ~~~ DEMONSTRATION 2 of 3: Training at its simplest
### ~~~

#
# ~~~ Hyperparameters (basically, "what variety of gradient descent will we use?")
b = 50
epochs = 3
eta = 1e-2

#
# ~~~ Organize data, build model, and such
dataloader = torch.utils.data.DataLoader( MNIST_train, batch_size=b, shuffle=True ) # ~~~ see the exercises from two weeks ago
loss_fn = nn.CrossEntropyLoss()                                                     # ~~~ see the exercises from two weeks ago
X_test, y_test = convert_Dataset_to_Tensors(MNIST_test)                             # ~~~ see the exercises from two weeks ago
torch.manual_seed(680)
model = nn.Sequential( nn.Flatten(), nn.Linear(28*28, 512), nn.ReLU(), nn.Linear(512, 10) )

#
# ~~~ Do training
for e in range(epochs):             # ~~~ do the following `epochs` times
    for batch in tqdm(dataloader):  # ~~~ split the data into batches
        X,y = batch
        loss = loss_fn(model(X),y)  # ~~~ compute the loss on this batch of the data
        loss.backward()             # ~~~ compute the derivative with respect to anything that has a grad attribute (which model's parameters do by default)
        with torch.no_grad():
            for p in model.parameters():
                p -= eta*p.grad     # ~~~ perform the GD update
                p.grad = None       # ~~~ zero out the gradient (this is a just quirk of pytorch)
    #
    # ~~~ After each complete cycle through the data set, check the loss on the last batch, as well as the model's accuracy on a test data set
    print(f"Loss after {e+1}/{epochs} epochs is {loss.item()}")
    print(f"Accuracy after {e+1}/{epochs} epochs is {measure_accuracy(model,X_test,y_test)}")



### ~~~
## ~~~ DEMONSTRATION 3 of 3: Training using an Optimizer class
### ~~~

#
# ~~~ A greatly simplified analog to torch.optim.SGD
class Optimizer:
    def __init__(self,params,lr):
        self.param_list = list(params)
        self.lr = lr
    def step(self):
        with torch.no_grad():
            for p in self.param_list:
                p -= self.lr*p.grad
    def zero_grad(self):
        with torch.no_grad():
            for p in self.param_list:
                p.grad = None

#
# ~~~ Exact same code as in the previous DEMONSTRATION (defining hyper-parameters)
b = 50
epochs = 3
eta = 1e-2

#
# ~~~ Exact same code as in the previous DEMONSTRATION (organize data, build model, and such)
dataloader = torch.utils.data.DataLoader( MNIST_train, batch_size=b, shuffle=True ) # ~~~ see the exercises from two weeks ago
loss_fn = nn.CrossEntropyLoss()                                                     # ~~~ see the exercises from two weeks ago
X_test, y_test = convert_Dataset_to_Tensors(MNIST_test)                             # ~~~ see the exercises from two weeks ago
torch.manual_seed(680)
model = nn.Sequential( nn.Flatten(), nn.Linear(28*28, 512), nn.ReLU(), nn.Linear(512, 10) )

#
# ~~~ Do training
optimizer = Optimizer( model.parameters(), lr=eta ) # ~~~ try, also, optimizer = torch.optim.SGD( model.parameters(), lr=eta )
for e in range(epochs):             # ~~~ do the following `epochs` times
    for batch in tqdm(dataloader):  # ~~~ split the data into batches
        X,y = batch
        loss = loss_fn(model(X),y)  # ~~~ compute the loss on this batch of the data
        loss.backward()             # ~~~ compute the derivative with respect to anything that has a grad attribute (which model's parameters do by default)
        optimizer.step()            # ~~~ perform the GD update
        optimizer.zero_grad()       # ~~~ zero out the gradient (this is a just quirk of pytorch)
    #
    # ~~~ After each complete cycle through the data set, check the loss on the last batch, as well as the model's accuracy on a test data set
    print(f"Loss after {e+1}/{epochs} epochs is {loss.item()}")
    print(f"Accuracy after {e+1}/{epochs} epochs is {measure_accuracy(model,X_test,y_test)}")
