# todo NN in base numpy, and example using tensorflow



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
from collections import OrderedDict


from quality_of_life.my_torch_utils import convert_Dataset_to_Tensors, convert_Tensors_to_Dataset, hot_1_encode_an_integer
from quality_of_life.my_base_utils import support_for_progress_bars


### ~~~
## ~~~ EXERCISE 1 of ?: Download MNIST
###  ~~~

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
## ~~~ EXERCISE 2 of ?: Get the actual data out of pytorch's Dataset class
### ~~~

if exercise_mode:
    #
    # ~~~ write a function that excracts the actual matrices from `MNIST_train` and `MNIST_test`
    def get_data(object_of_class_Dataset):
        # HINT objects of class torch.utils.data.Dataset aren't matrices but you can still " do [index] to them;" try printing `MNIST_train[0]` (this works thanks to the __getitem__ method; see https://pytorch.org/tutorials/beginner/basics/data_tutorial.html#getitem)
        return X, y
else:
    from quality_of_life.my_torch_utils import convert_Dataset_to_Tensors as get_data


X_test, y_test = convert_Dataset_to_Tensors(MNIST_test) # ~~~ correct implementation
my_X_test, my_y_test = get_data(MNIST_test)             # ~~~ my implementation
assert (X_test==my_X_test).min() and (y_test==my_y_test).min()  # ~~~ check that the results are identical




### ~~~
## ~~~ EXERCISE 3 of ?: Build a torch.utils.data.Dataset out of plain old numbers
### ~~~

if exercise_mode:
    #
    # ~~~ write a function that assembles actual matrices into objects of class torch.utils.data.Dataset like `MNIST_train` and `MNIST_test`
    def assemble_data(object_of_class_Dataset):
        # HINT this is highly googlable
        return Dataset
else:
    #
    # ~~~ load my left inverse to get_data; note that my implelemntation is *not* a bijection: we have get(assemble(v))=v, but *not* assemble(get(w))=w
    from quality_of_life.my_torch_utils import convert_Tensors_to_Dataset as assemble_data



my_X, my_y = get_data(assemble_data(X_test,y_test))     # ~~~ build a torch.utils.data.Dataset and then call the tensors that we built it out of
assert (X_test==my_X).min() and (y_test==my_y).min()    # ~~~ check that we got back exactly what we started with



### ~~~
## ~~~ EXERCISE 4 of ?: Identify where a,b,c,d,e,f,g,h below are valid, and which of them are identical to which others
### ~~~

if exercise_mode:
    #
    # ~~~ Delete any of these that are invalid; determine which are identical to which others
    a = torch.Tensor([3,1,0])
    b = torch.tensor([3,1,0])
    c = torch.Tensor([3.,1.,0.])
    d = torch.tensor([3.,1.,0.])
    e = torch.Tensor([3.,1.,0.], dtype=torch.int32)
    f = torch.tensor([3.,1.,0.], dtype=torch.int32)
    g = torch.Tensor([3.,1.,0.], dtype=torch.int64)
    h = torch.tensor([3.,1.,0.], dtype=torch.int64)




# Example of target with class indices
predicted = torch.tensor([[-1.1933,  0.2766,  1.1250, -0.4440,  2.7818],
                          [-1.3816, -0.3540, -1.1034,  2.2402,  0.1682],
                          [-3.2324, -2.3961, -0.1085,  0.0205, -0.3228]])
targets = torch.tensor([3,1,0])
loss = nn.CrossEntropyLoss()
loss(predicted,targets)



def my_cross_entropy(predicted,targets):
    encode = hot_1_encode_an_integer(n_class=5)
    t = encode(targets)
    p = predicted.softmax(dim=1)
    return ( -t*p.log() ).sum(axis=1).mean()


my_cross_entropy(predicted,targets)






class Bar:
    def __init__(self, arg):
        self.attribute = arg
    def method(self, extra):
        print(self.attribute + extra)

foo = Bar("x")
foo.method(foo,"extra")


