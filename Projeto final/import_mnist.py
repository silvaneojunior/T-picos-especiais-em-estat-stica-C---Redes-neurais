from mlxtend.data import loadlocal_mnist
import config
import numpy as np
import dill

file=open('mnist_expanded.dill','rb')
expanded_training_data, validation_data, test_data=dill.load(file)
file.close()

data_x=np.asarray(expanded_training_data[0]).T
data_y=np.asarray(expanded_training_data[1])

data_l=np.array([[0]*len(data_x[0])]*10,dtype=config.float_type)

for i,j in enumerate(data_y):
    data_l[j,i]=1

test_x=np.asarray(test_data[0]).T
test_y=np.asarray(test_data[1])

test_l=np.array([[0]*test_x.shape[1]]*10,dtype=config.float_type)

for i,j in enumerate(test_y):
    test_l[j,i]=1
    
data_x=data_x.T
data_l=data_l.T
test_x=test_x.T
test_l=test_l.T

