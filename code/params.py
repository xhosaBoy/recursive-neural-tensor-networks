import os
import math

data_number = 0  # 0 - Wordnet, 1 - Freebase

if data_number == 0:
    data_name = 'Wordnet'
else:
    data_name = 'Freebase'

file = os.path.realpath(__file__)
project = os.path.split(os.path.dirname(file))[0]

data_path = os.path.join(project, 'data', data_name)
output_path = os.path.join(project, 'output', data_name)

num_iter = 100
train_both = False
batch_size = 10000
corrupt_size = 10  # how many negative examples are given for each positive example?
embedding_size = 100
slice_size = 3  # depth of tensor for each relation
regularization = 0.0001  # parameter \lambda used in L2 normalization
in_tensor_keep_normal = False
save_per_iter = 101
val_per_iter = 10
learning_rate = 0.01

output_dir = ''
