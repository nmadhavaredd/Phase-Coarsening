import cv2
import numpy as np
import os
import torch
from torch.utils.data import Dataset
from openstl.api import BaseExperiment
from openstl.utils import create_parser, default_parser


class CustomDataset(Dataset):
    def __init__(self, X, Y, normalize=False):
        super(CustomDataset, self).__init__()
        self.X = X
        self.Y = Y
        self.mean = None
        self.std = None

        if normalize:
            # get the mean/std values along the channel dimension
            mean = data.mean(axis=(0, 1, 2, 3)).reshape(1, 1, -1, 1, 1)
            std = data.std(axis=(0, 1, 2, 3)).reshape(1, 1, -1, 1, 1)
            data = (data - mean) / std
            self.mean = mean
            self.std = std

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, index):
        data = torch.tensor(self.X[index]).float()
        labels = torch.tensor(self.Y[index]).float()
        return data, labels
batch_size = 1  
pre_seq_length = 10 #input_length
aft_seq_length = 50 #output_length

file_path = "/home1/nmadhavaredd2021/spectralETD/hp_phase/csimulation/new_PhaseC10in50out5w_200datasets/dataset200_i10_o50_w5_new_hp.npz"  # Replace with the correct file path
data = np.load(file_path) 
#data = np.load('Data/phdataset.npz')
X_train = data['x_train']
Y_train = data['y_train']
X_val = data['x_val']
Y_val = data['y_val']
X_test = data['x_test']
Y_test = data['y_test']
# X_train, X_val, X_test, Y_train, Y_val, Y_test = data['X_train'], data['X_val'], data['X_test'], data['Y_train'], data['Y_val'], data['Y_test']
print(f"X_train shape: {X_train.shape}")
print(f"Y_train shape: {Y_train.shape}")
print(f"X_val shape: {X_val.shape}")
print(f"Y_val shape: {Y_val.shape}")
print(f"X_test shape: {X_test.shape}")
print(f"Y_test shape: {Y_test.shape}")

#
train_set = CustomDataset(X=X_train, Y=Y_train)
val_set = CustomDataset(X=X_val, Y=Y_val)
test_set = CustomDataset(X=X_test, Y=Y_test)
dataloader_train = torch.utils.data.DataLoader(
    train_set, batch_size=batch_size, shuffle=True, pin_memory=True)
dataloader_val = torch.utils.data.DataLoader(
    val_set, batch_size=batch_size, shuffle=True, pin_memory=True)
dataloader_test = torch.utils.data.DataLoader(
    test_set, batch_size=batch_size, shuffle=True, pin_memory=True)

custom_training_config = {
    'pre_seq_length': pre_seq_length,
    'aft_seq_length': aft_seq_length,
    'total_length': pre_seq_length + aft_seq_length,
    'batch_size': batch_size,
    'val_batch_size': batch_size,
    'epoch': 200,
    'lr': 1e-4,   
    'metrics': ['mse', 'mae'],

    'ex_name': 'custom_exp',
    'dataname': 'custom',
    'in_shape': [10, 1, 256, 256] #[10, 1, 64, 64],
}

custom_model_config = {
    # For MetaVP models, the most important hyperparameters are: 
    # N_S, N_T, hid_S, hid_T, model_type
    'method': 'SimVP',
    # Users can either using a config file or directly set these hyperparameters 
    'config_file': 'configs/taxibj/ConvLSTM.py',    
    # Here, we directly set these parameters
    'model_type': 'gSTA',
    'N_S': 4,
    'N_T': 8,
    'hid_S': 64,
    'hid_T': 256
}
   
"""
# reverse scheduled sampling
'reverse_scheduled_sampling': 0,
'r_sampling_step_1': 25000,
'r_sampling_step_2': 50000,
'r_exp_alpha': 5000,
# scheduled sampling
'scheduled_sampling': 1,
'sampling_stop_iter': 50000,
'sampling_start_value': 1.0,
'sampling_changing_rate': 0.00002,
# model
'num_hidden': '128,128,128,128',
'filter_size': 5,
'stride': 1,
'patch_size': 4,
'layer_norm': 0,
# training
'batch_size': 16,
'sched': 'cosine',
'warmup_epoch': 1
"""


args = create_parser().parse_args([])
config = args.__dict__

# update the training config
config.update(custom_training_config)
# update the model config
config.update(custom_model_config)
# fulfill with default values
default_values = default_parser()
for attribute in default_values.keys():
    if config[attribute] is None:
        config[attribute] = default_values[attribute]

exp = BaseExperiment(args, dataloaders=(dataloader_train, dataloader_val, dataloader_test))
print('>'*35 + ' training ' + '<'*35)
exp.train()

print('>'*35 + ' testing  ' + '<'*35)
exp.test()


