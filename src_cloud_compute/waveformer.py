import os
seed = 13
os.environ['TORCH_COMPILE_DISABLE'] = '1'
os.environ['PYTHONHASHSEED'] = str(seed)
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

import numpy as np
import torch
import torch.nn as nn

np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class WaveLayer(nn.Module):
    def __init__(self, n:int): #n is the number of layers in causal conv
        super().__init__()
        self.n = n
        self.conv_list = nn.ModuleList([
             nn.Conv2d(1, 1, (2,1), stride = (2,1))
            for _ in range(self.n)
        ])


    def forward(self, x):
        for conv in self.conv_list:
            x = conv(x)
        return x

class WaveBlock(nn.Module):
    def __init__(self, log_input_size:int, n_features:int):
        super().__init__()

        self.q = WaveLayer(log_input_size)
        self.k = WaveLayer(log_input_size)
        self.softmax = torch.nn.Softmax(dim = -1)
        self.attention_denominator = np.sqrt(n_features)

        self.ff = nn.Linear(n_features, n_features)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.layerlike_norm = nn.InstanceNorm1d(n_features, affine = True, device = device)

    def forward(self, x):

        q = self.q(x)
        k = self.k(x)

        attention = self.softmax(torch.matmul(q.transpose(-1, -2), k)/self.attention_denominator)
        res_x = x

        x = torch.matmul(x, attention)

        x = res_x + x

        x = torch.squeeze(x, 1)
        norm = self.layerlike_norm(torch.transpose(x, -2, -1))
        x = torch.transpose(norm, 2, 1)
        x = torch.unsqueeze(x, 1)

        res_x = x
        x = self.ff(x)

        x = res_x + x

        x = torch.squeeze(x, 1)
        norm = self.layerlike_norm(torch.transpose(x, -2, -1))
        x = torch.transpose(norm, 2, 1)
        x = torch.unsqueeze(x, 1)

        return x

class WaveFormer(nn.Module):
    def __init__(self, input_length:int, output_length:int, n_features:int, n_blocks:int = 1):
        super().__init__()

        self.n_blocks = n_blocks
        self.input_length = input_length
        self.output_length = output_length
        self.prefinallayer = nn.Linear(n_features, 1)
        self.finallayer = nn.Linear(input_length, output_length)

        self.wave_list = nn.ModuleList([
             WaveBlock(int(np.log2(input_length)), n_features)
            for _ in range(self.n_blocks)
        ])

        self.prelu = nn.PReLU()

    def forward(self, x:torch.Tensor):
        mean_is_series = torch.mean(x[:,:,:,0], -1, True)
        x[:,:,:,0] = x[:,:,:,0] - mean_is_series

        for block in self.wave_list:
            x = block(x)

        x = self.prefinallayer(x).reshape(-1, 1, 1, self.input_length)
        #x = torch.concatenate((x, mean_is_series.unsqueeze(-1)), dim = -1) #I think this is the way, but didn't come up with solution yet

        x = self.prelu(x)
        x = self.finallayer(x)

        x[:,:,:,0] = x[:,:,:,0] + mean_is_series

        return x

def initialize_weights(m):

    if isinstance(m, nn.Linear):
        nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Conv2d):
        nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.InstanceNorm1d):
        if m.weight is not None:
            nn.init.ones_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.PReLU):
        nn.init.constant_(m.weight, 0.25)