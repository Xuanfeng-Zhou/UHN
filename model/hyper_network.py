import torch
import torch.nn as nn
from model.model.model import ModelUtils
from model.layer.layer import LayerUtils, IndexType

class ResidualBlock(nn.Module):
    def __init__(self, input_size):
        super(ResidualBlock, self).__init__()
        self.fc1 = nn.Linear(input_size, input_size)
        self.fc2 = nn.Linear(input_size, input_size)

        self.norm1 = nn.LayerNorm(input_size)
        self.norm2 = nn.LayerNorm(input_size)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        # Save the input as the residual
        residual = x
        
        # Apply ReLU and normalization (pre-activation)
        x = self.relu(x)
        x = self.norm1(x)
        
        # First linear layer
        x = self.fc1(x)
        
        # Apply ReLU and normalization (pre-activation) again
        x = self.relu(x)
        x = self.norm2(x)
        
        # Second linear layer
        x = self.fc2(x)
        
        # Add the residual (skip connection)
        x = x + residual

        return x
    
class FourierFeatures(nn.Module):
    def __init__(self, input_dim, num_frequencies, sigma=100.0):
        super(FourierFeatures, self).__init__()
        self.input_dim = input_dim
        self.num_frequencies = num_frequencies
        self.sigma = sigma

        # Initialize random frequencies with shape (input_dim, num_frequencies)
        # We do not need to learn these frequencies, so we set requires_grad=False
        self.weight = nn.Parameter(torch.randn(input_dim, num_frequencies) * self.sigma, requires_grad=False)

    def forward(self, x):
        # Apply random Fourier feature transformation without needing transpose
        projections = torch.matmul(x, self.weight)
        # Concatenate sine and cosine components to form the final feature vector
        return torch.cat([torch.cos(projections), 
                          torch.sin(projections)], dim=-1)  # Shape: (batch_size, 2 * num_frequencies)

class FourierEncoder(nn.Module):
    '''
    Fourier Encoder for generating weights of the base network, including
        input normalization and fourier encoding.
    '''
    def __init__(self, input_dim, input_minmax, fourier_n_freqs, fourier_sigma=100.0):
        super(FourierEncoder, self).__init__()
        # Calculate the mean and variance for normalization
        self.input_mean = nn.Parameter((input_minmax[0] + input_minmax[1]) / 2, requires_grad=False)
        self.input_var = nn.Parameter((input_minmax[1] - input_minmax[0]) ** 2 / 12, requires_grad=False)
        # Fourier encoding
        self.encoder = FourierFeatures(input_dim, fourier_n_freqs, fourier_sigma)

    def forward(self, x):
        # Normalize the input
        x = (x - self.input_mean) / torch.sqrt(self.input_var + 1e-5)
        # Fourier encoding
        x = self.encoder(x)
        return x
    
class RawEncoder(nn.Module):
    '''
    Raw Encoder for generating weights of the base network, including
        only input normalization.
    '''
    def __init__(self, input_dim, input_minmax):
        super(RawEncoder, self).__init__()
        # Calculate the mean and variance for normalization
        self.input_mean = nn.Parameter((input_minmax[0] + input_minmax[1]) / 2, requires_grad=False)
        self.input_var = nn.Parameter((input_minmax[1] - input_minmax[0]) ** 2 / 12, requires_grad=False)

    def forward(self, x):
        # Normalize the input
        x = (x - self.input_mean) / torch.sqrt(self.input_var + 1e-5)
        return x

class PositionalEncoder(nn.Module):
    '''
    Positional Encoder for generating weights of the base network, including
        input normalization and positional encoding.
    '''
    def __init__(self, input_dim, input_minmax, n_freqs, alpha=1.0, sigma=100.0):
        super(PositionalEncoder, self).__init__()
        # Calculate the mean and variance for normalization
        self.input_mean = nn.Parameter((input_minmax[0] + input_minmax[1]) / 2, requires_grad=False)
        self.input_var = nn.Parameter((input_minmax[1] - input_minmax[0]) ** 2 / 12, requires_grad=False)
        # Frequency matrix for positional encoding, with frequency increasing exponentially
        self.frequency = nn.Parameter(alpha * (sigma ** (torch.linspace(0, n_freqs - 1, n_freqs) / n_freqs)), requires_grad=False)
        
    def forward(self, x):
        # Normalize the input
        x = (x - self.input_mean) / torch.sqrt(self.input_var + 1e-5)
        # Apply positional encoding
        x = torch.einsum('bi,j->bij', x, self.frequency).view(x.shape[0], -1)
        # Apply sine and cosine transformations
        x = torch.cat([torch.sin(x), torch.cos(x)], dim=-1)
        return x  # Shape: (batch_size, 2 * n_freqs * input_dim)

# Transformer Encoder Layer
class TransformerLayer(nn.Module):
    def __init__(self, input_dim, num_heads, num_layers):
        super(TransformerLayer, self).__init__()
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=input_dim, 
                                       nhead=num_heads,
                                       dim_feedforward=input_dim,
                                       dropout=0.0,
                                       batch_first=True
                                       ),
            num_layers=num_layers
        )
    
    def forward(self, x):
        # Apply Transformer encoder to encoded layers
        x = self.transformer(x)
        # Mean pooling over sequence dimension to get fixed feature
        return x.mean(dim=1)
    
class HyperNetwork(nn.Module):
    '''
    HyperNetwork for generating weights of the base network.
    '''
    def __init__(self, 
                 param_dict,
                 global_structure_minmax,
                 local_structure_minmax,
                 index_encoding_minmax,
                 with_structure=True,
                 ):
        super(HyperNetwork, self).__init__()

        # Hidden layers to process indices
        hyper_hidden_size = param_dict['hidden_size']

        # ----------------- Hypernetwork Structure -----------------
        # # Fourier Encoder
        self.with_structure = with_structure
        if self.with_structure:
            structure_input_dim = len(ModelUtils.encode_model_structure()) + len(LayerUtils.encode_layer_structure())
            structure_minmax = torch.cat((global_structure_minmax, local_structure_minmax), dim=1)
            structure_fourier_n_freqs = param_dict['structure_fourier_n_freqs']

            structure_fourier_encoder = FourierEncoder(structure_input_dim, structure_minmax,
                                                       structure_fourier_n_freqs)
            structure_n_heads = param_dict['structure_n_heads']
            structure_n_layers = param_dict['structure_n_layers']

            structure_transformer_encoder = TransformerLayer(input_dim=structure_fourier_n_freqs * 2,
                                                           num_heads=structure_n_heads,
                                                           num_layers=structure_n_layers)
            
            self.structure_input_network = nn.Sequential(
                structure_fourier_encoder,
                structure_transformer_encoder,
                nn.Linear(structure_fourier_n_freqs * 2, hyper_hidden_size),
                nn.ReLU(),
                nn.Linear(hyper_hidden_size, hyper_hidden_size)
            )   
            # Zero initialization for the last linear layer of the structure input network
            nn.init.constant_(self.structure_input_network[-1].weight, 0.0)
            nn.init.constant_(self.structure_input_network[-1].bias, 0.0)

        # Index encoding
        index_input_dim = IndexType.get_length()
        index_encoding_type = param_dict['index_encoding_type']
        if index_encoding_type == 'raw':
            index_fourier_encoder = RawEncoder(index_input_dim, index_encoding_minmax)
            index_encoding_output_dim = index_input_dim
        elif index_encoding_type == 'positional':
            index_positional_n_freqs = param_dict['index_positional_n_freqs']
            index_positional_sigma = param_dict['index_positional_sigma']
            index_fourier_encoder = PositionalEncoder(index_input_dim, index_encoding_minmax,
                index_positional_n_freqs, sigma=index_positional_sigma)
            index_encoding_output_dim = index_positional_n_freqs * 2 * index_input_dim
        else:
            index_fourier_n_freqs = param_dict['index_fourier_n_freqs']
            index_fourier_encoder = FourierEncoder(index_input_dim, index_encoding_minmax, 
                                                        index_fourier_n_freqs)
            index_encoding_output_dim = index_fourier_n_freqs * 2

        # Input network
        index_input_network = nn.Linear(index_encoding_output_dim, hyper_hidden_size)
        self.index_input_network = nn.Sequential(
            index_fourier_encoder,
            index_input_network
        )

        # Hidden layers, residual blocks
        hyper_block_num = param_dict['block_num']
        self.hidden_networks = nn.ModuleList([ResidualBlock(hyper_hidden_size) for _ in range(hyper_block_num)])

        # Output network
        self.relu = nn.ReLU()
        self.output_network = nn.Linear(hyper_hidden_size, 1)

    def forward(self, global_structure, local_structures, idxes):
        # Structure encoding
        if self.with_structure:
            structure = torch.cat((global_structure.expand(1, local_structures.size(1), -1), 
                                local_structures), dim=2)
            structure_output = self.structure_input_network(structure)
        else:
            structure_output = 0

        # Index encoding
        x = self.index_input_network(idxes)
        # Hidden layers
        for hidden_network in self.hidden_networks:
            x = hidden_network(x)

        # Combine structure output
        if self.with_structure:
            x = x + structure_output

        # Output network
        x = self.relu(x)
        weight_value = self.output_network(x)
        return weight_value.squeeze()
