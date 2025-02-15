"""Reimplement TimeGAN-pytorch Codebase.

Reference: Jinsung Yoon, Daniel Jarrett, Mihaela van der Schaar,
"Time-series Generative Adversarial Networks,"
Neural Information Processing Systems (NeurIPS), 2019.

Paper link: https://papers.nips.cc/paper/8789-time-series-generative-adversarial-networks

Last updated Date: October 18th 2021
Code author: Zhiwei Zhang (bitzzw@gmail.com), Biaolin Wen (robinbg@foxmail.com)

-----------------------------

model.py: Network Modules

(1) Encoder
(2) Recovery
(3) Generator
(4) Supervisor
(5) Discriminator
"""

import torch
import torch.nn as nn
import torch.nn.init as init

Activation = nn.LeakyReLU #nn.ReLU #  nn.Sigmoid

#--- resnet impl here is based on https://blog.paperspace.com/writing-resnet-from-scratch-in-pytorch/
class ResidualBlock(nn.Module):
    def __init__(self, opt, stride = 1, downsample = None):
        super().__init__()
        in_channels, out_channels = opt.z_dim, opt.hidden_dim
        self.conv1 = nn.Sequential(
                        nn.Conv1d(in_channels, out_channels, kernel_size = 3, stride = stride, padding = 1),
                        nn.BatchNorm1d(out_channels),
                        nn.ReLU())
        self.conv2 = nn.Sequential(
                        nn.Conv1d(out_channels, out_channels, kernel_size = 3, stride = 1, padding = 1),
                        nn.BatchNorm1d(out_channels))
        self.downsample = downsample
        self.relu = nn.ReLU()
        self.out_channels = out_channels
        
    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.conv2(out)
        if self.downsample:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out

def _RNN(opt):
    if opt.module == 'gru':
        return nn.GRU
    elif opt.module == 'lstm':
        return nn.LSTM
    elif opt.module == 'rnn':
        return nn.RNN
    else:
        raise ValueError(f'unknonw RNN module "{opt.module}"')

def _weights_init(m):
    classname = m.__class__.__name__
    if isinstance(m, nn.Linear):
        init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0)
    elif classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('Norm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)
    elif classname.find("GRU") != -1:
      for name,param in m.named_parameters():
        if 'weight_ih' in name:
          init.xavier_uniform_(param.data)
        elif 'weight_hh' in name:
          init.orthogonal_(param.data)
        elif 'bias' in name:
          param.data.fill_(0)


class Encoder(nn.Module):
    """Embedding network between original feature space to latent space.

        Args:
          - input: input time-series features. (L, N, X) = (24, ?, 6)
          - h3: (num_layers, N, H). [3, ?, 24]

        Returns:
          - H: embeddings
        """
    def __init__(self, opt):
        super(Encoder, self).__init__()
        _rnn = _RNN(opt)
        self.rnn = _rnn(input_size=opt.z_dim, hidden_size=opt.hidden_dim, num_layers=opt.num_layer, batch_first = True)
       # self.norm = nn.BatchNorm1d(opt.hidden_dim)
        self.fc = nn.Linear(opt.hidden_dim, opt.hidden_dim)
        self.sigmoid = Activation() #nn.Sigmoid()
        self.apply(_weights_init)

    def forward(self, input, sigmoid=True):
        e_outputs, _ = self.rnn(input)
        H = self.fc(e_outputs)
        if sigmoid:
            H = self.sigmoid(H)
        return H


class Recovery(nn.Module):
    """Recovery network from latent space to original space.

    Args:
      - H: latent representation
      - T: input time information

    Returns:
      - X_tilde: recovered data
    """
    def __init__(self, opt):
        super(Recovery, self).__init__()
        _rnn = _RNN(opt)
        self.rnn = _rnn(input_size=opt.hidden_dim, hidden_size=opt.z_dim_out, num_layers=opt.num_layer, batch_first = True)
        
      #  self.norm = nn.BatchNorm1d(opt.z_dim)
        self.fc = nn.Linear(opt.z_dim_out, opt.z_dim_out)
        self.sigmoid = Activation() #nn.Sigmoid()
        self.apply(_weights_init)

    def forward(self, input, sigmoid=True):
        r_outputs, _ = self.rnn(input)
        X_tilde = self.fc(r_outputs)
        if sigmoid:
            X_tilde = self.sigmoid(X_tilde)
        return X_tilde

class Generator(nn.Module):
    """Generator function: Generate time-series data in latent space.

    Args:
      - Z: random variables
      - T: input time information

    Returns:
      - E: generated embedding
    """
    def __init__(self, opt, note_embedder):
        super(Generator, self).__init__()
            
        self.embed = note_embedder #nn.Embedding(NUM_MIDI_TOKENS, opt.embedding_dim)
        history_len_samples = opt.z_dim #--- this is hacky, TODO (input dim of encoder input is the history len)
        rnn_inp_dim = opt.latent_dim + (opt.embedding_dim + 2) * history_len_samples
        
        _rnn = _RNN(opt)
        self.rnn = _rnn(input_size = rnn_inp_dim, hidden_size = opt.hidden_dim, num_layers = opt.num_layer_gen, batch_first = True)
     #   self.norm = nn.LayerNorm(opt.hidden_dim)
        self.fc = nn.Linear(opt.hidden_dim, opt.hidden_dim)
        self.sigmoid = Activation() #n.Sigmoid()
        self.apply(_weights_init)

    def forward(self, z, note_ids, note_en, is_note, sigmoid = True):
        batch_size, seq_len = z.shape[0:2]
        note_emb = self.embed(note_ids).reshape(batch_size, seq_len, -1)
        z = torch.concat([z, note_emb, note_en, is_note], dim = 2)
        g_outputs, _ = self.rnn(z)
      #  g_outputs = self.norm(g_outputs)
        E = self.fc(g_outputs)
        if sigmoid:
            E = self.sigmoid(E)
        return E

class Supervisor(nn.Module):
    """Generate next sequence using the previous sequence.

    Args:
      - H: latent representation
      - T: input time information

    Returns:
      - S: generated sequence based on the latent representations generated by the generator
    """
    def __init__(self, opt):
        super(Supervisor, self).__init__()
        _rnn = _RNN(opt)
        self.rnn = _rnn(input_size=opt.hidden_dim, hidden_size=opt.hidden_dim, num_layers=opt.num_layer, batch_first = True)
      #  self.norm = nn.LayerNorm(opt.hidden_dim)
        self.fc = nn.Linear(opt.hidden_dim, opt.hidden_dim)
        self.sigmoid = Activation() #n.Sigmoid()
        self.apply(_weights_init)

    def forward(self, input, sigmoid=True):
        s_outputs, _ = self.rnn(input)
      #  s_outputs = self.norm(s_outputs)
        S = self.fc(s_outputs)
        if sigmoid:
            S = self.sigmoid(S)
        return S


class Discriminator(nn.Module):
    """Discriminate the original and synthetic time-series data.

    Args:
      - H: latent representation
      - T: input time information

    Returns:
      - Y_hat: classification results between original and synthetic time-series
    """
    def __init__(self, opt, note_embedder):
        super(Discriminator, self).__init__()

        self.embed = note_embedder
        history_len_samples = opt.z_dim #--- this is hacky, TODO (input dim of encoder input is the history len)
        rnn_inp_dim = opt.hidden_dim + (opt.embedding_dim + 2) * history_len_samples
        
        _rnn = _RNN(opt)
        self.rnn = _rnn(input_size = rnn_inp_dim, hidden_size = opt.hidden_dim, num_layers = opt.num_layer_discrim, batch_first = True, dropout = 0.)
      #  self.norm = nn.LayerNorm(opt.hidden_dim)
        fc_out_dim = 1 #opt.hidden_dim # 1 # that's the original impl, but makes sense to set to 1 (binary classification)
        self.fc = nn.Linear(opt.hidden_dim, fc_out_dim)
        self.sigmoid = nn.Sigmoid()
        self.apply(_weights_init)

    def forward(self, h_input, note_ids, note_en, is_note, sigmoid = False, use_last_hidden = False): #True):
        #--- NOTE(2) I set the default value of "sigmoid" to False, since I change the BCE loss to BCEWithLogitsLoss
        #--- NOTE(1) that if use_last_hidden is set to False, output will be of size [batch X seq_len X 1], and the bce loss will average over the 2nd dim (that's its default)
        #--- TODO add the option to reduce using a concatenation of max and avg pooling over seq of hidden states, aka "concat pooling" 
        #--- (see https://medium.com/@sonicboom8/sentiment-analysis-with-variable-length-sequences-in-pytorch-6241635ae130)
        batch_size, seq_len = h_input.shape[0:2]
        note_emb = self.embed(note_ids).reshape(batch_size, seq_len, -1)
        h_input = torch.concat([h_input, note_emb, note_en, is_note], dim = 2)
        d_outputs, _ = self.rnn(h_input)
        if use_last_hidden:
            d_outputs = d_outputs[:, -1, :]
        Y_hat = self.fc(d_outputs)
        if sigmoid:
            Y_hat = self.sigmoid(Y_hat)
        return Y_hat
