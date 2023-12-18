""" Code adjusted and fixed for envelope-timegan (itamar katz)
Reimplement TimeGAN-pytorch Codebase.

Reference: Jinsung Yoon, Daniel Jarrett, Mihaela van der Schaar,
"Time-series Generative Adversarial Networks,"
Neural Information Processing Systems (NeurIPS), 2019.

Paper link: https://papers.nips.cc/paper/8789-time-series-generative-adversarial-networks

Last updated Date: October 18th 2021
Code author: Zhiwei Zhang (bitzzw@gmail.com), Biaolin Wen(robinbg@foxmail.com)

-----------------------------

timegan.py

Note: Use original data as training set to generater synthetic data (time-series)
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt

from .data import batch_generator
from ..utils import extract_time, random_generator, NormMinMax
from .model import Encoder, Recovery, Generator, Discriminator, Supervisor


NUM_MIDI_TOKENS = 128 #--- TODO we actually don't use the whole range

class BaseModel():
  """ Base Model for timegan
  """

  def __init__(self, opt, train_dataloader, val_dataloader):
    # Seed for deterministic behavior
    self.seed(opt.manualseed)

    # Initalize variables.
    self.opt = opt
    self.train_dataloader = train_dataloader
    self.val_dataloader = val_dataloader
    #self.ori_data, self.min_val, self.max_val = NormMinMax(ori_data)
    #self.ori_time, self.max_seq_len = extract_time(self.ori_data)
    #self.data_num, _, _ = np.asarray(ori_data).shape    # 3661; 24; 6

    #--- set output folders
    self.trn_dir = os.path.join(self.opt.outf, self.opt.name, 'train')
    self.tst_dir = os.path.join(self.opt.outf, self.opt.name, 'test')
    self.device = torch.device("cuda:0" if self.opt.device != 'cpu' else "cpu")
    self.log_batch = False
    self.writer = SummaryWriter(os.path.join(self.trn_dir, 'log'))

  def seed(self, seed_value):
    """ Seed

    Arguments:
        seed_value {int} -- [description]
    """
    # Check if seed is default value
    if seed_value == -1:
      return

    # Otherwise seed all functionality
    import random
    random.seed(seed_value)
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)
    np.random.seed(seed_value)
    torch.backends.cudnn.deterministic = True

  def save_weights(self, epoch):
    """Save net weights for the current epoch.

    Args:
        epoch ([int]): Current epoch number.
    """

    weight_dir = os.path.join(self.opt.outf, self.opt.name, 'train', 'weights')
    if not os.path.exists(weight_dir): 
      os.makedirs(weight_dir)

    torch.save({'epoch': epoch + 1, 'state_dict': self.net_note_embed.state_dict()},
               '%s/netNE.pth' % (weight_dir))
    torch.save({'epoch': epoch + 1, 'state_dict': self.nete.state_dict()},
               '%s/netE.pth' % (weight_dir))
    torch.save({'epoch': epoch + 1, 'state_dict': self.netr.state_dict()},
               '%s/netR.pth' % (weight_dir))
    torch.save({'epoch': epoch + 1, 'state_dict': self.netg.state_dict()},
               '%s/netG.pth' % (weight_dir))
    torch.save({'epoch': epoch + 1, 'state_dict': self.netd.state_dict()},
               '%s/netD.pth' % (weight_dir))
    torch.save({'epoch': epoch + 1, 'state_dict': self.nets.state_dict()},
               '%s/netS.pth' % (weight_dir))


  def train_one_iter_er(self, X0, X0_out, T):
    """ Train the model for one epoch.
    """

    self.nete.train()
    self.netr.train()

    # set mini-batch
    self.X0, self.X0_out, self.T = X0, X0_out, T #batch_generator(self.ori_data, self.ori_time, self.opt.batch_size)
    self.X = self.X0.to(self.device)
    self.X_out = self.X0_out.to(self.device)

    # train encoder & decoder
    self.optimize_params_er()

  def train_one_iter_er_(self, X0, X0_out, T):
    """ Train the model for one epoch.
    """

    self.nete.train()
    self.netr.train()

    # set mini-batch
    self.X0, self.X0_out, self.T = X0, X0_out, T #batch_generator(self.ori_data, self.ori_time, self.opt.batch_size)
    #--- TODO check if we need to copy again to device, e.g if a generator train step was done before embedder_ train step
    self.X = self.X0.to(self.device)
    self.X_out = self.X0_out.to(self.device)

    # train encoder & decoder
    self.optimize_params_er_()
 
  def train_one_iter_s(self, X0, T):
    """ Train the model for one epoch.
    """

    #self.nete.eval()
    self.nete.train()
    self.nets.train()

    # set mini-batch
    self.X0, self.T = X0, T #batch_generator(self.ori_data, self.ori_time, self.opt.batch_size)
    self.X = self.X0.to(self.device)
    
    # train superviser
    self.optimize_params_s()

  def train_one_iter_g(self, X0, X0_out, T, note_ids0, note_en0, is_note0):
    """ Train the model for one epoch.
    """

    """self.netr.eval()
    self.nets.eval()
    self.netd.eval()"""
    self.netg.train()

    # set mini-batch
    self.X0, self.X0_out, self.T = X0, X0_out, T #batch_generator(self.ori_data, self.ori_time, self.opt.batch_size)
    self.note_ids0, self.note_en0, self.is_note0 = note_ids0, note_en0, is_note0
    self.X = self.X0.to(self.device)
    self.X_out = self.X0_out.to(self.device)
    self.note_ids = self.note_ids0.to(self.device)
    self.note_en = self.note_en0.to(self.device)
    self.is_note = self.is_note0.to(self.device)

    batch_size = min(len(self.T), self.opt.batch_size) #--- last batch in epoch is probably smaller
    self.Z = random_generator(batch_size, self.opt.latent_dim, self.T, self.max_seq_len)

    # train superviser
    self.optimize_params_g()

  def train_one_iter_d(self, X0, T, note_ids0, note_en0, is_note0):
    """ Train the model for one epoch.
    """
    """self.nete.eval()
    self.netr.eval()
    self.nets.eval()
    self.netg.eval()"""
    self.netd.train()

    # set mini-batch
    self.X0, self.T = X0, T #batch_generator(self.ori_data, self.ori_time, self.opt.batch_size)
    self.note_ids0, self.note_en0, self.is_note0 = note_ids0, note_en0, is_note0
    self.X = self.X0.to(self.device)
    self.note_ids = self.note_ids0.to(self.device)
    self.note_en = self.note_en0.to(self.device)
    self.is_note = self.is_note0.to(self.device)
    
    batch_size = min(len(self.T), self.opt.batch_size) #--- last batch in epoch is probably smaller
    self.Z = random_generator(batch_size, self.opt.latent_dim, self.T, self.max_seq_len)

    # train superviser
    self.optimize_params_d()

  def evaluate_d(self):
    self.nete.eval()
    self.netd.eval()
    self.netg.eval()
    self.nets.eval()

    acc_real = 0.
    acc_fake = 0.
    acc_fake_e = 0.
    sz_tot = 0
    with torch.no_grad():
      for batch in self.val_dataloader: # range(self.opt.iteration):
        seq, seq_out, seq_len, note_ids, note_en, is_note = batch
        
        self.X = seq.to(self.device)
        self.note_ids = note_ids.to(self.device)
        self.note_en = note_en.to(self.device)
        self.is_note = is_note.to(self.device)
    
        batch_size, T = seq.shape[0:2]
        T = [T] * batch_size
        sz_tot += batch_size
        
        self.Z = random_generator(batch_size, self.opt.latent_dim, T, self.max_seq_len)
        
        self.forward_e()
        self.forward_g()
        self.forward_sg()
        self.forward_d()
        #self.forward_dg()
      
        acc_real += batch_size * (self.Y_real.detach().mean(1).cpu().numpy() >= 0.).mean()
        acc_fake += batch_size * (self.Y_fake.detach().mean(1).cpu().numpy() < 0.).mean()
        acc_fake_e += batch_size * (self.Y_fake_e.detach().mean(1).cpu().numpy() < 0.).mean()
        #print(f'sz {batch_size} acc real/fake/fake_e {acc_real:.3f}/{acc_fake:.3f}/{acc_fake_e:.3f}')
    
    self.nete.train()
    self.netd.train()
    self.netg.train()
    self.nets.train()

    acc_real /= sz_tot
    acc_fake /= sz_tot
    acc_fake_e /= sz_tot
    
    return acc_real, acc_fake, acc_fake_e
  
  def evaluate_s(self):
    self.nete.eval()
    self.nets.eval()

    with torch.no_grad():
      loss = 0.0
      for batch in self.val_dataloader: # range(self.opt.iteration):
        seq, _, _, _, _, _ = batch
        # Train for one iter
        self.X = seq.to(self.device)
        self.forward_e()
        self.forward_s()
        err_s = self.l_mse(self.H[:,1:,:], self.H_supervise[:,:-1,:]).item()
        loss += err_s
    
    loss /= len(self.val_dataloader)
    
    self.nete.train()
    self.nets.train()
    
    return loss
  
  def evaluate_er(self):
    self.nete.eval()
    self.netr.eval()
      
    with torch.no_grad():
      loss = 0.0
      for batch in self.val_dataloader: # range(self.opt.iteration):
        seq, seq_out, seq_len, _, _, _ = batch
        # Train for one iter
        self.X = seq.to(self.device)
        self.X_out = seq_out.to(self.device)
        self.forward_er()
        err_er = self.l_mse(self.X_tilde, self.X_out).item()
        loss += err_er

    loss /= len(self.val_dataloader)

    self.nete.train()
    self.netr.train()

    return loss

  def get_validation_examples(self, n_samples = 5, seed = 42, get_from_train_set = False, decode = True):
    ''' get the 'autoencoder' part validation examples (encode and decode)
        also return the condional data (notes) for use with the generator
    '''
    rng = np.random.default_rng(seed)
    val_data = self.val_dataloader.dataset if not get_from_train_set else self.train_dataloader.dataset
    n_val = len(val_data)
    val_example_idx = rng.integers(0, n_val, n_samples)
    val_examples_in = np.zeros((n_samples, self.max_seq_len, self.opt.z_dim), dtype = np.float32)
    val_examples_out = np.zeros((n_samples, self.max_seq_len, self.opt.z_dim_out), dtype = np.float32)
    val_examples_note_ids = np.zeros_like(val_examples_in)
    val_examples_note_en = np.zeros_like(val_examples_in)
    val_examples_is_note = np.zeros_like(val_examples_in)
    for iidx, idx in enumerate(val_example_idx):
        x_in, x_out, _, note_id, note_en, is_note = val_data[idx]
        val_examples_in[iidx] = x_in
        val_examples_out[iidx] = x_out
        val_examples_note_ids[iidx] = note_id
        val_examples_note_en[iidx] = note_en
        val_examples_is_note[iidx] = is_note
    
    val_examples_tilde = None
    if decode:
        self.nete.eval()
        self.netr.eval()
        with torch.no_grad():
            val_examples_in = torch.tensor(val_examples_in, dtype = torch.float32).to(self.device)
            val_examples_tilde = self.netr(self.nete(val_examples_in))
    
        #val_examples = val_examples.cpu().numpy()
        val_examples_tilde = val_examples_tilde.detach().cpu().numpy()
    
        self.nete.train()
        self.netr.train()

    return val_examples_out, val_examples_tilde, val_examples_note_ids, val_examples_note_en, val_examples_is_note

  def train(self, log_batch = False):
    """ Train the model
    """
    iter_all = 0
    for epoch in range(self.opt.num_epochs_es):
        train_loss = 0.0
        for batch in self.train_dataloader: # range(self.opt.iteration):
            #seq, seq_out, seq_len, note_ids, note_en, is_note = batch
            seq, seq_out, seq_len, _, _, _ = batch
            # Train for one iter
            self.train_one_iter_er(seq, seq_out, seq_len)
            train_loss += self.err_er.item()
            iter_all += 1

        #--- report at epoch end
        #  if iter > 0 and iter % self.opt.print_freq == 0:
        train_loss /= len(self.train_dataloader)
        val_loss = self.evaluate_er()
        val_x, val_x_tilde, _, _, _= self.get_validation_examples()
        self.writer.add_scalars('Loss Embedder', dict(train = train_loss, validation = val_loss), epoch)
        #fig_list = []
        k = 0 # which channel to plot (the auto-encoder decodes all channels - TODO consider just decode 1 channel??)
        for ifig in range(val_x.shape[0]):
            fig, ax = plt.subplots()
            ax.plot(val_x[ifig,:,k], 'o')
            ax.plot(val_x_tilde[ifig,:,k], 'x')
            ax.legend(['x','x_tilde'])
            ax.grid()
            self.writer.add_figure(f'Embedder Val/{ifig}', fig, epoch)
            plt.close(fig)
            #fig_list.append(fig)

        self.writer.flush()
        print(f'Encoder training epoch: {epoch}/{self.opt.num_epochs_es} training loss {train_loss:.5f}, validation loss {val_loss:.5f}')
    
    for epoch in range(self.opt.num_epochs_es):
        train_loss = 0.0
        for batch in self.train_dataloader: # range(self.opt.iteration):
          seq, _, seq_len, _, _, _ = batch
          
          # Train for one iter
          self.train_one_iter_s(seq, seq_len)
          train_loss += self.err_s.item()

        #if iter > 0 and iter % self.opt.print_freq == 0:
        train_loss /= len(self.train_dataloader)
        val_loss_s = self.evaluate_s()
        val_loss_e = self.evaluate_er() #--- DEBUG make sure that back-prop updates embedder (self.nete) as well, so loss must change
        self.writer.add_scalars('Loss Supervised', dict(train = train_loss, validation = val_loss_s), epoch)
        self.writer.flush()
        print(f'Supervisor training epoch: {epoch}/{self.opt.num_epochs_es} training loss {train_loss:.5f}, validation loss {val_loss_s:.5f} (embedder loss {val_loss_e:.5f})')
    
    for epoch in range(self.opt.num_epochs):
        running_loss_g = 0.0
        running_loss_er_ = 0.0
        running_loss_d = 0.0
        for batch in self.train_dataloader: # range(self.opt.iteration):
          seq, seq_out, seq_len, note_ids, note_en, is_note = batch
    
          # Train for one iter
          for kk in range(2):
            self.train_one_iter_g(seq, seq_out, seq_len, note_ids, note_en, is_note)
            self.train_one_iter_er_(seq, seq_out, seq_len)

          self.train_one_iter_d(seq, seq_len, note_ids, note_en, is_note)
          
          running_loss_g += self.err_g.item()
          running_loss_er_ += self.err_er_.item()
          running_loss_d += self.err_d.item()

        #--- eval discriminator accuracy TODO add to tensorboard writer
        acc_real, acc_fake, acc_fake_e = self.evaluate_d()
        #if iter > 0 and iter % self.opt.print_freq == 0:
        running_loss_g /= len(self.train_dataloader)
        running_loss_er_ /= len(self.train_dataloader)
        running_loss_d /= len(self.train_dataloader)
        self.writer.add_scalar('Joint Loss/ g train', running_loss_g, epoch)
        self.writer.add_scalar('Joint Loss/ er_ train', running_loss_er_, epoch)
        self.writer.add_scalar('Joint Loss/ d train', running_loss_d, epoch)
            
        #--- plot generated samples
        num_to_generate = 8
        generated_samples, conditioned_env = self.generation(num_to_generate)
        k = 0 # which channel to plot (the auto-encoder decodes all channels - TODO consider just decode 1 channel??)
        for ifig in range(num_to_generate):
            fig, ax = plt.subplots()
            ax.plot(generated_samples[ifig,:,k], '.-')
            ax.plot(conditioned_env[ifig,:,k], '.:')
            #ax.legend(['x','x_tilde'])
            ax.grid()
            self.writer.add_figure(f'Generator/{ifig}', fig, epoch)

        self.writer.flush()
        print(f'Joint training epoch: {epoch}/{self.opt.num_epochs} loss gen: {running_loss_g:.5f} disc: {running_loss_d:.5f} disc acc real/fake/fake_e: {acc_real:.3f}/{acc_fake:.3f}/{acc_fake_e:.3f}')
    
    self.save_weights(self.opt.iteration)
    self.generated_data = self.generation(self.opt.batch_size)
    self.writer.close()
    print('Finish Synthetic Data Generation')

  #  self.evaluation()


  """def evaluation(self):
    ## Performance metrics
    # Output initialization
    metric_results = dict()

    # 1. Discriminative Score
    discriminative_score = list()
    for _ in range(self.opt.metric_iteration):
      temp_disc = discriminative_score_metrics(self.ori_data, self.generated_data)
      discriminative_score.append(temp_disc)

    metric_results['discriminative'] = np.mean(discriminative_score)

    # 2. Predictive score
    predictive_score = list()
    for tt in range(self.opt.metric_iteration):
      temp_pred = predictive_score_metrics(self.ori_data, self.generated_data)
      predictive_score.append(temp_pred)

    metric_results['predictive'] = np.mean(predictive_score)

    # 3. Visualization (PCA and tSNE)
    visualization(self.ori_data, self.generated_data, 'pca')
    visualization(self.ori_data, self.generated_data, 'tsne')

    ## Print discriminative and predictive scores
    print(metric_results)
"""

  #def generation(self, num_samples, mean = 0.0, std = 1.0): #--- see commnet in random_generator method (itamar katz)
  def generation(self, num_samples, uni_min = 0.0, uni_max = 1.0):
    if num_samples == 0:
      return None, None
    ## Synthetic data generation
    #self.X0, self.T = batch_generator(self.ori_data, self.ori_time, self.opt.batch_size)
    seq_out, _, note_ids, note_en, is_note = self.get_validation_examples(n_samples = num_samples, decode = False)
    T = [self.max_seq_len] * num_samples
    self.Z = random_generator(num_samples, self.opt.latent_dim, T, self.max_seq_len, uni_min, uni_max) #mean, std)
    self.Z = torch.tensor(self.Z, dtype=torch.float32).to(self.device)
    
    note_ids = torch.tensor(note_ids, dtype = torch.int).to(self.device)
    note_en = torch.tensor(note_en, dtype = torch.int).to(self.device)
    is_note = torch.tensor(is_note, dtype = torch.int).to(self.device)
    
    self.netg.eval()
    self.nets.eval()
    self.netr.eval()
    with torch.no_grad():
        self.E_hat = self.netg(self.Z, note_ids, note_en, is_note)    # [?, 24, 24]
        self.H_hat = self.nets(self.E_hat)  # [?, 24, 24]
        self.Y_fake = self.netd(self.H_hat, note_ids, note_en, is_note)
        self.Y_fake_e = self.netd(self.E_hat, note_ids, note_en, is_note)
        generated_data_curr = self.netr(self.H_hat).cpu().detach().numpy()  # [?, 24, 24]
    
    self.netg.train()
    self.nets.train()
    self.netr.train()

    return generated_data_curr, seq_out

    #generated_data = list()
    #for i in range(num_samples):
    #  temp = generated_data_curr[i, :self.ori_time[i], :]
    #  generated_data.append(temp)
    # 
    # # Renormalization
    # generated_data = generated_data * self.max_val
    # generated_data = generated_data + self.min_val
    # return generated_data



class EnvelopeTimeGAN(BaseModel):
    """TimeGAN Class
    """

    @property
    def name(self):
      return 'EnvelopeTimeGAN'

    def __init__(self, opt, train_dataloader, val_dataloader):
      super().__init__(opt, train_dataloader, val_dataloader)

      # -- Misc attributes
      self.epoch = 0
      self.times = []
      self.total_steps = 0

      # Create and initialize networks.
      self.net_note_embed = nn.Embedding(NUM_MIDI_TOKENS, opt.embedding_dim)
      self.nete = Encoder(self.opt).to(self.device)
      self.netr = Recovery(self.opt).to(self.device)
      self.netg = Generator(self.opt, self.net_note_embed).to(self.device)
      self.netd = Discriminator(self.opt, self.net_note_embed).to(self.device)
      self.nets = Supervisor(self.opt).to(self.device)

      if self.opt.resume != '':
        print("\nLoading pre-trained networks.")
        self.opt.iter = torch.load(os.path.join(self.opt.resume, 'netG.pth'))['epoch']
        self.net_note_embed.load_state_dict(torch.load(os.path.join(self.opt.resume, 'netNE.pth'))['state_dict'])
        self.nete.load_state_dict(torch.load(os.path.join(self.opt.resume, 'netE.pth'))['state_dict'])
        self.netr.load_state_dict(torch.load(os.path.join(self.opt.resume, 'netR.pth'))['state_dict'])
        self.netg.load_state_dict(torch.load(os.path.join(self.opt.resume, 'netG.pth'))['state_dict'])
        self.netd.load_state_dict(torch.load(os.path.join(self.opt.resume, 'netD.pth'))['state_dict'])
        self.nets.load_state_dict(torch.load(os.path.join(self.opt.resume, 'netS.pth'))['state_dict'])
        print("\tDone.\n")

      # loss
      self.l_mse = nn.MSELoss()
      self.l_r = nn.L1Loss()
      self.l_bce = nn.BCEWithLogitsLoss() #snn.BCELoss()

      # Setup optimizer
      if self.opt.isTrain:
        self.nete.train()
        self.netr.train()
        self.netg.train()
        self.netd.train()
        self.nets.train()
        self.optimizer_e = optim.Adam(self.nete.parameters(), lr=self.opt.lr, betas=(self.opt.beta1, 0.999))
        self.optimizer_r = optim.Adam(self.netr.parameters(), lr=self.opt.lr, betas=(self.opt.beta1, 0.999))
        self.optimizer_g = optim.Adam(self.netg.parameters(), lr=self.opt.lr, betas=(self.opt.beta1, 0.999))
        self.optimizer_d = optim.Adam(self.netd.parameters(), lr=self.opt.lr, betas=(self.opt.beta1, 0.999))
        self.optimizer_s = optim.Adam(self.nets.parameters(), lr=self.opt.lr, betas=(self.opt.beta1, 0.999))


    def forward_e(self):
      """ Forward propagate through netE
      """
      self.H = self.nete(self.X)

    def forward_er(self):
      """ Forward propagate through netR
      """
      self.H = self.nete(self.X)
      self.X_tilde = self.netr(self.H)

    def forward_g(self):
      """ Forward propagate through netG
      """
      self.Z = torch.tensor(self.Z, dtype=torch.float32).to(self.device)
      self.E_hat = self.netg(self.Z, self.note_ids, self.note_en, self.is_note)
    
    def forward_dg(self):
      """ Forward propagate through netD
      """
      self.Y_fake = self.netd(self.H_hat, self.note_ids, self.note_en, self.is_note)
      self.Y_fake_e = self.netd(self.E_hat, self.note_ids, self.note_en, self.is_note)

    def forward_rg(self):
      """ Forward propagate through netR
      """
      self.X_hat = self.netr(self.H_hat)

    def forward_s(self):
      """ Forward propagate through netS
      """
      self.H_supervise = self.nets(self.H)
      # print(self.H, self.H_supervise)

    def forward_sg(self):
      """ Forward propagate through netS
      """
      self.H_hat = self.nets(self.E_hat)

    def forward_d(self):
      """ Forward propagate through netD
      """
      self.Y_real = self.netd(self.H, self.note_ids, self.note_en, self.is_note)
      self.Y_fake = self.netd(self.H_hat, self.note_ids, self.note_en, self.is_note)
      self.Y_fake_e = self.netd(self.E_hat, self.note_ids, self.note_en, self.is_note)

    def backward_er(self):
      """ Backpropagate through netE
      """
      self.err_er = self.l_mse(self.X_tilde, self.X_out)
      self.err_er.backward(retain_graph=True)
      if self.log_batch:
          print(f"Loss: {self.err_er.item():.4f}")

    def backward_er_(self):
      """ Backpropagate through netE
      """
      self.err_er_ = self.l_mse(self.X_tilde, self.X_out) 
      self.err_s = self.l_mse(self.H_supervise[:,:-1,:], self.H[:,1:,:])
      self.err_er = 10 * torch.sqrt(self.err_er_) + 0.1 * self.err_s
      self.err_er.backward(retain_graph=True)

    #  print("Loss: ", self.err_er_, self.err_s)
    def backward_g(self):
      """ Backpropagate through netG
      """
      if self.opt.average_seq_before_loss:
          score_fake = self.Y_fake.mean(1)
          score_fake_e = self.Y_fake_e.mean(1)
      else:
          score_fake = self.Y_fake
          score_fake_e = self.Y_fake_e

      self.err_g_U = self.l_bce(score_fake, torch.ones_like(score_fake))
      self.err_g_U_e = self.l_bce(score_fake_e, torch.ones_like(score_fake_e))
      #self.err_g_U = self.l_bce(self.Y_fake, torch.ones_like(self.Y_fake))
      #self.err_g_U_e = self.l_bce(self.Y_fake_e, torch.ones_like(self.Y_fake_e))
      
      self.err_g_V1 = torch.mean(torch.abs(torch.sqrt(torch.std(self.X_hat,[0])[1] + 1e-6) - torch.sqrt(torch.std(self.X_out,[0])[1] + 1e-6)))   # |a^2 - b^2|
      self.err_g_V2 = torch.mean(torch.abs((torch.mean(self.X_hat,[0])[0]) - (torch.mean(self.X_out,[0])[0])))  # |a - b|
      self.err_s = self.l_mse(self.H_supervise[:,:-1,:], self.H[:,1:,:])
      self.err_g = self.err_g_U + \
                   self.err_g_U_e * self.opt.w_gamma + \
                   self.err_g_V1 * self.opt.w_g + \
                   self.err_g_V2 * self.opt.w_g + \
                   torch.sqrt(self.err_s) 
      self.err_g.backward(retain_graph=True)
      if self.log_batch:
          print(f"Loss G: {self.err_g.item():.4f}")

    def backward_s(self):
      """ Backpropagate through netS
      """
      self.err_s = self.l_mse(self.H[:,1:,:], self.H_supervise[:,:-1,:])
      self.err_s.backward(retain_graph=True)
      if self.log_batch:
          print(f"Loss S: {self.err_s.item():.4f}")
   #   print(torch.autograd.grad(self.err_s, self.nets.parameters()))

    def backward_d(self):
      """ Backpropagate through netD
      """
      if self.opt.average_seq_before_loss:
          score_real = self.Y_real.mean(1)
          score_fake = self.Y_fake.mean(1)
          score_fake_e = self.Y_fake_e.mean(1)
      else:
          score_real = self.Y_real
          score_fake = self.Y_fake
          score_fake_e = self.Y_fake_e
      
      self.err_d_real = self.l_bce(score_real, torch.ones_like(score_real))
      self.err_d_fake = self.l_bce(score_fake, torch.zeros_like(score_fake))
      self.err_d_fake_e = self.l_bce(score_fake_e, torch.zeros_like(score_fake_e))
      #self.err_d_real = self.l_bce(self.Y_real, torch.ones_like(self.Y_real))
      #self.err_d_fake = self.l_bce(self.Y_fake, torch.zeros_like(self.Y_fake))
      #self.err_d_fake_e = self.l_bce(self.Y_fake_e, torch.zeros_like(self.Y_fake_e))
      self.err_d = self.err_d_real + \
                   self.err_d_fake + \
                   self.err_d_fake_e * self.opt.w_gamma
      if self.err_d > 0.15:
        self.err_d.backward(retain_graph=True)

     # print("Loss D: ", self.err_d)

    def optimize_params_er(self):
      """ Forwardpass, Loss Computation and Backwardpass.
      """
      # Forward-pass
      self.forward_er()

      # Backward-pass
      # nete & netr
      self.optimizer_e.zero_grad()
      self.optimizer_r.zero_grad()
      self.backward_er()
      self.optimizer_e.step()
      self.optimizer_r.step()

    def optimize_params_er_(self):
      """ Forwardpass, Loss Computation and Backwardpass.
      """
      # Forward-pass
      self.forward_er()
      self.forward_s()
      # Backward-pass
      # nete & netr
      self.optimizer_e.zero_grad()
      self.optimizer_r.zero_grad()
      self.backward_er_()
      self.optimizer_e.step()
      self.optimizer_r.step()

    def optimize_params_s(self):
      """ Forwardpass, Loss Computation and Backwardpass.
      """
      # Forward-pass
      self.forward_e()
      self.forward_s()

      # Backward-pass
      # nets
      self.optimizer_s.zero_grad()
      self.backward_s()
      self.optimizer_s.step()

    def optimize_params_g(self):
      """ Forwardpass, Loss Computation and Backwardpass.
      """
      # Forward-pass
      self.forward_e()
      self.forward_s()
      self.forward_g()
      self.forward_sg()
      self.forward_rg()
      self.forward_dg()

      # Backward-pass
      # nets
      self.optimizer_g.zero_grad()
      self.optimizer_s.zero_grad()
      self.backward_g()
      self.optimizer_g.step()
      self.optimizer_s.step()

    def optimize_params_d(self):
      """ Forwardpass, Loss Computation and Backwardpass.
      """
      # Forward-pass
      self.forward_e()
      self.forward_g()
      self.forward_sg()
      self.forward_d()
      self.forward_dg()

      # Backward-pass
      # nets
      self.optimizer_d.zero_grad()
      self.backward_d()
      self.optimizer_d.step()
