import os
import configparser
import operator
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import truncnorm
from functools import reduce
from operator import mul

import operator
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from scipy.stats import truncnorm


from torch.autograd import Variable
from torchvision import datasets, transforms
import time

dtype = torch.FloatTensor

### Metal Model (i.e. Optimizee)  ###


class MetaModel:
    def __init__(self, model):
        self.model = model

    def reset(self):
        
        
        for p in self.model.children():
            if len(p._parameters) ==0:
                for pp in p.children():
                    if(len(pp._parameters) == 0):
                        for ppp in pp.children():
                            if (len(ppp._parameters) == 0):
                                for pppp in ppp.children():
                                    pppp._parameters['weight'] = Variable(
                                        pppp._parameters['weight'].data)
                                    try:
                                        pppp._parameters['bias'] = Variable(
                                            pppp._parameters['bias'].data)
                                    except:
                                        pass
                            else:
                                ppp._parameters['weight'] = Variable(
                                    ppp._parameters['weight'].data)
                                try:
                                    ppp._parameters['bias'] = Variable(
                                        ppp._parameters['bias'].data)
                                except:
                                    pass
                                
                                
                else:
                        pass
            else:
                p._parameters['weight'] = Variable(
                    p._parameters['weight'].data)
                try:
                    p._parameters['bias'] = Variable(
                        p._parameters['bias'].data)
                except:
                    pass
        
        

    def get_flat_params(self):
        params = []
        offset = 0
        for p in self.model.children():
            if len(p._parameters) ==0:
                for pp in p.children():
                    if(len(pp._parameters) == 0):
                        for ppp in pp.children():
                            if (len(ppp._parameters) == 0):
                                for pppp in ppp.children():
                                    params.append(pppp._parameters['weight'].view(-1))
                                    try:
                                        params.append(pppp._parameters['bias'].view(-1))
                                    except:
                                        pass
                            else:
                                params.append(ppp._parameters['weight'].view(-1))
                                try:
                                    params.append(ppp._parameters['bias'].view(-1))
                                except:
                                    pass
                else:
                        pass
            else:
                params.append(p._parameters['weight'].view(-1))
                try:
                    params.append(p._parameters['bias'].view(-1))
                except:
                    pass
        return torch.cat(params)

    def set_flat_params(self, flat_params):
        # Restore original shapes
        params = []
        offset = 0
        for p in self.model.children():
            if len(p._parameters) ==0:
                for pp in p.children():
                    if(len(pp._parameters) == 0):
                        for ppp in pp.children():
                            if (len(ppp._parameters) == 0):
                                for pppp in ppp.children():
                                    weight_shape = pppp._parameters['weight'].size()
                                    weight_flat_size = reduce(mul,weight_shape,1)
                                    pppp._parameters['weight'] = flat_params[
                                        offset:offset + weight_flat_size].view(*weight_shape)
                                    try:
                                        bias_shape = pppp._parameters['bias'].size()
                                        bias_flat_size = reduce(mul,bias_shape,1)
                                        pppp._parameters['bias'] = flat_params[
                                            offset + weight_flat_size:offset + weight_flat_size + bias_flat_size].view(*bias_shape)
                                    except:
                                            bias_flat_size = 0
                                    offset += weight_flat_size + bias_flat_size
                            else:
                                weight_shape = ppp._parameters['weight'].size()
                                weight_flat_size = reduce(mul,weight_shape,1)
                                ppp._parameters['weight'] = flat_params[
                                    offset:offset + weight_flat_size].view(*weight_shape)
                                try:
                                    bias_shape = ppp._parameters['bias'].size()
                                    bias_flat_size = reduce(mul,bias_shape,1)
                                    ppp._parameters['bias'] = flat_params[
                                        offset + weight_flat_size:offset + weight_flat_size + bias_flat_size].view(*bias_shape)
                                except:
                                        bias_flat_size = 0
                                offset += weight_flat_size + bias_flat_size
                    else:
                        pass
            else:
                weight_shape = p._parameters['weight'].size()
                weight_flat_size = reduce(mul,weight_shape,1)

                p._parameters['weight'] = flat_params[
                    offset:offset + weight_flat_size].view(*weight_shape)
                try:
                    bias_shape = p._parameters['bias'].size()
                    bias_flat_size = reduce(mul,bias_shape,1)
                    p._parameters['bias'] = flat_params[
                        offset + weight_flat_size:offset + weight_flat_size + bias_flat_size].view(*bias_shape)
                except:
                        bias_flat_size = 0
                offset += weight_flat_size + bias_flat_size

    def copy_params_from(self, model):
        for modelA, modelB in zip(self.model.parameters(), model.parameters()):
            modelA.data.copy_(modelB.data)

    def copy_params_to(self, model):
        for modelA, modelB in zip(self.model.parameters(), model.parameters()):
            modelB.data.copy_(modelA.data)



## MNIST
batch_size = 64
test_batch_size = 1000
train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('data/patrick-data/pytorch/data/', train=True, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])),
    batch_size=batch_size, shuffle=True)

test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('data/patrick-data/pytorch/data/', train=False, transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])),
    batch_size=test_batch_size, shuffle=True)


### The Model to Train on ###

class Model(nn.Module):

    def __init__(self):
        super(Model, self).__init__()
        self.linear1 = nn.Linear(28 * 28, 32)
        self.linear2 = nn.Linear(32, 10)

    def forward(self, inputs):
        x = inputs.view(-1, 28 * 28)
        x = F.relu(self.linear1(x))
        x = self.linear2(x)
        return F.log_softmax(x)




    
def initialize_rnn_hidden_state(dim_sum,n_layers,n_params):
    h0 = Variable(torch.zeros(n_layers,n_params,dim_sum),requires_grad = True)
    if args.cuda:
        h0.data = h0.data.cuda(args.cuda_num)
    return h0

class MetaOptimizerRNN(nn.Module):

    def __init__(self, model, num_layers, input_dim, hidden_size):
        super(MetaOptimizerRNN, self).__init__()
        self.meta_model = model
        self.first_order = nn.RNN(input_dim,hidden_size,num_layers,batch_first = True,bias = False)
        #self.second_order = nn.RNN(input_dim-2,hidden_size,num_layers,batch_first = True,bias = False)

        self.outputer = nn.Linear(hidden_size,1,bias=False)
        self.outputer.weight.data.mul_(0.1)
        #self.outputer.bias.data.fill_(0.0)
        self.hidden_size = hidden_size
        #self.outputer2 = nn.Linear(hidden_size,1,bias=False)
        #self.outputer2.weight.data.mul_(0.1)
        #self.outputer2.bias.data.fill_(0.0)

    def cuda(self):
        super(MetaOptimizerRNN, self).cuda(args.cuda_num)
        self.first_order.cuda(args.cuda_num)
        #self.second_order.cuda(args.cuda_num)
    def reset_lstm(self, keep_states=False, model=None, use_cuda=False):
        self.meta_model.reset()
        self.meta_model.copy_params_from(model)
        if keep_states > 0:
            self.h0 = Variable(self.h0.data)
        else:
            self.h0 = initialize_rnn_hidden_state(args.hidden_size,args.num_layers,self.meta_model.get_flat_params().size(0))

    def forward(self, x):
        # Gradients preprocessing
#        x = F.tanh(self.ln1(self.linear1(x)))
        output1, hn1 = self.first_order(x,self.h0)
        self.h0 = hn1
        o1 = self.outputer(output1)
        #output2, hn2 = self.second_order(x[:,:,2:],self.h1)
        #self.h1 = hn2
        #o2 = self.outputer2(output2)
        #final_update = o2[:,0,0] * o1[:,0,0]
        return o1.squeeze()

    def meta_update(self, model_with_grads, loss):
        # First we need to create a flat version of parameters and gradients
        grads = []

        for p in model_with_grads.children():
            if len(p._parameters) ==0:
                for pp in p.children():
                    if(len(pp._parameters) == 0):
                        for ppp in pp.children():
                            if (len(ppp._parameters) == 0):
                                for pppp in ppp.children():
                                    grads.append(pppp._parameters['weight'].grad.data.view(-1))
                                    try:
                                        grads.append(pppp._parameters['bias'].grad.data.view(-1))
                                    except:
                                        pass
                            else:
                                grads.append(ppp._parameters['weight'].grad.data.view(-1))
                                try:
                                    grads.append(ppp._parameters['bias'].grad.data.view(-1))
                                except:
                                    pass
                else:
                        pass
            else:
                grads.append(p._parameters['weight'].grad.data.view(-1))
                try:
                    grads.append(p._parameters['bias'].grad.data.view(-1))
                except:
                    pass

        
        flat_params = self.meta_model.get_flat_params()
        flat_grads = torch.cat(grads)
        inputs = Variable(flat_grads.view(-1,1).unsqueeze(1))

        # Meta update itself
        flat_params = flat_params + self(inputs)

        self.meta_model.set_flat_params(flat_params)

        # Finally, copy values from the meta model to the normal one.
        self.meta_model.copy_params_to(model_with_grads)
        return self.meta_model.model


args = configparser.ConfigParser()
args.batch_size = 32
args.optimizer_steps = 200
args.truncated_bptt_step = 20
args.updates_per_epoch = 10
args.max_epoch = 20
args.hidden_size = 10
args.num_layers = 1
args.no_cuda = True
args.input_dim = 1
args.cuda = not args.no_cuda and torch.cuda.is_available()
args.cuda_num = 0
assert args.optimizer_steps % args.truncated_bptt_step == 0
prefix_loss = 0.1
meta_model = Model()
if args.cuda:
    meta_model.cuda(args.cuda_num)

meta_optimizer = MetaOptimizerRNN(MetaModel(meta_model), args.num_layers, args.input_dim, args.hidden_size)
if args.cuda:
    meta_optimizer.cuda()
optimizer = optim.Adam(meta_optimizer.parameters(), lr=1e-3)

for epoch in range(args.max_epoch):
    decrease_in_loss = 0.0
    final_loss = 0.0
    train_iter = iter(train_loader)
    for i in range(args.updates_per_epoch):
        # Sample a new model
        model = Model()        
        if args.cuda:
            model.cuda(args.cuda_num)

        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = Variable(data), Variable(target)
            break
        if args.cuda:
            data, target = data.cuda(args.cuda_num), target.cuda(args.cuda_num)

        # Compute initial loss of the model
        f_x = model(data)
        initial_loss = F.nll_loss(f_x, target)
        
        for k in range(args.optimizer_steps // args.truncated_bptt_step):
            # Keep states for truncated BPTT
            
            meta_optimizer.reset_lstm(
                    keep_states=k > 0, model=model, use_cuda=args.cuda)
                
            loss_sum = 0
            prev_loss = torch.zeros(1)
            if args.cuda:
                prev_loss = prev_loss.cuda(args.cuda_num)
            for j in range(args.truncated_bptt_step):                
                for batch_idx, (data, target) in enumerate(train_loader):
                    data, target = Variable(data), Variable(target)
                    break
                if args.cuda:
                    data, target = data.cuda(args.cuda_num), target.cuda(args.cuda_num)

                # First we need to compute the gradients of the model
                f_x = model(data)
                loss = F.nll_loss(f_x, target)
                model.zero_grad()
                loss.backward()

                # Perfom a meta update using gradients from model
                # and return the current meta model saved in the optimizer
                
                meta_model = meta_optimizer.meta_update(model, loss.data)

                # Compute a loss for a step the meta optimizer
                f_x = meta_model(data)
                loss = F.nll_loss(f_x, target)
                loss_sum += (k * args.truncated_bptt_step ) * (loss - Variable(prev_loss))
                prev_loss = loss.data

            # Update the parameters of the meta optimizer
            
            
            
            meta_optimizer.zero_grad()
            loss_sum.backward()
            for param in meta_optimizer.parameters():
                param.grad.data.clamp_(-1, 1)
            optimizer.step()

        # Compute relative decrease in the loss function w.r.t initial
        # value
        decrease_in_loss += loss.item() / initial_loss.item()
        final_loss += loss.item()

    print("Epoch: {}, final loss {}, average final/initial loss ratio: {}".format(epoch, final_loss / args.updates_per_epoch,
                                                                       decrease_in_loss / args.updates_per_epoch))


