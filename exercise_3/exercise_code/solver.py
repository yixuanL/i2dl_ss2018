from random import shuffle
import numpy as np

import torch
from torch.autograd import Variable


class Solver(object):
    default_adam_args = {"lr": 1e-4,
                         "betas": (0.9, 0.999),
                         "eps": 1e-8,
                         "weight_decay": 0.0}

    def __init__(self, optim=torch.optim.Adam, optim_args={},
                 loss_func=torch.nn.CrossEntropyLoss()):
        optim_args_merged = self.default_adam_args.copy()
        optim_args_merged.update(optim_args)
        self.optim_args = optim_args_merged
        self.optim = optim
        self.loss_func = loss_func

        self._reset_histories()

    def _reset_histories(self):
        """
        Resets train and val histories for the accuracy and the loss.
        """
        self.train_loss_history = []
        self.train_acc_history = []
        self.val_acc_history = []
        self.val_loss_history = []

    def check_accuracy(self, model, X, y):
        """
        Check accuracy of the model on the provided data.

        Inputs:
        - X: Array of data, of shape (N, d_1, ..., d_k)
        - y: Array of labels, of shape (N,)
        - num_samples: If not None, subsample the data and only test the model
          on num_samples datapoints.
        - batch_size: Split X and y into batches of this size to avoid using too
          much memory.

        Returns:
        - acc: Scalar giving the fraction of instances that were correctly
          classified by the model.
        """
        
        #loss = self.loss_func(X, y).detach().numpy()
        y_pred = np.argmax(X.detach().numpy(), axis=1)
        #y_pred = np.hstack(y_pred)
        #print(y.detach().numpy())
        acc = np.mean(y_pred == y.detach().numpy())

        return acc


    def train(self, model, train_loader, val_loader, num_epochs=20, log_nth=100):
        """
        Train a given model with the provided data.

        Inputs:
        - model: model object initialized from a torch.nn.Module
        - train_loader: train data in torch.utils.data.DataLoader
        - val_loader: val data in torch.utils.data.DataLoader
        - num_epochs: total number of training epochs
        - log_nth: log training accuracy and loss every nth iteration
        """
        optim = self.optim(model.parameters(), **self.optim_args)
        self._reset_histories()
        iter_per_epoch = len(train_loader)
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model.to(device)

        print('START TRAIN.')
        ########################################################################
        # TODO:                                                                #
        # Write your own personal training method for our solver. In each      #
        # epoch iter_per_epoch shuffled training batches are processed. The    #
        # loss for each batch is stored in self.train_loss_history. Every      #
        # log_nth iteration the loss is logged. After one epoch the training   #
        # accuracy of the last mini batch is logged and stored in              #
        # self.train_acc_history. We validate at the end of each epoch, log    #
        # the result and store the accuracy of the entire validation set in    #
        # self.val_acc_history.                                                #
        #                                                                      #
        # Your logging could like something like:                              #
        #   ...                                                                #
        #   [Iteration 700/4800] TRAIN loss: 1.452                             #
        #   [Iteration 800/4800] TRAIN loss: 1.409                             #
        #   [Iteration 900/4800] TRAIN loss: 1.374                             #
        #   [Epoch 1/5] TRAIN acc/loss: 0.560/1.374                            #
        #   [Epoch 1/5] VAL   acc/loss: 0.539/1.310                            #
        #   ...                                                                #
        ########################################################################
        
        num_iterations = num_epochs*iter_per_epoch
        # create an optimizer
        optimizer = optim

        for epoch in range(num_epochs):
            print('[Epoch ', epoch+1, '/', num_epochs,']')
            
            running_loss = 0.0
            running_corrects = 0
            
            best_train_acc = 0.0
            best_val_acc = 0.0
            
            for step, (input_, target) in enumerate(train_loader):
                #input_ : batch_X, target : batch_y
                
                # a single step of an example training loop
                optimizer.zero_grad()   # zero the gradient buffers
                output = model(input_)
                train_loss = self.loss_func(output, target)
                train_loss.backward()
                optimizer.step()    # Does the update based on the accumalted gradients
                if step%log_nth == 0:
                    print('[Iteration ', step+1+iter_per_epoch*epoch, '/', num_iterations,'] Train loss:', train_loss)
                
                self.train_loss_history.append(train_loss)
                
                if step == iter_per_epoch-1:
                    train_acc = self.check_accuracy(model, output, target)
                    self.train_acc_history.append(train_acc)
                    print('Train acc/loss:',train_acc, '/', train_loss)
                
            for step, (input_, target) in enumerate(train_loader):
                
                # a single step of an example training loop
                optimizer.zero_grad()   # zero the gradient buffers
                output = model(input_)
                val_loss = self.loss_func(output, target)
                
                #print('[input: ', input_.numpy(), '] [target: ', target.numpy(),']')
                self.val_loss_history.append(val_loss)
                
                if step == iter_per_epoch-1:
                    val_acc = self.check_accuracy(model, output, target)
                    self.val_acc_history.append(val_acc)
                    print('Train acc/loss:',val_acc,'/',val_loss)
                    
        ########################################################################
        #                             END OF YOUR CODE                         #
        ########################################################################
        print('FINISH.')
