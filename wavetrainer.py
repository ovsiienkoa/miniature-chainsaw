from waveformer import WaveFormer, initialize_weights
from gradio_app.case import Case

import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter

import numpy as np
import torch
import torch.nn as nn

from sklearn.metrics import root_mean_squared_error, r2_score

class TorchDS(torch.utils.data.Dataset):
    def __init__(self, X, y):
        super().__init__()
        self.len = len(y)
        self.X = torch.tensor(X, dtype = torch.float)
        self.y = torch.tensor(y, dtype = torch.float)

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class WaveNN:
    def __init__(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")
        self.model = None
        self.n_features = None
        self.context_size = None
        self.n_blocks = 1

    def _train_architecture(self,
                            model_name:str,
                            train_dataloader:torch.utils.data.DataLoader,
                            eval_dataloader:torch.utils.data.DataLoader,
                            input_length:int,
                            output_length:int,
                            n_blocks:int,
                            verbose:int,
                            verbose_update_freq:int,
                            learning_rate:float,
                            weight_decay:float,
                            n_epochs:int,
                            max_grad_norm:float):
        if verbose ==2:
            writer = SummaryWriter()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = WaveFormer(input_length = input_length, output_length = output_length, n_blocks = n_blocks, n_features = self.n_features)
        self.model.apply(initialize_weights)
        self.model.to(device)

        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate, weight_decay=weight_decay)

        self.model.train()
        cum_loss_ev = 0
        for epoch in range(n_epochs):
            if verbose ==2:
                cum_loss = 0
                cum_norm = 0

            for features, targets in train_dataloader:

                features = features.reshape(-1, 1, input_length, self.n_features) #batch_size, n_channels, (h,w)
                targets = targets.reshape(-1, 1, output_length)
                features, targets = features.to(device), targets.to(device)

                optimizer.zero_grad()

                outputs = self.model(features)
                outputs = torch.squeeze(outputs)
                targets = torch.squeeze(targets)

                loss = criterion(outputs, targets)
                if verbose == 2:
                    cum_loss +=loss.item()
                loss.backward()

                #clip grad
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm = max_grad_norm)
                optimizer.step()

                if verbose == 2:
                    norm = 0
                    for p in self.model.parameters():
                        if p.grad is not None:
                            param_norm = p.grad.data.norm(2)
                            norm += param_norm.item() ** 2

                    cum_norm += norm ** (1./2)

            if (epoch % verbose_update_freq == 0) and (verbose == 2):
                writer.add_scalar(f'grad/{model_name}/train', cum_norm, epoch)

            if verbose != 0:
                with torch.no_grad():
                    cum_loss_ev = 0
                    for features, targets in eval_dataloader:
                        features = features.reshape(-1, 1, input_length, self.n_features) #batch_size, n_channels, (h,w)
                        targets = targets.reshape(-1, 1, output_length)
                        features, targets = features.to(device), targets.to(device)
                        outputs = self.model(features)
                        targets = torch.squeeze(targets)
                        outputs = torch.squeeze(outputs)
                        loss = criterion(outputs, targets)
                        cum_loss_ev +=loss.item()
                if verbose == 2:
                    writer.add_scalars(f'losses/{model_name}',{"train": cum_loss, "eval": cum_loss_ev}, epoch)

        if verbose ==2:
            writer.close()

        return cum_loss_ev

    def train(self,
              case:Case,
              epochs:int,
              verbose:int = 0,  #0 - nothing happens; 1 - eval for optuna; 2 - eval, params, tensorboard etc
              verbose_update_freq:int = 10,
              experiment:bool = True,
              distill_loops:int = 0):

        distill_loops = distill_loops+1
        self.context_size = case.context_size

        batch_size = 64

        train_dict = case.sample('train')

        self.n_features = case.n_features #after .sample because case.n_features is reinitialized during .sample
        if not experiment:
            train_dict = case.sample('full')

        eval_dict = case.sample('eval')
        X_eval, y_eval = eval_dict['features'], eval_dict['targets']
        eval_ds = TorchDS(np.array(X_eval), np.array(y_eval))
        eval_dataloader = torch.utils.data.DataLoader(eval_ds, batch_size = len(eval_ds), drop_last = False, shuffle = False)

        X_train = train_dict['features']

        for i in range(distill_loops):
            if i == 0:
                y_train = train_dict['targets']
            else:
                y_train = self.predict(np.array(X_train))

            torch_ds = TorchDS(np.array(X_train), np.array(y_train))
            dataloader = torch.utils.data.DataLoader(torch_ds, batch_size = batch_size, drop_last = False, shuffle = False)

            self._train_architecture(train_dataloader = dataloader,
                                             eval_dataloader = eval_dataloader,
                                             input_length= case.context_size,
                                             output_length= case.predict_days_size,
                                             n_blocks = self.n_blocks,
                                             verbose = verbose,
                                             verbose_update_freq = verbose_update_freq,
                                             learning_rate = 0.002,
                                             weight_decay = 3e-5,#3e-4?
                                             max_grad_norm = 1,
                                             n_epochs = epochs,
                                             model_name = case.case_name
                                             )

    def evaluate(self, X_test = None, y_test= None, case_sample = None, plot:bool = False):

        if case_sample is not None:
            test_dict = case_sample
            X_test, y_test, actual_prices = test_dict['features'], test_dict['targets'], test_dict['actual_prices']


        pred = self.predict(np.array(X_test))
        test = np.array(y_test).squeeze()

        if plot:
            plt.figure(figsize=(15, 12))
            try:
                if len(actual_prices) > 0:
                    actual_prices = np.array(actual_prices).squeeze()
                    week_lag = np.concatenate((np.ones(7)*np.mean(actual_prices[:7]), actual_prices[:-7]))
                    plt.plot(actual_prices)
                    plt.plot(week_lag)
            except:
                week_lag = np.concatenate((np.ones(7) * np.mean(test[:7]), test[:-7]))
                plt.plot(week_lag)
            #running_mean = np.mean(np.array(X_test)[:,:,0], 1)
            plt.grid()
            plt.plot(test)
            plt.plot(pred)

            plt.show()

        return root_mean_squared_error(test, pred), r2_score(test, pred)

    def predict(self, X) -> np.array:

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(device)
        X = torch.tensor(X, dtype = torch.float)
        X = X.reshape(-1, 1, self.context_size, self.n_features).to(device)
        self.model.eval()
        with torch.no_grad():
            y = self.model(X)
        y = torch.squeeze(y)
        y = y.numpy(force = True)

        return y
