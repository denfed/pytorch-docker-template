import os
import time

import numpy as np
import pandas as pd
import torch
from torchvision.utils import make_grid

from base import BaseTrainer
from model.metric import MetricTracker, AUROC, AUPRC
from utils import inf_loop


class Trainer(BaseTrainer):
    """
    Trainer class
    """
    def __init__(self, config):
        super().__init__(config)

        if config.resume is not None:
            self._resume_checkpoint(config.resume, finetune=self.finetune)

        # data_loaders
        self.train_loader = self.train_data_loaders['data']
        self.valid_loader = self.valid_data_loaders['data']
        self.do_validation = self.valid_loader is not None
        if self.len_epoch is None:
            # epoch-based training
            self.len_epoch = len(self.train_loader)
        else:
            # iteration-based training
            self.train_loader = inf_loop(self.train_loader)
        self.seq_len = self.train_datasets['data'].seq_len
        self.log_step = int(np.sqrt(self.train_loader.batch_size))

        # models
        self.model = self.models['model']

        # losses
        self.criterion = self.losses['loss']

        # metrics
        keys_loss = ['loss']
        keys_iter = [m.__name__ for m in self.metrics_iter]
        keys_epoch = [m.__name__ for m in self.metrics_epoch]
        self.train_metrics = MetricTracker(keys_loss + keys_iter, keys_epoch, writer=self.writer)
        self.valid_metrics = MetricTracker(keys_loss + keys_iter, keys_epoch, writer=self.writer)

        # optimizers
        self.optimizer = self.optimizers['model']

        # learning rate schedulers
        self.do_lr_scheduling = len(self.lr_schedulers) > 0
        self.lr_scheduler = self.lr_schedulers['model']

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains average loss and metric in this epoch.
        """
        start = time.time()
        self.model.train()
        self.train_metrics.reset()
        hidden = self.model.initHidden(self.seq_len)
        if len(self.metrics_epoch) > 0:
            outputs = torch.FloatTensor().to(self.device)
            targets = torch.FloatTensor().to(self.device)
        for batch_idx, (data, target) in enumerate(self.train_loader):
            data, target = data.to(self.device), target.to(self.device)

            self.optimizer.zero_grad()
            output, hidden = self.model(data, hidden.data)
            if len(self.metrics_epoch) > 0:
                outputs = torch.cat((outputs, output))
                targets = torch.cat((targets, target))
            loss = self.criterion(output, target)
            loss.backward()
            self.optimizer.step()

            self.writer.set_step((epoch - 1) * self.len_epoch + batch_idx)
            self.train_metrics.iter_update('loss', loss.item())
            for met in self.metrics_iter:
                self.train_metrics.iter_update(met.__name__, met(output, target))

            if batch_idx % self.log_step == 0:
                epoch_debug = f"Train Epoch: {epoch} {self._progress(batch_idx)} "
                current_metrics = self.train_metrics.current()
                metrics_debug = ', '.join(f"{key}: {value:.6f}" for key, value in current_metrics.items())
                self.logger.debug(epoch_debug + metrics_debug)
                # self.writer.add_image('input', make_grid(data.cpu(), nrow=8, normalize=True))

            if batch_idx == self.len_epoch:
                break

        for met in self.metrics_epoch:
            self.train_metrics.epoch_update(met.__name__, met(outputs, targets))

        train_log = self.train_metrics.result()

        if self.do_validation:
            valid_log = self._valid_epoch(epoch)
            valid_log.set_index('val_' + valid_log.index.astype(str), inplace=True)

        if self.do_lr_scheduling:
            self.lr_scheduler.step()

        log = pd.concat([train_log, valid_log])
        end = time.time()
        ty_res = time.gmtime(end - start)
        res = time.strftime("%H hours, %M minutes, %S seconds", ty_res)
        epoch_log = {'epochs': epoch,
                     'iterations': self.len_epoch * epoch,
                     'Runtime': res}
        epoch_info = ', '.join(f"{key}: {value}" for key, value in epoch_log.items())
        logger_info = f"{epoch_info}\n{log}"
        self.logger.info(logger_info)

        return log

    def _valid_epoch(self, epoch):
        """
        Validate after training an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains information about validation
        """
        self.model.eval()
        self.valid_metrics.reset()
        hidden = self.model.initHidden(self.seq_len)
        with torch.no_grad():
            if len(self.metrics_epoch) > 0:
                outputs = torch.FloatTensor().to(self.device)
                targets = torch.FloatTensor().to(self.device)
            for batch_idx, (data, target) in enumerate(self.valid_loader):
                data, target = data.to(self.device), target.to(self.device)

                output, hidden = self.model(data, hidden.data)
                loss = self.criterion(output, target)
                if len(self.metrics_epoch) > 0:
                    outputs = torch.cat((outputs, output))
                    targets = torch.cat((targets, target))

                self.writer.set_step((epoch - 1) * self.len_epoch + batch_idx, 'valid')
                self.valid_metrics.iter_update('loss', loss.item())
                for met in self.metrics_iter:
                    self.valid_metrics.iter_update(met.__name__, met(output, target))
                # self.writer.add_image('input', make_grid(data.cpu(), nrow=8, normalize=True))

        for met in self.metrics_epoch:
            self.valid_metrics.epoch_update(met.__name__, met(outputs, targets))

        # # add histogram of model parameters to the tensorboard
        # for name, param in self.model.named_parameters():
        #     self.writer.add_histogram(name, param, bins='auto')

        valid_log = self.valid_metrics.result()

        return valid_log

    def _progress(self, batch_idx):
        ratio = '[{}/{} ({:.0f}%)]'
        return ratio.format(batch_idx, self.len_epoch, 100.0 * batch_idx / self.len_epoch)
