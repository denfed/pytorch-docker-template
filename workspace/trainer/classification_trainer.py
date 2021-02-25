
import numpy as np
import torch
from torchvision.utils import make_grid
from base import BaseTrainer
from utils import inf_loop, MetricTracker
import wandb

from torch_lr_finder import LRFinder
from matplotlib import pyplot as plt 

class ClassificationTrainer(BaseTrainer):
    """
    Trainer class
    """
    def __init__(self, model, criterion, metric_ftns, optimizer, config, data_loader,
                 valid_data_loader=None, len_epoch=None):
        super().__init__(model, criterion, metric_ftns, optimizer, config)
        # Save this code to wandb
        wandb.save(__file__, base_path="./")
        
        self.config = config
        self.data_loader = data_loader
        if len_epoch is None:
            # epoch-based training
            self.len_epoch = len(self.data_loader)
        else:
            # iteration-based training
            self.data_loader = inf_loop(data_loader)
            self.len_epoch = len_epoch
        self.valid_data_loader = valid_data_loader
        self.do_validation = self.valid_data_loader is not None
        
        self.log_step = int(np.sqrt(data_loader.batch_size))

        self.train_metrics = MetricTracker('loss', *[m.__name__ for m in self.metric_ftns], writer=self.writer)
        self.valid_metrics = MetricTracker('loss', *[m.__name__ for m in self.metric_ftns], writer=self.writer)
        
#         lr_finder = LRFinder(model, optimizer, criterion, device=self.device)
#         lr_finder.range_test(data_loader, end_lr=100, num_iter=100)
#         fig, ax = plt.subplots()
#         lr_finder.plot(ax=ax) # to inspect the loss-learning rate graph
#         fig.savefig("test.png")
#         lr_finder.reset()
        
        self.lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(
            self.optimizer, 
            max_lr=0.0002, 
            steps_per_epoch=len(data_loader), epochs=self.epochs)
        
        

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch
        :param epoch: Integer, current training epoch.
        :return: A log that contains average loss and metric in this epoch.
        """
        self.model.train()
        self.train_metrics.reset()
        for batch_idx, (data, target, audio_ids) in enumerate(self.data_loader):
            data, target, audio_ids = data.to(self.device), target.to(self.device), audio_ids.to(self.device)
            
#             print("data", data.shape)
#             print("target", target.shape)

            self.optimizer.zero_grad()
            output = self.model(data)
#             print("outout", output.shape)
#             print(output)
#             batchs, times, labels = output.size()
#             batchs_t, times_t, labels_t = target.size()
#             output = output.view(batchs*times, labels)
#             target = target.view(batchs_t*times_t, labels_t).squeeze(1)
#             print("viewed output", output.shape)
#             print("viewed target", target.shape)
            loss = self.criterion(output, target)
            loss.backward()
            self.optimizer.step()
            
            self.lr_scheduler.step()

            self.writer.set_step((epoch - 1) * self.len_epoch + batch_idx)
            self.train_metrics.update('loss', loss.item())
            wandb.log({'loss':loss})
            wandb.log({'learning_rate': self.optimizer.param_groups[0]['lr']})
            for met in self.metric_ftns:
                self.train_metrics.update(met.__name__, met(output, target))
                wandb.log({met.__name__: met(output, target)})

            if batch_idx % self.log_step == 0:
                self.logger.debug('Train Epoch: {} {} Loss: {:.6f}'.format(
                    epoch,
                    self._progress(batch_idx),
                    loss.item()))
                # self.writer.add_image('input', make_grid(data.cpu(), nrow=8, normalize=True))

            if batch_idx == self.len_epoch:
                break
        log = self.train_metrics.result()
        
        wandb.run.summary['epochs_trained'] = epoch

        if self.do_validation:
            val_log = self._valid_epoch(epoch)
            wandb.log({'val_'+k : v for k, v in val_log.items()})
            log.update(**{'val_'+k : v for k, v in val_log.items()})

#         if self.lr_scheduler is not None:
#             self.lr_scheduler.step(val_log.get('loss'))
            
        return log

    def _valid_epoch(self, epoch):
        """
        Validate after training an epoch
        :param epoch: Integer, current training epoch.
        :return: A log that contains information about validation
        """
        self.model.eval()
        self.valid_metrics.reset()
        with torch.no_grad():
            for batch_idx, (data, target, audio_ids) in enumerate(self.valid_data_loader):
                data, target, audio_ids = data.to(self.device), target.to(self.device), audio_ids.cpu()

                output = self.model(data)
                
#                 batchs, times, labels = output.size()
#                 batchs_t, times_t, labels_t = target.size()
#                 output = output.view(batchs*times, labels)
#                 target = target.view(batchs_t*times_t, labels_t).squeeze(1)
                
                loss = self.criterion(output, target)

                self.writer.set_step((epoch - 1) * len(self.valid_data_loader) + batch_idx, 'valid')
                self.valid_metrics.update('loss', loss.item())
                for met in self.metric_ftns:
                    self.valid_metrics.update(met.__name__, met(output, target))
                # self.writer.add_image('input', make_grid(data.cpu(), nrow=8, normalize=True))

        # add histogram of model parameters to the tensorboard
        # for name, p in self.model.named_parameters():
        #     self.writer.add_histogram(name, p, bins='auto')
        return self.valid_metrics.result()

    def _progress(self, batch_idx):
        base = '[{}/{} ({:.0f}%)]'
        if hasattr(self.data_loader, 'n_samples'):
            current = batch_idx * self.data_loader.batch_size
            total = self.data_loader.n_samples
        else:
            current = batch_idx
            total = self.len_epoch
        return base.format(current, total, 100.0 * current / total)