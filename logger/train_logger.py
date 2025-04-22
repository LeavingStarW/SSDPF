import util

from time import time
from .base_logger import BaseLogger


class TrainLogger(BaseLogger):
    def __init__(self, args, dataset_len):
        super(TrainLogger, self).__init__(args, dataset_len)

        assert args.is_training
        
       

        self.iters_per_print = args.bs

        self.experiment_name = args.name

        self.epochs = args.epochs
        self.loss_meters = self._init_loss_meters()
        self.loss_meter = util.AverageMeter()


    def _init_loss_meters(self):
        loss_meters = {}
        loss_meters['cls_loss'] = util.AverageMeter()

        return loss_meters

    def _reset_loss_meters(self):
        #w reset是AverageMeter这个类中的方法
        for v in self.loss_meters.values():
            v.reset()

    def _update_loss_meters(self, n, cls_loss = None):
        self.loss_meters['cls_loss'].update(cls_loss, n)

    def _get_avg_losses(self, as_string = False):
        if as_string:
            s = ', '.join('{}: {:.4g}'.format(k, v.avg) for k, v in self.loss_meters.items())
            return s
        else:
            loss_dict = {'batch_{}'.format(k): v.avg for k, v in self.loss_meters.items()}
            return loss_dict

    #w 计算时间差
    def start_iter(self):
        self.iter_start_time = time()

    def log_iter(self, inputs, cls_logits, targets, cls_loss, optimizer):
        """Log results from a training iteration."""
        cls_loss = None if cls_loss is None else cls_loss.item()
        self._update_loss_meters(inputs.size(0), cls_loss)

        # Periodically write to the log and TensorBoard
        if self.iter % self.iters_per_print == 0:

            # Write a header for the log entry
            avg_time = (time() - self.iter_start_time) / self.bs
            message = '[epoch: {}, iter: {} / {}, time: {:.2f}, {}]' \
                .format(self.epoch, self.iter, self.dataset_len, avg_time, self._get_avg_losses(as_string=True))

            # Write all errors as scalars to the graph
            scalar_dict = self._get_avg_losses()
            scalar_dict.update({'train/lr{}'.format(i): pg['lr'] for i, pg in enumerate(optimizer.param_groups)})

            self._reset_loss_meters()

            self.write(message)

    def end_iter(self):
        """Log info for end of an iteration."""
        self.iter += self.bs
        self.global_step += self.bs

    def start_epoch(self):
        """Log info for start of an epoch."""
        self.epoch_start_time = time()
        self.iter = 0
        self.write('[start of epoch {}]'.format(self.epoch))



    #w
    def end_epoch(self, metric, loss):
        self.write('[end of epoch {}, epoch time:{:.4g}]'.format(self.epoch, time() - self.epoch_start_time))

        self._log_loss(loss)
        self._log_metric(metric)

        self.epoch += 1




    def is_finished_training(self):
        """Return True if finished training, otherwise return False."""
        return 0 < self.epochs < self.epoch
