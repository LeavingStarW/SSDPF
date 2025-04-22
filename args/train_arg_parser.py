import util #w 用到了io_util中的str_to_bool
from .base_arg_parser import BaseArgParser #w 基础参数类，用于继承


class TrainArgParser(BaseArgParser):
    def __init__(self):
        super(TrainArgParser, self).__init__()
        self.is_training = True #w 用于参数解析函数中，如果是True则是配置训练和验证相关的



        #w Lung
        self.parser.add_argument('--epochs', type = int, required = True)
        self.parser.add_argument('--lr', type = float, required = True)
        self.parser.add_argument('--lr_scheduler', type = str, required = True, default = 'cosine_warmup', 
                                    choices = ('step', 'cosine_warmup'))
        self.parser.add_argument('--lr_decay_step', type = int, required = True, default = 600000)
        self.parser.add_argument('--lr_decay_gamma', type = float, default = 0.1)
        self.parser.add_argument('--lr_warmup_steps', type = int, default = 10000)
        
        self.parser.add_argument('--weight_decay', type = float, default = 1e-3)
        self.parser.add_argument('--best_metric', type = str, default = 'loss', choices = ('loss', 'AUROC'))
  
        self.parser.add_argument('--optimizer', type = str, default = 'sgd', choices = ('sgd', 'adam'))
        self.parser.add_argument('--adam_beta_1', type = float, default = 0.9)
        self.parser.add_argument('--adam_beta_2', type = float, default = 0.999)
        self.parser.add_argument('--sgd_momentum', type = float, default = 0.9)
        self.parser.add_argument('--sgd_dampening', type = float, default = 0.9)
        
        self.parser.add_argument('--do_hflip', type = util.str_to_bool, default = True)
        self.parser.add_argument('--do_vflip', type = util.str_to_bool, default = False)
        self.parser.add_argument('--do_rotate', type = util.str_to_bool, default = True)
        self.parser.add_argument('--do_jitter', type = util.str_to_bool, default = True)


        