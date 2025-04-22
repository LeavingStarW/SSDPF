import torch


#w 加载预训练参数
def load_pretrained_model(module, ckpt_path):
    trained_dict = torch.load(ckpt_path)['model_state']
    model_dict = module.state_dict()
    trained_dict = {k[len('module.'):]:v for k, v in trained_dict.items()}
    trained_dict = {k:v for k, v in trained_dict.items() if k in model_dict}
    model_dict.update(trained_dict)
    module.load_state_dict(model_dict)
    print('load pretrained model:' + module.__class__.__name__)