import torch
from collections import OrderedDict
from termcolor import colored
from torch_lr_finder import LRFinder, TrainDataLoaderIter, ValDataLoaderIter
import os

def load_checkpoint_cascast(checkpoint_path, model):
    checkpoint_dict = torch.load(checkpoint_path, map_location=torch.device('cpu'))
    checkpoint_model = checkpoint_dict['model']
    ckpt_submodels = list(checkpoint_model.keys())
    submodels = ['autoencoder_kl']
    key = 'autoencoder_kl'
    if key not in submodels:
        print(f"warning!!!!!!!!!!!!!: skip load of {key}")
    new_state_dict = OrderedDict()
    for k, v in checkpoint_model[key].items():
        name = k
        if name.startswith("module."):
            name = name[len("module."):]
        if name.startswith("net."):
            name = name[len("net."):]
        new_state_dict[name] = v
    model.load_state_dict(new_state_dict, strict=True)
    print(colored(f"loaded {key} successfully the game is on", 'green'))
    return model

def lr_range_test(model, optimizer, train_dataloader, val_dataloader, criterion, cfg):
    """
    generally /10 were lr explodes gradients.
    """
    class CustomTrainIter(TrainDataLoaderIter):
        def inputs_labels_from_batch(self, batch_data):
            #batch data = [b, h, w, t]
            data = batch_data.permute(0, 3, 1, 2)
            inp, target = data[:, :cfg.input_frames], data[:, cfg.input_frames:]
            return inp, target

    class CustomValIter(ValDataLoaderIter):
        def inputs_labels_from_batch(self, batch_data):
            #batch data = [b, h, w, t]
            data = batch_data.permute(0, 3, 1, 2)
            inp, target = data[:, :cfg.input_frames], data[:, cfg.input_frames:]
            return inp, target

    custom_train_dataloader = CustomTrainIter(train_dataloader)
    custom_val_dataloader = CustomValIter(val_dataloader)
    outputs_path = os.path.join(cfg.experiment_path, 'outputs')
    os.makedirs(outputs_path, exist_ok = True)

    lr_finder = LRFinder(model, optimizer, criterion, device = f"cuda:{cfg.devices[0]}")
    lr_finder.range_test(custom_train_dataloader, val_loader = custom_val_dataloader, end_lr = cfg.lr_range_test.max_lr, num_iter = cfg.lr_range_test.num_iter)
    fig = lr_finder.plot()
    fig.savefig(os.path.join(outputs_path, 'lr_range_test.png'))
    lr_finder.reset()



