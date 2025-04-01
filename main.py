from train import train
from omegaconf import OmegaConf
from utils import draw
from model import Transformer
import torch

if __name__ == '__main__':
    conf = OmegaConf.load('config.yaml')
    input_dim = conf["train"]["input_dim"]
    hidden_dim = conf["train"]["hidden_dim"]
    num_layers = conf["train"]["num_layers"]
    num_heads = conf["train"]["num_heads"]
    num_classes = conf["train"]["num_classes"]
    dropout = conf["train"]["dropout"]
    model = Transformer(input_dim=input_dim,
                        hidden_dim=hidden_dim,
                        num_layers=num_layers,
                        num_heads=num_heads,
                        num_classes=num_classes,
                        dropout=dropout)
    model.load_state_dict(torch.load('./pth_256_4/model_epoch_150.pth', weights_only=True))
    with torch.no_grad():
        model.eval()
        draw(model, conf)
    # train(conf)