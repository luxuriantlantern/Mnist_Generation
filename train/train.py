import time
import torch
import os

from torch.utils.tensorboard import SummaryWriter
from torch import nn, Tensor

from flow_matching.path.scheduler import CondOTScheduler
from flow_matching.path import AffineProbPath

from torchvision import transforms, datasets
from transformers.pytorch_utils import meshgrid

from model import Transformer
from utils import draw

from omegaconf import OmegaConf

def train(conf : OmegaConf) -> None:
    lr = conf["train"]["lr"]
    batch_size = conf["train"]["batch_size"]
    num_epochs = conf["train"]["num_epochs"]
    device = conf["train"]["device"]
    path = conf["train"]["path"]
    dataset = conf["train"]["dataset"]
    image_size = conf["train"]["image_size"]
    input_dim = conf["train"]["input_dim"]
    hidden_dim = conf["train"]["hidden_dim"]
    num_layers = conf["train"]["num_layers"]
    num_heads = conf["train"]["num_heads"]
    num_classes = conf["train"]["num_classes"]
    dropout = conf["train"]["dropout"]
    log_dir = conf["train"]["log_dir"]
    pth_dir = conf["train"]["pth_dir"]
    start_epoch = conf["train"]["start_epoch"]

    writer = SummaryWriter(log_dir=log_dir)

    # Load the dataset
    if dataset == "MNIST":
        train_dataset = datasets.MNIST(
            root=path,
            train=True,
            download=True,
            transform=transforms.ToTensor()
        )

    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
    )

    # Define the model
    model = Transformer(input_dim = input_dim,
                        hidden_dim = hidden_dim,
                        num_layers = num_layers,
                        num_heads = num_heads,
                        num_classes = num_classes,
                        dropout = dropout)
    # model.load_state_dict(torch.load(pth_dir + f"/model_epoch_{start_epoch}.pth", weights_only=True))
    model = model.to(device)

    optim = torch.optim.SGD(model.parameters(), lr=lr)

    Path = AffineProbPath(scheduler=CondOTScheduler())

    grid = torch.arange(image_size)
    x, y = meshgrid(grid, grid, indexing="ij")
    coords = torch.stack([x, y], dim=-1).float()
    coords = coords / image_size

    global_step = 0
    for _ in range(start_epoch, num_epochs):
        model.train()
        epoch_loss = 0
        start_time = time.time()

        for batch_idx, (data, labels) in enumerate(train_loader):

            data = data.to(device)
            labels = labels.to(device)
            B = data.shape[0]

            x_1 = data.reshape(B, -1, 1)
            x_0 = torch.rand_like(x_1).to(device)
            batch_coords = coords.unsqueeze(0).reshape(1, -1, 2).repeat(B, 1, 1).to(device)

            t = torch.rand(B, device=device) # [B]

            class_idx = labels.reshape(B, 1, 1)

            optim.zero_grad()
            path_sample = Path.sample(t = t, x_0 = x_0, x_1 = x_1)

            t = t[:, None, None]  # [B, 1, 1]
            vt_pred = model(x = path_sample.x_t, t = t, coords = batch_coords, class_idx = class_idx)

            target = x_1 - x_0
            loss = nn.MSELoss()(vt_pred, target)
            loss.backward()

            optim.step()
            epoch_loss += loss.item()

            writer.add_scalar('Training/BatchLoss', loss.item(), global_step)
            global_step += 1

        epoch_loss /= len(train_loader)
        end_time = time.time()
        print(f"Epoch [{_+1}/{num_epochs}], Loss: {epoch_loss:.4f}, Time: {end_time - start_time:.2f}s")

        if (_ + 1) % 5 == 0 or _ == num_epochs - 1:
            torch.save(model.state_dict(), f"{pth_dir}/model_epoch_{_+1}.pth")

        if (_ + 1) % 5 == 0 or _ == num_epochs - 1:
            model.eval()
            with torch.no_grad():
                draw(model, conf)
