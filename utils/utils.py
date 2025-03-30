from examples.text.logic.generate import WrappedModel
from jedi.plugins.stdlib import Wrapped

from model import Transformer
import torch
import numpy as np
from omegaconf import OmegaConf
import matplotlib.pyplot as plt
from transformers.pytorch_utils import meshgrid
from flow_matching.path.scheduler import CondOTScheduler
from flow_matching.path import AffineProbPath
from torchdiffeq import odeint
from flow_matching.utils import ModelWrapper
from flow_matching.solver import ODESolver, Solver

class WrapperModel(ModelWrapper):
    def forward(self, x:torch.Tensor, coords:torch.Tensor, t:torch.tensor, class_idx : torch.tensor) -> torch.Tensor:
        return self.model(x, coords, t, class_idx)


@torch.no_grad()
def draw(model: torch.nn.Module, conf: OmegaConf) -> None:
    num_classes = conf["train"]["num_classes"]
    image_size = conf["train"]["image_size"]
    device = conf["train"]["device"]

    # 确保模型在正确的设备上
    model.to(device)
    model.eval()  # 设置为评估模式

    # 时间步长
    t_span = torch.linspace(0, 1, 10).to(device)

    # 创建画布
    fig, axes = plt.subplots(num_classes, 10, figsize=(20, 2 * num_classes))
    if num_classes == 1:
        axes = axes[np.newaxis, :]  # 单类别时确保axes是二维的

    # 生成坐标网格
    grid = torch.arange(image_size, device=device)
    x, y = torch.meshgrid(grid, grid, indexing="ij")
    coords = torch.stack([x, y], dim=-1)  # [H, W, 2]
    coords = coords.reshape(1, image_size * image_size, 2).expand(num_classes, -1, -1)
    coords = coords.float() / image_size  # 归一化到[0, 1]
    coords.to(device)

    # 初始噪声和类别索引
    x_t = torch.rand((num_classes, image_size * image_size, 1), device=device)
    class_idx = torch.arange(num_classes, device=device).view(-1, 1, 1)

    # 存储所有时间步的结果
    all_steps = []

    # 逐步计算并更新
    for i, t in enumerate(t_span):
        # 计算速度场vt
        t_tensor = t.expand(num_classes, 1, 1)  # [num_classes, 1, 1]
        with torch.no_grad():
            vt = model(x_t, coords, t_tensor, class_idx)

        # 更新图像 (这里使用简单的欧拉方法)
        dt = 0.1 if i == 0 else (t_span[i] - t_span[i - 1]).item()
        x_t = x_t + vt * dt

        # 存储当前步的结果
        all_steps.append(x_t.detach().cpu().numpy())

    # 转换为numpy并reshape以便可视化
    sol = np.stack(all_steps)  # [time_steps, num_classes, H*W, 1]
    sol = sol.reshape(10, num_classes, image_size, image_size)

    # 可视化 - 修改为灰度显示
    for c in range(num_classes):
        for t in range(10):
            ax = axes[c, t]
            img = sol[t, c]

            # 归一化到[0,1]范围
            img_normalized = (img - img.min()) / (img.max() - img.min() + 1e-8)

            # 使用灰度colormap
            ax.imshow(img_normalized, cmap='gray', vmin=0, vmax=1)
            ax.set_xticks([])
            ax.set_yticks([])
            if c == 0:
                ax.set_title(f't={t_span[t].item():.1f}')

    plt.tight_layout()
    plt.show()