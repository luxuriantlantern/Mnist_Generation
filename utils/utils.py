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
from flow_matching.solver import ODESolver, Solver

@torch.no_grad()
def draw(model: torch.nn.Module, conf: OmegaConf) -> None:
    num_classes = conf["train"]["num_classes"]
    image_size = conf["train"]["image_size"]
    device = conf["train"]["device"]

    model.to(device)
    model.eval()

    # 生成100个时间步对应的时间点（共101个点，包含t=1.0）
    num_steps = 100
    t_span = torch.linspace(0, 1, num_steps + 1).to(device)  # [0.0, 0.01, ..., 1.0]

    # 创建画布（11列对应t=0.0到1.0）
    fig, axes = plt.subplots(num_classes, 11, figsize=(22, 2 * num_classes))
    if num_classes == 1:
        axes = axes[np.newaxis, :]  # 确保单类别时axes维度正确

    # 生成坐标网格（归一化到[0,1]）
    grid = torch.arange(image_size)
    x, y = meshgrid(grid, grid, indexing="ij")
    coords = torch.stack([x, y], dim=-1).float()
    coords = coords / image_size

    x_t = torch.rand((num_classes, image_size * image_size, 1), device=device)
    class_idx = torch.arange(num_classes, device=device).view(-1, 1, 1)

    all_steps = []
    all_steps.append(x_t.detach().cpu().numpy())  # 保存初始状态t=0.0

    # # 逐步积分（欧拉方法）
    # for i in range(num_steps):
    #     t_current = t_span[i]
    #     t_next = t_span[i + 1]
    #     dt = (t_next - t_current).item()
    #
    #     # 准备时间张量 [num_classes, 1, 1]
    #     t_tensor = t_current.view(1, 1, 1).expand(num_classes, 1, 1)
    #
    #     batch_coords = coords.unsqueeze(0).reshape(1, -1, 2).repeat(num_classes, 1, 1).to(device)
    #
    #     # 调用模型（注意参数顺序与训练一致）
    #     vt = model(x=x_t, t=t_tensor, coords=batch_coords, class_idx=class_idx)
    #
    #     # 更新图像
    #     x_t = x_t + vt * dt
    #
    #     # 每10步保存一次结果（对应t=0.1,0.2,...,1.0）
    #     if (i + 1) % 10 == 0:
    #         all_steps.append(x_t.detach().cpu().numpy())
    #
    # # 转换为numpy并reshape
    # sol = np.stack(all_steps)  # [11, num_classes, H*W, 1]
    # sol = sol.reshape(11, num_classes, image_size, image_size)
    #
    # # 可视化（灰度图）
    # for c in range(num_classes):
    #     for t_idx in range(11):
    #         ax = axes[c, t_idx]
    #         img = sol[t_idx, c]
    #         img_norm = (img - img.min()) / (img.max() - img.min() + 1e-8)
    #         ax.imshow(img_norm, cmap='gray', vmin=0, vmax=1)
    #         ax.axis('off')
    #         if c == 0:
    #             ax.set_title(f't={t_idx * 0.1:.1f}')
    #
    # plt.tight_layout()
    # plt.show()

    coords = coords.unsqueeze(0).reshape(1, -1, 2).repeat(num_classes, 1, 1).to(device)

    def ode_func(t:torch.Tensor, x:torch.Tensor) -> torch.Tensor:
        t_expanded = t.expand(x.size(0))[:, None, None]
        return model(x, coords, t_expanded, class_idx)

    # 生成11个均匀分布的时间点(t=0.0到t=1.0)
    t_eval = torch.linspace(0.0, 1.0, 11, device=device)

    # 解ODE（自适应步长），获取所有时间点的结果
    generated = odeint(
        ode_func,
        x_t,
        t_eval,
        rtol=1e-5,
        atol=1e-5,
        method='dopri5'
    )

    # 转换为numpy并reshape
    generated_np = generated.detach().cpu().numpy()  # 形状为[11, num_classes, H*W, 1]
    images = generated_np.reshape(11, num_classes, image_size, image_size)

    # 可视化所有时间点的图像
    for c in range(num_classes):
        for t_idx in range(11):
            ax = axes[c, t_idx]
            img = images[t_idx, c]
            img_norm = (img - img.min()) / (img.max() - img.min() + 1e-8)
            ax.imshow(img_norm, cmap='gray', vmin=0, vmax=1)
            ax.axis('off')
            if c == 0:
                ax.set_title(f't={t_idx * 0.1:.1f}')

    plt.tight_layout()
    plt.show()
