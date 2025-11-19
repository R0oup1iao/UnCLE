import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import os
from mpl_toolkits.axes_grid1 import make_axes_locatable

def plot_static_heatmap(matrix, title, save_path=None):
    """绘制并保存静态热力图"""
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(matrix, cmap='hot', interpolation='nearest')
    ax.set_title(title)
    ax.set_xlabel("Source")
    ax.set_ylabel("Target")
    
    # Colorbar
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im, cax=cax)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        plt.close(fig)
    else:
        plt.show()

def create_dynamic_gif(dynamic_adj, save_path, fps=10):
    """
    生成动态因果图 GIF
    dynamic_adj: (N, N, T) - 注意最后一维是时间
    save_path: output.gif
    """
    # dynamic_adj shape: (Source, Target, Time) or (Target, Source, Time)?
    # UnCLENet output strengths is [Source, Target, Time].
    # 习惯上 Heatmap 是 Row=y(Target), Col=x(Source) 或者反过来
    # 我们统一显示：Y轴为 Target, X轴为 Source
    
    # 现在的 strengths shape: (N, N, T) -> (j, i, t) -> j causes i
    # j 是 source, i 是 target
    # imshow 默认 (row, col) -> (i, j) -> (target, source)
    # 所以需要 transpose(1, 0, 2) 变成 (Target, Source, Time) 以符合直觉
    data_to_plot = dynamic_adj.transpose(1, 0, 2) 
    
    N, _, T = data_to_plot.shape
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # 初始帧 (为了定 Colorbar Range，我们取全局最大值)
    vmax = np.percentile(data_to_plot, 99) # 忽略极端异常值
    vmin = 0
    
    im = ax.imshow(data_to_plot[:, :, 0], cmap='hot', vmin=vmin, vmax=vmax, interpolation='nearest')
    title = ax.set_title(f"Dynamic Causal Strength (t=0)")
    ax.set_xlabel("Source Variable")
    ax.set_ylabel("Target Variable")
    
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im, cax=cax)
    
    def update(frame):
        im.set_data(data_to_plot[:, :, frame])
        title.set_text(f"Dynamic Causal Strength (t={frame})")
        return im, title
    
    # 只有 T 帧
    ani = animation.FuncAnimation(fig, update, frames=range(T), blit=True, interval=1000/fps)
    
    # Save
    try:
        ani.save(save_path, writer='pillow', fps=fps)
        print(f"GIF saved to {save_path}")
    except Exception as e:
        print(f"Failed to save GIF (check if pillow/imagemagick is installed): {e}")
        
    plt.close(fig)