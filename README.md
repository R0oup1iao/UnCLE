# UnCLe: UnCoupling for Causal Discovery (Dev Branch)

本项目实现了 **UnCLENet (ST-CausalFormer)** 模型，旨在解决非线性时序系统中的动态因果发现问题。

> **⚠️ 注意**：当前为开发分支 (`dev`)。与主分支不同，本分支的训练、推理与可视化逻辑已整合至单一入口 `src/run.py` 中。

## 📂 目录结构 (Directory Structure)

```text
.
├── data/               # 数据存放目录
├── results/            # 实验输出 (模型权重、可视化图片)
├── scripts/            # 辅助脚本
│   └── generate_data.py # 合成数据生成工具
├── src/                # 核心代码
│   ├── run.py          # [NEW] 主程序入口 (Training + Visualization)
│   ├── model.py        # ST_CausalFormer 模型定义
│   ├── dataloader.py   # 数据加载与处理
│   ├── tcn.py          # TCN 模块
│   ├── metrics.py      # 评估指标
│   └── ...
└── requirements.txt    # 依赖库
```

## 🛠️ 安装 (Installation)

请确保安装了 PyTorch 2.0+ 及其他依赖：

```bash
pip install -r requirements.txt
```

## 🚀 使用流程 (Usage Workflow)

整个流程分为两步：首先生成数据，然后使用 `src/run.py` 进行训练和评估。

### 1\. 生成合成数据 (Generate Data)

使用 `scripts/generate_data.py` 生成带有 Ground Truth 的合成数据集。

```bash
# 1. Lorenz96 (N=128, 默认) - 模拟大气对流
python scripts/generate_data.py --dataset lorenz96 --num_replicas 5 --p 128

# 2. NC8 (N=8) - 非线性多变量关系
python scripts/generate_data.py --dataset nc8 --num_replicas 5

# 3. TVSEM (N=2) - 时变因果关系
python scripts/generate_data.py --dataset tvsem --num_replicas 5
```

生成的数据默认保存在 `data/synthetic/` 目录下。

### 2\. 训练与可视化 (Training & Visualization)

使用 **`src/run.py`** 启动实验。该脚本会自动执行以下流程：

1.  **Phase 1 (Coarse)**: 训练粗粒度模型，学习 Patch 间的关系。
2.  **Mask Update**: 基于 Coarse 阶段结果生成空间掩码 (Spatial Mask)。
3.  **Phase 2 (Fine)**: 训练细粒度模型，进行节点级的因果发现。
4.  **Visualization**: 训练结束后自动生成包含 6 个子图的结果汇总图。

#### 运行示例

**场景 A: Lorenz96 (128 变量)**
适合测试大规模因果发现与 Patch 聚类能力。

```bash
python src/run.py \
    --dataset lorenz96 \
    --N 128 \
    --k_patches 8 \
    --epochs_coarse 30 \
    --epochs_fine 30 \
    --batch_size 64 \
    --output_dir results/lorenz96_exp
```

**场景 B: NC8 (8 变量)**
变量较少，建议减少 Patch 数量或视情况调整。

```bash
python src/run.py \
    --dataset nc8 \
    --N 8 \
    --k_patches 2 \
    --epochs_coarse 50 \
    --epochs_fine 50 \
    --output_dir results/nc8_exp
```

**场景 C: TVSEM (2 变量)**
极小规模验证。

```bash
python src/run.py \
    --dataset tvsem \
    --N 2 \
    --k_patches 1 \
    --epochs_coarse 20 \
    --epochs_fine 20 \
    --output_dir results/tvsem_exp
```

### 关键参数说明

| 参数 | 描述 | 默认值 |
| :--- | :--- | :--- |
| `--dataset` | 数据集名称 (`lorenz96`, `nc8`, `tvsem`) | `lorenz96` |
| `--N` | 变量 (Node) 数量，需与生成数据一致 | `128` |
| `--k_patches` | 将变量聚类为多少个 Patch (Coarse 粒度) | `8` |
| `--epochs_coarse` | 第一阶段训练轮数 | `30` |
| `--epochs_fine` | 第二阶段训练轮数 | `30` |
| `--output_dir` | 结果保存路径 | `./results` |

## 📊 输出结果 (Outputs)

运行结束后，请检查 `output_dir` (例如 `results/`) 下的 **`result_full.png`**。该图包含：

1.  **Spatial Layout**: 节点的空间分布及模型学习到的 Patch 聚类颜色。
2.  **GT Coarse**: (仅供参考) 真实的粗粒度因果图。
3.  **Est Coarse**: 第一阶段学习到的粗粒度因果图。
4.  **GT Fine**: 真实的节点级 (Node-level) 因果图。
5.  **Est Fine**: 最终预测的节点级因果图。
6.  **Adaptive Spatial Mask**: 模型生成的稀疏掩码，用于过滤无关区域。

此外，如果你配置了 WandB (`--wandb_entity`)，所有指标和图片也会同步上传至 Weights & Biases。

## 💡 高级用法

本项目集成了 HuggingFace `accelerate`，支持分布式训练。如果需要在多 GPU 上运行：

```bash
accelerate launch src/run.py --dataset lorenz96 ...
```