# UnCLe: UnCoupling for Causal Discovery (Dev Branch)

本项目实现了 **UnCLENet (ST-CausalFormer)** 模型，旨在解决非线性时序系统中的动态因果发现问题。

## 📂 目录结构 (Directory Structure)

```text
.
├── data/                       # 数据存放目录
├── results/                    # 实验输出 (模型权重、可视化图片)
├── scripts/                    # 数据生成脚本
│   ├── generate_data.py        # 标准合成数据 (Lorenz96/NC8/TVSEM)
│   └── generate_cluster_data.py # [DiffPool专用] 生成具有空间聚类特性的 Lorenz 数据
├── src/                        # 核心代码
│   ├── run.py                  # 主程序入口 (Training + Visualization)
│   ├── model.py                # ST_CausalFormer 模型定义 (Transformer-based)
│   ├── dataloader.py           # 数据加载与处理
│   ├── visualize.py            # 可视化工具
│   └── metrics.py              # 评估指标
└── requirements.txt            # 依赖库
````

## 🛠️ 安装 (Installation)

请确保安装了 PyTorch 2.0+ 及其他依赖：

```bash
pip install -r requirements.txt
```

## 🚀 使用流程 (Usage Workflow)

### 1\. 生成合成数据 (Generate Data)

#### 选项 A: 标准基准数据

使用 `scripts/generate_data.py` 生成标准的 Lorenz96、NC8 或 TVSEM 数据。

```bash
# 1. Lorenz96 (N=128, 默认) - 模拟大气对流
python scripts/generate_data.py --dataset lorenz96 --num_replicas 5 --p 128

# 2. NC8 (N=8) - 非线性多变量关系
python scripts/generate_data.py --dataset nc8 --num_replicas 5
```

#### 选项 B: DiffPool 空间聚类数据 (推荐)

使用 `scripts/generate_cluster_data.py` 生成具有明显空间簇结构的数据，适合测试层级聚类 (Hierarchical Pooling) 效果。

```bash
# 生成 4 个簇，每个簇 32 个节点 (总 N=128)
python scripts/generate_cluster_data.py --num_groups 4 --nodes_per_group 32 --num_replicas 5
```

数据默认保存在 `data/synthetic/cluster_lorenz/`。

### 2\. 训练与可视化 (Training & Visualization)

使用 **`src/run.py`** 启动实验。该脚本会自动执行 Coarse 训练、Mask 更新、Fine 训练以及最终的可视化。

#### 运行示例

**场景 A: Lorenz96 (DiffPool 模式)**
适合测试 `generate_cluster_data.py` 生成的数据。使用 `--hierarchy` 指定分层结构。

```bash
# 假设数据在 data/synthetic/cluster_lorenz
python src/run.py \
    --dataset cluster_lorenz \
    --N 128 \
    --hierarchy 32 8 \
    --epochs_coarse 100 \
    --epochs_fine 100 \
    --batch_size 64 \
    --output_dir results/cluster_exp
```

  * `--hierarchy 32 8`: 表示第一层将 128 个节点聚类为 32 个 Patch，第二层进一步聚类为 8 个 Patch。

**场景 B: 标准 Lorenz96**

```bash
python src/run.py \
    --dataset lorenz96 \
    --N 128 \
    --hierarchy 16 \
    --epochs_coarse 50 \
    --epochs_fine 50 \
    --output_dir results/lorenz96_exp
```

**场景 C: NC8 (小规模验证)**

```bash
python src/run.py \
    --dataset nc8 \
    --N 8 \
    --hierarchy 2 \
    --epochs_coarse 50 \
    --epochs_fine 50 \
    --output_dir results/nc8_exp
```

### 关键参数说明

| 参数 | 描述 | 默认值 |
| :--- | :--- | :--- |
| `--dataset` | 数据集名称 (`lorenz96`, `cluster_lorenz`, `nc8`) | `lorenz96` |
| `--N` | 变量 (Node) 数量，需与生成数据一致 | `128` |
| `--hierarchy` | **[核心]** 层级结构列表。例如 `32 8` 表示两层 Coarse 模型。 | `32 8` |
| `--epochs_coarse` | 第一阶段 (Coarse Hierarchy) 训练轮数 | `100` |
| `--epochs_fine` | 第二阶段 (Fine) 训练轮数 | `100` |
| `--output_dir` | 结果保存路径 | `./results` |

## 📊 输出结果 (Outputs)

运行结束后，请检查 `output_dir` 下的 **`result_full.png`**。该图包含：

1.  **Spatial Clusters**: 模型学习到的第一层 Patch 聚类结果。
2.  **GT Coarse / Fine**: 真实的粗/细粒度因果图 (如有)。
3.  **Est Coarse / Fine**: 预测的因果图。
4.  **DiffPool Generated Mask**: 模型生成的稀疏掩码，用于指导细粒度发现。

此外，如果你配置了 WandB (`--wandb_entity`)，动态演化图 (`causal_evolution.gif`) 和指标也会同步上传。
