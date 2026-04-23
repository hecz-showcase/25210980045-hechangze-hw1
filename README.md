# 25210980045-hechangze-hw1
深度学习与空间智能 HW1：从零开始构建三层神经网络分类器，实现地表覆盖图像分类。

## 1. 环境依赖

推荐环境：

- Python 3.10 - 3.12（建议 3.12）
- pip

本项目使用的第三方库：

- numpy
- matplotlib
- Pillow

安装方式（Windows PowerShell）：

```powershell
python -m pip install numpy matplotlib pillow
```

## 2. 数据准备要求

请将 EuroSAT RGB 数据集按如下目录放置（仓库根目录下）：

```text
EuroSAT_RGB/
	AnnualCrop/
	Forest/
	HerbaceousVegetation/
	Highway/
	Industrial/
	Pasture/
	PermanentCrop/
	Residential/
	River/
	SeaLake/
```

## 3. 运行顺序（训练 + 测试）

注意：训练和测试都依赖 `outputs/eurosat_split.npz`，因此必须先执行数据预处理。

### Step 1: 生成训练/验证/测试划分

```powershell
python scripts/prepare_data.py
```

输出：

- `outputs/eurosat_split.npz`
- `outputs/eurosat_split_meta.json`

### Step 2: 训练模型

```powershell
python scripts/train.py
```

输出：

- 最优权重：`checkpoints/best_model.npz`
- 训练日志：`outputs/logs/train_history.json`
- 曲线图：`outputs/figures/loss_curve.png`、`outputs/figures/val_accuracy_curve.png`

### Step 3: 测试模型

```powershell
python scripts/test.py
```

输出：

- 指标文件：`outputs/logs/test_results.json`、`outputs/logs/test_results.txt`
- 混淆矩阵：`outputs/logs/test_confusion_matrix.csv`
- 错误样本可视化：`outputs/figures/test_error_pair_matrix_n*_m*.png`

## 4. 超参数搜索（Random / Grid）

如果你想先搜索较优超参数，再固定配置做完整训练，可以使用搜索脚本。

### Step 1: 先确保已完成数据预处理

搜索脚本依赖训练/验证划分数据：

```powershell
python scripts/prepare_data.py
```

### Step 2: 在配置中选择搜索模式

编辑 `src/config.py` 中 `SearchConfig`：

- `mode = "random"`：随机搜索
- `mode = "grid"`：网格搜索

常用参数：

- `num_trials`：随机搜索的试验次数
- `epochs_per_trial`：每组超参数训练轮数（>0 时覆盖 `TRAIN.epochs`）

### Step 3: 运行搜索

```powershell
python scripts/search.py
```

输出：

- 排名结果（JSON）：`outputs/logs/random_search_results.json` 或 `outputs/logs/grid_search_results.json`
- 排名结果（CSV）：`outputs/logs/random_search_results.csv` 或 `outputs/logs/grid_search_results.csv`
- 搜索阶段最佳模型：`checkpoints/best_model_random.npz` 或 `checkpoints/best_model_grid.npz`

提示：

- 搜索脚本会在终端打印最优 trial 的参数与验证集准确率。
- 搜索完成后，若要按固定配置做完整训练，继续运行 `python scripts/train.py`。

## 5. 常见报错排查

- 报错 `Split file not found`：先运行 `python scripts/prepare_data.py`
- 报错 `Checkpoint not found`：先运行 `python scripts/train.py`
- 报错数据目录不存在：确认 `EuroSAT_RGB/` 在仓库根目录
