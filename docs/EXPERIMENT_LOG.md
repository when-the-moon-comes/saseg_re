# SASeg 修订实验总记录（权威版）

> 最后更新：2026-11-01 20:45（服务器时间）
> 代码版本：`e1_train.py` md5 `2b3efdc407606ac4033f3ed77aad726e`
> 本文件是唯一权威结果记录。旧的 `EXPERIMENT_LOG` 版本、V1/V2 计划、`EXPERIMENT_STATUS.md` 已归档到 `archive/`，不再作为结果依据。
> 所有数字均来自服务器 `runs/formal_eval*` 下的 `final/metrics.json`，可逐项追溯。

---

## 1. 实验设定

### 1.1 数据与协议

| 数据集 | 用途 | train / val / test | 备注 |
|---|---|---|---|
| Synapse / BTCV | 主实验 | 14 / 4 / 12 cases（1027 / 282 / 869 slices） | case-level split，test 为独立 12 cases |
| AMOS22 CT-only | 第二个医学数据集 | 180 / 20 / 100 cases（17,208 / 1,815 / 11,256 slices） | 官方 val 作 test，train 中划 20 cases 作 val |
| PASCAL VOC 2012 | 跨域方向性验证 | 1,318 / 146 / 1,449 images | 官方 val 作 test，train 中划 146 张作 val |
| Cityscapes | **未做** | — | 按 V3 最小计划舍弃 |

- **A1（seen-unknown structured abstention）**：训练与测试均使用 seen-unknown 类别；指标衡量结构化弃权。
- **A2（unseen-unknown detection generalization）**：训练仅见 seen-unknown，测试标签重编码为 unseen-unknown；只做检测/泛化评测。
- A1/A2 训练监督完全相同，因此 A2 用 `--eval-only` 复用 A1 checkpoint，不重复训练。
- 所有 split、阈值、early stopping 只使用 validation；test 只在最终评测使用。

### 1.2 模型变体

| label | 实现 | 说明 |
|---|---|---|
| `perpixel_k` | PerPixelSegmenter(K+1) | 闭集 K 类 + 背景；unknown 用 MSP / Energy / MaxLogit 后处理 |
| `perpixel_k1` | PerPixelSegmenter(K+2) | 显式 K+1 类，seen-unknown 作为第 K+1 类直接输出 |
| `saseg` | SASeg，M=2 | native unknown query + prototype-sphere + boundary 抑制 |
| `saseg_m0` | SASeg，`--unknown-queries 0` | 完整重训的 M=0 对照（不是推理时删 query） |
| `saseg_generic` | SASeg，M 个 query 监督为前景 objectness | 参数匹配的 generic query 对照；unknown 用 MSP |
| `saseg_classagnostic` | ClassAgnosticSASeg | MaskFormer 式类无关 query + 共享分类器（背景/K 类/unknown） |
| `binary_branch` | BinaryBranchSegmenter | 独立 K+1 known 头 + 二值 unknown 头 |

**数据集覆盖**：
- Synapse：全部 7 个变体 × 3 seeds
- AMOS22：PerPixel-K、K+1、SASeg × 3 seeds；M=0、binary × seed42
- VOC2012：PerPixel-K、SASeg、M=0 × 3 seeds
- generic / class-agnostic 按 V3 计划只在 Synapse 跑

### 1.3 训练配置

- Backbone：Swin-T，ImageNet-21k（`swin_tiny_patch4_window7_224.ms_in22k`），前两个 stage 冻结
- Pixel decoder：FPN，d=256；SASeg decoder：3 层，8 heads
- 输入 512×512
- Synapse：physical batch 1 + accumulation 8；AMOS/VOC：physical batch 2 + accumulation 4（等效 batch 8）
- AdamW：lr 1e-4（backbone 1e-5），500-step warmup，cosine
- AMP；CPU 单线程；数据从 `/dev/shm` 读取
- 100 epochs；early stop：validation known macro Dice，patience 10 个验证周期，min_delta 0.001
- AMOS/VOC：`val-every 3`、`max-val-batches 200/100`；训练阶段 `--skip-final-eval`，最终指标由流式评测补

### 1.4 指标定义

- **AUROC / AUPR / FPR@TPR95**：仅在前景像素（`target != 0` 且 `!= -100`）上计算，即论文的 `*_fg`
- **F1 / IoU / Precision / Recall**：全像素，阈值只在 validation 上选一次后冻结
- **known test Dice**：known 类 macro Dice，只平均 truth 中出现的类
- **component F1**：per-slice connected components 的平均（min IoU 0.1）
- AMOS/VOC 使用**流式直方图评测**（`--stream-eval`），内存与 test 规模无关

---

## 2. Synapse 结果（42 / 42 eval，完整）

### 2.1 A1

| 模型 | known test Dice | 主分数 | AUROC_fg | AUPR_fg | F1 | IoU |
|---|---|---|---|---|---|---|
| PerPixel-K + MSP | 0.774 ± 0.006 | `msp` | 0.693 ± 0.029 | 0.124 ± 0.012 | 0.103 ± 0.001 | 0.054 ± 0.000 |
| PerPixel-K+1 | 0.774 ± 0.003 | `unknown_class` | 0.983 ± 0.005 | 0.860 ± 0.029 | 0.661 ± 0.016 | 0.494 ± 0.017 |
| SASeg (M=2) | 0.492 ± 0.034 | `unknown_union` | 0.970 ± 0.006 | 0.802 ± 0.028 | **0.615 ± 0.013** | 0.444 ± 0.013 |
| SASeg M=0 | 0.492 ± 0.059 | `msp` | 0.896 ± 0.010 | 0.444 ± 0.033 | **0.011 ± 0.000** | 0.006 ± 0.000 |
| SASeg generic | 0.605 ± 0.007 | `msp` | 0.905 ± 0.005 | 0.506 ± 0.038 | 0.011 ± 0.000 | 0.006 ± 0.000 |
| SASeg class-agnostic | 0.787 ± 0.005 | `unknown_class` | 0.970 ± 0.004 | 0.851 ± 0.012 | 0.675 ± 0.007 | 0.510 ± 0.008 |
| Binary branch | 0.755 ± 0.001 | `unknown_branch` | 0.983 ± 0.002 | 0.867 ± 0.014 | **0.681 ± 0.009** | 0.516 ± 0.010 |

### 2.2 A2

| 模型 | known test Dice | 主分数 | AUROC_fg | AUPR_fg | F1 | IoU |
|---|---|---|---|---|---|---|
| PerPixel-K | 0.780 ± 0.005 | `msp` | 0.644 ± 0.057 | 0.005 ± 0.001 | 0.003 ± 0.000 | 0.002 ± 0.000 |
| PerPixel-K+1 | 0.780 ± 0.002 | `unknown_class` | 0.892 ± 0.030 | 0.021 ± 0.005 | 0.012 ± 0.001 | 0.006 ± 0.001 |
| SASeg (M=2) | 0.493 ± 0.034 | `ensemble` | 0.941 ± 0.016 | 0.048 ± 0.008 | 0.011 ± 0.000 | 0.005 ± 0.000 |
| SASeg M=0 | 0.494 ± 0.060 | `msp` | 0.967 ± 0.004 | 0.080 ± 0.013 | 0.000 ± 0.000 | 0.000 ± 0.000 |
| SASeg generic | 0.608 ± 0.007 | `msp` | 0.969 ± 0.002 | 0.078 ± 0.005 | 0.001 ± 0.000 | 0.000 ± 0.000 |
| SASeg class-agnostic | 0.790 ± 0.005 | `unknown_class` | 0.815 ± 0.044 | 0.018 ± 0.005 | 0.012 ± 0.002 | 0.006 ± 0.001 |
| Binary branch | 0.763 ± 0.001 | `unknown_branch` | 0.866 ± 0.031 | 0.020 ± 0.005 | 0.012 ± 0.002 | 0.006 ± 0.001 |

### 2.3 Synapse 补充分析

**Support score（prototype-sphere，SASeg，3 seeds）**

| 协议 | AUROC_fg | AUPR_fg | F1 | IoU |
|---|---|---|---|---|
| A1 | 0.638 ± 0.027 | 0.142 ± 0.024 | 0.027 ± 0.026 | 0.014 ± 0.014 |
| A2 | 0.731 ± 0.015 | 0.008 ± 0.001 | 0.0005 ± 0.0003 | 0.0002 ± 0.0002 |

**Component F1（A1，3 seeds）**

| 模型 | 主分数 | component F1 | 像素 F1 |
|---|---|---|---|
| **SASeg (M=2)** | `unknown_union` | **0.672 ± 0.005** | 0.615 ± 0.013 |
| PerPixel-K+1 | `unknown_class` | 0.598 ± 0.003 | 0.661 ± 0.016 |
| Binary branch | `unknown_branch` | 0.617 ± 0.014 | 0.681 ± 0.009 |
| SASeg M=0 | `msp` | 0.000 | 0.011 |

**A2 candidate analysis（SASeg seed42）**

| 指标 | unknown query 候选 | 检测分数 top-r proposal |
|---|---|---|
| purity | 0.0011 | 0.003–0.010 |
| union IoU | 0.0055 | 0.003–0.004 |
| hit@IoU≥0.1 | 0.0069 | 0.034–0.064 |
| component recall | 0.788 | 0.808–0.827 |
| component precision | 0.0049 | 0.005–0.010 |

**Known mask 阈值诊断（SASeg seed42, val）**

| threshold | 0.1 | 0.2 | 0.3 | 0.4 | 0.5 |
|---|---|---|---|---|---|
| macro Dice | 0.477 | 0.475 | 0.474 | 0.471 | 0.469 |

**效率表（Swin-T, 512×512, batch 1, RTX 3090）**

| 变体 | Params (M) | FLOPs (G) | Forward (ms) | 峰值显存 (MB) |
|---|---|---|---|---|
| PerPixel-K | 30.25 | 78.3 | 32.4 | 874 |
| PerPixel-K+1 | 30.25 | 78.3 | 12.5 | 1590 |
| SASeg | 33.55 | 93.4 | 35.7 | 2862 |
| SASeg generic | 33.55 | 93.4 | 37.1 | 3989 |
| SASeg class-agnostic | 33.55 | 92.1 | 29.3 | 4660 |
| Binary branch | 30.25 | 78.3 | 14.1 | 5413 |

---

## 3. AMOS22 结果（22 个 metrics，完整）

### 3.1 A1

| 模型 | known test Dice | 主分数 | AUROC_fg | AUPR_fg | F1 | IoU | seeds |
|---|---|---|---|---|---|---|---|
| PerPixel-K | 0.878 ± 0.012 | `msp` | 0.444 ± 0.015 | 0.100 ± 0.000 | 0.082 ± 0.003 | 0.043 ± 0.001 | 3/3 |
| PerPixel-K+1 | 0.885 ± 0.005 | `unknown_class` | 0.983 ± 0.004 | 0.951 ± 0.009 | 0.807 ± 0.008 | 0.676 ± 0.011 | 3/3 |
| SASeg (M=2) | 0.856 ± 0.006 | `unknown_union` | 0.964 ± 0.002 | 0.917 ± 0.005 | **0.816 ± 0.003** | 0.688 ± 0.005 | 3/3 |
| SASeg M=0 | 0.845 | `msp` | 0.971 | 0.769 | **0.012** | 0.006 | 1/3 |
| Binary branch | 0.879 | `unknown_branch` | 0.962 | 0.914 | **0.832** | 0.712 | 1/3 |

### 3.2 A2

| 模型 | known test Dice | 主分数 | AUROC_fg | AUPR_fg | F1 | IoU | seeds |
|---|---|---|---|---|---|---|---|
| PerPixel-K | 0.886 ± 0.011 | `msp` | 0.336 ± 0.018 | 0.024 ± 0.000 | 0.005 ± 0.001 | 0.003 ± 0.000 | 3/3 |
| PerPixel-K+1 | 0.890 ± 0.005 | `unknown_class` | 0.762 ± 0.023 | 0.250 ± 0.034 | 0.101 ± 0.005 | 0.053 ± 0.003 | 3/3 |
| SASeg (M=2) | 0.859 ± 0.005 | `ensemble` | 0.987 ± 0.001 | 0.549 ± 0.019 | 0.079 ± 0.001 | 0.041 ± 0.000 | 3/3 |
| SASeg M=0 | 0.848 | `msp` | 0.986 | 0.523 | 0.003 | 0.001 | 1/3 |
| Binary branch | 0.885 | `unknown_branch` | 0.569 | 0.070 | 0.047 | 0.024 | 1/3 |

### 3.3 说明
- AMOS SASeg 已完成 3 seeds（A1 F1 0.816，A2 F1 0.079）
- AMOS 上 M=0 崩塌成立：A1 AUROC 0.971 但 F1 0.012
- AMOS 上 post-hoc 的 AUROC **低于 0.5**（0.44 / 0.34），说明 unknown 被高置信误判为已知类；与 Synapse 不同，写论文时需单独解释
- AMOS A2 上 Support AUPR 高达 0.86（seed42），值得在分析中单列
- generic / class-agnostic 未在 AMOS 上训练（V3 计划限定 Synapse）

---

## 4. VOC2012 结果（18 / 18 eval，完整）

### 4.1 A1

| 模型 | known test Dice | 主分数 | AUROC_fg | AUPR_fg | F1 | IoU |
|---|---|---|---|---|---|---|
| PerPixel-K | 0.818 ± 0.005 | `msp` | 0.650 ± 0.005 | 0.193 ± 0.004 | 0.148 ± 0.017 | 0.080 ± 0.010 |
| SASeg (M=2) | 0.851 ± 0.002 | `unknown_union` | 0.893 ± 0.003 | 0.756 ± 0.007 | **0.660 ± 0.005** | 0.493 ± 0.006 |
| SASeg M=0 | 0.847 ± 0.011 | `msp` | 0.945 ± 0.016 | 0.691 ± 0.047 | **0.056 ± 0.003** | 0.029 ± 0.002 |

### 4.2 A2

| 模型 | known test Dice | 主分数 | AUROC_fg | AUPR_fg | F1 | IoU |
|---|---|---|---|---|---|---|
| PerPixel-K | 0.830 ± 0.007 | `msp` | 0.505 ± 0.003 | 0.100 ± 0.002 | 0.084 ± 0.001 | 0.044 ± 0.001 |
| SASeg (M=2) | 0.854 ± 0.002 | `ensemble` | 0.972 ± 0.006 | 0.776 ± 0.026 | 0.060 ± 0.001 | 0.031 ± 0.000 |
| SASeg M=0 | 0.852 ± 0.010 | `msp` | 0.971 ± 0.010 | 0.767 ± 0.023 | 0.057 ± 0.009 | 0.030 ± 0.005 |

### 4.3 说明
- A1 M=0 崩塌在 VOC 上同样成立（0.660 → 0.056）
- A2 的 gap 仍存在；VOC 的 A2 F1（0.060）高于医学数据集，说明自然图像 unknown 区域更大、更连贯
- A2 上 M=0 与 SASeg 接近，说明 unseen-unknown 的结构化收益有限

---

## 5. 关键结论（可直接用于论文）

1. **显式 unknown 输出空间是 A1 结构化的必要条件**
   - Synapse：M=0 F1 0.011 vs SASeg 0.615
   - AMOS：M=0 F1 0.012 vs SASeg 0.813
   - VOC：M=0 F1 0.056 vs SASeg 0.660
   - 三个数据集方向一致

2. **具体实现形式不唯一，但 native query 的区域连贯性最好**
   - Synapse 像素 F1：binary 0.681 ≈ class-agnostic 0.675 > K+1 0.661 > SASeg 0.615
   - Synapse component F1：**SASeg 0.672** > binary 0.617 > K+1 0.598
   - AMOS pixel F1：SASeg **0.816** > K+1 0.807，binary 0.832（1 seed）
   - VOC pixel F1：SASeg 0.660，post-hoc 0.148
   - 论文应写成"显式输出空间必要；native query 在区域连贯性上有优势"

3. **A2 的 detection–structuring gap 稳定存在**
   - Synapse：AUROC 0.89–0.98，F1 0.00–0.013
   - AMOS：AUROC 0.76–0.99，F1 0.003–0.101
   - VOC：AUROC 0.97，F1 0.060
   - 高检测排名不能自动转化为可用 mask

4. **A2 瓶颈不能只归因于 candidate generation**
   - unknown query 候选 union IoU 0.0055
   - 检测分数 top-r proposal upper bound 也只有 union IoU 0.003–0.004
   - 更准确的表述：检测信号的空间定位精度 + 极端类别不平衡

5. **Support / prototype-sphere 不是更强的 detector**
   - A1 AUPR 0.142（略高于 MSP 0.124），F1 仅 0.027
   - A2 AUROC 0.731，明显弱于 MSP/Ensemble
   - 应定位为几何支持信号，而不是检测器

6. **Synapse 上 native SASeg 的 known 分割是真实短板**
   - known test Dice 0.492 vs 其他 0.755–0.787
   - 阈值诊断（0.1–0.5）仅从 0.477 变到 0.469，不是二值化问题
   - AMOS（0.853）与 VOC（0.851）上并无此问题，说明与 Synapse 的小器官类别和验证集规模有关，需在 limitations 说明

---

## 6. 未完成 / 限制

> 实验状态：**全部必要实验已完成**。Synapse 42/42、AMOS 22/22、VOC 18/18；补充分析（Support、component、candidate、阈值诊断）与定性图均完成。代码仓库由用户自行处理。

### 6.1 定性图

| 文件 | 内容 |
|---|---|
| `figures/qualitative/fig_a1_success.png` | Synapse A1，3 个不同病例：Image / GT（红=unknown）/ SASeg unknown mask |
| `figures/qualitative/fig_a2_gap.png` | Synapse A2：GT unknown / ensemble score heatmap / 阈值化 mask，展示高 AUROC 下 mask 仍严重误检 |
| `figures/qualitative/fig_m0_failure.png` | A1：SASeg unknown vs M=0 的 MSP score heatmap，展示 M=0 无法定位 unknown |

生成脚本：`src/saseg/tools/make_qualitative.py`；输出目录为 `figures/`。

| 项目 | 状态 |
|---|---|
| AMOS SASeg seed44 训练 + 评测 | ✅ 完成（A1 F1 0.816，A2 F1 0.079） |
| AMOS component F1 | 未计算（可选；R2 的 component 要求已在 Synapse 满足） |
| 定性图（A1 成功 / A2 高 AUROC 低 F1 / M=0 失败） | ✅ 已生成（见下） |
| 匿名代码仓库 | 用户自行处理 |
| Cityscapes | 按 V3 计划舍弃 |
| AMOS generic / class-agnostic | 按 V3 计划不做 |
| AMOS M=0 / binary 仅 seed42 | 计划内降级 |

---

## 7. 数据来源与复现

| 内容 | 路径 |
|---|---|
| Synapse metrics | `runs/formal_eval/synapse/{A1,A2}/{label}/seed{42,43,44}/final/metrics.json` |
| AMOS metrics | `runs/formal_eval_amos/{A1,A2}/{label}/seed*/final/metrics.json` |
| VOC metrics | `runs/formal_eval_voc/{A1,A2}/{label}/seed*/final/metrics.json` |
| Synapse checkpoints | `runs/formal/synapse/A1/{label}/seed*/best.pt` |
| AMOS checkpoints | `runs/formal/amos22/A1/{label}/seed*/best.pt` |
| VOC checkpoints | `runs/formal/voc2012/A1/{label}/seed*/best.pt` |
| Support / component | 同 metrics.json 中的 `support` 与 `component_*` 字段 |
| Candidate analysis | `runs/candidates/saseg_seed42_a2.json` |
| 阈值诊断 | `runs/diagnostics/synapse_saseg_seed42_val.json` |
| 效率表 | `outputs/metrics/efficiency_synapse.json` |
| 协议/manifest | `e1/protocols/*.json`、`e0/manifests/*.json` |

重新生成某数据集表格：

```bash
cd $PROJECT_ROOT
source $PROJECT_ROOT/activate_saseg.sh
export PYTHONPATH=$PWD/src:$PYTHONPATH
python -m saseg.tools.build_experiment_log \
  --root $PROJECT_ROOT/runs/formal_eval_amos \
  --output $PROJECT_ROOT/runs/LOG_amos.md --dataset AMOS22
```
