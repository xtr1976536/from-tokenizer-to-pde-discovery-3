# From tokenizer to PDE discovery — stage 3

本仓库归档 **LLM 提议显式隐 ODE、数值筛选、学生完整训练一个 epoch，再重新搜索** 的代码，以及全局 2/4/8 维、带冻结 U-Net 教师的训练快照。学生编码器、解码器从随机权重开始；筛选候选期间不更新学生。

## 结果状态

`results/snapshot_summary.json` 给出本次快照每组已完成 epoch 数及最佳验证 MSE。**这些学生实验仍在服务器训练，没有最终测试结果**；这里只上传已经完成的周期和对应最佳检查点，不把尚未完成的周期当作结果。

历史基线的已完成评价：

| 模型 | 验证窗口 | 验证 RMSE | 测试窗口 | 测试 RMSE |
|---|---:|---:|---:|---:|
| U-Net | 832 | 0.011022209711051839 | 1853 | 0.01196736866900218 |
| FNO | 832 | 0.012262220003058573 | 1853 | 0.013332415715399906 |

原始报告、检查点哈希及逐轨迹误差在 `results/baselines/`。学生报告使用 MSE，比较时应先取平方根，或将基线 RMSE 平方。

U-Net 的验证最佳权重通过 GitHub Release 提供；FNO 历史权重及基线原始 mean_std.pt 在服务器记录路径下已缺失，因此 FNO 只归档代码、配置和现存评价结果，不能声称已经重新运行复现。学生所需的两通道归一化数值保存在检查点和缓存清单中；它们不足以替代完整基线三通道原始归一化文件。

## 当前全局学生流程

- 输入：20 帧、64×128 的 u/v 速度场；输出未来 20 帧。
- 编码器将整个输入窗口压为 2/4/8 个全局隐变量；解码器恢复每个未来场。
- 每个 epoch 固定编码器/解码器，重新编码训练窗口，LLM 最多调用三次、每次提出八个 ODE 结构。
- 按原始轨迹 ID 划分系数拟合和训练内部留出组；数值拟合、积分检查、结构去重后，按内部留出场 MSE 选一个方程。
- 候选试训更新次数为零；选定结构后，学生完整遍历 8147 个训练窗口、batch=12，共 679 次更新，再验证全部 832 个窗口。
- 训练损失为真实场 MSE + 冻结教师预测 MSE，权重均为 1。编码器、解码器和方程系数一起更新，教师不更新。
- 随机种子 525、AdamW、固定学习率 1e-4；最多 200 个交替 epoch，耐心值 10，有效改进阈值 1e-6。
- LLM 不接收正式验证误差；验证集用于模型选择和早停。当前学生实验没有读取测试集。

`stage3_latent_ode/` 同时保留最新代码中新增 U-Net 空间显式瓶颈和无教师选项，但**本次学生结果仅对应全局 2/4/8 维带教师路线**。`architecture=compact` 表示全局隐向量，`unet_spatial_ode` 表示空间特征图，二者不是同一个模型。

## 目录

- `stage3_latent_ode/`：运行入口、训练、显式方程、API 接口、缓存准备和评价。
- `stage3_neural_pde/models.py`：全局学生使用的卷积 Block。
- `ast_bpe_multifield_v1/`：方程解析、AST 数值运行时。
- `ast_bpe_multifield_v4/`：数据访问依赖。
- `baseline_code/`、`baseline_configs/`：基线训练代码和正式配置。
- `research/realpdebench/upstream/`：基线 U-Net/FNO 和数据读取依赖；保留上游许可。
- `checkpoints/`：三个全局学生的最佳快照权重；学生归一化数值保存在检查点与缓存清单中。
- `data_protocol/`：含窗口条目的缓存清单、划分哈希、数据协议配置；不包含完整原始数据或预测缓存。
- `provenance_manifest.json`：从服务器取得的文件、来源路径、大小及 SHA256。

## 运行

需要 Python 3.12 和匹配的 PyTorch/CUDA 环境。依赖参考 `research/realpdebench/upstream/pyproject.toml`，另需 NumPy、SciPy、SymPy、PyArrow。完整数据缓存需自行准备并放入指定位置；缓存 `manifest.json` 中有划分、归一化、教师和来源哈希。配置中的服务器绝对路径是来源记录，迁移机器时需改为本机路径。

```bash
export PYTHONPATH="$PWD/stage3_neural_pde:$PWD/research/realpdebench/upstream:$PWD/ast_bpe_multifield_v1:$PWD/ast_bpe_multifield_v4"
# AIWANWU_API_KEY 通过本地安全方式设置；不要提交到 Git。
python stage3_latent_ode/run_llm_student.py \
  --root "$PWD" --cache /path/to/full/cache \
  --from-scratch --alternating --architecture compact --dim 2 --width 32 \
  --lambda-teacher 1 --lambda-real 1 \
  --epochs 200 --patience 10 --batch-size 12 --gpu-memory-gib 20 \
  --calls-per-cycle 3 --candidates-per-call 8 \
  --model gpt-5.6 --base-url https://www.aiwanwu.cc/v1 \
  --output /path/to/new/output
```

将 `--dim` 改为 4 或 8 可运行其他全局维数。输出目录必须新建；恢复原目录使用同一参数并加 `--resume-run`。恢复以已完成 epoch 为边界，中断 epoch 会从该 epoch 起点重跑，已保存 API 响应可复用。

## 数据与可复现性限制

当前使用 cylinder 的 `real` 分支与 seen 划分。不能仅凭分支名称认定数据全部来自物理实验测量。缓存清单保留 clean 协议及有效索引哈希；上游时间滤波因果性仍未验证，不能据此声明所有潜在信息泄漏均已排除。原 clean manifest 和独立划分索引 JSON 在服务器原路径缺失，本仓库保留含训练/验证窗口条目的缓存清单及各划分哈希；测试索引需按原协议恢复。

FNO/U-Net 历史评价与学生采用同一 seen 索引、20→20、u/v 两通道反归一化场误差；训练预算和监督方式不同，应报告这些差异。本仓库不包含其他尚未完成的新实验结果。

## 权重与来源

U-Net 权重：见仓库 Release `snapshot-2026-11-18`。原始文件约 92 MB，通过 Release 单独下载；下载后可依据 `provenance_manifest.json` 检查 SHA256。

上游 RealPDEBench 代码遵守其 CC BY-NC 4.0 许可，详见 `research/realpdebench/upstream/LICENSE`。本仓库不改变第三方文件的许可。

## 仓库完整性检查

```bash
python scripts/verify_snapshot.py
```

该检查不需要数据、GPU 或 API：核对归档文件哈希，加载 2/4/8 维检查点，验证对应指标，并运行编码、显式积分和解码前向。另有 `scripts/recover_cache_indices.py --output /new/path`，可从缓存清单恢复训练/验证窗口条目；不恢复原文件排版哈希，也不生成缺失的测试索引。

复现状态：代码和三个学生检查点的无数据前向已验证。完整训练/基线重评仍需原始数据、缓存、缺失的测试索引/clean manifest、基线完整归一化文件；FNO 重新评价还需找回其权重。保留的历史指标不等于本仓库已重新完成这些评价。
