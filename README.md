# SpanCrack

论文 *SpanCrack: Co-designing a compact decoder and continuity-aware supervision for pavement crack segmentation* 的官方实现。

SpanCrack 沿用预训练的 SegFormer-B0 编码器，重新设计了解码器和监督方式：

- **EMD**（等权多尺度解码器）：把四个编码阶段投影到 64 通道后取平均，再做深度可分离卷积细化；
- **AMF**（自适应多尺度融合）：逐像素预测四个尺度的权重，并增强融合特征；
- **CAL**（连续性感知损失）：soft-clDice 加上只在训练时使用的骨架和边界辅助监督。

SpanCrack-Lite 只保留 EMD，用基础损失（BCE + Dice）训练。

## 命名对照

代码沿用开发时的内部名 `tcs_seg`，对应关系如下：

| 论文中的名称 | 配置目录 | `model.variant` |
| --- | --- | --- |
| SpanCrack（EMD + AMF + CAL） | `configs/tcs_seg/` | `tcs_seg` |
| SpanCrack-Lite（仅 EMD） | `configs/tcs_seg_lite/` | `tcs_seg` |

两者用同一个模型类 `TCSSegFormer`，区别只在 `model.arch` 的模块开关和 `training.loss` 的损失权重。

## 目录

```text
configs/
  tcs_seg/           SpanCrack：train/、eval/、infer/ 各 4 个数据集，flops.yaml
  tcs_seg_lite/      SpanCrack-Lite：train/ 4 个数据集，flops.yaml
models/
  segformer_base.py          TCSSegFormer：SegFormer-B0 编码器 + 本文解码器
  width_adaptive_decoder.py  EMD 与 AMF
  inference_export.py        去掉推理时用不到的原 SegFormer 解码头
  crack_selective_adapter.py, texture_suppression.py
                             早期模块，论文配置中关闭，保留是因为模型类仍引用它们
tasks/               train、eval、infer、flops 四个任务，由 main.py 调用
utils/               数据加载、损失、指标
scripts/eval_topology_v3.py  论文的评测流程：验证集选阈值，测试集算像素和结构指标
pretrained/mit-b0/   SegFormer-B0 的配置文件（权重需另行下载）
```

## 环境

```bash
pip install -r requirements.txt
```

论文实验使用 PyTorch 2.6 和单张 NVIDIA H20。

transformers 4.x 和 5.x 中 SegFormer 的参数名不同，检查点要在同一大版本的 transformers 下训练和加载。

## 预训练权重

编码器从 NVIDIA 发布的 SegFormer-B0 检查点初始化（ImageNet-1k 预训练后在 ADE20K 上微调）。仓库里的 `pretrained/mit-b0/config.json` 与该检查点一致，只需下载权重文件：

```bash
huggingface-cli download nvidia/segformer-b0-finetuned-ade-512-512 model.safetensors --local-dir pretrained/mit-b0
```

## 数据

配置中的数据路径相对于仓库根目录，默认数据放在仓库的上一级：

```text
../data/<数据集>/
  train_img/  train_lab/
  val_img/    val_lab/
  test_img/   test_lab/
```

图像为 `.jpg` 或 `.png`，标注为同名 `.png`，白色（255）为裂缝。四个数据集按约 7:1:2 划分：

| 数据集 | 目录名 | 训练 | 验证 | 测试 |
| --- | --- | ---: | ---: | ---: |
| DeepCrack | `DeepCrack` | 368 | 53 | 106 |
| CamCrack789 | `CamCrack789` | 553 | 79 | 157 |
| CrackMap | `CrackMap` | 84 | 12 | 24 |
| Crack500 | `Crack500` | 2357 | 336 | 675 |

Crack500 按原始图像划分。训练和评测时所有图像都缩放到 512×512。

## 训练

```bash
python main.py train --config configs/tcs_seg/train/deepcrack.yaml
python main.py train --config configs/tcs_seg_lite/train/deepcrack.yaml
```

论文对每个配置用 2025–2029 五个种子各训练一次，改 `training.seed` 即可，输出目录 `weights/spancrack/<数据集>/seed_<种子>/` 也要相应修改。训练 100 轮，每轮在验证集上按阈值 0.5 的 F1 保存最优检查点 `checkpoints/best.pth`。

## 评测

论文表格中的指标由下面的脚本计算：在验证集上选二值化阈值（0.01–0.99），再在测试集上计算 P、R、F1、mIoU、clDice、BF1（2 像素容差）和 Betti-0 误差。脚本需要一张 CUDA GPU，用 `CUDA_VISIBLE_DEVICES` 指定：

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/eval_topology_v3.py --config configs/tcs_seg/train/deepcrack.yaml
```

结果写入配置中 `structure_evaluation.output_dir` 下的 `summary.json`（含选出的阈值 `val_selected_threshold`）和逐图的 `per_image.csv`。

`main.py eval` 给出固定阈值和阈值扫描的像素指标，可用于快速检查：

```bash
python main.py eval --config configs/tcs_seg/eval/deepcrack.yaml
```

## 推理

把 `configs/tcs_seg/infer/<数据集>.yaml` 里的 `inference.threshold` 改成评测得到的阈值，`data.image_dir` 改成待预测的图像目录：

```bash
python main.py infer --config configs/tcs_seg/infer/deepcrack.yaml
```

输出 512×512 的二值 PNG，白色为裂缝。

## 参数量与计算量

```bash
python main.py flops --config configs/tcs_seg/flops.yaml
```

这里统计的模型包含推理时不参与计算的原 SegFormer 解码头（约 0.40 M 参数）。论文表格只统计推理路径，可用 `models/inference_export.py` 中的 `compact_inference_model` 去掉这部分后再统计。

## 许可

本仓库代码以 [MIT](LICENSE) 协议发布。SegFormer-B0 预训练权重的使用条款以 NVIDIA 的发布页为准。
