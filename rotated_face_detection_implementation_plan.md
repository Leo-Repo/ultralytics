# YOLO26-OBB 旋转人脸检测适配实施计划

## 1. 文档目的

本文档用于指导代码工作区中的 Agent，在一个新克隆的 Ultralytics 仓库中完成旋转人脸检测模型的第一阶段适配。

本阶段目标不是重新设计完整检测器，也不是立即完成极致硬件优化，而是在尽量保留 YOLO26-OBB 训练、损失、标注格式、评估和导出生态的基础上，得到一个满足当前硬约束、可以训练、验证和导出的旋转人脸检测基线。

Agent 应把本文档视为实施规格，而不是只读调研材料。每个修改都必须有对应验证结果，不能只修改 YAML 后结束任务。

---

## 2. 项目背景

### 2.1 业务任务

- 输入：RGB 图像。
- 固定输入尺寸：`320×192`（宽×高），网络 Tensor 形状为 `N×3×192×320`。
- 场景：约 1 米范围内的人脸检测，典型场景包括婴儿监视、家居环境人脸检测、正视摄像头和俯视摄像头。
- 人脸可能以任意平面旋转角度出现，包括横向、倒立及其间任意角度。
- 家居背景复杂，可能存在床品纹理、玩具、相框、屏幕人脸、遮挡、暗光和运动模糊等干扰。
- 业务只要求输出能够包围人脸的旋转矩形，不要求向上层业务提供或评价独立的角度值。
- 精度优先，在业务召回达标的前提下尽量降低误检。

### 2.2 端侧约束

- 最终部署模型参数量小于 1M。
- 硬件优先支持 `Conv + BN + Act + Pool` 级联加速。
- 激活函数优先使用 ReLU。
- 不希望在网络中间计算图中出现 Sigmoid 或 Softmax。
- 检测结果解码、置信度变换、NMS 等后处理位置的 Sigmoid 不属于第一阶段禁止范围。
- Resize、Upsample、Concat、Reshape、Transpose 等数据搬运算子暂不在第一阶段优化范围内。
- 预处理固定为 RGB 顺序和：

  ```text
  normalized = (pixel - 128.0) / 128.0
  ```

- 输入像素的理论映射应为：

  | uint8 输入 | 归一化输出 |
  | ---: | ---: |
  | 0 | -1.0 |
  | 128 | 0.0 |
  | 255 | 0.9921875 |

### 2.3 工程诉求

- 优先使用公开模型和成熟框架适配，不从零编写旋转检测训练框架。
- 需要保留成熟的数据格式、训练入口、验证指标、推理接口和模型导出能力。
- 标注工具需要支持旋转矩形及 YOLO OBB 四角点格式。
- 当前代码工作区已经位于新克隆的 Ultralytics 仓库中。

---

## 3. 已确定的技术决策

### 3.1 任务表示

本任务属于二维 Oriented Object Detection，不求解人脸真实三维姿态。

采用标准旋转矩形表示：

- 数据集文件：Ultralytics YOLO OBB 四角点格式；
- 框架内部：`xywhr` 五参数；
- 最终业务接口：输出四个角点或旋转矩形；
- 不向业务侧暴露角度字段。

旋转矩形本身没有方向，旋转 180° 后与原矩形相同。因此业务所说的“360°旋转人脸检测”是指人脸可能以任意 roll 角出现，并不要求检测框区分 0° 和 180°。

### 3.2 不采用八参数自由四边形头

第一阶段不采用 QBB 八参数回归，原因如下：

- 业务目标是旋转矩形，而不是一般四边形；
- 五参数旋转框已经足够描述目标；
- YOLO26-OBB 已有完整的数据转换、损失、解码、评估和导出链路；
- 八参数头会额外引入顶点顺序、凸性约束、四边形 NMS 和自定义损失。

外部标注使用八个角点坐标，不代表网络采用八参数自由四边形回归。Ultralytics 会在内部将角点转换为 `xywhr`。

### 3.3 模型基线

选用 YOLO26-OBB 作为工程母版。

使用方式分为两层：

1. 官方 `yolo26n-obb`：用于验证原始流程、迁移权重、建立精度参考；它不是最终部署模型。
2. 自定义 `YOLO26-OBB-Face-Lite`：用于最终小于 1M 的端侧部署。

官方 nano OBB 模型超过 1M，且原始结构包含 C2PSA 和 SiLU，因此不能直接部署。Agent 不得把“成功加载官方 yolo26n-obb”视为适配完成。

### 3.4 损失函数

第一阶段保留 YOLO26-OBB 当前官方损失与分配策略，不切换 Chamfer-IoU、KLD 或自定义角点损失。

重点保留：

- 旋转框 ProbIoU 几何损失；
- YOLO26 的周期角度损失；
- `reg_max=1` 的 DFL-free 回归；
- 框架原生 Rotated Task-Aligned Assigner。

只有在真实数据实验明确显示近正方形人脸框仍存在严重训练不稳定或旋转 IoU 异常时，后续阶段才增加 KLD/GWD 对照实验。

### 3.5 Sigmoid/Softmax 边界

第一阶段禁止范围：

- Backbone、Neck 和检测 Head 的中间特征处理中，不允许依赖 Sigmoid/Softmax 的注意力或门控结构。

第一阶段允许范围：

- 分类置信度后处理中的 Sigmoid；
- 推理结果解码或阈值处理；
- 只在训练服务器中使用、不会进入部署推理图的损失计算。

YOLO26-OBB 的 `reg_max=1` 已使 DFL Softmax失效；OBB26 角度分支输出 raw angle。当前主要需要移除的中间结构是 C2PSA。Agent 仍必须通过源码和导出图审计确认不存在其他中间 Sigmoid/Softmax，不能仅凭假设结束检查。

### 3.6 第一阶段暂不处理的内容

以下内容记录为后续阶段事项，本阶段不要主动扩大修改范围：

- Upsample、Resize、Concat 的替换或消除；
- 无 FPN/下采样式 Neck 重构；
- CPU/NPU 任务切分优化；
- 自定义 Rot-NMS 内核；
- 量化感知训练和端侧编译器适配；
- 蒸馏方案；
- KLD、GWD、Chamfer-IoU 对照实验；
- P2 检测层；
- 自定义八参数 QBB Head。

---

## 4. 第一阶段目标架构

目标模型暂定命名：

```text
YOLO26-OBB-Face-Lite-ReLU
```

目标属性：

| 属性 | 第一阶段目标 |
| --- | --- |
| 任务 | 单类别 OBB 人脸检测 |
| 类别数 | `nc=1` |
| 输入 | 固定尺寸 RGB |
| 输入分辨率 | `320×192`（W×H），Tensor 为 `N×3×192×320` |
| 归一化 | `(x-128)/128` |
| 主激活 | ReLU |
| Attention | 无 C2PSA，无中间 Softmax/Sigmoid Attention |
| 框回归 | YOLO26 OBB26，`reg_max=1` |
| 标注输入 | YOLO OBB 四角点 |
| 内部框 | `xywhr` |
| 参数量 | 最终融合推理模型 `<1M`；建议设计目标 `<=0.90M` |
| FPN | 第一阶段保留原有 Upsample/Concat/P3-P5 |
| 后处理 | 保留框架原生逻辑 |

`320×192` 的宽和高都是 32 的整数倍。对 P3/P4/P5 三个检测尺度，预期空间尺寸为：

| 检测层 | Stride | 特征图尺寸（W×H） | Tensor 空间维度（H×W） |
| --- | ---: | ---: | ---: |
| P3 | 8 | `40×24` | `24×40` |
| P4 | 16 | `20×12` | `12×20` |
| P5 | 32 | `10×6` | `6×10` |

Agent 必须验证训练、验证、推理和导出均实际使用该矩形尺寸，不能只把 `imgsz` 写成 320 后得到 `320×320` 输入。

关于参数量口径，Agent 必须同时记录：

1. 训练态参数量；
2. `model.fuse()` 后的推理态参数量；
3. 导出 ONNX 中 initializer 的参数总量。

默认以最终融合推理模型小于 1M 作为部署验收口径。如果产品定义要求训练态模型也小于 1M，必须在报告中单独指出，并进一步调整 end-to-end 双头或模型宽度，不能静默更改验收口径。

---

## 5. Agent 实施原则

### 5.1 保持默认 Ultralytics 行为不变

所有适配尽量采用新增配置、可选参数或独立模块完成。不得为了一个人脸模型，无条件改变所有 YOLO 模型的默认预处理或默认激活。

例如：

- 不应直接把整个仓库所有输入都永久改成 `(x-128)/128`；
- 不应让加载普通 YOLO26、YOLO11 时也被强制改成 ReLU；
- 不应破坏官方 `yolo26n-obb.yaml`。

新增模型配置应独立于官方配置，建议文件名：

```text
ultralytics/cfg/models/26/yolo26-face-obb.yaml
```

### 5.2 单变量修改

按以下顺序实施，每一步提交前都运行验证：

1. 官方模型基线；
2. 自定义 YAML 和单类别；
3. C2PSA 替换；
4. ReLU；
5. 新归一化；
6. 模型宽度/深度缩放到小于 1M；
7. 综合训练和导出验证。

不要在第一次修改中同时改 Head、Loss、Assigner、Neck、激活和预处理。

### 5.3 先检查当前提交，再决定具体代码位置

Ultralytics 更新较快，本文给出的路径是排查入口，不保证当前提交的精确行号。Agent 必须先通过 `rg` 定位真实实现，再修改。

---

## 6. 详细实施步骤

## 阶段 0：仓库基线和环境记录

### 0.1 检查工作区

执行并记录：

```bash
git status --short
git rev-parse --show-toplevel
git rev-parse HEAD
git branch --show-current
python --version
python -c "import torch; print(torch.__version__)"
```

如果工作区不是干净的新克隆，不覆盖或删除已有修改。先报告冲突文件，再决定如何避开。

### 0.2 安装开发版本

优先使用可编辑安装：

```bash
python -m pip install -e .
```

确认 Python 加载的是当前工作区代码，而不是环境中另一个 Ultralytics：

```bash
python -c "import ultralytics; print(ultralytics.__file__)"
```

### 0.3 建立官方 OBB 基线

至少完成：

- 从官方 YAML 构建 `yolo26n-obb`；
- 随机 RGB Tensor 前向；
- 打印模型结构、参数量和输出形状；
- 如网络可用，下载并加载官方 `yolo26n-obb.pt`；
- 导出一次原始 ONNX，作为后续算子对照。
- 基线前向和导出使用固定 Tensor `N×3×192×320`；如框架命令行对矩形 `imgsz` 的语法随版本变化，应先通过 Python API确认实际输入，不得假定参数顺序。

基线输出保存为文本，例如：

```text
artifacts/baseline/model_info.txt
artifacts/baseline/onnx_ops.txt
```

不要把大权重和生成 ONNX 提交到 Git。

验收：原始模型可以构建、前向和导出，失败时先修复环境，不进入结构适配。

---

## 阶段 1：创建旋转人脸专用 YAML

### 1.1 复制而不是修改官方 YAML

以当前仓库的 `yolo26-obb.yaml` 为母版，创建：

```text
ultralytics/cfg/models/26/yolo26-face-obb.yaml
```

保留以下核心属性：

```yaml
nc: 1
reg_max: 1
```

第一轮应尽量保持官方 `end2end` 设置，先隔离验证其他改动。后续根据参数量口径和精度实验决定是否切换，而不是未经评估直接关闭。

### 1.2 模型命名和元数据

确保训练、验证和导出日志中可以明确区分：

- 官方 `yolo26n-obb`；
- 自定义 `yolo26-face-obb`；
- 后续不同宽度的 Lite 变体。

验收：仅把类别数改为 1 后，模型仍可构建、前向、训练一个 batch 和导出。

---

## 阶段 2：替换 C2PSA

### 2.1 先进行算子定位

使用 `rg` 检查：

```bash
rg -n "C2PSA|Softmax|softmax|Sigmoid|sigmoid" ultralytics/nn ultralytics/cfg/models/26
```

重点检查：

```text
ultralytics/cfg/models/26/yolo26-obb.yaml
ultralytics/nn/modules/block.py
ultralytics/nn/modules/head.py
ultralytics/nn/tasks.py
```

需要区分：

- 中间特征 Attention/门控；
- OBB26 raw angle；
- 分类结果后处理 Sigmoid；
- 训练 Loss 中的 Sigmoid/Softmax。

不要把所有搜索结果一律删除。

### 2.2 替换策略

第一候选是使用已有的纯卷积 CSP/C3 类模块替换 C2PSA，并保持：

- 输入输出通道一致；
- 特征图尺寸一致；
- YAML 层索引不变化；
- 后续 Head 的输入来源不变化。

建议优先试验同仓库已有 `C3k2` 变体；如果当前实现参数开销过大，则使用较少 repeat 或更轻的纯卷积模块。第一阶段允许其内部存在 Split/Concat，因为数据搬运优化不在本阶段范围。

不建议简单删除 C2PSA 后直接把上一层接给 Head，除非 Agent 同时给出参数、精度和特征表达能力的对照依据。

### 2.3 C2PSA 替换验收

- `model.modules()` 中不再出现 C2PSA；
- 自定义模型的中间图不再出现 Attention Softmax；
- 输出尺度和 Head 输入通道与替换前一致；
- 随机输入前向无 NaN/Inf；
- ONNX 导出成功；
- 不影响官方原始 YAML 的构建。

---

## 阶段 3：将模型激活修改为 ReLU

### 3.1 修改路径选择

先检查当前版本的模型解析器是否支持在 YAML 中配置全局 activation：

```bash
rg -n "activation|default_act" ultralytics/nn ultralytics/cfg
```

优先级如下：

1. 如果框架已经支持 YAML 级 activation，为自定义 YAML 指定 `nn.ReLU`；
2. 如果不支持，增加仅由自定义模型启用的 activation 配置；
3. 最后才考虑新增独立 ReLU Conv 模块。

禁止通过永久修改共享类全局默认值，使所有其他模型都被动切换到 ReLU。

### 3.2 覆盖范围

需要覆盖：

- Backbone；
- Neck；
- OBB Head 中带激活的卷积块；
- 替换 C2PSA 后的新模块。

最终线性预测卷积不应添加 ReLU，否则会限制分类 logit、距离或角度的输出范围。

### 3.3 权重迁移说明

SiLU 改成 ReLU 后，官方预训练权重不再与原模型完全等价，但卷积权重仍可作为初始化。加载时应：

- 明确记录成功迁移的参数比例；
- 不冻结全部 Backbone；
- 允许 BN 统计量重新更新；
- 分别保存“随机初始化”和“迁移初始化”的最小对照结果。

### 3.4 ReLU 验收

- 自定义模型中 `nn.SiLU` 数量为 0；
- 中间层激活均为 ReLU 或 Identity；
- 最终预测层保持线性；
- 官方模型仍维持原始激活；
- 前向、反向和 ONNX 导出正常。

---

## 阶段 4：实现可选的 `(RGB-128)/128` 归一化

### 4.1 先定位所有预处理入口

执行：

```bash
rg -n "255\.0|/ 255|/255|preprocess_batch|def preprocess" ultralytics
```

至少检查：

- 训练 batch 预处理；
- Validator 预处理；
- Predictor 预处理；
- ONNX/其他格式导出后的官方推理封装；
- INT8 校准数据入口（只记录，量化本阶段不实施）。

### 4.2 设计要求

归一化必须是模型/任务可选项，默认 Ultralytics 行为继续使用 `/255`。

推荐定义统一模式名，例如：

```text
input_norm=minus128_div128
```

不要在多个入口复制不同的数学表达式。应尽量共用一个归一化函数或统一配置解释，避免训练和推理不一致。

### 4.3 RGB 顺序

Ultralytics 的部分输入由 OpenCV 读取为 BGR，随后在 Predictor 中转换到 RGB。Agent 必须确认新归一化发生在通道顺序转换之后，不能只验证数值范围而忽略 RGB/BGR。

### 4.4 固定矩形输入

本项目不采用默认方形输入。所有入口必须统一为：

```text
width  = 320
height = 192
tensor = NCHW = N×3×192×320
```

Agent 需要检查当前版本的 `imgsz` 参数、LetterBox、batch collate、rect training 和 export 实现，确认以下事项：

- 原图按既定策略 resize/letterbox 到 320×192；
- 不会被配置解析器折叠成单个 `320` 并生成 320×320；
- batch 中图像具有固定的 `192×320` 空间尺寸；
- Padding 值经过 `(x-128)/128` 后与训练约定一致；
- OBB 四角点在非等比例 resize/letterbox 后同步正确变换；
- Predictor、Validator 和导出模型的宽高顺序一致；
- ONNX 输入形状固定为 `[1, 3, 192, 320]`，除非测试时明确导出动态 batch；
- P3/P4/P5 输出空间尺寸分别为 `24×40`、`12×20`、`6×10`。

优先使用保持长宽比的 LetterBox，不允许为适配固定输入而直接对图像进行非等比例拉伸，除非后续实验明确证明这种拉伸可接受。

### 4.5 单元测试

至少覆盖：

```python
0   -> -1.0
128 ->  0.0
255 ->  127.0 / 128.0
```

并比较同一张图在以下路径中的网络输入 Tensor：

- 训练 Dataset/Trainer；
- Predictor；
- Validator。

同时断言三条路径最终 Tensor 空间形状均为：

```python
(height, width) == (192, 320)
```

允许图像 resize/pad 不同，但在同一有效像素位置上的通道顺序和归一化公式必须一致。

### 4.6 归一化与输入尺寸验收

- 自定义人脸 OBB 模型使用 `(RGB-128)/128`；
- 普通官方模型仍使用默认 `/255`；
- 训练、验证、推理三条路径一致；
- ONNX 模型输入约定被写入文档或 metadata；
- 预处理不被错误重复执行两次。
- 训练、验证、推理输入均为 `N×3×192×320`；
- ONNX 固定输入为 `[1,3,192,320]`；
- 三个检测尺度的空间形状与预期一致。

---

## 阶段 5：缩放到小于 1M

### 5.1 缩放原则

不改动第一阶段保留的 P3/P4/P5 和 Upsample/Concat，只调整：

- depth multiplier；
- width multiplier；
- max channels；
- C2PSA 替换模块的 repeats；
- Head 中间通道（仅在必要时）。

建议从约为官方 nano 一半的宽度起步，例如把 nano 的宽度系数从 `0.25` 降到 `0.125`，但这只是初始候选，不是参数量结论。

每次构图后必须实际统计参数量。检测 Head 的最小通道约束、对齐规则和双 Head 都可能使参数量不按宽度平方精确变化。

### 5.2 候选梯度

至少形成三个候选并记录：

| 候选 | 目的 |
| --- | --- |
| A：接近 0.9M | 精度优先 |
| B：接近 0.7M | 精度/余量平衡 |
| C：接近 0.5M | 硬件和后续增量余量 |

禁止只保留最小模型。最终应通过真实数据选择精度达标的最大允许模型。

### 5.3 参数量验收

- 实际融合推理参数 `<1,000,000`；
- 建议正式候选 `<=900,000`；
- 记录训练态、融合态和 ONNX 参数量；
- 记录输入尺寸对应 FLOPs，但本阶段不设置 FLOPs 硬门槛；
- 对每个候选保存 YAML 和模型信息文本。

---

## 阶段 6：训练和数据链路

### 6.1 标注格式

采用 Ultralytics YOLO OBB 格式：

```text
class_id x1 y1 x2 y2 x3 y3 x4 y4
```

坐标归一化到图像宽高。类别只有：

```text
0: face
```

主标注工具推荐 CVAT，使用旋转矩形，并直接导出 Ultralytics YOLO Oriented Bounding Boxes。X-AnyLabeling 可作为离线备选。

### 6.2 数据组成

训练数据至少包含：

- 正常正视成人和婴儿人脸；
- 俯视婴儿脸；
- 躺卧、横向、倒立人脸；
- 0-360°各角度区间；
- 被子、枕头、床栏、手部和玩具遮挡；
- 暗光 RGB、曝光变化和运动模糊；
- 娃娃、相框、海报、电视/手机屏幕中的人脸等困难负样本；
- 无人脸的真实家居背景。

WIDER FACE 的普通水平框不能直接作为最终 OBB 真值。它可以用于人脸检测预训练或生成待修正伪标签，但最终旋转框评估必须使用真实 OBB 标注。

### 6.3 旋转增强

优先使用框架已有的 OBB 几何增强，配置覆盖 `[-180°, 180°]`。如果使用 `degrees=180`，必须通过可视化确认：

- 图像旋转正确；
- 四个角点同步变换；
- 旋转后仍为合法矩形；
- 越界裁剪没有生成错误框；
- 接近正方形的框没有因角点排序导致异常。

不得只看训练不报错，必须生成增强可视化样本进行人工检查。

### 6.4 数据划分

- 按人物、家庭场景或视频序列划分 train/val/test；
- 相邻视频帧不得跨集合；
- 旋转增强后的同源图像不得跨集合；
- 测试集必须包含真实横向/倒立/俯视样本，不能全部由合成旋转组成。

### 6.5 第一轮训练策略

建议按以下顺序对照：

1. 官方 yolo26n-obb 结构微调，建立精度参考；
2. Lite + 原 SiLU，用于隔离模型缩放影响；
3. Lite + C2PSA 替换；
4. Lite + C2PSA 替换 + ReLU；
5. 完整目标模型 + 新归一化。

如果资源有限，至少保留“官方参考”和“完整目标模型”两组，但代码适配阶段必须做逐步前向/导出测试。

---

## 阶段 7：测试与验收

### 7.1 必需测试层级

#### A. 静态结构测试

- 模型能够从 YAML 构建；
- `nc=1`；
- `reg_max=1`；
- 无 C2PSA；
- 无 SiLU；
- 中间层无 Softmax/Sigmoid；
- 预测层输出维度正确；
- 参数量小于目标。
- 输入 Tensor 为 `N×3×192×320`；
- P3/P4/P5 空间尺寸依次为 `24×40`、`12×20`、`6×10`。

#### B. 数值测试

- 固定随机种子；
- 随机 RGB 输入前向；
- 输出无 NaN/Inf；
- 前向两次结果一致；
- 一个最小 batch 能完成 loss 计算和反向传播；
- 所有主要分支获得有限梯度。

#### C. 预处理测试

- 0/128/255 精确映射；
- RGB 通道顺序正确；
- Trainer、Validator、Predictor 一致；
- 默认模型仍保持 `/255`。

#### D. 导出测试

- 固定输入 `[1,3,192,320]` 的 ONNX 导出成功；
- ONNX Runtime 与 PyTorch 输出在合理容差内一致；
- 导出图无 Softmax；
- Sigmoid 只允许出现在分类后处理位置；
- 无 Attention 节点或 C2PSA 残留子图；
- 输出可以被现有 OBB 后处理解析为旋转框。

#### E. 训练冒烟测试

- 使用 DOTA8、合成单类 OBB mini dataset 或自有小样本运行 1-3 个 epoch；
- train loss 有限；
- val 流程完成；
- predict 能输出 OBB；
- 导出的 ONNX 能对同一张图推理。

### 7.2 ONNX 算子审计工具

建议新增一个小型只读脚本，例如：

```text
tools/inspect_onnx_ops.py
```

输出至少包括：

- 各 op_type 数量；
- Sigmoid/Softmax 节点名称；
- 这些节点的输入、输出和直接消费者；
- 参数 initializer 总量；
- 模型输入输出名称和形状。

脚本应支持 CI 使用：检测到中间 Softmax 或未白名单 Sigmoid 时返回非零退出码。

### 7.3 业务指标

最终真实数据评估至少报告：

- rotated mAP50 和 mAP50-95；
- Precision、Recall；
- 在指定 Precision 下的 Recall；
- 每小时误检数/漏检数（视频场景）；
- 按人脸像素尺寸分桶；
- 按 roll 角分桶，例如每 45° 一档；
- 正视、俯视、横向、倒立、遮挡、暗光分别统计；
- 旋转 IoU，而不是角度 MAE。

业务不要求角度信息，因此不要把角度误差作为主要验收指标。只要最终旋转矩形的几何位置正确即可。

---

## 7. 推荐代码改动范围

Agent 应先用 `rg` 确认当前提交中的真实路径，预计涉及：

```text
ultralytics/cfg/models/26/yolo26-face-obb.yaml       # 新增专用模型配置
ultralytics/nn/tasks.py                              # 如需解析可选 activation/normalization 元数据
ultralytics/nn/modules/conv.py                       # 仅在需支持模型级 ReLU 配置时修改
ultralytics/nn/modules/block.py                      # 审计 C2PSA，不建议直接破坏原模块
ultralytics/nn/modules/head.py                       # 审计 OBB26，不应无理由改写 Head
ultralytics/utils/loss.py                            # 审计，不在第一阶段改损失
ultralytics/models/yolo/*/train.py                   # 定位训练预处理
ultralytics/engine/predictor.py                      # 定位推理预处理
ultralytics/engine/validator.py                      # 定位验证预处理
tests/                                               # 新增结构、预处理和导出测试
tools/inspect_onnx_ops.py                            # 可选的算子审计脚本
```

尽量不修改：

```text
官方 yolo26-obb.yaml
官方预训练权重
OBB 数据格式定义
现有 Rotated Assigner
现有 OBB loss
现有结果对象与四角点转换接口
```

---

## 8. 建议提交拆分

每个提交只解决一个问题，建议：

1. `test: add yolo26 obb baseline and graph inspection`
2. `feat: add single-class yolo26 face obb config`
3. `feat: replace c2psa in face obb variant`
4. `feat: add relu activation option for face obb model`
5. `feat: add centered-128 rgb normalization mode`
6. `feat: add sub-1m face obb model scales`
7. `test: cover face obb forward train export and preprocessing`
8. `docs: add rotated face obb training and export guide`

如果用户没有明确要求提交，Agent 可以完成修改但不要自行 push。提交前必须确认工作区中不存在需要保留的用户未提交改动。

---

## 9. 每阶段交付物

### 第一阶段代码交付

- 人脸专用 YOLO26-OBB YAML；
- C2PSA-free 模型；
- ReLU 模型级配置；
- 可选 `(RGB-128)/128` 预处理；
- 至少一个小于 1M 的候选；
- 结构和预处理单元测试；
- ONNX 算子审计脚本；
- 最小训练、验证、推理、导出使用说明。

### 第一阶段报告

Agent 最终回复必须包含：

- 修改文件列表；
- 当前 Git commit；
- 训练、验证、推理和 ONNX 的实际输入形状；
- 实际参数量三种口径；
- 激活模块统计；
- ONNX Sigmoid/Softmax 审计结果；
- 归一化一致性测试结果；
- 前向、反向、训练冒烟和导出结果；
- 未解决问题；
- 下一阶段建议。

---

## 10. 决策与停止条件

### 10.1 参数量未达标

如果融合模型仍大于等于 1M：

1. 先降低 width multiplier；
2. 再降低非关键阶段 repeats；
3. 再压缩 Head 中间通道；
4. 本阶段不要直接删除 P3 或重构 FPN；
5. 记录每次调整的参数量变化。

### 10.2 出现中间 Softmax/Sigmoid

如果替换 C2PSA 后仍存在：

1. 确认它属于中间网络还是后处理/Loss；
2. 追踪 ONNX 节点消费者；
3. 中间 Attention/门控必须替换；
4. 后处理分类 Sigmoid暂时保留并在报告中白名单说明；
5. 不得为了让搜索结果为零而错误删除分类逻辑。

### 10.3 ReLU 后精度明显下降

本阶段不恢复 SiLU 作为最终方案。优先：

- 延长微调；
- 更新 BN；
- 使用官方模型教师蒸馏（下一阶段）；
- 适当选择接近 0.9M 的宽模型；
- 增加真实困难样本。

### 10.4 新归一化导致迁移权重失效

检查是否存在：

- RGB/BGR 错误；
- 训练与推理公式不一致；
- 重复归一化；
- 预训练后冻结过多层；
- BN 未更新。

确认实现正确后，再通过完整微调适应输入分布，不要静默改回 `/255`。

### 10.5 第一阶段完成条件

必须同时满足：

- 自定义模型独立于官方 YAML；
- 五参数 OBB 训练/推理链路可用；
- C2PSA 已替换；
- 中间网络无 Sigmoid/Softmax；
- 全部隐藏激活为 ReLU/Identity；
- RGB `(x-128)/128` 在训练、验证、推理一致；
- 固定 `320×192（W×H）` 输入在训练、验证、推理和导出中一致；
- 融合推理模型参数小于 1M；
- 最小训练、验证、预测、ONNX 导出通过；
- 官方默认模型行为没有被破坏；
- 真实业务数据尚未达标时，被明确标记为“工程适配完成、精度待训练验证”，不得声称项目整体完成。

---

## 11. 后续阶段路线图

第一阶段完成后再依次评估：

1. 真实家居/婴儿监控 OBB 数据训练和 hard-negative mining；
2. 输入尺寸与最小人脸像素分析；
3. 完整 YOLO26-OBB 教师到 Lite 学生的蒸馏；
4. INT8 QAT 和量化前后精度对比；
5. 端侧编译和逐层算子支持检查；
6. Upsample/Concat/Resize 的实际延迟分析；
7. 必要时重构硬件友好 Neck；
8. 业务视频上的误检/漏检每小时指标；
9. 许可证与闭源商用合规确认。

---

## 12. 给代码 Agent 的简短执行指令

> 在当前新克隆的 Ultralytics 仓库中，先用固定输入 `N×3×192×320` 验证官方 yolo26n-obb 的构建、前向和 ONNX 导出。随后新增独立的 `yolo26-face-obb.yaml`，不要修改官方 YAML。保持 YOLO26 OBB26、`reg_max=1`、现有旋转框损失和 P3/P4/P5 Neck；把类别改为单类 face，用无 Attention 的纯卷积模块替换 C2PSA，把自定义模型所有隐藏激活改为 ReLU，并增加仅由该模型启用的 RGB `(x-128)/128` 归一化。训练、验证、推理和导出必须统一使用 `320×192（W×H）`，不得退化为 320×320。暂不优化 Upsample/Concat，也不删除分类后处理中的 Sigmoid。通过实际构图逐步缩放模型到融合推理参数小于 1M，并增加结构、输入形状、归一化、训练冒烟和 ONNX 算子审计测试。所有修改必须保持普通 Ultralytics 模型默认行为不变，最终报告修改文件、参数量、算子审计、实际输入形状、测试结果和剩余风险。

---

## 13. 调研依据

- Ultralytics YOLO26：<https://docs.ultralytics.com/models/yolo26>
- Ultralytics OBB 任务：<https://docs.ultralytics.com/tasks/obb>
- YOLO26-OBB 模型 YAML：<https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/models/26/yolo26-obb.yaml>
- OBB26 Head：<https://docs.ultralytics.com/reference/nn/modules/head>
- Ultralytics OBB Loss：<https://docs.ultralytics.com/reference/utils/loss>
- CVAT 旋转矩形：<https://docs.cvat.ai/docs/annotation/manual-annotation/shapes/annotation-with-rectangles/>
- CVAT Ultralytics YOLO OBB 导出：<https://docs.cvat.ai/docs/dataset_management/formats/format-yolo-ultralytics/>
