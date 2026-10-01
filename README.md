# 阿拉德英雄传：资源提取与合成数据集

## 第三阶段：具体类名，每类 1,000 张

本轮输出目录是 **`dataset_named_1000/`**；原来的 `dataset/` 是两类基线，仍保留。训练新版本时使用新目录内的 YAML，不能继续使用旧 `dataset/data.yaml`。

- 116 个具体资源类别：91 个怪物资源类别、25 个角色外观类别。
- 每类在 1,000 张不同场景图中出现：训练 800、验证 100、测试 100。每图同一类只有一个实例，混合图同时计入多个类别。
- 固定随机计划共 24,102 张图：23,629 张正样本、473 张空场景负样本，116,000 个实例。实际完成状态与逐图检查结果以 `dataset_named_1000/validation.json` 为准。
- `dataset_named_1000/catalog.html`：可搜索类名、每类数量、原始帧数量、别名与合成限制。
- `preview_named.html`：24 个场景的交互预览。`dataset_named_1000/debug/index.html`：24 张随机 A/B/C/D 检查图。
- `dataset_named_1000/data_detection.yaml` 和 `data_segmentation.yaml` 分别对应 YOLO 检测和实例分割；`data.yaml` 也可用于检测。
- 原始 Alpha、完整／可见 Mask、完整／可见框、脚点、遮挡比例、层级与 Polygon 全部保留。默认检测使用可见框，分割使用可见轮廓；Polygon 是像素 Mask 的有损派生格式。

本轮全量像素校验已通过：24,102 张图内容唯一、116,000 个实例通过检查、116 类的 800／100／100 配额一致。共 1,487 个 Polygon 代理低于配置的 0.95 栅格 IoU 目标，已逐实例标记；最低约 0.817。对应的原始 Alpha 和完整／可见像素 Mask 全部保留，不因轮廓简化或代理误差而删除。原有 11 项核心测试与新增 4 项配额、来源、旧 renderer 回归测试均通过。独立检查结果见 `validation.json`、`delivery_audit.json`。

**名称和覆盖范围。** 所有名字来自已提取资源包，保留中文、英文和后缀。不把未经核实的英文资源 ID 翻译成游戏显示名，也不将 avatar 目录自动认定为当前玩家。类别范围是已有提取清单中的怪物与角色外观，不包含尚未提取的 UI、道具等资源。129 个原包全部有处理记录：122 个实体资源包归入 116 类，6 个重复／共享角色包作为别名归并；7 个纯武器、特效、影子包不作为独立实体类别。详见 `named_catalog.json` 和数据集内 `catalog.snapshot.json`。

**4 类装备合成的边界。** 哥布林、投掷哥布林、哥布林士兵和哥布林弓箭手共享裸身帧。直接给相同身体图标不同名字会制造标签冲突。这四类使用原包图层的人工位置合成，每类仅两个已查看的姿态；它们不是已解码或实机验证的原始游戏装配。`sprites.snapshot.json` 的 `assembly.layers` 保留原层 ID、路径、原包 PNG 哈希和位置，原始层 PNG 也未改动。合成后 Alpha、Mask 精确对应生成图，但外观与原游戏装配的一致性需要实机验证。其他类别的部分完整身体帧也没有重建独立装备层。

**动作和低多样性类别。** `animation=null`、`animation_status=not_decoded` 仍然如实保留，帧号不是动作名；脚点继续标记真实来源，本轮无游戏原始 pivot。共使用 1,204 个去重后帧／合成姿态。大多数类别按原始外观哈希隔离训练、验证和测试；8 类有效帧不足 5 个，三种划分共享原始外观：上述四类哥布林、铁锤石巨人、cazen2、rokosi、siroco。铁锤石巨人只有一个完整实体帧，另一个提取图是独立手臂。每类 1,000 张代表不同合成场景，不代表 1,000 张原始游戏帧，也不保证动作覆盖或实际识别精度。

原有 `renderer.compose()` 只增加可选的指定来源、细分类名和尺寸／可见性配置，旧默认路径已进行 RGB 像素回归检查。`extractor.py` 没有改写或重新执行；原地图也直接复用。新入口负责配额调度、多进程、断点和逐场景校验，继续调用原有 `ground_truth.export_scene()` 和 `validate_scene_v2()`。

```powershell
# 复用已经整理好的目录，不扫描或重新提取资源；相同配置支持断点续跑。
.\Work\run_named.ps1 -PerClass 1000 -Workers 16 -Output dataset_named_1000

# 独立命令（需要另一份数据时选择新的输出目录）。
python -X utf8 Work/code/render_named_dataset.py --per-class 1000 --workers 16 --output dataset_named_1000
python -X utf8 Work/code/debug_ground_truth.py --dataset dataset_named_1000 --count 24 --seed 20261002
python -X utf8 Work/code/make_preview.py --dataset dataset_named_1000
python -X utf8 Work/code/named_catalog_page.py --dataset dataset_named_1000
python -X utf8 Work/code/test_named_dataset.py
```

生成过程中查看 `dataset_named_1000/progress.json` 和 `reports/named_1000_generation.log`。每个 metadata 检查点都在该图 Alpha、Mask、框、可见性、Polygon、类别与来源通过独立校验后才提交；中途停止可用相同命令恢复。恢复会重新核验已有检查点，配置／类别清单不同则拒绝混用。最后再核对全量图片唯一性、每类各划分配额和来源帧隔离，并建立标准 YOLO `images/labels` 视图。

完成的是可用于训练的数据，不包含模型训练或实机准确率评测。训练前可以先从类别清单查看装备合成、源帧不足和英文资源名称这些限制。

---

已完成从本地游戏资源到 `image + YOLO + mask + metadata` 的流水线。全部代码、提取资源、地图、数据集和报告都位于本 `Work` 目录，原游戏资源只读。

**第二阶段已在现有实现上增量增加完整 Ground Truth。** 直接补充原有 2,000 张场景，不重新扫描、提取资源或生成地图，也不重新渲染这批 RGB。旧检测标签、实例 ID mask 和元数据字段保持兼容。每个实例新增完整／可见框与 Mask、原始 Alpha、可见 Alpha 贡献、分割 Polygon、脚点来源和使用完整 Sprite 面积作分母的 `visibility`。

## 直接查看

- 双击 [preview.html](preview.html)：离线检查 24 个代表场景，可切换检测框和实例 mask。
- [场景总览](reports/preview/contact_sheet.jpg)
- [图片／YOLO 框／mask 对照](reports/preview/image_yolo_mask.png)
- [数据集统计](dataset/summary.json)
- [全量校验报告](dataset/validation.json)
- [YOLO 数据配置](dataset/data.yaml)
- [Detection 训练配置](dataset/data_detection.yaml) / [Segmentation 训练配置](dataset/data_segmentation.yaml)
- [24 张随机四项 debug 对照](dataset/debug/index.html)：A 检测、B 分割、C 脚点、D 遮挡。
- [逐张目视检查记录](dataset/debug/visual_review.json)：包含 24 张图的检查结论与文件哈希。
- [增量升级报告](dataset/ground_truth_upgrade.json)

## Ground Truth v2

原始数据链是 `Sprite Alpha → Full Mask → 原有 Z-order / 前景遮挡 → Visible Mask → bbox / polygon / visibility`。Polygon 始终是派生数据，不会覆盖或修改像素 Mask。

| 实例字段 | 含义与存储 |
| --- | --- |
| `bbox_full` | 完整二值 Alpha Mask 的最小框，场景坐标，允许伸出画面 |
| `bbox_visible` | 最终可见像素的最小框；默认 Detection 目标 |
| `mask_full` | 未遮挡、未裁掉画面外部分的完整二值 Mask；PNG，0/255，尺寸为变换后 Sprite 大小 |
| `mask_full_origin` | 完整 Mask 左上角在场景中的坐标；可为负数 |
| `mask_full_in_frame` | 相同完整 Mask 放入画布后的版本；用于画内 amodal 任务，各实例允许重叠 |
| `mask_visible` | 最终画布大小的可见二值 Mask；默认 Instance Segmentation 目标 |
| `alpha_full` | 变换后 Sprite 的原始 8 位 Alpha，完整保留 0–255 值，未阈值化；原始未变换 RGBA 仍保留在 `assets` |
| `alpha_visible` | NPZ 中的 float32 `alpha`，场景大小；保留经过前方 Alpha 衰减后的连续贡献，未二值化 |
| `polygon_visible` | 从可见 Mask 提取的多区域／孔洞轮廓、YOLO 导出点、简化参数、栅格 IoU |
| `mask_full_area` / `mask_visible_area` | 完整／可见二值 Mask 的非零像素数 |
| `visibility` | 严格等于 `mask_visible_area / mask_full_area`，分母包括画面外的完整 Sprite 像素 |
| `visibility_in_frame` | 画内可见率；用于区分遮挡和画面截断 |
| `foot_point` / `foot_point_source` | 优先变换原资源 pivot/anchor；否则取 `bbox_full` 底边中心，明确标记 `bbox_bottom_center_fallback` |
| `z_order` | 实际绘制顺序；兼容旧 `render_order` |

完整 Mask 使用局部坐标保存，是为了保留负坐标和超出画布的像素；不是只保存可见窗口内的部分。使用 `mask_full_origin` 可以准确放回场景。若需要固定画布大小，使用 `mask_full_in_frame`。完整形状不能放进互斥的单张 ID 图，因为不同实例的完整 Mask 会相互重叠。

二值训练标签沿用已有 alpha ≥ 128 的归属规则，以保持既有检测结果和实例 ID 一致。8 位 `alpha_full` 是更上游的精确数据；`alpha_visible` 额外记录 source-over 合成的连续贡献 `alpha_i × ∏(1 − alpha_front)`，包含前景遮挡，因此半透明对象后方仍有贡献的像素不会只剩一个互斥 ID。这两种 Alpha 表示都不会受 Polygon 简化影响。

旧 `visible_fraction` 字段仍表示“可见面积／画内未遮挡面积”，与新 `visibility` 的分母不同。旧放置筛选规则继续沿用，避免为了改指标而改变已有图像；因此被画面截断的实例，新 `visibility` 可以低于旧筛选阈值。

### Polygon 配置

在 `config.json` 修改：

```json
"polygon": {
  "enabled": true,
  "simplify": true,
  "epsilon": 1.0,
  "epsilon_units": "pixels",
  "min_iou": 0.95
}
```

`epsilon` 是 Douglas–Peucker 简化容差，单位为像素；设 `simplify=false` 可关闭简化。若栅格 IoU 低于目标，导出器逐步降低实际 epsilon，直到达到目标或退回原始轮廓；实际使用值写入 `effective_epsilon`。`enabled=false` 只关闭 Polygon 导出，完整／可见 Mask 和 Alpha 仍照常保存，此时不应使用分割标签训练。

元数据 `components` 保存每块区域的 `outer` 和 `holes`。YOLO 每个实例使用一行，因此将分离区域和孔洞用重复的零宽连接路径组合成一个导出轮廓，不把同一个对象拆成多个训练实例。YOLO 的整数栅格化可能把连接线变成细线，简化也可能有误差；这种代理格式无法对所有多区域 Mask 保证无损。每个实例保存 `raster_iou` 和 `quality_pass`，未达到目标的实例会在校验报告中单独统计，原始 Mask 不受影响。精确分割、Tracking、Amodal 和遮挡推理应使用 PNG／Alpha 数据。

本批次 10,139 个实例中，9,756 个 Polygon 达到 0.95 栅格 IoU 目标，383 个复杂实例低于目标并已标记；最低约 0.705。所有实例均保留精确二值 Mask 和完整 Alpha，不因 Polygon 质量标记而删除。导出顶点总数从 OpenCV 初始轮廓的 2,930,237 减少到 1,387,902；简化后的顶点数量仍取决于对象复杂程度和孔洞数。

YOLO 分割格式遵循 [Ultralytics 官方说明](https://docs.ultralytics.com/datasets/segment/)：每个对象一行 `class x1 y1 x2 y2 ...`，坐标按图片宽高归一化。`labels_segmentation` 是派生 Polygon，`labels_detection` 默认使用 `bbox_visible`。

### 新增目录

```text
dataset/
├── images/{train,val,test}/scene_000000.png        # 原图不变
├── labels_detection/{train,val,test}/*.txt
├── labels_segmentation/{train,val,test}/*.txt
├── masks/instances/{train,val,test}/*.png          # 16 位 ID mask
├── masks/full/{split}/{scene}/instance_001.png      # 完整局部 mask
├── masks/full_in_frame/{split}/{scene}/*.png
├── masks/visible/{split}/{scene}/*.png
├── masks/alpha_full/{split}/{scene}/*.png           # 原始 8 位 Alpha
├── masks/alpha_visible/{split}/{scene}/*.npz        # float32 连续贡献
├── metadata/{train,val,test}/*.json
├── yolo/{detection,segmentation}/{images,labels}/   # 分别对应标准 YOLO 目录
├── data_detection.yaml
├── data_segmentation.yaml
└── debug/                                         # 24 张 A/B/C/D 对照及索引
```

旧 `labels/`、`masks/instance/`、`masks/semantic/`、`data.yaml` 保留。两个训练视图各自拥有匹配的 `images/labels`，避免仅在 YAML 中改标签目录而训练器仍读到旧 detection 标签。Windows 支持时使用文件硬链接，不重复占用 RGB 图片的磁盘空间；不支持时复制。请将这些训练视图视为只读，编辑硬链接图像会影响同一数据集中的原文件。

### 增量补齐和新场景生成

```powershell
# 已完成本次补齐。修改 polygon 参数后，可只重导标注；不会扫描或提取资源。
python Work/code/upgrade_ground_truth.py --dataset dataset
python Work/code/validate_dataset.py --dataset dataset
python Work/code/debug_ground_truth.py --dataset dataset --count 24 --seed 20261001

# 新增另一批场景，复用现有精灵和地图，同时输出 v2 Ground Truth
python Work/code/run_pipeline.py --skip-extract --reuse-maps --count 2000 --output dataset_v2
```

升级前的代码、配置和原始 metadata 在 `reports/ground_truth_upgrade_backup/` 中留有副本，重复升级不会覆盖这些副本。增量实现集中在新模块 `ground_truth.py`；`renderer.py` 仅增加导出调用和原始 anchor 可用时的放置支持，extractor 未更改。

读取精确 Alpha 与局部完整 Mask：

```python
import json
import numpy as np
from PIL import Image
from pathlib import Path

dataset = Path("Work/dataset")
meta = json.loads((dataset / "metadata/train/scene_000000.json").read_text(encoding="utf-8"))
instance = meta["instances"][0]
full_alpha = np.array(Image.open(dataset / instance["alpha_full"]))
full_mask = np.array(Image.open(dataset / instance["mask_full"])) > 0
visible_mask = np.array(Image.open(dataset / instance["mask_visible"])) > 0
with np.load(dataset / instance["alpha_visible"], allow_pickle=False) as data:
    visible_alpha = data["alpha"]  # float32，场景坐标
origin_x, origin_y = instance["mask_full_origin"]
```

## 已生成的数据

| 项目 | 数量／含义 |
| --- | --- |
| 图片 | 2,000 张，640 × 480，RGB PNG |
| 划分 | train 1,600 / val 200 / test 200 |
| 类别 | `0: player`、`1: monster` |
| 实例 | player 1,900；monster 8,239；合计 10,139 |
| 空场景 | 100 张，YOLO 标注文件为空 |
| 角色 | swordman、aganzo、seelen_b 三套外观 |
| 怪物 | 16 种，名称保存在每个实例的元数据里 |
| 渲染帧 | 人工检查后选入 184 个完整角色帧 |
| 地图 | 5 个主题、50 个合成变体；每个都有背景、前景和碰撞图 |
| 完整提取 | 129 个 EMPAK 包，13,705 张 RGBA PNG |

数据集使用洛兰、幽暗密林、冰霜丛林、格拉卡、雷鸣废墟的原始 PNG 素材。随机化内容包括地图变体、帧、位置、遮挡、镜像、缩放和少量整体亮度／对比度变化。

同一地图变体只属于一个划分；三个划分仍共享原始地图主题和角色资源。因此验证集衡量的是已有素材上的新组合，不代表对全新游戏素材的泛化能力。尚未训练检测模型或测量实机检测精度。

## 目录

```text
Work/
├── code/
│   ├── extractor.py          # EMPAK 解包，输出 RGBA 和来源索引
│   ├── build_maps.py         # 背景、前景、合成碰撞图
│   ├── renderer.py           # 图像、YOLO、mask、元数据
│   ├── validate_dataset.py   # 全量校验与 mask 重建
│   ├── make_preview.py       # 离线 HTML 与预览图片
│   ├── run_pipeline.py       # 一键流水线
│   ├── ground_truth.py       # 完整 Alpha / mask / polygon 增量导出
│   ├── upgrade_ground_truth.py # 现有场景直接补齐标注
│   ├── validate_ground_truth.py # v2 独立像素与字段校验
│   ├── debug_ground_truth.py # A/B/C/D 四项可视化
│   ├── test_pipeline.py      # 解包／遮挡／裁切等核心测试
│   ├── common.py
│   ├── inspect_assets.py     # 初始资源检查与接触表
│   └── probe_selection.py    # 帧筛选预览
├── assets/
│   ├── manifest.json
│   ├── sprites/{player,monster}/名称/名称_frame_0000.png
│   ├── auxiliary/            # 保留的未解析非图像记录
│   └── maps/map_00_00/
│       ├── map_background.png
│       ├── map_foreground.png
│       ├── collision.png
│       └── map.json
├── dataset/
│   ├── images/{train,val,test}/scene_000000.png
│   ├── labels/{train,val,test}/scene_000000.txt
│   ├── masks/instance/{train,val,test}/scene_000000.png
│   ├── masks/semantic/{train,val,test}/scene_000000.png
│   ├── metadata/{train,val,test}/scene_000000.json
│   ├── data.yaml
│   ├── index.json
│   ├── summary.json
│   ├── validation.json
│   └── *.snapshot.json
├── reports/                  # 初始检查、24 张试渲染、最终预览
├── config.json
├── requirements.txt
├── run.ps1
└── preview.html
```

## 重新运行

以下命令在游戏根目录执行。已使用 Python 3.13、Pillow 12.2.0、NumPy 2.4.4、OpenCV 5.0.0.93 验证。现有环境已具备依赖。

```powershell
# 使用已提取的帧，生成另一份 2,000 张数据并校验、更新预览
python Work/code/run_pipeline.py --skip-extract --count 2000 --output dataset_v2

# 或者使用 PowerShell 入口
.\Work\run.ps1 -Count 2000 -Output dataset_v2 -SkipExtract

# 从原始 EMPAK 重新执行全部步骤
python Work/code/run_pipeline.py --count 2000 --output dataset_full_rerun
```

输出目录必须位于 `Work` 内且为空。已有 `dataset` 会被保护，不会直接覆盖。流水线重跑会更新 `assets` 下的共享地图与 `preview.html`；更改地图配置前如需保留旧数据集的复核环境，应一并保留其 `assets`。每份数据集包含配置、地图和所选帧的 JSON 快照，像素资源仍引用 `Work/assets`。

单独执行步骤：

```powershell
python Work/code/extractor.py
python Work/code/build_maps.py
python Work/code/renderer.py --count 2000 --output dataset_v3
python Work/code/validate_dataset.py --dataset dataset_v3
python Work/code/make_preview.py --dataset dataset_v3
python Work/code/test_pipeline.py
```

`config.json` 控制随机种子、数量、尺寸、怪物数量、缩放、透明阈值和帧白名单。改变尺寸或地图配置后需要重建地图。同一代码、依赖版本、配置、素材和种子可复现同一输出。

若在新环境缺少依赖，可将环境和下载缓存也放在 `Work`：

```powershell
python -m venv Work/.venv
Work/.venv/Scripts/python -m pip install --cache-dir Work/.pip-cache -r Work/requirements.txt
Work/.venv/Scripts/python Work/code/run_pipeline.py --skip-extract --output dataset_new
```

## 标注约定

**YOLO 检测格式：** `labels/` 和 `labels_detection/` 每行 `class_id center_x center_y width height`，后四个值均归一化到 `[0,1]`。边界框由最终可见 mask 的最小外接矩形计算。分割多边形另存 `labels_segmentation/`，两种格式不混用。

**实例 mask：** 16 位单通道 PNG。`0` 为背景，正整数对应同名 JSON 中的 `instance_id`；实例 ID 每张图重新编号。普通图片查看器中可能近乎全黑，请使用 `preview.html` 的彩色显示，或保持 16 位数值读取。

**语义 mask：** 8 位单通道 PNG。`0` 为背景，`1` 为 player，`2` 为 monster。这里的值比 YOLO 类别 ID 大 1，以给背景保留 0。

**可见性：** alpha ≥ 128 的像素参与 mask。角色按脚点 y 坐标从后往前绘制；随后叠加地图前景。前方实例和前景会覆盖后方实例的 mask。少量低于透明阈值的边缘和阴影不归入实例。某个实例可见像素少于 32，或可见面积占其画内完整面积不足 15% 时，整张场景重新采样，不会留下“画了物体却不标注”的小碎片。

**坐标：** JSON 的 `bbox_xyxy` 是左上包含、右下不包含，即 `[x0,y0,x1,y1)`；`amodal_bbox_xyxy` 是遮挡前的完整精灵框，可以超出画面。`visible_fraction` 分母为裁切到画面后、遮挡前的 mask 面积，因此画面截断和遮挡分开记录。

**元数据：** 每个实例包括类别、具体名称、帧编号、源包／源 PNG 校验值、裁剪、缩放、镜像、位置、脚点、绘制顺序、可见面积、截断标志和遮挡前／后的框。场景还记录种子、地图、碰撞图、亮度及对比度变化。

读取实例 mask：

```python
from PIL import Image
import numpy as np

ids = np.array(Image.open("Work/dataset/masks/instance/train/scene_000000.png"))
first_instance = ids == 1
```

`data.yaml` 可用于支持标准 YOLO 检测目录格式的训练器，当前 `path` 为生成时的绝对路径。移动整个目录后应修改该字段。训练检测只需要 images、labels 和 data.yaml；复核 mask 来源与重新渲染还需要 `assets`。

## 已确认的格式与边界

本游戏实际使用 `Dat/Pak/monster/*.empak`、`Dat/Pak/avatar/*.empak` 和 `Dat/dnf_mapimg/**/*.png`，并非示意图中的 `.xxx`。

通过本地文件验证的 EMPAK 布局为：30 字节文件头；若干条 `0D 0F 3E 03 + uint32 解压长度 + zlib 流`；末尾 uint32 是最后一条记录的压缩记录长度。提取器检查流结束、长度、边界和尾部字段，输出 RGBA PNG 并保留来源偏移、原始 PNG 和像素校验值。

动作名称、帧时长、原始锚点和非图像记录尚未解析。因此文件诚实地命名为 `哥布林_frame_0000.png`，`animation=null`；不会把未知帧自动叫作 `goblin_attack_001`。这里导出静态场景，不声称重现原始攻击动画时间线。脚点使用裁剪后图像的底边中心，是合成放置约定。

原始 `Dat/map.dat` 尚未解析。地图是用原始素材重组的合成场景，`collision.png` 是生成的脚点约束：黑色 0 可行走，白色 255 禁止放置。它包含上方背景区域、底边和前景物件的近似占地，不代表原游戏精确碰撞几何。没有重建游戏 UI、独立武器装配和特效系统。

完整提取的 13,705 帧里包含碎片和特效，不应全部作为实体。当前白名单来自人工查看的完整角色帧；增加类型时，先通过接触表检查，再修改 `sprite_selection`。

## 校验范围

核心测试覆盖记录边界、损坏压缩数据、遮挡、负坐标裁切、前景覆盖、alpha 阈值、输出目录限制和划分数量。全量校验对每张图片检查尺寸及哈希、文件配对、重复内容、实例 ID、类别、可见面积、YOLO 坐标和语义 mask，并从原始提取帧及记录的变换独立重建实例 mask。还检查脚点不落在碰撞区域、地图变体不跨划分，以及源包和原始地图 PNG 的校验值未变化。

第二阶段还逐实例核对完整 Alpha、完整／可见 Mask、画外面积、两种可见率、脚点来源、Polygon 归一化及实际栅格 IoU，并以独立 float64 计算验证保存的 float32 可见 Alpha 贡献。新增测试包含半透明遮挡、原始 pivot 变换、孔洞／多区域轮廓，以及 offscreen full-mask 分母。24 张新 renderer 回归样本与第一阶段 RGB 哈希完全一致；最终 2,000 张原图通过索引哈希检查确认未变。
