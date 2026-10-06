# 阿拉德英雄传 · Work 工具集

本目录是从游戏资源衍生的工具集，目标是**可复用、与具体数据集解耦**：网页识别程序不依赖任何特定数据集，换模型权重即可识别其他内容。

- 数据集生成、标注格式、提取流水线的完整文档：[DATASET.md](DATASET.md)
- 模型训练目录说明：[train/README.txt](train/README.txt)
- 网页识别程序详细说明：[webui/web/README.txt](webui/web/README.txt)

## 本地图片识别网页（webui）

一个本地运行的图片目标检测网页：上传图片 → 加载 YOLO 检测模型 → 显示目标框、类别、置信度、推理耗时和 GPU 状态。模型可在页面中随时切换，程序本身与任何数据集无关。

### 启动

在游戏根目录（本项目根目录）执行：

```powershell
powershell -ExecutionPolicy Bypass -File .\Work\webui\start_web.ps1
```

启动成功后自动打开浏览器，访问 <http://127.0.0.1:8766>。

可选参数：

| 参数 | 默认 | 说明 |
| --- | --- | --- |
| `-Port` | `8766` | 服务监听端口 |
| `-NoBrowser` | — | 只启动服务，不自动打开浏览器 |

```powershell
# 指定端口且不打开浏览器
powershell -ExecutionPolicy Bypass -File .\Work\webui\start_web.ps1 -Port 8767 -NoBrowser
```

启动行为说明：

- 首次启动会加载并预热模型，可能需要等待（最长 120 秒）。
- **服务已在运行时再次执行命令会直接复用**，不会重复加载模型。
- 服务在后台运行，关闭浏览器或本命令窗口不会停止服务。
- 服务 PID、地址和日志位置记录在 `Work/webui/web/service.json`；日志在 `Work/webui/web/logs/`。
- 仅监听 `127.0.0.1`，不对外网开放；图片与结果都保存在本机。

### 页面使用

1. 选择图片、拖入图片、拖入整个文件夹，或点击左侧"添加文件夹"按钮；文件夹会递归收集所有 PNG/JPG/WEBP/BMP（上限 200 张），非图片自动跳过。
2. 在"识别设置"顶部下拉框选择模型，切换即时生效（已加载的模型会缓存，再次切换秒切）。
3. 点击"开始识别"。模型给出目标框、类名、置信度和像素坐标。
4. 识别结果区显示检测目标数、类别数、纯推理耗时；性能面板展开显示预处理／模型推理／后处理／绘制／端到端五段耗时，GPU 行显示利用率和显存占用。
5. 可调整置信度、重叠框抑制和输入尺寸，点击"重新识别"生效。
6. 可切换原图、筛选类别、悬停高亮；下载标框 PNG 或结果 JSON。

限制：单张图片最大 25 MB / 2500 万像素；输入尺寸支持 640、960、1280；有 CUDA 时使用 GPU，否则回退 CPU。

### 模型

程序按注册表自动扫描可用的 `.pt` 检测权重（见 `web/server.py` 的 `build_registry`），不需要修改网页代码即可更换模型。当前注册表指向 `Work/train` 下的训练产物和官方 COCO 预训练权重：

| 注册键 | 来源 | 说明 |
| --- | --- | --- |
| `named116_best` | `Work/train/runs/named116_yolo11n/weights/best.pt` | 训练产物（默认） |
| `named116_last` | 同目录 `last.pt` | 最近权重 |
| `pretrained_*` | `Work/train/pretrained/*.pt` | 官方 COCO 权重（80 类，英文类名，仅供速度对比） |

**复用到其他模型／项目**：把新的 `.pt` 权重放到注册表扫描的目录（或修改 `build_registry` 的扫描路径），并在模型清单中提供对应类别文件，即可识别其他内容；网页前端、推理接口和结果输出均无需改动。

### HTTP 接口

程序同时提供本地 REST 接口，可脱离页面直接调用：

| 方法 | 路径 | 说明 |
| --- | --- | --- |
| `GET` | `/api/health` | 模型注册表、当前模型、设备、指标 |
| `GET` | `/api/gpu` | 当前 GPU 利用率与显存 |
| `POST` | `/api/model` | `{"key": "named116_best"}` 切换模型，返回新 health |
| `POST` | `/api/predict?filename=xxx.png&conf=0.25&iou=0.7&imgsz=640` | 请求体为图片原始二进制（`application/octet-stream`），返回检测结果 JSON |

`/api/predict` 返回 JSON：类别、置信度、原图坐标 xyxy、`timing_ms` 分段耗时、GPU 状态、权重哈希和结果下载地址。各耗时字段含义：

- `timing_ms.inference_pure`：纯模型推理（Ultralytics 内核计时，不含画图、前后处理）
- `timing_ms.preprocess`：读图、EXIF 纠正、RGB→BGR、缩放入网络前
- `timing_ms.postprocess`：NMS、框坐标回映、置信度排序
- `timing_ms.drawing`：保存原图／标框 PNG 的绘制耗时
- `timing_ms.total`：端到端（含上述全部 + JSON 写盘）

### 结果与测试

- 每次识别输出到 `Work/webui/web/results/<识别ID>/`：`original.png`、`annotated.png`、`result.json`。
- 自动化测试（需要 playwright，使用本机 Chrome；日常运行不需要）：

```powershell
& 'D:\APP\miniconda\envs\yolo\python.exe' Work/webui/web/test_web.py
```

### webui 目录结构

```text
Work/webui/
├── start_web.ps1     启动脚本（从项目根目录运行）
├── runtime.py        网页专用运行时（缓存目录、字体、Ultralytics 设置）
└── web/
    ├── launch_web.py   后台启动并等待模型就绪
    ├── server.py       本地 HTTP 服务 + 推理（Python 标准库，无 Web 框架）
    ├── static/         前端页面（HTML/CSS/JS，无在线 CDN 依赖）
    ├── service.json    运行中服务的 PID、地址、日志位置
    ├── results/        识别结果输出
    ├── logs/           服务日志
    └── qa/             浏览器测试报告与截图
```

## 其他子目录速览

| 目录 | 内容 | 文档 |
| --- | --- | --- |
| `webui/` | 本地图片识别网页（本文档主体） | [webui/web/README.txt](webui/web/README.txt) |
| `train/` | 模型训练、推理脚本与训练产物 | [train/README.txt](train/README.txt) |
| `Datas/` | 生成的数据集 | [DATASET.md](DATASET.md) |
| `code/` | 资源提取与数据集生成流水线 | [DATASET.md](DATASET.md) |
| `assets/` | 提取的精灵、地图等共享素材 | [DATASET.md](DATASET.md) |
| `reports/` | 检查与生成过程报告 | [DATASET.md](DATASET.md) |
