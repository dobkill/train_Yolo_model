模型测试工作台 v2 — 详细说明
===========================

界面
----
参考布局：顶部任务／模式切换，左侧模型和数据集，中间画面预览，右侧参数与结果。
预览支持原图、结果叠加、左右对比、缩放、前后切换、文件搜索和分页加载。
显示设置支持类别名称、置信度、检测框与分割掩码。
摘要显示当前图片／视频帧的目标数量、平均置信度、纯推理用时和类别统计。
没有结果时显示空状态；所有结果值来自实际推理。

模型
----
支持 Ultralytics YOLO 检测和实例分割的 .pt 与兼容 YOLO 的 .onnx 模型。
登记时指定检测／分割，运行时检查 PT 的实际任务或 ONNX task 元数据。
模型类别使用权重内部定义，不读固定数据集 classes.txt。
ONNX 需要 onnx 和 onnxruntime，已加入 webui/requirements.txt。
自动设备：PT 有 CUDA 则使用 GPU；ONNX 按其可用 provider 选择。
每次测试固定模型与参数，切换不会混入另一模型的结果。
模型按测试需要加载，支持没有任何模型／数据集时启动登记页面。

数据集
------
可登记本机文件夹、单个媒体文件或 YOLO 数据集 YAML。
图片：PNG、JPG/JPEG、WEBP、BMP、TIF/TIFF；单图最多 2500 万像素。
视频：MP4、AVI、MOV、MKV、WEBM、M4V、MPG/MPEG；具体解码由本机 OpenCV 决定。
普通文件夹递归扫描，登记图片目录可避免把 mask、debug 等辅助图片一起纳入测试。
YAML 默认预览 test，缺少 test 时使用 val，再缺少时使用 train。
YAML 的旧绝对根目录失效时，会尝试相邻目录里的有效图片路径。
适配原项目的通用 yolo/detection、yolo/segmentation 目录；原 YAML 不会被覆盖。
启动检查重新扫描已登记文件夹，文件移动／删除后会更新计数或标记记录失效。

定量评测
--------
需要图片集 YAML，提供 names、有效的 test 或 val、images / labels 对应结构。
检测标签为 YOLO box 格式，实例分割标签为 YOLO polygon 格式。
模型类别定义必须匹配数据集。没有标注时不会用推理结果冒充准确率。
评测使用当前尺寸和阈值，结果包括 Precision、Recall、mAP50、mAP50-95。
分割指标区分 Box / Mask。配置副本与指标保存在 web/results/<测试ID>/。
Ultralytics 可能在原标注目录旁生成 labels.cache；不会改写原图片、标注或 YAML。
测试期间一次只运行一个推理／评测任务，停止在当前推理或评测批次结束后生效。

视频
----
先预计算，再播放。界面区分环境初始化、模型读取、首帧预热、逐帧预计算与 MP4 整理。
测试所有帧或按“视频帧间隔”抽帧，显示处理进度与已等待时间。
frames.sqlite3 保存每个测试帧的框、掩码、置信度与耗时，不一次加载整段视频到内存。
测试后生成原视频和带标注视频的 H.264 MP4，浏览器直接播放缓存并支持拖动和左右对比。
播放不进行推理、不逐帧请求 JPEG、不反复打开原视频；帧统计低频读取 SQLite。
即使抽帧推理，输出仍保留原始帧率、所有源帧和时长，抽帧之间沿用最近一次预测。
相同模型、源文件和参数复用完成的结果；源文件／权重／参数改变则重新计算。
旧的 AVI 结果会自动转换为 MP4，无需重新加载模型或推理。
下载 annotated.mp4 为带框／掩码的视频（不含原视频音轨）。
需要 FFmpeg，优先系统安装，其次 webui/tools/ffmpeg.exe 或 imageio-ffmpeg 包。
视频标注样式按生成时的显示参数保存；修改显示参数后再次测试生效。
后台计算期间可切换任务、模型、数据集和图片／视频浏览，当前计算使用启动时的参数。
结果图片下载对应最后测试帧，JSON 为视频摘要。每帧完整数据保存在 frames.sqlite3。
浏览器不能直接播放原视频的编码时，会显示静态画面，测试后的按帧回看仍可用。

持久化
------
web/data/library.sqlite3：models、datasets、files、settings、jobs、outcomes。
路径按 Windows 大小写规则去重。登记同一路径可以更新名称和模型任务。
启动会检查失效路径；记录保留供修复，失效项禁止选择。
上次选择、设备和推理参数持久化；任务中断后下次启动会标记“已中断”。
删除登记只移除 SQLite 记录，不删除用户模型、数据集或结果。
web/results/<结果ID>/：original.png、annotated.png、result.json。
视频结果额外包含 annotated.mp4、original.mp4 与 frames.sqlite3；旧 AVI 文件保留。
数据仅在本机处理，服务监听 127.0.0.1；无前端 CDN 依赖。

验证
----
使用独立测试环境：
  Work/webui/.qa-env/Scripts/python.exe Work/webui/web/test_web.py
测试临时启动独立端口，数据库、样例、日志、结果及截图都位于 web/qa。
检测／评测使用现有检测权重与少量原数据；ONNX 从 QA 中的权重副本导出。
分割使用本地生成的未训练权重验证真实掩码流程，不代表训练后的分割质量。
覆盖路径去重、启动失效检查、记录恢复、图片／视频／分割／ONNX推理、
停止、定量评测、下载、浏览器交互和桌面／手机布局。
报告：qa/test_report.json；截图：qa/workstation_desktop.png、workstation_video.png 等。

API
---
GET  /api/health                            工作台状态、库和设置
POST /api/models                           {path,name,task}
POST /api/datasets                         {path,name}
POST /api/validate                         重新检查登记记录
POST /api/remove                           {category:models|datasets,id}
POST /api/settings                         上次选择与参数
POST /api/picker                           {kind:model|folder|yaml|media}
GET  /api/datasets/<id>/files               kind/search/offset/limit/model_id
GET  /api/media/<file_id>                   原始媒体（支持 Range）
GET  /api/preview/<file_id>                 标准化图片／视频首帧
GET  /api/thumb/<file_id>                   缩略图
POST /api/jobs                             {model_id,dataset_id,kind,mode,file_id?,parameters}
GET  /api/jobs/<id>                         进度与结果
POST /api/jobs/<id>/stop                    停止任务
GET  /api/history                          最近100个测试任务
GET  /api/results/<id>/<file>               测试产物
GET  /api/results/<id>/frame?index=N        视频帧结果
GET  /api/results/<id>/frame.jpg?index=N    视频原始帧
POST /api/results/<id>/playback             准备旧结果的 MP4 缓存
GET  /api/results/<id>/playback              视频缓存转换状态

推理接口参考：https://docs.ultralytics.com/modes/predict/
评测接口参考：https://docs.ultralytics.com/modes/val/
