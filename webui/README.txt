模型测试工作台 v2
=================

从项目根目录启动：
  powershell -ExecutionPolicy Bypass -File Work/webui/start_web.ps1

默认地址：http://127.0.0.1:8766
不打开浏览器：追加 -NoBrowser
更换端口：追加 -Port 8767
指定 Python：追加 -Python 'D:\your-env\python.exe'

使用：
1. 选择“物体检测”或“图像分割”。
2. 点击“登记模型”，选择本地 YOLO .pt / .onnx 文件并填写模型任务。
3. 点击“登记数据集”，选择图片／视频文件夹、单个媒体文件或 YOLO 数据集 YAML。
4. 选择文件，调整设备、尺寸、阈值和显示设置，点击“测试当前”或“批量测试”。
5. 视频先预计算并缓存 H.264 MP4，随后按原帧率流畅播放，支持时间轴、左右对比和下载。
   相同模型／源文件／参数复用缓存。后台计算期间可以切回图片浏览。
6. 使用带标注 YAML 的图片集时，可以切换“定量评测”计算 Precision / Recall / mAP。

SQLite 记录：web/data/library.sqlite3
测试输出：web/results/<结果ID>/
服务日志与 PID：web/logs/、web/service.json
自动检查：每次后台服务启动时检查所有登记路径，失效记录保留并禁止选择。
运行中也可在模型库／数据集库中点击“重新检查路径”。
相同本地路径复用同一记录；上次模型、数据集与参数会自动恢复。
移除登记记录不会删除原始文件或已保存的测试结果。

本页面按模型自身的类别定义推理，不依赖固定数据集或 classes.txt。
首次运行只将相邻 Work/train 中已有权重作为可选登记；该目录不存在仍能启动。
训练代码未改动。网页的缓存、记录、评测配置副本和输出位于 webui 内。

环境：启动器优先使用 webui/.venv，其次现有 yolo 环境，或通过 -Python 指定环境。
当前 .venv 复用已有 PyTorch / Ultralytics，并独立安装 ONNX 依赖，不修改训练环境。
迁移到其他电脑时重新建立虚拟环境，安装合适的 PyTorch 和 requirements.txt。
ONNX 自动设备选择会根据 ONNX Runtime provider 决定；当前 CPU 版使用 CPU。

详细说明：web/README.txt
旧界面备份：.backup_ui_v1/
