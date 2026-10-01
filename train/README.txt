116 类资源名称目标检测训练

训练已完成（2026-10-01，40 轮）。合成测试集 mAP50 96.23%、mAP50–95 85.77%。
本地图片识别网页（v1.1，已独立到 Work/webui）：
  powershell -ExecutionPolicy Bypass -File Work/webui/start_web.ps1
打开 http://127.0.0.1:8766 ，选择/拖拽图片或整个文件夹后点击“开始识别”。
支持下拉切换模型（best/last/COCO 预训练）、中文类名、置信度调整、类别筛选、
原图对比、纯推理耗时与 GPU 利用率/显存面板、下载标框图和 JSON。
说明：Work/webui/web/README.txt。识别结果：Work/webui/web/results/。

Python: D:\APP\miniconda\envs\yolo\python.exe
入口: powershell -ExecutionPolicy Bypass -File Work/train/run.ps1
训练参数: train_config.json。默认 YOLO11n 预训练、640、batch 32、40 epochs、patience 10。
使用 data_detection.yaml 的 bbox_visible，保持 train/val/test，所有数据参与各自划分。
不会重新提取、合成资源，也不会训练 segmentation。

输出: Work/train/runs/named116_yolo11n/
  weights/best.pt: 验证集表现最佳权重，用于推理
  weights/last.pt: 最近权重；训练中断时可恢复
  results.csv / results.png: 每轮指标和训练曲线
  preflight.json: 实际标签核验、116 类顺序、环境及数据哈希
  requested_config.json / args.yaml: 请求参数 / 实际训练参数
  initialization.json: 初始权重来源及哈希
  status.json: 训练/评估/完成/失败状态，约 30 秒刷新
  test_metrics.json: 训练完毕后最佳模型的独立测试指标和逐类 AP
  predictions/: 20 张测试样本，左侧可见框 GT，右侧预测（conf >= 0.25）
日志和进程 PID: Work/train/launch.json、Work/train/logs/
查看进度:
  & 'D:\APP\miniconda\envs\yolo\python.exe' Work/train/status.py
训练后检测图片（结果写入 Work/train/inference/）:
  & 'D:\APP\miniconda\envs\yolo\python.exe' Work/train/predict.py 'D:\path\screenshot.png'

使用其他本机预训练权重:
  powershell -ExecutionPolicy Bypass -File Work/train/run.ps1 -Model 'D:\path\model.pt'
中断续训（不要对同一运行重复启动）:
  powershell -ExecutionPolicy Bypass -File Work/train/run.ps1 -Resume '完整路径\weights\last.pt'
官方在正常训练结束时会从 last.pt/best.pt 移除优化器；完成后需要新运行继续微调。
若训练已经完成而测试中断，可单独评估:
  & 'D:\APP\miniconda\envs\yolo\python.exe' Work/train/train.py --evaluate '完整路径\weights\best.pt'
重复运行保护: 已有 last.pt 时拒绝覆盖，请续训或在配置中改 name。

数据范围: 91 个怪物资源身份、25 个角色外观身份。名称继承数据集 classes.txt。
这是合成场景的检测模型；测试指标不代表真实游戏画面表现。
8 个少帧类别在不同划分共享原始外观；4 个哥布林类别采用人工装备位置合成。
原资源的动作、原始 pivot、角色中文名及可操控身份仍未全部解析。
配置采用较轻的颜色增强以保留不同外观类别的颜色差异。

安装只补充 Ultralytics 依赖，保留当前支持 V100 的 torch/torchvision。
关闭图片 RAM/磁盘缓存；仍会生成较小的 YOLO 标签索引缓存（位于 Work 数据集内）。
训练使用 4 个读取进程，验证/测试使用 0 个额外读取进程，以控制 Windows 内存占用。
验证进程数记录在 requested_config.json 的 val_workers，由训练回调配置。
第三方训练上传/遥测集成已关闭。预训练权重可能从官方 GitHub 下载。
官方训练参数说明: https://docs.ultralytics.com/modes/train/
