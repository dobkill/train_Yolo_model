Work/webui —— 本地图片识别网页（独立目录）
================================================

本目录只包含网页识别相关的代码与运行数据，与模型训练完全分离：

  start_web.ps1   启动脚本（从游戏目录运行）
  runtime.py      网页专用运行时（缓存、字体、Ultralytics 设置）
  web/server.py   本地 HTTP 服务 + 推理
  web/static/     前端页面
  web/results/    识别结果输出
  web/logs/       服务日志
  web/qa/         测试报告

启动:
  powershell -ExecutionPolicy Bypass -File Work/webui/start_web.ps1
打开 http://127.0.0.1:8766

模型来源（只读，不修改训练目录）:
  Work/train/runs/named116_yolo11n/weights/best.pt   默认模型
  Work/train/runs/named116_yolo11n/weights/last.pt
  Work/train/pretrained/*.pt                          官方 COCO 权重

类别清单来自 Work/dataset_named_1000/classes.txt。
训练相关请看 Work/train/README.txt；详细使用说明看 web/README.txt。
