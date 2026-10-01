阿拉德 · 本地图片识别网页 v1.1

启动(从游戏目录运行):
  powershell -ExecutionPolicy Bypass -File Work/webui/start_web.ps1
默认地址:http://127.0.0.1:8766
已启动时会复用服务,不重复加载模型。启动脚本会打开浏览器。
不打开浏览器:追加 -NoBrowser。修改端口:追加 -Port 8767。
关闭网页不停止后台服务。服务 PID、地址及日志位置记录在 web/service.json。

使用:
1. 选择图片、拖入图片、拖入整个文件夹、或点击左侧"添加文件夹"按钮;
   文件夹会递归收集所有 PNG/JPG/WEBP/BMP(上限 200 张),非图片自动跳过。
2. 在"识别设置"顶部下拉框选择模型,切换即时生效(已加载的模型会缓存,再次切换秒切)。
3. 点击"开始识别"。模型给出目标框、资源类名、置信度和像素坐标。
4. 识别结果区显示:检测目标数、类别数、纯推理耗时;性能面板展开显示
   预处理 / 模型推理 / 后处理 / 绘制 / 端到端五段耗时,GPU 行显示利用率和显存占用。
5. 可调整置信度、重叠框抑制和输入尺寸,点击"重新识别"生效。
6. 可切换原图、筛选类别、悬停高亮;下载标框 PNG 或结果 JSON。

耗时口径说明:
  timing_ms.inference_pure  纯模型推理(Ultralytics 内核计时,不含画图、前后处理)
  timing_ms.preprocess      读图、EXIF 纠正、RGB->BGR、缩放入网络前
  timing_ms.postprocess     NMS、框坐标回映、置信度排序
  timing_ms.drawing         保存原图/标框 PNG 的绘制耗时
  timing_ms.total           端到端(含上述全部 + JSON 写盘)
  GPU 统计:安装 pynvml 时显示整机 GPU 利用率与显存占用;未安装时仅显示
  进程内显存(cuda memory_reserved),利用率行隐藏。

可选模型(注册表,server.py build_registry 自动扫描):
  named116_best   Work/train/runs/named116_yolo11n/weights/best.pt(默认,116 类)
  named116_last   同目录 last.pt(最近权重)
  pretrained_*    Work/train/pretrained/*.pt 官方 COCO 权重(80 类,类名英文)
说明: 网页代码在 Work/webui,模型权重与训练产物只读自 Work/train,二者互不影响。
  切换到 COCO 权重后类名是英文通用物体,仅供速度对比。

默认模型指标(2026-10-01 训练,40 轮):
  Precision 95.08%、Recall 91.98%、mAP50 96.23%、mAP50-95 85.77%(合成测试集)。
  合成测试集指标不代表真实游戏截图效果;类别沿用原游戏资源名称。
  网页默认使用 CUDA/V100;没有 CUDA 时改用 CPU。启动时加载并预热一次。
  图片以原始方向和分辨率处理;EXIF 方向会纠正,透明区域以白色合成。
  检测框是模型预测的可见框,不是精确 Alpha Mask;此页面使用目标检测模型。

目录:
  server.py               Python 标准库 HTTP 服务,复用现有 Ultralytics 环境
  launch_web.py           后台启动并等待模型就绪
  static/                 HTML/CSS/JavaScript,无在线 CDN 依赖
  results/<识别ID>/        original.png、annotated.png、result.json(含 timing_ms 与 gpu)
  logs/                   服务日志
  qa/                     浏览器测试报告、界面截图和下载验证文件

接口:
  GET  /api/health                     模型注册表、当前模型、设备、指标
  GET  /api/gpu                        当前 GPU 利用率与显存
  POST /api/model                      {"key": "named116_best"} 切换模型,返回新 health
  POST /api/predict?filename=xxx.png&conf=0.25&iou=0.7&imgsz=640
       请求体是图片原始二进制,Content-Type: application/octet-stream。
       返回 JSON:类别、置信度、原图坐标 xyxy、timing_ms 分段耗时、gpu 状态、
       权重哈希和结果下载地址。
  单张最多 25 MB / 2500 万像素;imgsz 支持 640、960、1280。
  仅监听 127.0.0.1;输入图片和结果保存在本机,不会发送到外部服务。

验证:
  & 'D:\APP\miniconda\envs\yolo\python.exe' Work/webui/web/test_web.py
测试需要 playwright,使用本机 Chrome,无需额外下载浏览器;服务运行不需要 playwright。
官方推理接口说明:https://docs.ultralytics.com/modes/predict/
