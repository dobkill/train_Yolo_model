"""A user-triggered native dialog, isolated from HTTP worker threads."""
import json
import sys
import tkinter as tk
from tkinter import filedialog

root = tk.Tk()
root.withdraw()
root.attributes('-topmost', True)
kind = sys.argv[1]
if kind == 'folder':
    path = filedialog.askdirectory(parent=root, title='选择本地图片／视频数据集文件夹')
elif kind == 'model':
    path = filedialog.askopenfilename(parent=root, title='选择本地 YOLO 模型', filetypes=[('YOLO 模型', '*.pt *.onnx')])
elif kind == 'yaml':
    path = filedialog.askopenfilename(parent=root, title='选择 YOLO 数据集 YAML', filetypes=[('数据集配置', '*.yaml *.yml')])
else:
    path = filedialog.askopenfilename(parent=root, title='选择图片或视频', filetypes=[('媒体文件', '*.png *.jpg *.jpeg *.webp *.bmp *.mp4 *.avi *.mov *.mkv *.webm'), ('所有文件', '*.*')])
print(json.dumps({'path': path}, ensure_ascii=True))
root.destroy()
