"use strict";
const $ = id => document.getElementById(id);
const MAX_QUEUE = 200;
const state = {items: [], selected: null, ready: false, busy: false, activeJob: null, boxes: true, filter: "all", highlight: null, models: [], activeModel: ""};
const current = () => state.items.find(item => item.id === state.selected);
const loadImage = src => new Promise((resolve, reject) => { const img = new Image(); img.onload = () => resolve(img); img.onerror = () => reject(new Error("图片预览加载失败，请重新选择图片。")); img.src = src; });
function notice(message = "", error = false) { $("notice").textContent = message; $("notice").hidden = !message; $("notice").classList.toggle("error", error); }
function buttonState() {
  const item = current();
  $("detect").disabled = !state.ready || !item || state.busy;
  $("detect").querySelector("span").textContent = state.busy ? "正在识别…" : item?.result ? "重新识别" : "开始识别";
  $("processing").hidden = state.activeJob !== item?.id;
}
function renderQueue() {
  const queue = $("queue"); queue.replaceChildren(); $("queue-count").textContent = state.items.length;
  if (!state.items.length) { const p = document.createElement("p"); p.className = "queue-empty"; p.textContent = "把想识别的画面添加到这里。"; queue.append(p); }
  state.items.forEach(item => {
    const button = document.createElement("button"); button.className = "queue-item" + (item.id === state.selected ? " active" : ""); button.title = item.name;
    const img = document.createElement("img"); img.src = item.preview; img.alt = "";
    const content = document.createElement("div"), name = document.createElement("strong"), status = document.createElement("small");
    name.textContent = item.name; status.textContent = state.activeJob === item.id ? "正在识别" : item.result ? `${item.result.count} 个目标` : item.fromFolder ? "文件夹" : "等待识别";
    content.append(name, status); button.append(img, content); button.onclick = () => selectItem(item.id); queue.append(button);
  });
}
function selectItem(id) {
  state.selected = id; state.filter = "all"; state.highlight = null; notice();
  renderQueue(); renderResults(); draw(); buttonState();
}
async function addFiles(files, fromFolder = false) {
  let first = null, errors = [];
  const images = Array.from(files).filter(file => file.type.startsWith("image/") || /\.(png|jpe?g|webp|bmp)$/i.test(file.name));
  const skipped = Array.from(files).length - images.length;
  for (const file of images) {
    if (state.items.length >= MAX_QUEUE) { errors.push(`队列已满（最多 ${MAX_QUEUE} 张），其余图片未添加`); break; }
    if (!/\.(png|jpe?g|webp|bmp)$/i.test(file.name)) { errors.push(`${file.name}：格式不支持`); continue; }
    if (file.size > 25 * 1024 * 1024) { errors.push(`${file.name}：超过 25 MB`); continue; }
    const preview = URL.createObjectURL(file);
    try {
      const img = await loadImage(preview);
      if (img.naturalWidth * img.naturalHeight > 25000000) throw new Error("超过 2500 万像素");
      const item = {id: crypto.randomUUID(), name: file.name, file, preview, img, result: null, fromFolder};
      state.items.push(item); if (!first) first = item.id;
    } catch (error) { URL.revokeObjectURL(preview); errors.push(`${file.name}：${error.message}`); }
  }
  if (first) selectItem(first); else renderQueue();
  if (fromFolder) notice(`已从文件夹添加图片${skipped ? `，跳过 ${skipped} 个非图片文件` : ""}。`);
  if (errors.length) notice(errors.join("；"), true);
}
function filtered() { return (current()?.result?.detections || []).filter(d => state.filter === "all" || String(d.class_id) === state.filter); }
function draw() {
  const item = current(), canvas = $("canvas"), stage = $("dropzone");
  canvas.hidden = !item; $("empty-state").hidden = !!item; stage.classList.toggle("has-image", !!item);
  if (!item) return;
  const img = item.img, w = img.naturalWidth, h = img.naturalHeight;
  canvas.width = w; canvas.height = h;
  const ctx = canvas.getContext("2d"); ctx.drawImage(img, 0, 0);
  if (state.boxes && item.result) {
    const size = Math.max(13, Math.min(36, Math.round(Math.max(w, h) / 45)));
    ctx.font = `600 ${size}px "Microsoft YaHei", sans-serif`; ctx.textBaseline = "top";
    for (const d of filtered().slice().reverse()) {
      const [x1,y1,x2,y2] = d.bbox_xyxy, active = state.highlight === d.id;
      ctx.strokeStyle = active ? "#fff" : d.color; ctx.lineWidth = active ? Math.max(4, w/180) : Math.max(2, w/450);
      if (active) { ctx.fillStyle = d.color + "33"; ctx.fillRect(x1,y1,x2-x1,y2-y1); }
      ctx.strokeRect(x1,y1,x2-x1,y2-y1);
      const label = `${d.class_name} ${(d.confidence*100).toFixed(0)}%`, tw = ctx.measureText(label).width + 10;
      const tx = Math.max(0,Math.min(x1,w-tw)), ty = Math.max(0,y1-size-10);
      ctx.fillStyle = d.color; ctx.fillRect(tx,ty,tw,size+8); ctx.fillStyle = "#07111d"; ctx.fillText(label,tx+5,ty+3);
    }
  }
  $("file-info").textContent = `${item.name} · ${w} × ${h}`;
  $("preview-hint").textContent = item.result ? state.boxes ? "类别 · 置信度" : "原图预览" : "准备就绪";
  $("view-boxes").classList.toggle("active",state.boxes); $("view-original").classList.toggle("active",!state.boxes);
  $("view-boxes").setAttribute("aria-pressed",state.boxes); $("view-original").setAttribute("aria-pressed",!state.boxes);
}
function renderPerf(result) {
  const panel = $("perf-panel");
  if (!result) { panel.hidden = true; return; }
  const t = result.timing_ms || {};
  $("perf-pre").textContent = t.preprocess != null ? `${t.preprocess.toFixed(1)} ms` : "—";
  $("perf-infer").textContent = t.inference_pure != null ? `${t.inference_pure.toFixed(1)} ms` : "—";
  $("perf-post").textContent = t.postprocess != null ? `${t.postprocess.toFixed(1)} ms` : "—";
  $("perf-draw").textContent = t.drawing != null ? `${t.drawing.toFixed(1)} ms` : "—";
  $("perf-total").textContent = t.total != null ? `${t.total.toFixed(1)} ms` : "—";
  const gpu = result.gpu || {};
  const hasUtil = gpu.util_pct != null, hasMem = gpu.mem_used_mb != null && gpu.available;
  $("gpu-row").hidden = !hasUtil; $("gpu-mem-row").hidden = !hasMem;
  if (hasUtil) $("gpu-util").textContent = `${gpu.util_pct.toFixed(0)}%`;
  if (hasMem) $("gpu-mem").textContent = `${gpu.mem_used_mb.toFixed(0)} / ${gpu.mem_total_mb.toFixed(0)} MB` + (gpu.mem_pct != null ? `（${gpu.mem_pct.toFixed(0)}%）` : "");
  panel.hidden = false;
}
function renderResults(rebuildFilter = true) {
  const result = current()?.result;
  $("target-count").textContent = result ? result.count : "—"; $("class-count").textContent = result ? result.class_count : "—";
  $("result-badge").textContent = result ? result.count : "0";
  const infer = result?.timing_ms?.inference_pure;
  $("result-time").textContent = infer != null ? `${infer.toFixed(0)} ms 纯推理` : "—";
  $("infer-time").textContent = infer != null ? `${infer.toFixed(0)}ms` : "—";
  renderPerf(result);
  const select = $("class-filter"); select.disabled = !result || !result.count;
  if (rebuildFilter) {
    select.replaceChildren(new Option("全部类别", "all"));
    if (result) {
      const counts = new Map();
      result.detections.forEach(d => { const c = counts.get(d.class_id) || {name:d.class_name,count:0}; c.count++; counts.set(d.class_id,c); });
      [...counts].sort((a,b) => a[1].name.localeCompare(b[1].name,"zh-CN")).forEach(([id,c]) => select.add(new Option(`${c.name} (${c.count})`, id)));
    }
    select.value = state.filter;
  }
  const list = $("result-list"); list.replaceChildren();
  if (!result || !result.count) {
    const div = document.createElement("div"); div.className = "result-empty";
    const icon=document.createElement("span"),p=document.createElement("p"),small=document.createElement("small");
    icon.textContent="⌖"; p.textContent=result ? "未检测到目标" : "目标会出现在这里";
    small.textContent=result ? "可降低阈值或尝试更清晰的游戏画面" : "识别后查看类别、置信度与位置";
    div.append(icon,p,small); list.append(div);
  } else {
    filtered().forEach(d => {
      const row=document.createElement("button"); row.className="result-row"; row.dataset.id=d.id;
      const swatch=document.createElement("span"); swatch.className="result-swatch"; swatch.style.background=d.color;
      const content=document.createElement("span"),name=document.createElement("strong"),coords=document.createElement("small"),score=document.createElement("span");
      content.className="result-row-content"; name.textContent=d.class_name; coords.textContent=d.bbox_xyxy.map(Math.round).join(" · ");
      score.className="result-confidence"; score.textContent=`${(d.confidence*100).toFixed(1)}%`;
      content.append(name,coords); row.append(swatch,content,score);
      row.onmouseenter=()=>{state.highlight=d.id;draw();}; row.onmouseleave=()=>{state.highlight=null;draw();};
      row.onfocus=()=>{state.highlight=d.id;draw();}; row.onblur=()=>{state.highlight=null;draw();};
      row.onclick=()=>{state.highlight=d.id;state.boxes=true;draw();}; list.append(row);
    });
  }
  for (const [id,url] of [["download-image",result?.annotated_url],["download-json",result?.json_url]]) {
    const link=$(id); link.classList.toggle("disabled",!result); link.setAttribute("aria-disabled",!result);
    if (result) { link.href=url+"?download=1"; link.download=id==="download-image"?"annotated.png":"result.json"; } else link.removeAttribute("href");
  }
}
async function detect() {
  const item=current(); if (!item || state.busy || !state.ready) return;
  state.busy=true; state.activeJob=item.id; notice(); buttonState(); renderQueue();
  const query=new URLSearchParams({filename:item.name,conf:Number($("confidence").value)/100,iou:Number($("iou").value)/100,imgsz:$("imgsz").value});
  try {
    const response=await fetch("/api/predict?"+query,{method:"POST",headers:{"Content-Type":"application/octet-stream"},body:item.file});
    const result=await response.json(); if (!response.ok) throw new Error(result.error || "识别失败，请重试。");
    const img=await loadImage(result.original_url); item.result=result; item.img=img;
    if (state.selected===item.id) { state.boxes=true; state.filter="all"; state.highlight=null; renderResults();draw();notice(`识别完成，找到 ${result.count} 个目标（纯推理 ${result.timing_ms.inference_pure.toFixed(0)} ms）。`); }
  } catch(error) { if (state.selected===item.id) notice(error.message || "连接失败，请确认本地服务正在运行。",true); }
  finally { state.busy=false;state.activeJob=null;buttonState();renderQueue(); }
}
async function switchModel(key) {
  if (!key || key === state.activeModel) return;
  $("model-loading").hidden = false; $("model-select").disabled = true;
  try {
    const response = await fetch("/api/model", {method: "POST", headers: {"Content-Type": "application/json"}, body: JSON.stringify({key})});
    const info = await response.json(); if (!response.ok) throw new Error(info.error || "模型切换失败。");
    applyInfo(info); notice(`已切换到 ${info.model}。之前的识别结果保留，重新识别使用新模型。`);
  } catch (error) { notice(error.message, true); $("model-select").value = state.activeModel; }
  finally { $("model-loading").hidden = true; $("model-select").disabled = false; }
}
function applyInfo(info) {
  state.ready = info.ready; state.models = info.models || []; state.activeModel = info.active;
  const select = $("model-select");
  select.replaceChildren(...state.models.map(m => new Option(m.label, m.key)));
  select.value = state.activeModel;
  $("model-state").textContent = "模型已就绪"; $("model-dot").classList.add("ready");
  $("model-name").textContent = info.model || "—";
  $("model-help").textContent = `${info.classes_note || ""} · ${info.device}${info.weights ? " · " + info.weights : ""}`;
  $("device-name").textContent = `${info.device} · ${info.classes} 类`;
  $("summary-model").textContent = `模型 ${info.model || "—"}`;
  $("summary-classes").textContent = `${info.classes} 类`;
  $("ready-pill").classList.add("ready"); $("ready-pill").querySelector("span").textContent="模型已就绪";
  if (info.test_map50 != null) { $("score-wrap").hidden = false; $("test-score").textContent = (info.test_map50*100).toFixed(2)+"%"; }
  else $("score-wrap").hidden = true;
  buttonState();
}
async function initialize() {
  try {
    const response=await fetch("/api/health"); if (!response.ok) throw new Error(); const info=await response.json();
    applyInfo(info);
    for (const sample of info.examples) {
      const button=document.createElement("button"); button.className="example-button"; button.dataset.example=sample.id; button.title="选择 "+sample.name;
      const image=document.createElement("img"),label=document.createElement("span"),arrow=document.createElement("span");
      image.src=sample.url; image.alt="测试场景 "+(sample.id+1); label.textContent="示例 "+String(sample.id+1).padStart(2,"0"); arrow.className="arrow"; arrow.textContent="↗";
      label.append(arrow); button.append(image,label); button.onclick=async()=>{
        button.disabled=true;
        try { const response=await fetch(sample.url); if(!response.ok)throw new Error("示例图片读取失败"); const blob=await response.blob();await addFiles([new File([blob],sample.name,{type:"image/png"})]); }
        catch(error){notice(error.message,true);}finally{button.disabled=false;}
      }; $("examples").append(button);
    }
    buttonState();
  }catch(error){$("model-state").textContent="服务连接失败";$("ready-pill").querySelector("span").textContent="未连接";notice("无法连接模型。请运行 Work/webui/start_web.ps1，再刷新此页。",true);}
}
for(const id of ["choose-top","choose-center","add-more"]) $(id).onclick=()=>$("file-input").click();
$("add-folder").onclick=()=>$("folder-input").click();
$("file-input").onchange=async event=>{await addFiles(event.target.files);event.target.value="";};
$("folder-input").onchange=async event=>{await addFiles(event.target.files,true);event.target.value="";};
$("model-select").onchange=event=>switchModel(event.target.value);
$("detect").onclick=detect;
$("confidence").oninput=()=>{$("confidence-value").textContent=$("confidence").value+"%";if(current()?.result)notice("参数已调整，点击“重新识别”后生效。");};
$("iou").oninput=()=>{$("iou-value").textContent=(Number($("iou").value)/100).toFixed(2);if(current()?.result)notice("参数已调整，点击“重新识别”后生效。");};
$("imgsz").onchange=()=>{if(current()?.result)notice("参数已调整，点击“重新识别”后生效。");};
$("class-filter").onchange=event=>{state.filter=event.target.value;state.highlight=null;renderResults(false);draw();};
$("view-boxes").onclick=()=>{state.boxes=true;draw();};$("view-original").onclick=()=>{state.boxes=false;draw();};
const dropzone=$("dropzone"); let dragDepth=0;
document.addEventListener("dragover",event=>event.preventDefault());document.addEventListener("drop",event=>event.preventDefault());
dropzone.addEventListener("dragenter",event=>{event.preventDefault();dragDepth++;dropzone.classList.add("dragging");});
dropzone.addEventListener("dragleave",()=>{dragDepth--;if(dragDepth<=0)dropzone.classList.remove("dragging");});
dropzone.addEventListener("drop",async event=>{
  event.preventDefault();dragDepth=0;dropzone.classList.remove("dragging");
  const items = event.dataTransfer.items;
  if (items && typeof items[0]?.webkitGetAsEntry === "function") {
    const entries = Array.from(items, item => item.webkitGetAsEntry()).filter(Boolean);
    if (entries.some(entry => entry.isDirectory)) { await addFolderEntries(entries); return; }
  }
  await addFiles(event.dataTransfer.files);
});
async function addFolderEntries(entries) {
  const files = [];
  const walk = async entry => {
    if (entry.isFile) {
      await new Promise(resolve => entry.file(file => { files.push(file); resolve(); }, resolve));
    } else if (entry.isDirectory) {
      const reader = entry.createReader();
      const readBatch = () => new Promise(resolve => reader.readEntries(async batch => {
        if (!batch.length) return resolve();
        for (const child of batch) await walk(child);
        resolve(readBatch());
      }, resolve));
      await readBatch();
    }
  };
  for (const entry of entries) await walk(entry);
  await addFiles(files, true);
}
dropzone.addEventListener("keydown",event=>{if(event.target===dropzone&&(event.key==="Enter"||event.key===" ")){event.preventDefault();$("file-input").click();}});
initialize();
