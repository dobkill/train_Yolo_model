"use strict";
const $ = id => document.getElementById(id);
const esc = value => String(value ?? "").replace(/[&<>"']/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[c]));
const state = {task:"detect", mode:"preview", kind:"image", view:"overlay", models:[], datasets:[], modelId:"", datasetId:"", files:[], total:0, selected:"", result:null, baseResult:null, image:null, job:null, busy:false, offset:0, listTicket:0, previewTicket:0, registerType:"models", libraryType:"models", frameTimer:null, pollTimer:null, frameTicket:0, lastFrameRequest:0, lastPreviewAt:0, listKey:"", preparingResultId:"", playbackResultId:""};
const selected = () => state.files.find(f => f.id === state.selected);
const model = () => state.models.find(m => m.id === state.modelId);
const dataset = () => state.datasets.find(d => d.id === state.datasetId);
const taskName = task => task === "segment" ? "图像分割" : "物体检测";
const statusName = status => ({queued:"排队中",running:"运行中",stopping:"停止中",completed:"已完成",completed_with_errors:"部分失败",failed:"失败",cancelled:"已停止",interrupted:"已中断"}[status] || status);
const isRunning = job => job && ["queued","running","stopping"].includes(job.status);
async function api(path, body) {
  const options = body === undefined ? {} : {method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify(body)};
  options.signal = AbortSignal.timeout(path === "/api/picker" ? 300000 : 30000);
  const response = await fetch(path, options);
  const value = await response.json();
  if (!response.ok) throw new Error(value.error || "本地服务请求失败");
  return value;
}
function notice(message = "", error = false) { $("notice").textContent = message; $("notice").hidden = !message; $("notice").classList.toggle("error", error); }
function errorNotice(error) { notice(error.message, true); }
function guard(action) { return (...args) => Promise.resolve().then(() => action(...args)).catch(errorNotice); }
function segmented(id, value, key) {
  $(id).querySelectorAll("button").forEach(button => { const active = button.dataset[key] === value; button.classList.toggle("active", active); button.setAttribute("aria-pressed", active); });
}
function params() {
  return {device:$("device").value,imgsz:Number($("imgsz").value),conf:Number($("confidence-number").value),iou:Number($("iou-number").value),stride:Number($("stride").value),show_names:$("show-names").checked,show_conf:$("show-conf").checked,show_boxes:$("show-boxes").checked,show_masks:$("show-masks").checked};
}
let saveTimer;
function remember() {
  clearTimeout(saveTimer);
  saveTimer = setTimeout(() => api("/api/settings", {task:state.task,mode:state.mode,kind:state.kind,model_id:state.modelId,dataset_id:state.datasetId,parameters:params()}).catch(errorNotice), 250);
  footer();
}
function footer(lastTime) {
  $("last-selection").textContent = `上次选择：${model()?.name || "未选择模型"} / ${dataset()?.name || "未选择数据集"}`;
  if (lastTime) $("last-test").textContent = `最近测试时间：${new Date(lastTime).toLocaleString("zh-CN",{hour12:false})}`;
}
function buttons() {
  const can = !!model()?.valid && !!dataset()?.valid && !state.busy;
  $("test-current").disabled = !can || !selected() || state.mode === "evaluate";
  $("test-batch").disabled = !can || !state.total;
  $("stop").disabled = !state.busy || state.job?.status === "stopping";
  $("test-current").querySelector("span").textContent = state.kind === "video" ? "测试当前视频" : "测试当前图片";
  $("test-batch").querySelector("span").textContent = state.mode === "evaluate" ? "开始定量评测" : "批量测试";
  $("previous").disabled = !state.selected || state.files.findIndex(f => f.id === state.selected) <= 0;
  $("next").disabled = !state.selected || state.files.findIndex(f => f.id === state.selected) >= state.total - 1;
  ["register-model","register-dataset","refresh-files","library-add","library-recheck","recheck-top"].forEach(id => $(id).disabled = state.busy);
  ["model-select","dataset-select"].forEach(id => $(id).disabled = false);
  ["task-switch","mode-switch","kind-switch"].forEach(id => $(id).querySelectorAll("button").forEach(b => b.disabled = false));
  $("stride-row").hidden = state.kind !== "video";
  $("evaluation-help").hidden = state.mode !== "evaluate";
  $("previous").querySelector("span").textContent = state.kind === "video" ? "上一个" : "上一张";
  $("next").querySelector("span").textContent = state.kind === "video" ? "下一个" : "下一张";
  const testingHere = state.busy && state.job?.current_file_id === state.selected && state.job?.model_id === state.modelId;
  $("processing").hidden = !testingHere && !state.preparingResultId;
}
function catalogs() {
  const models = state.models.filter(m => m.task === state.task);
  if (!models.some(m => m.id === state.modelId && m.valid)) state.modelId = models.find(m => m.valid)?.id || "";
  if (!state.datasets.some(d => d.id === state.datasetId && d.valid)) state.datasetId = state.datasets.find(d => d.valid)?.id || "";
  $("model-select").replaceChildren(new Option(models.length ? "请选择模型" : `暂无${taskName(state.task)}模型`, ""));
  models.forEach(m => { const option = new Option(m.name + (m.valid ? "" : " · 已失效"), m.id); option.disabled = !m.valid; $("model-select").add(option); });
  $("model-select").value = state.modelId;
  $("dataset-select").replaceChildren(new Option("请选择数据集", ""));
  state.datasets.forEach(d => { const option = new Option(d.name + (d.valid ? "" : " · 已失效"), d.id); option.disabled = !d.valid; $("dataset-select").add(option); });
  $("dataset-select").value = state.datasetId;
  const m = model(), d = dataset();
  $("model-meta").innerHTML = m ? `<dl><div><dt>类型：</dt><dd>${taskName(m.task)}</dd></div><div><dt>格式：</dt><dd>YOLO / ${esc(m.format)}</dd></div><div><dt>状态：</dt><dd class="status"><i class="status-dot"></i> 已登记 · 测试时加载</dd></div></dl><span class="path" title="${esc(m.path)}">${esc(m.path)}</span>` : `<p>登记本地${taskName(state.task)}模型<br>支持 YOLO .pt / .onnx</p>`;
  $("dataset-meta").innerHTML = d ? `<dl><div><dt>数据：</dt><dd>图片 ${d.image_count} 张　视频 ${d.video_count} 个</dd></div><div><dt>标注：</dt><dd>${/\.ya?ml$/i.test(d.path) ? "YAML 配置 · 评测时校验" : d.annotated ? "存在 labels · 评测需 YAML" : "未关联 YAML"}</dd></div><div><dt>状态：</dt><dd class="status"><i class="status-dot"></i> 正常</dd></div></dl>` : `<p>登记图片／视频文件夹或单个文件<br>定量评测请选择 YOLO 数据集 YAML</p>`;
  footer(); buttons();
}
async function health(initial = false) {
  const info = await api("/api/health");
  state.models = info.models; state.datasets = info.datasets;
  $("connection").classList.remove("offline"); $("connection").querySelector("span").textContent = "SQLite 本地记录 · 已连接";
  $("connection").title = info.database;
  $("invalid-banner").hidden = !info.invalid.length;
  $("invalid-text").textContent = `启动检查：${info.invalid.length} 条模型／数据集记录已失效，已禁止选择。可重新登记原路径或移除记录。`;
  if (initial) {
    const s = info.settings;
    state.task = ["detect","segment"].includes(s.task) ? s.task : "detect";
    state.kind = s.kind === "video" ? "video" : "image";
    state.mode = s.mode === "evaluate" ? "evaluate" : "preview";
    state.modelId = s.model_id || ""; state.datasetId = s.dataset_id || "";
    const p = s.parameters || {};
    [["device","device"],["imgsz","imgsz"],["stride","stride"]].forEach(([id,key]) => { if (p[key] != null) $(id).value = String(p[key]); });
    [["confidence","conf"],["iou","iou"]].forEach(([id,key]) => { if (p[key] != null) { $(id).value = p[key]; $(id+"-number").value = Number(p[key]).toFixed(2); } });
    [["show-names","show_names"],["show-conf","show_conf"],["show-boxes","show_boxes"],["show-masks","show_masks"]].forEach(([id,key]) => { $(id).checked = p[key] == null ? (key !== "show_masks" || state.task === "segment") : p[key]; });
    segmented("task-switch", state.task, "task"); segmented("kind-switch", state.kind, "kind"); segmented("mode-switch", state.mode, "mode");
    footer(s.last_test_at);
  }
  catalogs();
  if (initial && isRunning(info.active_job)) { state.job = info.active_job; state.busy = true; buttons(); setTimeout(poll, 500); }
  return info;
}
async function loadFiles(append = false, preservePreview = false) {
  const ticket = ++state.listTicket;
  if (!state.datasetId) { state.files = []; state.total = 0; state.selected = ""; renderFiles(); await preview(); return; }
  const offset = append ? state.files.length : 0;
  const query = new URLSearchParams({kind:state.kind,search:$("file-search").value,offset,limit:80,model_id:state.modelId});
  const response = await api(`/api/datasets/${state.datasetId}/files?${query}`);
  if (ticket !== state.listTicket) return;
  state.files = append ? state.files.concat(response.items) : response.items; state.total = response.total;
  const previous = state.selected;
  if (!state.files.some(f => f.id === state.selected)) state.selected = state.files[0]?.id || "";
  renderFiles();
  if ((!append && !preservePreview) || previous !== state.selected) await selectFile(state.selected);
}
function renderFiles() {
  const key = JSON.stringify([state.datasetId,state.modelId,state.kind,state.selected,state.total,state.busy ? state.job?.current_file_id : "",state.files.map(f=>[f.id,f.outcome?.status,f.outcome?.result_id])]);
  if (key === state.listKey) { buttons(); return; }
  state.listKey = key;
  const list = $("file-list"); list.replaceChildren();
  if (!state.files.length) { const empty = document.createElement("div"); empty.className = "list-empty"; empty.textContent = state.datasetId ? "没有符合条件的文件" : "选择数据集，浏览测试文件"; list.append(empty); }
  state.files.forEach(file => {
    const button = document.createElement("button"); button.className = "file-item" + (file.id === state.selected ? " active" : ""); button.title = file.path;
    const running = state.busy && state.job?.current_file_id === file.id;
    const status = running ? "运行中" : file.outcome?.status === "completed" ? "已完成" : file.outcome?.status === "failed" ? "失败" : file.outcome?.status === "cancelled" ? "已停止" : "未测试";
    button.innerHTML = `<img src="${file.thumbnail}" alt="" loading="lazy"><span class="file-content"><strong>${esc(file.name)}</strong><small>${file.kind === "video" ? "视频" : "图片"} · ${(file.size / 1024 / 1024).toFixed(2)} MB</small></span><span class="file-status ${running ? "running" : status === "已完成" ? "done" : status === "失败" ? "failed" : ""}">${status}</span>`;
    button.querySelector("img").onerror = event => { event.target.removeAttribute("src"); event.target.alt = file.kind === "video" ? "视频" : "图片"; };
    button.onclick = guard(() => selectFile(file.id)); list.append(button);
  });
  $("file-total").textContent = `${state.total} 个${state.kind === "video" ? "视频" : "图片"}`;
  $("load-more").hidden = state.files.length >= state.total;
  $("file-position").textContent = `${Math.max(0,state.files.findIndex(f => f.id === state.selected) + 1)} / ${state.total}`;
  buttons();
}
function stopPlayback() { clearTimeout(state.frameTimer); state.frameTimer = null; $("video").pause(); $("video-reference").pause(); $("play-frames").textContent = "▶"; }
async function selectFile(id) {
  stopPlayback(); state.preparingResultId="";state.playbackResultId="";state.frameTicket++;state.selected = id; state.result = null; state.baseResult = null; state.image = null;
  const ticket = ++state.previewTicket;
  renderFiles(); renderResults();
  const file = selected();
  if (file?.outcome?.result_id) {
    try {
      const report = await api(`/api/results/${file.outcome.result_id}/result.json`);
      if (ticket !== state.previewTicket) return;
      if (report.width) { state.result = report; state.baseResult = report; }
    } catch (error) { if (ticket === state.previewTicket) notice(`历史结果不可用：${error.message}`, true); }
  }
  if (ticket === state.previewTicket) await preview();
}
const loadImage = src => new Promise((resolve,reject) => { const image = new Image(); image.onload = () => resolve(image); image.onerror = () => reject(new Error("画面读取失败，请刷新数据集或检查源文件。")); image.src = src; });
async function preview() {
  const ticket = ++state.previewTicket, file = selected(), report = state.result;
  stopPlayback(); $("video-player").hidden = true; $("canvas").hidden = true;
  $("empty-state").hidden = !!file; $("video-timeline").hidden = true;
  if (!file) { state.image = null; $("file-caption").textContent = "等待选择测试文件"; renderResults(); return; }
  $("file-caption").textContent = `${file.name} · ${report ? `${report.width} × ${report.height} · ${taskName(report.task)}` : file.kind === "video" ? "原视频预览 · 点击测试生成标注视频" : "原始画面 · 等待测试"}`;
  try {
    if (report?.kind === "video" && report.playback_ready) { showVideo(report); renderResults(); return; }
    if (report?.kind === "video" && !["running","queued"].includes(report.status || "completed")) {
      state.image = await loadImage(report.original_url); if(ticket!==state.previewTicket)return;draw();renderResults();
      const ready = await prepareVideo(report, ticket);
      if (ready && ticket === state.previewTicket) { state.result=ready;state.baseResult=ready;showVideo(ready);renderResults(); }
      return;
    }
    let src;
    if (report) src = report.original_url + `?v=${report.processed_frames || report.id}`;
    else if (file.kind === "video") {
      $("video").src = file.url; $("video-player").hidden = false;$("video-player").classList.remove("compare");$("video-reference").hidden=true;
      $("video").onerror = guard(async () => { if (ticket !== state.previewTicket) return; $("video-player").hidden = true; state.image = await loadImage(`/api/preview/${file.id}`);if(ticket!==state.previewTicket)return;draw(); $("file-caption").textContent = `${file.name} · 原编码不支持浏览器播放，测试后将生成 MP4 预览`; });
      state.image = null; renderResults(); return;
    } else src = `/api/preview/${file.id}`;
    const image = await loadImage(src);
    if (ticket !== state.previewTicket) return;
    state.image = image;
    if (report?.kind === "video") $("file-caption").textContent += " · 正在预计算，完成后可流畅播放";
    draw(); renderResults();
  } catch (error) { if (ticket === state.previewTicket) errorNotice(error); }
}
async function prepareVideo(report, ticket) {
  state.preparingResultId=report.id;buttons();
  try {
    let task=await api(`/api/results/${report.id}/playback`,{});
    while(task.status==="running"){
      if(ticket!==state.previewTicket)return null;
      $("processing-text").textContent=task.message;$("file-caption").textContent="正在准备历史视频的 MP4 缓存，无需重新加载模型";
      await new Promise(resolve=>setTimeout(resolve,900));
      if(ticket!==state.previewTicket)return null;
      task=await api(`/api/results/${report.id}/playback`);
    }
    if(task.status==="failed")throw new Error(task.message);
    return task.report;
  } finally { if(state.preparingResultId===report.id){state.preparingResultId="";buttons();} }
}
function showVideo(report, keepTime=false) {
  const video=$("video"),reference=$("video-reference"),player=$("video-player"),compare=state.view==="compare";
  const previousTime=keepTime?video.currentTime:0,playing=keepTime&&!video.paused;
  state.playbackResultId=report.id;state.image=null;$("canvas").hidden=true;player.hidden=false;player.classList.toggle("compare",compare);reference.hidden=!compare;
  const src=state.view==="original"?report.original_video_url:report.video_url;
  if(video.getAttribute("src")!==src){video.src=src;video.onloadedmetadata=()=>{video.currentTime=Math.min(previousTime,Math.max(0,video.duration-.001));if(playing)video.play().catch(errorNotice);requestFrameStats(true);};}
  else requestFrameStats(true);
  if(compare){if(reference.getAttribute("src")!==report.original_video_url)reference.src=report.original_video_url;reference.currentTime=video.currentTime;if(!video.paused)reference.play().catch(()=>{});}else reference.pause();
  video.onerror=()=>notice("MP4 缓存播放失败，请重新测试该视频。",true);
  $("video-timeline").hidden=false;$("frame-seek").max=Math.max(0,(report.output_frames||report.total_frames||1)-1);$("frame-seek").step=1;
  $("file-caption").textContent=`${report.filename} · ${report.width} × ${report.height} · 预计算完成 · ${report.fps} FPS${report.status==="cancelled"?" · 部分结果":""}`;
  buttons();
}
function syncVideoClock() {
  const video=$("video"),reference=$("video-reference"),r=state.baseResult;
  if(!r||r.kind!=="video"||!r.playback_ready||$("video-player").hidden)return;
  const index=Math.min(Number($("frame-seek").max),Math.floor(video.currentTime*(r.fps||25)));
  $("frame-seek").value=index;$("frame-label").textContent=`第 ${index+1} 帧 · ${video.currentTime.toFixed(1)} s`;
  if(!reference.hidden&&Math.abs(reference.currentTime-video.currentTime)>.2)reference.currentTime=video.currentTime;
  requestFrameStats(false);
}
async function requestFrameStats(force) {
  const base=state.baseResult;
  if(!base?.playback_ready||state.playbackResultId!==base.id||$("video-player").hidden)return;
  if(!force&&performance.now()-state.lastFrameRequest<800)return;
  state.lastFrameRequest=performance.now();const ticket=++state.frameTicket;
  const index=Math.floor($("video").currentTime*(base.fps||25));
  try{const frame=await api(`/api/results/${base.id}/frame?index=${index}`);if(ticket!==state.frameTicket||state.baseResult?.id!==base.id)return;state.result={...base,...frame};renderResults();}
  catch(error){if(ticket===state.frameTicket)notice(`帧统计读取失败：${error.message}`,true);}
}
function drawOverlay(ctx, result, xOffset, w, h) {
  if (!result || state.view === "original") return;
  ctx.save(); ctx.translate(xOffset,0);
  const p = params();
  if (p.show_masks) for (const d of result.detections || []) {
    if (d.polygon?.length >= 3) { ctx.beginPath(); d.polygon.forEach(([x,y],i) => i ? ctx.lineTo(x,y) : ctx.moveTo(x,y)); ctx.closePath(); ctx.fillStyle = d.color+"60"; ctx.fill(); }
  }
  const size = Math.max(13,Math.min(32,Math.round(Math.max(w,h)/55)));
  ctx.font = `600 ${size}px "Microsoft YaHei",sans-serif`; ctx.textBaseline = "top";
  for (const d of (result.detections || []).slice().reverse()) {
    const [x1,y1,x2,y2] = d.bbox_xyxy;
    if (p.show_boxes) { ctx.strokeStyle = d.color; ctx.lineWidth = Math.max(2,w/450); ctx.strokeRect(x1,y1,x2-x1,y2-y1); }
    const label = [p.show_names ? d.class_name : "",p.show_conf ? d.confidence.toFixed(2) : ""].filter(Boolean).join(" ");
    if (label) { const width = ctx.measureText(label).width+10, x=Math.max(0,Math.min(x1,w-width)), y=Math.max(0,y1-size-8); ctx.fillStyle=d.color;ctx.fillRect(x,y,width,size+7);ctx.fillStyle="#fff";ctx.fillText(label,x+5,y+2); }
  }
  ctx.restore();
}
function draw() {
  if (!state.image) return;
  const canvas = $("canvas"), image = state.image, w = image.naturalWidth, h = image.naturalHeight;
  const compare = state.view === "compare"; canvas.hidden = false; canvas.width = compare ? w*2 : w; canvas.height = h;
  const ctx=canvas.getContext("2d");ctx.drawImage(image,0,0);
  if (compare) { ctx.drawImage(image,w,0); drawOverlay(ctx,state.result,w,w,h); ctx.strokeStyle="#fff";ctx.lineWidth=Math.max(3,w/200);ctx.beginPath();ctx.moveTo(w,0);ctx.lineTo(w,h);ctx.stroke(); }
  else drawOverlay(ctx,state.result,0,w,h);
  fitCanvas();
}
function fitCanvas() {
  const canvas=$("canvas"), stage=$("stage"); if (canvas.hidden) return;
  const ratio=$("zoom").value === "fit" ? Math.min(stage.clientWidth/canvas.width,stage.clientHeight/canvas.height) : Number($("zoom").value);
  canvas.style.width=`${Math.max(1,Math.round(canvas.width*ratio))}px`;canvas.style.height=`${Math.max(1,Math.round(canvas.height*ratio))}px`;
  stage.style.justifyContent=canvas.width*ratio>stage.clientWidth ? "flex-start" : "center";
  stage.style.alignItems=canvas.height*ratio>stage.clientHeight ? "flex-start" : "center";
}
function download(id,url) { const a=$(id);a.classList.toggle("disabled",!url);a.setAttribute("aria-disabled",!url);if(url){a.href=url+(url.includes("?")?"&":"?")+"download=1";a.setAttribute("download","");}else{a.removeAttribute("href");a.removeAttribute("download");} }
function renderResults() {
  const r=state.result, detections=r?.detections || [];
  $("target-count").textContent=r?.count ?? "—";$("average-conf").textContent=r?.average_confidence != null ? r.average_confidence.toFixed(2) : "—";
  $("infer-time").innerHTML=r?.timing_ms?.inference_pure != null ? `${Math.round(r.timing_ms.inference_pure)}<small>ms</small>` : "—";
  $("class-count").textContent=r ? `${r.class_count} 类` : "—";
  const groups=new Map(); for(const d of detections){if(!groups.has(d.class_id))groups.set(d.class_id,{name:d.class_name,color:d.color,count:0,sum:0});const g=groups.get(d.class_id);g.count++;g.sum+=d.confidence;}
  $("class-stats").innerHTML=groups.size ? [...groups.values()].map(g=>`<tr><td><i class="class-dot" style="background:${esc(g.color)}"></i>${esc(g.name)}</td><td>${g.count}</td><td>${(g.sum/g.count).toFixed(2)}</td></tr>`).join("") : `<tr><td colspan="3" class="table-empty">${r ? "未检测到目标" : "暂无测试结果"}</td></tr>`;
  $("result-detail").innerHTML=r ? `<div class="perf-row"><span>模型</span><span>${esc(r.model)}</span></div><div class="perf-row"><span>运行设备</span><span>${esc(r.device)}</span></div><div class="perf-row"><span>测试参数</span><span>${r.parameters?.imgsz} / conf ${r.parameters?.conf}</span></div>${Object.entries(r.timing_ms || {}).map(([key,value])=>`<div class="perf-row"><span>${({preprocess:"预处理",inference_pure:"模型推理",postprocess:"后处理",total:"单帧总耗时"})[key] || esc(key)}</span><span>${Number(value).toFixed(1)} ms</span></div>`).join("")}${detections.slice(0,100).map(d=>`<div class="result-object"><strong>${esc(d.class_name)} · ${d.confidence.toFixed(2)}</strong><small>坐标 ${d.bbox_xyxy.map(v=>Math.round(v)).join(", ")}${d.polygon?.length ? ` · 掩码 ${d.polygon.length} 个轮廓点` : ""}</small></div>`).join("")}` : '<p class="muted">测试后查看目标位置与性能数据</p>';
  download("download-image",r?.annotated_url);download("download-json",r?.json_url);download("download-video",r?.video_url);$("download-video").hidden=!r?.video_url;
  $("video-summary").hidden=r?.kind!=="video"; if(r?.kind==="video")$("video-summary").textContent=`已测试 ${r.processed_frames} 帧 / ${r.total_frames || "未知"} 帧 · 平均推理 ${r.mean_inference_ms ?? "—"} ms / 帧。目标数与类别统计对应当前帧。`;
}
async function navigate(direction) { let index=state.files.findIndex(f=>f.id===state.selected)+direction;if(index>=state.files.length&&state.files.length<state.total)await loadFiles(true);if(state.files[index])await selectFile(state.files[index].id); }
async function startTest(single) {
  if(state.busy)return;
  stopPlayback();notice();$("metrics-summary").hidden=true;state.busy=true;buttons();
  try{const job=await api("/api/jobs",{model_id:state.modelId,dataset_id:state.datasetId,kind:state.kind,mode:state.mode,file_id:single?state.selected:null,parameters:params()});
    state.job=job;state.lastPreviewAt=0;clearTimeout(state.pollTimer);buttons();footer(job.created_at);await poll();
  }catch(error){state.busy=false;buttons();throw error;}
}
function showMetrics(job) {
  const metrics=job.metrics;$("metrics-summary").hidden=!metrics;
  if(metrics)$("metrics-summary").innerHTML=`<strong>定量评测 · ${esc(job.split || "")}</strong>${Object.entries(metrics).map(([k,v])=>`${esc(k)}：${Number(v).toFixed(4)}<br>`).join("")}<a href="${job.metrics_url}?download=1" download>↓ 下载评测指标</a>`;
}
async function poll() {
  if(!state.job)return;
  const identifier=state.job.id;
  try {
    const job=await api(`/api/jobs/${identifier}`);if(state.job?.id!==identifier)return;state.job=job;state.busy=isRunning(job);
    const elapsed=Math.max(0,Math.floor((Date.now()-new Date(job.created_at).getTime())/1000));
    $("progress").value=job.progress;$("progress-value").textContent=`${job.progress}%`;$("progress-label").textContent=`${job.message}${state.busy?` · ${elapsed} s`:""}`;if(!state.preparingResultId)$("processing-text").textContent=job.message;
    for(const result of job.results){const file=state.files.find(f=>f.id===result.file_id);if(file)file.outcome={result_id:result.result_id,status:job.latest_result?.id===result.result_id ? job.latest_result.status || "completed" : "completed"};}
    renderFiles();
    if(job.latest_result?.file_id===state.selected && job.model_id===state.modelId && job.latest_result.width){const r=job.latest_result;const readyChanged=r.playback_ready&&!state.baseResult?.playback_ready;
      if(!state.result||state.baseResult?.id!==r.id||readyChanged||!state.busy||(performance.now()-state.lastPreviewAt>4000&&!state.baseResult?.playback_ready)){
        state.result=r;state.baseResult=r;state.lastPreviewAt=performance.now();await preview();
      }
    }
    showMetrics(job);
    if(state.busy)state.pollTimer=setTimeout(poll,1000);
    else {notice(job.error || job.message+(job.errors.length?`；${job.errors.slice(0,3).map(e=>`${e.file}：${e.error}`).join("；")}`:""),!!job.error||job.failed>0);buttons();await loadFiles(false,true);}
  }catch(error){if(state.job?.id!==identifier)return;notice("测试状态连接中断，正在重试；仍可切换浏览图片和视频。",true);state.pollTimer=setTimeout(poll,2500);}
}
async function seekFrame(index) {
  const base=state.baseResult;if(!base||base.kind!=="video")return;
  $("video").currentTime=index/(base.fps||25);if(!$("video-reference").hidden)$("video-reference").currentTime=$("video").currentTime;
  $("frame-label").textContent=`第 ${index+1} 帧 · ${(index/(base.fps||25)).toFixed(1)} s`;
  await requestFrameStats(true);
}
async function playFrames() {
  const video=$("video");if(!video.paused){stopPlayback();return;}if(video.ended)video.currentTime=0;await video.play();
}
function openRegister(category) {
  state.registerType=category;const isModel=category==="models";
  $("register-form").reset();$("form-error").hidden=true;$("entry-task").value=state.task;
  $("register-title").textContent=isModel?"登记模型":"登记数据集";
  $("register-description").textContent=isModel?"选择本地 YOLO 权重并指定任务类型，测试时会核验模型实际类型。":"登记文件夹、单个图片／视频，或带标注的 YOLO YAML。保存后无需重复选择。";
  $("entry-path").placeholder=isModel?"D:\\Models\\best.pt":"D:\\Datasets\\validation";
  $("entry-task-label").hidden=!isModel;$("model-browse").hidden=!isModel;$("dataset-browse").hidden=isModel;
  $("save-entry").disabled=false;$("save-entry").textContent="保存登记";$("register-dialog").showModal();
}
async function saveEntry(event) {
  event.preventDefault();const button=$("save-entry");button.disabled=true;button.textContent="检查并保存…";$("form-error").hidden=true;
  try{const entry=await api(`/api/${state.registerType}`,{name:$("entry-name").value.trim(),path:$("entry-path").value.trim(),task:$("entry-task").value});
    if(state.registerType==="models"){state.task=entry.task;state.modelId=entry.id;segmented("task-switch",state.task,"task");$("show-masks").checked=state.task==="segment";}else state.datasetId=entry.id;
    await health();await loadFiles();remember();$("register-dialog").close();notice("登记已保存；同一路径会复用原有记录。");
  }catch(error){$("form-error").textContent=error.message;$("form-error").hidden=false;}finally{button.disabled=false;button.textContent="保存登记";}
}
async function pick(button) {
  const original=button.textContent;button.disabled=true;button.textContent="等待系统窗口…";
  try{const result=await api("/api/picker",{kind:button.dataset.pick});if(result.path)$("entry-path").value=result.path;}catch(error){$("form-error").textContent=error.message;$("form-error").hidden=false;}finally{button.disabled=false;button.textContent=original;}
}
function renderLibrary() {
  const isModel=state.libraryType==="models",rows=isModel?state.models:state.datasets;
  $("library-title").textContent=isModel?"模型库":"数据集库";
  $("library-help").textContent="启动时自动检查本地路径；失效记录保留供修复，移除记录会保留源文件与测试结果。";
  $("library-content").innerHTML=rows.length?rows.map(r=>`<article class="library-row"><div><h3>${esc(r.name)}</h3><p>${isModel?`${taskName(r.task)} · ${esc(r.format)}`:`图片 ${r.image_count} 张 · 视频 ${r.video_count} 个`}</p><p>${esc(r.path)}</p>${r.valid?'<p class="blue">路径有效</p>':`<p class="invalid-message">已失效 · ${esc(r.issue)}</p>`}</div><div class="row-actions"><button data-use="${r.id}" ${!r.valid||state.busy?"disabled":""}>选择</button><button data-remove="${r.id}" ${state.busy?"disabled":""}>移除记录</button></div></article>`).join(""):'<p class="list-empty">暂无记录，点击新增登记</p>';
  $("library-content").querySelectorAll("[data-use]").forEach(button=>button.onclick=guard(async()=>{const row=rows.find(r=>r.id===button.dataset.use);if(isModel){state.task=row.task;state.modelId=row.id;segmented("task-switch",state.task,"task");$("show-masks").checked=state.task==="segment";}else state.datasetId=row.id;catalogs();await loadFiles();remember();$("library-dialog").close();}));
  $("library-content").querySelectorAll("[data-remove]").forEach(button=>button.onclick=guard(async()=>{await api("/api/remove",{category:state.libraryType,id:button.dataset.remove});await health();renderLibrary();await loadFiles();remember();}));
}
function openLibrary(category){state.libraryType=category;renderLibrary();$("library-dialog").showModal();}
async function recheck(){notice("正在检查登记路径与文件列表…");await api("/api/validate",{});await health();await loadFiles();if($("library-dialog").open)renderLibrary();notice("记录检查完成，失效记录已禁止选择。");}
async function history(){
  const rows=await api("/api/history");
  $("history-content").innerHTML=rows.length?rows.map(job=>`<article class="history-row"><header><h3>${esc(job.model_name)} / ${esc(job.dataset_name)}</h3><span class="history-state">${esc(statusName(job.status))}</span></header><p>${esc(new Date(job.created_at).toLocaleString("zh-CN",{hour12:false}))} · ${taskName(job.task)} · ${job.mode==="evaluate"?"定量评测":job.kind==="video"?"视频测试":"图片测试"} · 成功 ${job.completed} / ${job.total}</p>${job.error?`<p class="form-error">${esc(job.error)}</p>`:""}<details><summary>测试参数与错误明细</summary><p>${esc(JSON.stringify(job.parameters))}</p>${(job.errors||[]).map(e=>`<p>${esc(e.file)}：${esc(e.error)}</p>`).join("")}</details><div class="history-results">${(job.results||[]).slice(0,12).map((r,i)=>`<a href="/api/results/${r.result_id}/result.json?download=1" download>结果 ${i+1} · JSON</a>`).join("")}${job.metrics_url?`<a href="${job.metrics_url}?download=1" download>评测指标</a>`:""}</div>${job.metrics?`<p>${Object.entries(job.metrics).map(([k,v])=>`${esc(k)} ${Number(v).toFixed(4)}`).join(" · ")}</p>`:""}</article>`).join(""):'<p class="list-empty">暂无测试记录</p>';
  $("history-dialog").showModal();
}
$("task-switch").querySelectorAll("button").forEach(button=>button.onclick=guard(async()=>{state.task=button.dataset.task;state.modelId="";segmented("task-switch",state.task,"task");$("show-masks").checked=state.task==="segment";catalogs();await loadFiles();remember();}));
$("mode-switch").querySelectorAll("button").forEach(button=>button.onclick=()=>{state.mode=button.dataset.mode;segmented("mode-switch",state.mode,"mode");buttons();remember();});
$("kind-switch").querySelectorAll("button").forEach(button=>button.onclick=guard(async()=>{state.kind=button.dataset.kind;segmented("kind-switch",state.kind,"kind");state.selected="";await loadFiles();remember();}));
$("view-switch").querySelectorAll("button").forEach(button=>button.onclick=()=>{state.view=button.dataset.view;segmented("view-switch",state.view,"view");if(state.baseResult?.playback_ready)showVideo(state.baseResult,true);else draw();});
$("model-select").onchange=guard(async()=>{state.modelId=$("model-select").value;catalogs();await loadFiles();remember();});
$("dataset-select").onchange=guard(async()=>{state.datasetId=$("dataset-select").value;state.selected="";catalogs();await loadFiles();remember();});
let searchTimer;$("file-search").oninput=()=>{clearTimeout(searchTimer);searchTimer=setTimeout(guard(()=>loadFiles()),250);};
$("load-more").onclick=guard(()=>loadFiles(true));$("refresh-files").onclick=guard(recheck);
$("previous").onclick=guard(()=>navigate(-1));$("next").onclick=guard(()=>navigate(1));$("zoom").onchange=fitCanvas;
new ResizeObserver(fitCanvas).observe($("stage"));
$("test-current").onclick=guard(()=>startTest(true));$("test-batch").onclick=guard(()=>startTest(false));$("stop").onclick=guard(async()=>{state.job=await api(`/api/jobs/${state.job.id}/stop`,{});buttons();});
$("frame-seek").oninput=guard(async()=>{stopPlayback();await seekFrame(Number($("frame-seek").value));});$("play-frames").onclick=guard(playFrames);
for(const id of ["confidence","iou"]){$(id).oninput=()=>{$(id+"-number").value=Number($(id).value).toFixed(2);remember();};$(id+"-number").onchange=()=>{const value=Number($(id+"-number").value);if(Number.isFinite(value)&&value>=0&&value<=1){$(id).value=value;remember();}else notice("阈值必须在 0–1 之间。",true);};}
["device","imgsz","stride"].forEach(id=>$(id).onchange=remember);["show-names","show-conf","show-boxes","show-masks"].forEach(id=>$(id).onchange=()=>{draw();remember();if(state.baseResult?.playback_ready)notice("视频标注按测试时的显示设置生成，修改将在下一次视频预计算时生效。");});
$("video").ontimeupdate=syncVideoClock;
$("video").onseeked=()=>{syncVideoClock();requestFrameStats(true);};
$("video").onplay=()=>{$("play-frames").textContent="Ⅱ";const reference=$("video-reference");if(!reference.hidden){reference.currentTime=$("video").currentTime;reference.play().catch(()=>{});}};
$("video").onpause=()=>{$("play-frames").textContent="▶";$("video-reference").pause();};
$("video").onended=()=>{$("play-frames").textContent="▶";$("video-reference").pause();};
$("params-tab").onclick=()=>tab(false);$("result-tab").onclick=()=>tab(true);
function tab(result){$("parameters").hidden=result;$("result-detail").hidden=!result;$("params-tab").classList.toggle("active",!result);$("result-tab").classList.toggle("active",result);$("params-tab").setAttribute("aria-selected",!result);$("result-tab").setAttribute("aria-selected",result);}
$("register-model").onclick=()=>openRegister("models");$("register-dataset").onclick=()=>openRegister("datasets");$("register-center").onclick=()=>openRegister("datasets");
$("models-library").onclick=()=>openLibrary("models");$("datasets-library").onclick=()=>openLibrary("datasets");$("history-library").onclick=guard(history);
$("inspect-invalid").onclick=()=>openLibrary(state.datasets.some(d=>!d.valid)?"datasets":"models");$("recheck-top").onclick=guard(recheck);
$("library-add").onclick=()=>{$("library-dialog").close();openRegister(state.libraryType);};$("library-recheck").onclick=guard(recheck);
$("register-form").onsubmit=saveEntry;document.querySelectorAll("[data-pick]").forEach(button=>button.onclick=()=>pick(button));
document.querySelectorAll(".close-dialog").forEach(button=>button.onclick=()=>button.closest("dialog").close());
document.querySelectorAll("dialog").forEach(dialog=>dialog.addEventListener("click",event=>{if(event.target===dialog){const r=dialog.getBoundingClientRect();if(event.clientX<r.left||event.clientX>r.right||event.clientY<r.top||event.clientY>r.bottom)dialog.close();}}));
guard(async()=>{await health(true);await loadFiles();})().catch(errorNotice);
