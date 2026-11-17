import {COLORS,finiteVector,unit,dot,poseGeometry,project,decodeComparison,decodeCloud} from './geometry.js';
import {CloudView} from './cloud.js';

const el=id=>document.getElementById(id);
const bundle=JSON.parse(el('case-index').textContent),models=bundle.models;
const rows=models[0].rows;
const canvas=el('canvas'),ctx=canvas.getContext('2d');
const labels={success:'成功',failure:'失败',missed:'未抓取',unmarked:'未标注'};
const names={pretrained:'预训练',finetuned:'微调'},prefixes={pretrained:'P',finetuned:'F'};
const visibility={pretrained:'showPretrained',finetuned:'showFinetuned'};
const cloud=new CloudView(el('cloud'),el('axesHud'),message=>el('cloudStatus').textContent=message);
const fmt=(x,digits=4)=>typeof x==='number'&&Number.isFinite(x)?x.toFixed(digits):'—';
let filtered=[],position=0,entries=[],currentImage=null,requestVersion=0,controller=null;
let selectedOriginal=0,selectedPrediction=null,initialSelection=true;
const cache=new Map();
function quantile(values,p){if(!values.length)return null;const i=(values.length-1)*p,a=Math.floor(i),b=Math.ceil(i);return values[a]+(values[b]-values[a])*(i-a);}
function setLink(id,url){const a=el(id);a.hidden=!url;if(url)a.href=url;else a.removeAttribute('href');}
function makeCell(row,value){const td=document.createElement('td');td.textContent=String(value??'—');row.append(td);return td;}
function key(entry,c){return `${entry.model.id}:${c.prediction_index}`;}
function selectedCandidates(entry){return (entry.data.candidates??[]).filter(c=>c.scene_rank!=null).sort((a,b)=>a.scene_rank-b.scene_rank);}
function shownCandidates(entry){return selectedCandidates(entry).slice(0,Number(el('topk').value));}
function displayCandidates(entry){
  const shown=shownCandidates(entry),ids=new Set(shown.map(c=>c.prediction_index));
  if(el('showOriginal').checked)for(const pair of entry.comparison?.comparisons??[]){
    const candidate=entry.data.candidates.find(c=>c.prediction_index===pair.prediction_index);
    if(candidate&&!ids.has(candidate.prediction_index)){shown.push(candidate);ids.add(candidate.prediction_index);}
  }
  return shown;
}
function originals(){return entries[0]?.comparison?.originals??[];}
function activeOriginal(){return originals().find(o=>o.original_index===selectedOriginal);}
function activeCandidate(){
  const shown=entries.flatMap(entry=>displayCandidates(entry).map(c=>({entry,c})));
  return (shown.find(({entry,c})=>key(entry,c)===selectedPrediction)??shown[0])?.c;
}
function scoreName(model){return model.score_key==='task_success_score'?'任务评分':'q';}
function clearScene(){
  entries=[];currentImage=null;selectedOriginal=0;selectedPrediction=null;
  ctx.clearRect(0,0,canvas.width,canvas.height);cloud.clear();
  for(const id of ['candidates','comparisons','pairwise','originalSelect','modelLinks'])el(id).replaceChildren();
  for(const id of ['poses','geometry','provenance','comment','candidateCount','originalDetails','comparisonStatus','topComparison'])el(id).textContent='';
  el('originalSelect').disabled=true;el('focusOriginal').disabled=true;el('focusCandidate').disabled=true;
  if(cloud.available)el('cloudStatus').textContent='等待此帧观测点云…';
  setLink('cloudLink',null);
}
function applyFilters(){
  const old=filtered[position]?.sample,originalFilter=el('originalFilter').value;
  filtered=rows.filter(r=>(!el('day').value||r.day===el('day').value)&&(!el('status').value||r.status===el('status').value)&&(!originalFilter||(originalFilter==='present'?r.original_count>0:originalFilter==='multiple'?r.original_count>1:r.original_count===0)));
  position=Math.max(0,filtered.findIndex(r=>r.sample===old));
  if(initialSelection){position=Math.max(0,filtered.findIndex(r=>r.original_count>0));initialSelection=false;}
  el('sample').replaceChildren();filtered.forEach((r,i)=>{const option=document.createElement('option');option.value=String(i);option.textContent=r.sample+(r.original_count>1?` · ${r.original_count} 个原点`:r.original_count===0?' · 无原点':'');el('sample').append(option);});
  el('filteredCount').textContent=filtered.length;
  const samples=new Set(filtered.map(r=>r.sample));
  for(const model of models){const values=model.rows.filter(r=>samples.has(r.sample)).map(r=>r.score).filter(Number.isFinite).sort((a,b)=>a-b);el(`${model.id}Median`).textContent=fmt(quantile(values,.5));}
  selectCase(position);
}
function imageLoad(url){return new Promise((resolve,reject)=>{const image=new Image();image.onload=()=>resolve(image);image.onerror=()=>reject(new Error('无法读取原始 RGB'));image.src=url;});}
async function loadJSON(url,signal){
  let data=cache.get(url);if(data)return data;
  const response=await fetch(url,{signal});if(!response.ok)throw new Error(`HTTP ${response.status}`);
  data=await response.json();cache.set(url,data);if(cache.size>24)cache.delete(cache.keys().next().value);return data;
}
async function selectCase(index){
  const version=++requestVersion;controller?.abort();controller=new AbortController();const signal=controller.signal;
  clearScene();position=Math.max(0,Math.min(index,filtered.length-1));const row=filtered[position];
  el('previous').disabled=!row||position===0;el('next').disabled=!row||position===filtered.length-1;
  el('sample').value=String(position);el('position').textContent=row?`${position+1} / ${filtered.length}`:'0 / 0';setLink('rgbLink',row?.rgb);
  if(!row){el('identity').textContent='无匹配帧';el('manual').textContent='历史人工状态：—';el('message').textContent='当前筛选没有结果。';if(cloud.available)el('cloudStatus').textContent='无匹配帧。';return;}
  el('identity').textContent=row.sample;el('manual').textContent='历史人工状态：'+labels[row.status];el('message').textContent='正在同时读取两个模型与原规划记录…';
  try{
    // Publish one complete frame atomically; late responses from previous frames cannot mix models.
    const loaded=await Promise.all(models.map(async model=>{
      const r=model.rows.find(r=>r.sample===row.sample);if(!r)throw new Error(`${model.label} 缺少此帧`);
      const [data,rawComparison]=await Promise.all([loadJSON(r.json,signal),loadJSON(r.comparison,signal)]);
      if(data.sample_id!==row.sample||data.identity?.checkpoint_step!==model.checkpoint_step||(data.ranking?.score_field??'q')!==model.score_key)throw new Error(`${model.label} 样本、权重或排序身份不匹配`);
      const comparison=decodeComparison(rawComparison,row.sample);
      if(comparison.provenance?.checkpoint_sha256!==data.identity?.checkpoint_sha256)throw new Error(`${model.label} 对照文件权重不匹配`);
      return {model,row:r,data,comparison};
    }));
    if(version!==requestVersion)return;
    const reference=loaded[0];
    for(const entry of loaded.slice(1)){
      if(JSON.stringify(entry.comparison.originals)!==JSON.stringify(reference.comparison.originals))throw new Error('两个模型的原规划记录不一致');
      for(const field of ['intrinsics','image_size_hw','projection_convention'])if(JSON.stringify(entry.data.observation[field])!==JSON.stringify(reference.data.observation[field]))throw new Error('两个模型的观测坐标定义不一致');
    }
    entries=loaded;selectedOriginal=originals()[0]?.original_index??0;
    const first=selectedCandidates(reference)[0];selectedPrediction=first?key(reference,first):null;
    const data=reference.data,review=data.metadata?.raw_review??data.metadata?.manual_outcome;
    el('comment').textContent=review&&typeof review==='object'?`原备注：${review.commit??'无'}\n原机械臂标记：${review.arm??'无'}`:'无人工备注';
    el('topComparison').textContent=entries.map(e=>`${e.model.label}：Top1 ${e.model.score_label} ${fmt(selectedCandidates(e)[0]?.[e.model.score_key])}`).join('\n')+'\n两个模型的评分定义不同，不直接比较数值大小。三方共用同一帧观测点云。';
    el('geometry').textContent=JSON.stringify(data.metadata?.geometry_provenance??data.observation,null,2);
    el('provenance').textContent=JSON.stringify(entries.map(e=>({model:e.model.label,identity:e.data.identity,timing:e.data.timing,seed:e.data.seed})),null,2);
    for(const entry of entries)for(const [field,label]of [['json','JSON'],['npz','NPZ'],['comparison','原点对照 JSON'],['overlay','全部 NMS 叠图']]){
      if(!entry.row[field])continue;const a=document.createElement('a');a.href=entry.row[field];a.textContent=`${names[entry.model.id]} ${label}`;a.className=entry.model.id;if(field==='overlay'){a.target='_blank';a.rel='noopener';}else a.download='';el('modelLinks').append(a);
    }
    renderComparison();draw();
    await Promise.all([
      (async()=>{try{if(!row.rgb){el('message').textContent='此帧没有 RGB。';return;}const image=await imageLoad(row.rgb);if(version!==requestVersion)return;currentImage=image;el('message').textContent='';draw();}catch(error){if(version===requestVersion)el('message').textContent=error.message;}})(),
      loadCloud(reference,version,signal)
    ]);
  }catch(error){
    if(version!==requestVersion||error.name==='AbortError')return;
    el('message').textContent='读取失败：'+error.message;el('comparisonStatus').textContent='未显示不完整的三方对照。';if(cloud.available)el('cloudStatus').textContent='此帧未加载。';
  }
}
async function loadCloud(entry,version,signal){
  try{
    const data=entry.comparison;
    if(!data.cloud){if(cloud.available)el('cloudStatus').textContent='此帧无观测点云资产。';return;}
    const url=new URL(data.cloud.url,new URL(entry.row.comparison,location.href).href.replace(/comparison\/[^/]*$/,''));setLink('cloudLink',url.href);
    if(!cloud.available)return;
    el('cloudStatus').textContent='正在读取三方共用观测点云…';
    const response=await fetch(url,{signal});if(!response.ok)throw new Error(`点云 HTTP ${response.status}`);
    const buffer=await response.arrayBuffer();if(version!==requestVersion)return;
    cloud.setCloud(decodeCloud(buffer,data.cloud),data.cloud);
    const first=originals().find(o=>finiteVector(o.point_cam_m))??activeCandidate();
    if(first){cloud.focus(first);cloud.distance=.45;cloud.schedule();}
    el('cloudStatus').textContent=`三方共用观测深度点云 · 显示 ${data.cloud.count.toLocaleString()} 点（仅显示抽样）`;draw();
  }catch(error){if(version!==requestVersion||error.name==='AbortError')return;if(cloud.available)el('cloudStatus').textContent='三维资产未加载：'+error.message;}
}
function renderComparison(){
  el('originalSelect').replaceChildren();el('comparisons').replaceChildren();
  const source=originals();
  el('comparisonStatus').textContent=source.length?`${source.length} 个原规划吸点（品红），分别对比预训练（青色）和微调（黄色）。原规划均未验证实际执行。`:'此帧无可用原规划吸点；同时显示预训练与微调结果。';
  el('originalSelect').disabled=!source.length;
  for(const original of source){
    const option=document.createElement('option');option.value=String(original.original_index);option.textContent=`原点 ${original.original_index+1} · 实例 ${original.instance_id??'—'} / 物体 ${original.object_id??'—'}`;el('originalSelect').append(option);
    for(const entry of entries){
      const match=entry.comparison.comparisons.find(c=>c.original_index===original.original_index),tr=document.createElement('tr');
      const first=makeCell(tr,`原点 ${original.original_index+1}`),identity=document.createElement('small');identity.textContent=`${original.instance_id??'—'} / ${original.object_id??'—'}`;first.append(identity);
      const matched=match&&Number.isInteger(match.prediction_index),cell=makeCell(tr,'');cell.className=entry.model.id;
      const description=names[entry.model.id]+' · '+(matched?(match.scene_rank!=null?`Top${match.scene_rank}`:`全候选最高${scoreName(entry.model)}`):'无同实例候选');
      if(matched){const button=document.createElement('button');button.textContent=description;button.title='聚焦此同实例候选';button.addEventListener('click',()=>{selectedOriginal=original.original_index;selectedPrediction=key(entry,match);el('originalSelect').value=String(selectedOriginal);cloud.focus(match);draw();});cell.append(button);}else cell.textContent=description;
      makeCell(tr,fmt(match?.distance_mm,2));makeCell(tr,fmt(match?.outward_axis_angle_deg,2));el('comparisons').append(tr);
    }
  }
  el('originalSelect').value=String(selectedOriginal);
  el('originalDetails').textContent=JSON.stringify({note:'原 score 与两个模型的分数不等价。原位姿不是实测执行真值。',originals:source,models:entries.map(e=>({model:e.model.label,comparisons:e.comparison.comparisons}))},null,2);
}
function renderPairwise(){
  el('pairwise').replaceChildren();
  const top=Object.fromEntries(entries.map(e=>[e.model.id,selectedCandidates(e)[0]])),original=activeOriginal();
  const pairs=[['原点 ↔ 预训练 Top1',original,top.pretrained],['原点 ↔ 微调 Top1',original,top.finetuned],['预训练 ↔ 微调 Top1',top.pretrained,top.finetuned]];
  for(const [label,a,b]of pairs){
    let distance=null,angle=null;
    if(finiteVector(a?.point_cam_m)&&finiteVector(b?.point_cam_m))distance=Math.hypot(...a.point_cam_m.map((x,i)=>x-b.point_cam_m[i]))*1000;
    const na=unit(a?.normal_out_cam),nb=unit(b?.normal_out_cam);if(na&&nb)angle=Math.acos(Math.max(-1,Math.min(1,dot(na,nb))))*180/Math.PI;
    const tr=document.createElement('tr');makeCell(tr,label);makeCell(tr,fmt(distance,2));makeCell(tr,fmt(angle,2));el('pairwise').append(tr);
  }
}
function line(points,color,width=2){if(points.some(p=>!finiteVector(p,2)))return;ctx.beginPath();ctx.moveTo(...points[0]);for(const p of points.slice(1))ctx.lineTo(...p);ctx.strokeStyle=color;ctx.lineWidth=width;ctx.stroke();}
function point(pixel,color,original=false,label=''){
  if(!finiteVector(pixel,2)||pixel[0]<0||pixel[0]>=canvas.width||pixel[1]<0||pixel[1]>=canvas.height)return;
  ctx.fillStyle=color;ctx.strokeStyle='#061018';ctx.lineWidth=2;ctx.beginPath();ctx.arc(...pixel,original?6:4,0,Math.PI*2);ctx.fill();ctx.stroke();
  if(original){line([[pixel[0]-11,pixel[1]],[pixel[0]+11,pixel[1]]],color,2);line([[pixel[0],pixel[1]-11],[pixel[0],pixel[1]+11]],color,2);}
  const offset=original?-16:label.startsWith('P')?24:8;
  ctx.font='bold 16px sans-serif';ctx.lineWidth=4;ctx.strokeText(label,pixel[0]+12,pixel[1]+offset);ctx.fillText(label,pixel[0]+12,pixel[1]+offset);
}
function draw(){
  el('candidates').replaceChildren();if(!entries.length)return;
  const visibleCandidates=entries.flatMap(entry=>displayCandidates(entry).map(c=>({entry,c})));
  if(!visibleCandidates.some(({entry,c})=>key(entry,c)===selectedPrediction)){const first=visibleCandidates[0];selectedPrediction=first?key(first.entry,first.c):null;}
  el('candidateCount').textContent=entries.map(e=>`${names[e.model.id]}：TopK ${shownCandidates(e).length} / NMS ${selectedCandidates(e).length}，另有同实例对照 ${displayCandidates(e).length-shownCandidates(e).length} 个`).join('；');
  const poses=[];
  for(const entry of entries){
    const top=shownCandidates(entry),shown=displayCandidates(entry),regions=entry.data.metadata?.instance_masks?.regions??[];
    for(const c of shown){
      const tr=document.createElement('tr');tr.dataset.model=entry.model.id;tr.classList.toggle('selected',key(entry,c)===selectedPrediction);
      const cell=makeCell(tr,''),button=document.createElement('button');button.className=entry.model.id;button.textContent=`${names[entry.model.id]} ${c.scene_rank!=null?'Top'+c.scene_rank:'最高'+scoreName(entry.model)}${top.includes(c)?'':' · 对照'}`;button.addEventListener('click',()=>{selectedPrediction=key(entry,c);cloud.focus(c);draw();});cell.append(button);
      const region=regions.find(r=>r.instance_id===c.instance_id);makeCell(tr,`${c.instance_id} / ${region?.tracker_object_id??'—'}`);makeCell(tr,fmt(c.task_success_score));makeCell(tr,fmt(c.q));makeCell(tr,fmt(c.q0));el('candidates').append(tr);
      if(el(visibility[entry.model.id]).checked)poses.push({...c,color:COLORS[entry.model.id],kind:'candidate',model_id:entry.model.id,label:prefixes[entry.model.id]+(c.scene_rank??'*'),pixel_valid:entry.data.source_index_kind==='pixel'});
    }
  }
  if(el('showOriginal').checked)for(const o of originals())poses.push({...o,color:COLORS.original,kind:'original',label:`O${o.original_index+1}`,pixel_valid:true});
  el('poses').textContent=JSON.stringify(visibleCandidates.map(({entry,c})=>({model:entry.model.label,rank:c.scene_rank,instance:c.instance_id,point_cam_m:c.point_cam_m,normal_out_cam:c.normal_out_cam,source_index:c.source_index,source_pixel_uv:c.source_pixel_uv})),null,2);
  renderPairwise();
  el('focusOriginal').disabled=!cloud.available||!finiteVector(activeOriginal()?.point_cam_m);el('focusCandidate').disabled=!cloud.available||!finiteVector(activeCandidate()?.point_cam_m);
  cloud.setOptions({background:el('showBackground').checked,instanceColor:el('cloudColor').value==='instance',cups:el('showCups').checked,axes:el('showAxes').checked});cloud.setPoses(poses);
  if(!currentImage)return;
  canvas.width=currentImage.naturalWidth;canvas.height=currentImage.naturalHeight;ctx.drawImage(currentImage,0,0);
  const data=entries[0].data,obs=data.observation,k=obs.intrinsics,shape=data.rendering?.image_shape??obs.image_size_hw;
  const aligned=Array.isArray(shape)&&shape[0]===canvas.height&&shape[1]===canvas.width;
  const projection=aligned&&obs.projection_convention==='pinhole'&&Array.isArray(k)&&k.length===3&&k.every(r=>finiteVector(r));
  el('projectionNote').textContent=projection?'O 品红：原规划；P 青色：预训练；F 黄色：微调。两个模型各自 Top1 同时显示。圆和轴由相机三维位姿按针孔 K 投影；中心使用原记录像素。':'RGB 尺寸或针孔信息不完整，未绘制不可靠轮廓；仅在源尺寸一致时显示记录中心。';
  for(const pose of poses){
    const geometry=poseGeometry(pose);
    if(projection&&geometry){if(el('showCups').checked)line(geometry.rim.map(p=>project(p,k)),pose.color,pose.kind==='original'?3:2);if(el('showAxes').checked)line(geometry.axis.map(p=>project(p,k)),pose.color,pose.kind==='original'?3:2);}
    if(aligned&&pose.pixel_valid)point(pose.source_pixel_uv,pose.color,pose.kind==='original',pose.label);
  }
}
el('modelIdentity').textContent='原生产规划 + 预训练 E5 step 2880 + 微调 E5 step 1000 · 同帧同时显示';
el('rankingNote').textContent='预训练按 q 排序；微调按任务评分排序。任务评分未校准为真实机器人成功概率。';
el('allCount').textContent=rows.length;
for(const day of [...new Set(rows.map(r=>r.day))].sort()){const option=document.createElement('option');option.value=day;option.textContent=day;el('day').append(option);}
for(const id of ['day','status','originalFilter'])el(id).addEventListener('change',applyFilters);
el('sample').addEventListener('change',()=>selectCase(Number(el('sample').value)));
el('previous').addEventListener('click',()=>selectCase(position-1));el('next').addEventListener('click',()=>selectCase(position+1));
for(const id of ['topk','showOriginal','showPretrained','showFinetuned','showAxes','showCups','showBackground','cloudColor'])el(id).addEventListener('change',draw);
el('originalSelect').addEventListener('change',()=>{selectedOriginal=Number(el('originalSelect').value);cloud.focus(activeOriginal());draw();});
el('focusOriginal').addEventListener('click',()=>cloud.focus(activeOriginal()));el('focusCandidate').addEventListener('click',()=>cloud.focus(activeCandidate()));
el('resetCloud').addEventListener('click',()=>cloud.reset());el('frontCloud').addEventListener('click',()=>cloud.front());
for(const id of ['resetCloud','frontCloud','showBackground','cloudColor'])el(id).disabled=!cloud.available;
document.addEventListener('keydown',event=>{if(['SELECT','INPUT','TEXTAREA','BUTTON'].includes(event.target.tagName)||event.ctrlKey||event.metaKey||event.altKey)return;if(event.key==='ArrowLeft'&&position>0){event.preventDefault();selectCase(position-1);}if(event.key==='ArrowRight'&&position<filtered.length-1){event.preventDefault();selectCase(position+1);}});
applyFilters();
