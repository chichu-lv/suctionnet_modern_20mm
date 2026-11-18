// Geometry is camera XYZ in metres, shared by RGB projection and WebGL overlays.
export const CUP_RADIUS = 0.02;
export const AXIS_LENGTH = 0.03;
export const COLORS = {original:'#ff65d7', pretrained:'#45dfff', finetuned:'#ffdc28', top:'#ffdc28', candidate:'#32ebb4'};
export const finiteVector = (v, n=3) => Array.isArray(v) && v.length===n && v.every(Number.isFinite);
export const cross = (a,b) => [a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0]];
export const dot = (a,b) => a.reduce((s,x,i)=>s+x*b[i],0);
export function unit(v) {
  if (!finiteVector(v)) return null;
  const n=Math.hypot(...v);
  return n>1e-8 ? v.map(x=>x/n) : null;
}
export function project(p,k) {
  if (!finiteVector(p)||p[2]<=1e-6||!Array.isArray(k)||k.length!==3||!k.every(r=>finiteVector(r))) return null;
  const z=dot(k[2],p);
  return z>1e-6 ? [dot(k[0],p)/z,dot(k[1],p)/z] : null;
}
export function poseGeometry(pose) {
  const p=pose.point_cam_m,n=unit(pose.normal_out_cam);
  if (!finiteVector(p)||!n) return null;
  let smallest=0;
  for(let i=1;i<3;i++) if(Math.abs(n[i])<Math.abs(n[smallest])) smallest=i;
  const ref=[0,0,0];ref[smallest]=1;
  const t=unit(cross(n,ref)),b=cross(n,t),rim=[];
  for(let i=0;i<=64;i++) {
    const a=i*2*Math.PI/64;
    rim.push(p.map((x,j)=>x+CUP_RADIUS*(Math.cos(a)*t[j]+Math.sin(a)*b[j])));
  }
  return {center:p,rim,axis:[p,p.map((x,j)=>x+AXIS_LENGTH*n[j])]};
}
export const rgb = hex => [1,3,5].map(i=>parseInt(hex.slice(i,i+2),16)/255);

// The exporter owns pairing and comparison calculations; the UI never guesses a match.
export function decodeComparison(data, sample) {
  if(data.schema_version!=='suction20.target-comparison.v1'||data.sample_id!==sample||!Array.isArray(data.originals)||!Array.isArray(data.comparisons))
    throw new Error('原始对照格式或样本身份不匹配');
  return data;
}
export function decodeCloud(buffer, info) {
  if(info.dtype!=='float32'||info.endian!=='little'||info.stride!==8||info.stride_bytes!==32||!Number.isInteger(info.count)||info.count<0||buffer.byteLength!==info.count*32)
    throw new Error('点云格式或长度不匹配');
  const values=new Float32Array(info.count*8),view=new DataView(buffer);
  for(let i=0;i<values.length;i++) values[i]=view.getFloat32(i*4,true);
  for(let i=0;i<values.length;i++) if(!Number.isFinite(values[i])) throw new Error('点云含非有限数值');
  return values;
}
