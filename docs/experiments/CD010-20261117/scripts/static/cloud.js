import {cross,dot,finiteVector,poseGeometry,rgb} from './geometry.js';

const VERTEX=`
attribute vec3 a_position;
attribute vec3 a_color;
attribute float a_instance;
uniform vec3 u_eye, u_right, u_up, u_forward;
uniform float u_aspect, u_size, u_instanceColor;
varying vec3 v_color;
varying float v_instance;
void main() {
 vec3 d=a_position-u_eye;
 float z=dot(d,u_forward);
 gl_Position=vec4(dot(d,u_right)*2.41421356/u_aspect,dot(d,u_up)*2.41421356,1.00002*z-0.00200002,z);
 gl_PointSize=u_size;
 vec3 inst=0.3+0.7*fract(sin(vec3(a_instance*12.9898,a_instance*37.719,a_instance*78.233))*43758.5453);
 v_color=mix(a_color,a_instance<0.5?vec3(0.32,0.38,0.43):inst,u_instanceColor);
 v_instance=a_instance;
}`;
const FRAGMENT=`
precision mediump float;
varying vec3 v_color;
varying float v_instance;
uniform float u_background, u_points;
void main() {
 if(u_background<0.5&&v_instance<0.5) discard;
 if(u_points>0.5&&length(gl_PointCoord-vec2(0.5))>0.5) discard;
 gl_FragColor=vec4(v_color,1.0);
}`;

export class CloudView {
  constructor(canvas,hud,onStatus) {
    this.canvas=canvas;this.hud=hud;this.onStatus=onStatus;
    this.gl=canvas.getContext('webgl',{antialias:true,alpha:false});
    this.available=!!this.gl;this.count=0;this.poses=[];this.options={background:true,instanceColor:false,cups:true,axes:true};
    this.target=[0,0,1];this.distance=1;this.yaw=0;this.pitch=0;this.pending=false;
    if(!this.gl){onStatus('此浏览器不支持 WebGL。RGB 对照和数据下载仍可使用。');return;}
    const gl=this.gl;
    const compile=(type,source)=>{const s=gl.createShader(type);gl.shaderSource(s,source);gl.compileShader(s);if(!gl.getShaderParameter(s,gl.COMPILE_STATUS))throw new Error(gl.getShaderInfoLog(s));return s;};
    try {
      this.program=gl.createProgram();gl.attachShader(this.program,compile(gl.VERTEX_SHADER,VERTEX));gl.attachShader(this.program,compile(gl.FRAGMENT_SHADER,FRAGMENT));gl.linkProgram(this.program);
      if(!gl.getProgramParameter(this.program,gl.LINK_STATUS))throw new Error(gl.getProgramInfoLog(this.program));
      this.attributes=Object.fromEntries(['position','color','instance'].map(k=>[k,gl.getAttribLocation(this.program,'a_'+k)]));
      this.uniforms=Object.fromEntries(['eye','right','up','forward','aspect','size','instanceColor','background','points'].map(k=>[k,gl.getUniformLocation(this.program,'u_'+k)]));
      this.cloudBuffer=gl.createBuffer();this.lineBuffer=gl.createBuffer();this.markerBuffer=gl.createBuffer();
    } catch(error) {this.available=false;onStatus('WebGL 初始化失败：'+error.message);return;}
    canvas.addEventListener('webglcontextlost',event=>{event.preventDefault();this.available=false;onStatus('WebGL 上下文丢失，请刷新页面恢复三维视图。');});
    canvas.addEventListener('contextmenu',event=>event.preventDefault());
    let drag=null;
    canvas.addEventListener('pointerdown',event=>{if(event.button!==0&&event.button!==2)return;drag={x:event.clientX,y:event.clientY,pan:event.shiftKey||event.button===2};canvas.setPointerCapture(event.pointerId);});
    canvas.addEventListener('pointermove',event=>{
      if(!drag)return;const dx=event.clientX-drag.x,dy=event.clientY-drag.y;drag.x=event.clientX;drag.y=event.clientY;
      if(drag.pan){const {right,up}=this.basis(),scale=2*this.distance*Math.tan(Math.PI/8)/Math.max(1,canvas.clientHeight);this.target=this.target.map((v,i)=>v-dx*scale*right[i]+dy*scale*up[i]);}
      else {this.yaw-=dx*.008;this.pitch=Math.max(-1.48,Math.min(1.48,this.pitch-dy*.008));}
      this.schedule();
    });
    for(const name of ['pointerup','pointercancel','lostpointercapture'])canvas.addEventListener(name,()=>drag=null);
    canvas.addEventListener('wheel',event=>{event.preventDefault();this.distance=Math.max(.025,Math.min(30,this.distance*Math.exp(Math.max(-1,Math.min(1,event.deltaY*.001)))));this.schedule();},{passive:false});
    new ResizeObserver(()=>this.schedule()).observe(canvas);
    this.schedule();
  }
  basis() {
    const cy=Math.cos(this.yaw),sy=Math.sin(this.yaw),cp=Math.cos(this.pitch),sp=Math.sin(this.pitch);
    const forward=[sy*cp,sp,cy*cp],right=[cy,0,-sy],up=cross(right,forward);
    return {forward,right,up,eye:this.target.map((x,i)=>x-forward[i]*this.distance)};
  }
  clear() {this.count=0;this.poses=[];this.lineCount=0;this.markerCount=0;this.schedule();}
  setCloud(values,info) {
    if(!this.available)return;
    this.count=info.count;const gl=this.gl;gl.bindBuffer(gl.ARRAY_BUFFER,this.cloudBuffer);gl.bufferData(gl.ARRAY_BUFFER,values,gl.STATIC_DRAW);
    const lo=[Infinity,Infinity,Infinity],hi=[-Infinity,-Infinity,-Infinity];
    for(let i=0;i<values.length;i+=8)for(let j=0;j<3;j++){lo[j]=Math.min(lo[j],values[i+j]);hi[j]=Math.max(hi[j],values[i+j]);}
    this.homeTarget=this.count?lo.map((v,i)=>(v+hi[i])/2):[0,0,1];
    this.homeRadius=this.count?Math.max(.05,Math.hypot(...lo.map((v,i)=>hi[i]-v))/2):.5;
    this.reset();
  }
  reset() {if(!this.available)return;this.target=[...(this.homeTarget??[0,0,1])];this.distance=(this.homeRadius??.5)/Math.sin(Math.PI/8)*1.05;this.yaw=0;this.pitch=0;this.schedule();}
  front() {this.yaw=0;this.pitch=0;this.schedule();}
  focus(pose) {if(!finiteVector(pose?.point_cam_m))return;this.target=[...pose.point_cam_m];this.distance=.20;this.schedule();}
  setOptions(options) {Object.assign(this.options,options);this.setPoses(this.poses);}
  setPoses(poses) {
    this.poses=poses;if(!this.available)return;
    const lines=[],markers=[];
    for(const pose of poses){
      const color=rgb(pose.color),geometry=poseGeometry(pose);
      if(finiteVector(pose.point_cam_m))markers.push(...pose.point_cam_m,...color,1,0);
      if(!geometry)continue;
      const segment=(a,b)=>lines.push(...a,...color,1,0,...b,...color,1,0);
      if(this.options.cups)for(let i=1;i<geometry.rim.length;i++)segment(geometry.rim[i-1],geometry.rim[i]);
      if(this.options.axes)segment(...geometry.axis);
    }
    const gl=this.gl;this.lineCount=lines.length/8;this.markerCount=markers.length/8;
    for(const [buffer,data] of [[this.lineBuffer,lines],[this.markerBuffer,markers]]){gl.bindBuffer(gl.ARRAY_BUFFER,buffer);gl.bufferData(gl.ARRAY_BUFFER,new Float32Array(data),gl.DYNAMIC_DRAW);}
    this.schedule();
  }
  schedule() {if(this.pending||!this.available)return;this.pending=true;requestAnimationFrame(()=>{this.pending=false;this.draw();});}
  bind(buffer) {
    const gl=this.gl;gl.bindBuffer(gl.ARRAY_BUFFER,buffer);
    for(const [name,size,offset] of [['position',3,0],['color',3,12],['instance',1,24]]){const a=this.attributes[name];gl.enableVertexAttribArray(a);gl.vertexAttribPointer(a,size,gl.FLOAT,false,32,offset);}
  }
  draw() {
    if(!this.available)return;
    const gl=this.gl,dpr=Math.min(2,window.devicePixelRatio||1),width=Math.max(1,Math.round(this.canvas.clientWidth*dpr)),height=Math.max(1,Math.round(this.canvas.clientHeight*dpr));
    if(this.canvas.width!==width||this.canvas.height!==height){this.canvas.width=width;this.canvas.height=height;}
    gl.viewport(0,0,width,height);gl.clearColor(.025,.045,.065,1);gl.clear(gl.COLOR_BUFFER_BIT|gl.DEPTH_BUFFER_BIT);gl.useProgram(this.program);
    const basis=this.basis();for(const name of ['eye','right','up','forward'])gl.uniform3fv(this.uniforms[name],basis[name]);
    gl.uniform1f(this.uniforms.aspect,width/height);gl.uniform1f(this.uniforms.size,2.5*dpr);gl.uniform1f(this.uniforms.points,1);gl.uniform1f(this.uniforms.background,this.options.background?1:0);gl.uniform1f(this.uniforms.instanceColor,this.options.instanceColor?1:0);
    gl.enable(gl.DEPTH_TEST);gl.depthFunc(gl.LEQUAL);
    if(this.count){this.bind(this.cloudBuffer);gl.drawArrays(gl.POINTS,0,this.count);}
    // Pose overlays intentionally remain readable through points, never implying collision clearance.
    gl.disable(gl.DEPTH_TEST);gl.uniform1f(this.uniforms.instanceColor,0);gl.uniform1f(this.uniforms.background,1);gl.uniform1f(this.uniforms.points,0);
    if(this.lineCount){this.bind(this.lineBuffer);gl.drawArrays(gl.LINES,0,this.lineCount);}
    if(this.markerCount){gl.uniform1f(this.uniforms.points,1);gl.uniform1f(this.uniforms.size,8*dpr);this.bind(this.markerBuffer);gl.drawArrays(gl.POINTS,0,this.markerCount);}
    this.drawAxes(basis);
  }
  drawAxes({right,up}) {
    const c=this.hud,ctx=c.getContext('2d');ctx.clearRect(0,0,c.width,c.height);ctx.font='12px system-ui';ctx.lineWidth=2;
    for(const [axis,label,color] of [[[1,0,0],'X','#ff8585'],[[0,1,0],'Y','#7aefae'],[[0,0,1],'Z','#83baff']]) {
      const x=50+32*dot(axis,right),y=55-32*dot(axis,up);
      ctx.strokeStyle=color;ctx.fillStyle=color;ctx.beginPath();ctx.moveTo(50,55);ctx.lineTo(x,y);ctx.stroke();ctx.fillText(label,x+4,y-4);
    }
    ctx.fillStyle='#b7c8d5';ctx.fillText('相机 XYZ',15,100);
  }
}
