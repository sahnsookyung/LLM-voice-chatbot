import {BufferGeometry, CatmullRomCurve3, DoubleSide, Float32BufferAttribute, Matrix3, Matrix4, Mesh, MeshBasicMaterial, Quaternion, Raycaster, Vector3} from 'three';
import {parseWardrobe, type WardrobeSelection} from './wardrobe.js';

// This adapter owns the v9 export format. Face, body, facial morphs and humanoid
// nodes are copied untouched; new parts use the existing skeleton's bind space.
type JSONNode = Record<string, any>;
type Point = [number, number, number];
type Influence = {joints:number[]; weights:number[]};
type Surface = {positions:number[][]; uv:number[][]; influences:Influence[]; indices:number[]};
const smooth = (v:number) => {const t=Math.max(0,Math.min(1,v));return t*t*(3-2*t);};
const vector = (p:number[]) => new Vector3(p[0],p[1],p[2]);
const surface = ():Surface => ({positions:[],uv:[],influences:[],indices:[]});

class Model {
  g:JSONNode;
  views:Uint8Array[];
  worlds:Matrix4[]=[];
  constructor(buffer:ArrayBuffer) {
    const v=new DataView(buffer), n=v.getUint32(12,true);
    if(v.getUint32(0,true)!==0x46546c67 || v.getUint32(4,true)!==2)throw new Error('Invalid Ari model.');
    this.g=JSON.parse(new TextDecoder().decode(new Uint8Array(buffer,20,n)));
    this.views=this.g.bufferViews.map((b:JSONNode)=>new Uint8Array(buffer,28+n+(b.byteOffset??0),b.byteLength).slice());
    if(this.g.meshes[2]?.name!=='Ari fine tapered silk locks'||this.g.nodes.length!==127||this.g.materials.length!==16)
      throw new Error('Wardrobe parts require the finished Ari v9 base.');
    this.updateWorlds();
  }
  updateWorlds() {
    const g=this.g;
    const visit=(i:number,parent:Matrix4)=>{
      const n=g.nodes[i],m=n.matrix?new Matrix4().fromArray(n.matrix):new Matrix4().compose(vector(n.translation??[0,0,0]),new Quaternion(...(n.rotation??[0,0,0,1]) as [number,number,number,number]),vector(n.scale??[1,1,1]));
      this.worlds[i]=parent.clone().multiply(m);for(const c of n.children??[])visit(c,this.worlds[i]);
    };
    for(const i of g.scenes[g.scene??0].nodes)visit(i,new Matrix4());
  }
  accessor(index:number) {
    const a=this.g.accessors[index],v=this.g.bufferViews[a.bufferView],data=new DataView(this.views[a.bufferView].buffer);
    const width=({SCALAR:1,VEC2:2,VEC3:3,VEC4:4,MAT4:16} as Record<string,number>)[a.type];
    const bytes=a.componentType===5123?2:4,stride=v.byteStride??width*bytes;
    if(!width||a.sparse||![5123,5125,5126].includes(a.componentType))throw new Error('Unsupported wardrobe accessor.');
    const offset=(i:number,k:number)=>(a.byteOffset??0)+i*stride+k*bytes;
    return {a,get:(i:number)=>Array.from({length:width},(_,k)=>a.componentType===5126?data.getFloat32(offset(i,k),true):a.componentType===5123?data.getUint16(offset(i,k),true):data.getUint32(offset(i,k),true)),
      set:(i:number,p:number[])=>p.forEach((x,k)=>{if(!Number.isFinite(x))throw new Error('Non-finite wardrobe geometry.');if(a.componentType!==5126)throw new Error('Cannot edit integer attribute.');data.setFloat32(offset(i,k),x,true);})};
  }
  values(index:number) {const a=this.accessor(index);return Array.from({length:a.a.count},(_,i)=>a.get(i));}
  append(rows:number[][],type:string,componentType=5126) {
    const bytes=componentType===5123?2:4,width=rows[0]?.length??1,b=new Uint8Array(rows.length*width*bytes),d=new DataView(b.buffer);
    rows.forEach((p,i)=>p.forEach((x,k)=>{if(!Number.isFinite(x))throw new Error('Invalid wardrobe attribute.');const o=(i*width+k)*bytes;if(componentType===5126)d.setFloat32(o,x,true);else if(componentType===5123)d.setUint16(o,x,true);else d.setUint32(o,x,true);}));
    const view=this.g.bufferViews.push({buffer:0,byteLength:b.length})-1;this.views.push(b);
    const a:JSONNode={bufferView:view,componentType,count:rows.length,type};
    if(type==='VEC3'){a.min=[0,1,2].map(k=>rows.reduce((v,p)=>Math.min(v,p[k]),Infinity));a.max=[0,1,2].map(k=>rows.reduce((v,p)=>Math.max(v,p[k]),-Infinity));}
    return this.g.accessors.push(a)-1;
  }
  warp(primitive:JSONNode,fn:(p:number[])=>number[]) {
    const p=this.accessor(primitive.attributes.POSITION),n=this.accessor(primitive.attributes.NORMAL);
    for(const i of new Set(this.values(primitive.indices).flat())) {
      const point=p.get(i),normal=n.get(i),next=fn(point);
      const columns=[0,1,2].map(axis=>{const a=point.slice(),b=point.slice();a[axis]-=1e-5;b[axis]+=1e-5;const lo=fn(a),hi=fn(b);return hi.map((v,k)=>(v-lo[k])/2e-5);});
      const matrix=new Matrix3().set(...[0,1,2].flatMap(k=>columns.map(c=>c[k])) as [number,number,number,number,number,number,number,number,number]);
      if(Math.abs(matrix.determinant())<1e-5)throw new Error('Degenerate wardrobe deformation.');
      p.set(i,next);n.set(i,vector(normal).applyMatrix3(matrix.invert().transpose()).normalize().toArray());
    }
    const points=this.values(primitive.attributes.POSITION);
    p.a.min=[0,1,2].map(k=>points.reduce((a,v)=>Math.min(a,v[k]),Infinity));p.a.max=[0,1,2].map(k=>points.reduce((a,v)=>Math.max(a,v[k]),-Infinity));
  }
  geometry(s:Surface) {
    const geometry=new BufferGeometry();geometry.setAttribute('position',new Float32BufferAttribute(s.positions.flat(),3));geometry.setIndex(s.indices);geometry.computeVertexNormals();
    const normals=Array.from({length:s.positions.length},(_,i)=>[geometry.attributes.normal.getX(i),geometry.attributes.normal.getY(i),geometry.attributes.normal.getZ(i)]);geometry.dispose();
    return {attributes:{POSITION:this.append(s.positions,'VEC3'),NORMAL:this.append(normals,'VEC3'),TEXCOORD_0:this.append(s.uv,'VEC2'),JOINTS_0:this.append(s.influences.map(i=>i.joints),'VEC4',5123),WEIGHTS_0:this.append(s.influences.map(i=>i.weights),'VEC4')},indices:this.append(s.indices.map(i=>[i]),'SCALAR',s.positions.length>65535?5125:5123),mode:4};
  }
  addMesh(name:string,s:Surface,material:number,skin:number) {
    const mesh=this.g.meshes.push({name,primitives:[{...this.geometry(s),material}]})-1;
    const node=this.g.nodes.push({name,mesh,skin})-1;this.g.scenes[this.g.scene??0].nodes.push(node);
  }
  addMaterial(name:string,color:number[],detail?:string) {
    return this.g.materials.push({name,pbrMetallicRoughness:{baseColorFactor:[...color,1],metallicFactor:0,roughnessFactor:.9},doubleSided:true,...(detail?{extras:{ariSurfaceDetail:detail}}:{}),extensions:{VRMC_materials_mtoon:{specVersion:'1.0',shadeColorFactor:color.map(v=>v*.72),shadingToonyFactor:.8,shadingShiftFactor:0,outlineWidthMode:'none'}}})-1;
  }
  addFabric(name:string,color:number[],kind='cotton') {
    return this.g.materials.push({name,pbrMetallicRoughness:{baseColorFactor:[...color,1],metallicFactor:0,roughnessFactor:.88},doubleSided:true,extras:{ariFabric:kind}})-1;
  }
  addChain(name:string,points:number[][],parent:number,skinIndex:number,stiffness:number,drag:number,colliderGroups:number[]) {
    const skin=this.g.skins[skinIndex],matrices=this.values(skin.inverseBindMatrices),nodes:number[]=[],indices:number[]=[];
    let parentWorld=this.worlds[parent];
    for(const point of points){
      const world=new Matrix4().makeTranslation(...point as Point),local=parentWorld.clone().invert().multiply(world),p=new Vector3(),q=new Quaternion(),s=new Vector3();local.decompose(p,q,s);
      const node=this.g.nodes.push({name:`${name}_${nodes.length}`,translation:p.toArray(),rotation:q.toArray()})-1;
      (this.g.nodes[parent].children??=[]).push(node);parent=node;parentWorld=world;this.worlds[node]=world;
      nodes.push(node);indices.push(skin.joints.length);skin.joints.push(node);matrices.push(world.clone().invert().toArray());
    }
    skin.inverseBindMatrices=this.append(matrices,'MAT4');
    this.g.extensions.VRMC_springBone.springs.push({name,joints:nodes.map((node,i)=>i===nodes.length-1?{node}:{node,hitRadius:.008,stiffness:stiffness*(1-i/nodes.length*.3),dragForce:drag,gravityPower:0,gravityDir:[0,-1,0]}),colliderGroups});
    return indices;
  }
  influence(primitive:JSONNode) {
    const p=this.accessor(primitive.attributes.POSITION),j=this.accessor(primitive.attributes.JOINTS_0),w=this.accessor(primitive.attributes.WEIGHTS_0),ids=[...new Set(this.values(primitive.indices).flat())];
    return (point:number[]):Influence=>{let best=ids[0],distance=Infinity;for(const i of ids){const v=p.get(i),d=v.reduce((sum,x,k)=>sum+(x-point[k])**2,0);if(d<distance){distance=d;best=i;}}return {joints:j.get(best),weights:w.get(best)};};
  }
  finish() {
    let size=0;this.views.forEach((v,i)=>{size=(size+3)&~3;this.g.bufferViews[i].byteOffset=size;this.g.bufferViews[i].byteLength=v.length;size+=v.length;});size=(size+3)&~3;this.g.buffers=[{byteLength:size}];
    const text=new TextEncoder().encode(JSON.stringify(this.g)),jsonSize=(text.length+3)&~3,buffer=new ArrayBuffer(28+jsonSize+size),v=new DataView(buffer),bytes=new Uint8Array(buffer);
    v.setUint32(0,0x46546c67,true);v.setUint32(4,2,true);v.setUint32(8,buffer.byteLength,true);v.setUint32(12,jsonSize,true);v.setUint32(16,0x4e4f534a,true);bytes.fill(32,20,20+jsonSize);bytes.set(text,20);v.setUint32(20+jsonSize,size,true);v.setUint32(24+jsonSize,0x004e4942,true);
    this.views.forEach((b,i)=>bytes.set(b,28+jsonSize+this.g.bufferViews[i].byteOffset));return buffer;
  }
}

function tube(s:Surface,points:number[][],radius:(t:number)=>[number,number],influence:(p:number[],t:number)=>Influence,rows=40,sides=6,textureOffset=0) {
  const curve=new CatmullRomCurve3(points.map(vector)),base=s.positions.length;
  for(let row=0;row<=rows;row++){
    const t=row/rows,c=curve.getPoint(t),tangent=curve.getTangent(Math.max(.00001,Math.min(.99999,t))),axis=new Vector3(1,0,0).addScaledVector(tangent,-tangent.x).normalize(),normal=tangent.clone().cross(axis).normalize(),[rx,rz]=radius(t);
    for(let side=0;side<=sides;side++){
      const a=side/sides*Math.PI*2,p=c.clone().addScaledVector(axis,Math.cos(a)*rx).addScaledVector(normal,Math.sin(a)*rz).toArray();
      s.positions.push(p);s.uv.push([textureOffset+side/sides*.08,t]);s.influences.push(influence(p,t));
    }
  }
  for(let row=0;row<rows;row++)for(let side=0;side<sides;side++){const a=base+row*(sides+1)+side,b=a+sides+1;s.indices.push(a,a+1,b,a+1,b+1,b);}
}
function chainInfluence(joints:number[],t:number):Influence {
  const value=Math.min(.999999,Math.max(0,t))*(joints.length-1),a=Math.floor(value),f=value-a;
  return {joints:[joints[a],joints[a+1],0,0],weights:[1-f,f,0,0]};
}

// Sampled fabric panels, including their turned edge. Unlike a flat quad these
// keep a curved profile in side views and carry the same weights as the garment.
function clothPatch(s:Surface,point:(u:number,v:number)=>number[],bind:(p:number[])=>Influence,rows=16,cols=16,thickness=.0018) {
  const base=s.positions.length;
  for(let row=0;row<=rows;row++)for(let col=0;col<=cols;col++){
    const u=col/cols,v=row/rows,p=point(u,v);s.positions.push(p);s.uv.push([u,v]);s.influences.push(bind(p));
  }
  for(let row=0;row<rows;row++)for(let col=0;col<cols;col++){
    const a=base+row*(cols+1)+col,b=a+cols+1;s.indices.push(a,a+1,b,a+1,b+1,b);
  }
  const edge=[...Array.from({length:cols},(_,i)=>i),...Array.from({length:rows},(_,i)=>i*(cols+1)+cols),...Array.from({length:cols},(_,i)=>rows*(cols+1)+cols-i),...Array.from({length:rows},(_,i)=>(rows-i)*(cols+1))];
  const back=s.positions.length;
  for(const i of edge){const p=s.positions[base+i].slice();p[2]-=thickness;s.positions.push(p);s.uv.push(s.uv[base+i]);s.influences.push(bind(p));}
  for(let i=0;i<edge.length;i++){const j=(i+1)%edge.length;s.indices.push(base+edge[i],back+i,base+edge[j],base+edge[j],back+i,back+j);}
}

/** Bow, fastening band and tails share the hair's tie transform. */
function ribbon(s:Surface,curve:CatmullRomCurve3,joints:number[],tie:number,size=1,back=false) {
  const center=curve.getPoint(tie),down=curve.getTangent(tie),across=new Vector3(1,0,0).addScaledVector(down,-down.x).normalize(),front=down.clone().cross(across).normalize();
  if(back)front.negate();
  const binding=chainInfluence(joints,tie);
  const point=([x,y,z]:number[])=>center.clone().addScaledVector(across,x*size).addScaledVector(down,y*size).addScaledVector(front,z*size).toArray();
  const strip=(points:number[][],width:(t:number)=>number)=>{
    const path=new CatmullRomCurve3(points.map(vector)),base=s.positions.length,rows=28;
    for(let row=0;row<=rows;row++){
      const t=row/rows,c=path.getPoint(t),v=path.getTangent(Math.max(.0001,Math.min(.9999,t))),side=new Vector3(-v.y,v.x,0).normalize();
      for(let col=0;col<3;col++){
        const p=c.clone().addScaledVector(side,(col-1)*width(t));p.z+=(col===1?.002:0)*Math.sin(Math.PI*t);
        s.positions.push(point(p.toArray()));s.uv.push([t,col/2]);s.influences.push(binding);
      }
    }
    for(let row=0;row<rows;row++)for(let col=0;col<2;col++){const a=base+row*3+col,b=a+3;s.indices.push(a,a+1,b,a+1,b+1,b);}
  };
  for(const sign of [-1,1]){
    strip([[0,0,.015],[sign*.028,-.018,.018],[sign*.047,-.012,.020],[sign*.035,.005,.020],[0,0,.016]],t=>.008+.003*Math.sin(Math.PI*t));
    strip([[sign*.003,.005,.014],[sign*.018,.027,.013],[sign*.032,.045,.022],[sign*.036,.052,.029]],t=>.009*(1-.25*t));
  }
  // Closed band gathers the braid before the unbraided tuft begins.
  const bandPoints=Array.from({length:25},(_,i)=>point([Math.sin(i/24*Math.PI*2)*.0105,0,Math.cos(i/24*Math.PI*2)*.0105]));
  tube(s,bandPoints,()=>[.0045*size,.003*size],()=>binding,32,6);
  tube(s,[[0,-.007,.017],[0,0,.021],[0,.007,.017]].map(point),()=>[.007*size,.004*size],()=>binding,8,8);
}

function braidedTail(m:Model,hair:Surface,bows:Surface,guide:number[][],name:string,head:number) {
  const curve=new CatmullRomCurve3(guide.map(vector)),jointPoints=Array.from({length:9},(_,i)=>curve.getPoint(i/8).toArray());
  const joints=m.addChain(name,jointPoints,head,2,.5,.30,[2,3,10]),tie=.79;
  const spring=m.g.extensions.VRMC_springBone.springs.at(-1);
  spring.joints.forEach((joint:JSONNode,i:number)=>{
    if(i===jointPoints.length-1)return;
    // Firm gathered roots, then a gravity-dominated hanging length. Giving the
    // entire braid high stiffness and zero gravity made it hover beside Ari.
    joint.stiffness=i===0?1.1:i===1?.85:.5-(i-2)*.05;
    joint.gravityPower=i===0?.1:i===1?.35:.65;
  });
  const frame=(t:number)=>{const c=curve.getPoint(t),down=curve.getTangent(Math.max(.0001,Math.min(.9999,t))),across=new Vector3(1,0,0).addScaledVector(down,-down.x).normalize();return {c,across,front:down.clone().cross(across).normalize()};};
  // Three broad interwoven sections with fine parallel fibers. All gather into
  // the same narrow band; a separate loose tuft follows, rather than a cut end.
  tube(hair,Array.from({length:65},(_,i)=>curve.getPoint(i/64*tie).toArray()),t=>{
    const gather=1-.55*smooth((t-.85)/.15);return [.012*gather,.007*gather];
  },(_,t)=>chainInfluence(joints,t*tie),64,10,.17);
  for(let bundle=0;bundle<3;bundle++)for(let fiber=0;fiber<12;fiber++){
    const points=Array.from({length:65},(_,i)=>{
      const t=i/64*tie,{c,across,front}=frame(t),braided=smooth(t/.16),gather=1-.72*smooth((t-.70)/.09);
      const phase=bundle*Math.PI*2/3+Math.max(0,t-.10)/(tie-.10)*Math.PI*10;
      const x=Math.sin(phase)*.016*braided*gather+(fiber/11-.5)*.023*gather;
      const z=Math.cos(phase)*.009*braided*gather;
      return c.addScaledVector(across,x).addScaledVector(front,z).toArray();
    });
    tube(hair,points,t=>[.0025*(1-.4*smooth((t-.88)/.12)),.0019],(_,t)=>chainInfluence(joints,t*tie),64,6,(bundle*12+fiber)*.173);
  }
  for(let strand=0;strand<36;strand++){
    const end=.88+.12*(.5+.5*Math.sin(strand*2.3)),angle=strand/36*Math.PI*2;
    const points=Array.from({length:33},(_,i)=>{
      const u=i/32,t=tie+(1-tie)*u*end,{c,across,front}=frame(t),fan=(.006+.015*Math.sin(Math.PI*u*.95))*(1-.9*smooth((u-.65)/.35));
      return c.addScaledVector(across,Math.cos(angle)*fan+.008*Math.sin(u*Math.PI)*Math.sin(strand*1.7)).addScaledVector(front,Math.sin(angle)*fan*.55).toArray();
    });
    tube(hair,points,t=>[Math.max(.00004,.0027*(1-smooth((t-.65)/.35))),Math.max(.00004,.0017*(1-smooth((t-.65)/.35)))],(_,t)=>chainInfluence(joints,tie+(1-tie)*t*end),32,6,strand*.193);
  }
  ribbon(bows,curve,joints,tie);
}

function hair(m:Model,selection:WardrobeSelection) {
  if(selection.hair==='original')return;
  const g=m.g,p=g.meshes[2].primitives[0],springs=g.extensions.VRMC_springBone.springs;
  const head=g.extensions.VRMC_vrm.humanoid.humanBones.head.node,headJoint=g.skins[2].joints.indexOf(head);
  const fixed=()=>({joints:[headJoint,0,0,0],weights:[1,0,0,0]});
  if(['soft-waves','half-up','double-buns'].includes(selection.hair)){
    m.warp(p,([x,y,z])=>{
      const flow=smooth((1.43-y)/.20);
      return [x+Math.sign(x)*Math.sin((1.43-y)*19+z*5)*.020*flow,y,z+Math.sin((1.43-y)*17+Math.abs(x)*7)*.014*flow];
    });
    if(selection.hair==='soft-waves')return;
    const out:Surface={positions:m.values(p.attributes.POSITION),uv:m.values(p.attributes.TEXCOORD_0),indices:m.values(p.indices).flat(),influences:m.values(p.attributes.JOINTS_0).map((joints,i)=>({joints,weights:[]}))};
    const weights=m.values(p.attributes.WEIGHTS_0);out.influences.forEach((v,i)=>v.weights=weights[i]);
    if(selection.hair==='double-buns'){
      for(const sign of [-1,1]){
        const center=[sign*.090,1.568,-.025],base=out.positions.length,rows=18,sides=36;
        for(let row=0;row<=rows;row++)for(let col=0;col<=sides;col++){
          const phi=.001+(Math.PI-.002)*row/rows,a=col/sides*Math.PI*2;
          out.positions.push([center[0]+Math.sin(phi)*Math.cos(a)*.037,center[1]+Math.cos(phi)*.042,center[2]+Math.sin(phi)*Math.sin(a)*.037]);out.uv.push([col/sides,row/rows]);out.influences.push(fixed());
        }
        for(let row=0;row<rows;row++)for(let col=0;col<sides;col++){const a=base+row*(sides+1)+col,b=a+sides+1;out.indices.push(a,b,a+1,a+1,b,b+1);}
        for(let strand=0;strand<20;strand++){
          const points=Array.from({length:49},(_,i)=>{const t=i/48,phi=-1.4+t*2.8,a=strand/20*Math.PI*2+t*Math.PI*4;return [center[0]+Math.cos(phi)*Math.cos(a)*.038,center[1]+Math.sin(phi)*.043,center[2]+Math.cos(phi)*Math.sin(a)*.038];});
          tube(out,points,()=>[.0017,.001],fixed,48,5,strand*.13);
        }
      }
    }else{
      const curve=new CatmullRomCurve3([[0,1.535,-.13],[0,1.46,-.15],[.012,1.35,-.17],[.025,1.25,-.17]].map(vector)),points=Array.from({length:5},(_,i)=>curve.getPoint(i/4).toArray());
      const chain=m.addChain('HairHalfUp',points,head,2,.60,.25,[2,3,10]),bows=surface();
      for(let i=0;i<24;i++){
        const points=Array.from({length:33},(_,k)=>{const t=k/32,p=curve.getPoint(t),a=i/24*Math.PI*2,r=.014*Math.sin(Math.PI*t);return p.add(new Vector3(Math.cos(a)*r,0,Math.sin(a)*r)).toArray();});
        tube(out,points,t=>[.0024*(1-smooth((t-.8)/.2))+.00004,.001],(_,t)=>chainInfluence(chain,t),32,6,i*.13);
      }
      ribbon(bows,curve,chain,0,1.7,true);m.addMesh('Ari half-up ribbon',bows,m.addMaterial('Ari black ribbon',[.028,.031,.033]),2);
    }
    g.meshes[2]={name:`Ari ${selection.hair}`,primitives:[{...m.geometry(out),material:p.material}]};return;
  }
  // Keep the authored fringe. Replace the long locks with a dedicated tied rig.
  const old=m.values(p.attributes.POSITION),uv=m.values(p.attributes.TEXCOORD_0),js=m.values(p.attributes.JOINTS_0),ws=m.values(p.attributes.WEIGHTS_0),indices=m.values(p.indices).flat();
  const adj=new Map<number,Set<number>>();for(let i=0;i<indices.length;i+=3){const a=indices.slice(i,i+3);for(const n of a){if(!adj.has(n))adj.set(n,new Set());for(const k of a)adj.get(n)!.add(k);}}
  const seen=new Set<number>(),keep=new Set<number>();for(const start of adj.keys()){if(seen.has(start))continue;const queue=[start],group:number[]=[];seen.add(start);while(queue.length){const i=queue.pop()!;group.push(i);for(const n of adj.get(i)!)if(!seen.has(n)){seen.add(n);queue.push(n);}}if(group.every(i=>old[i][1]>1.38))for(const i of group)keep.add(i);}
  const out=surface(),map=new Map<number,number>();for(const i of keep){map.set(i,out.positions.length);out.positions.push(old[i]);out.uv.push(uv[i]);out.influences.push({joints:js[i],weights:ws[i]});}
  for(let i=0;i<indices.length;i+=3)if(keep.has(indices[i]))out.indices.push(...indices.slice(i,i+3).map(v=>map.get(v)!));
  g.extensions.VRMC_springBone.springs=springs.filter((s:JSONNode)=>s.name!=='Hair'||s.joints[0].node<33);
  if(selection.hair==='bob'||selection.hair==='low-pigtails'){
    // A bob needs its own silhouette and short guides. Compressing the long
    // locks crowded them against the cheek collider and produced round clumps.
    const chains:number[][]=[];
    const point=(a:number,y:number,rx:number,rz:number)=>[Math.sin(a)*rx,y,Math.cos(a)*rz-.019];
    for(let i=0;i<10;i++){
      const a=.78+i/9*(Math.PI*2-1.56);
      chains.push(m.addChain(`HairBob${i}`,[point(a,1.535,.089,.096),point(a,1.455,.109,.109),point(a,1.36,.095,.083)],head,2,.85,.30,[2,3]));
    }
    for(let i=0;i<100;i++){
      const a=.70+i/99*(Math.PI*2-1.40),end=1.355+.016*Math.sin(i*2.4)**2;
      const points=[point(a,1.588,.037,.039),point(a,1.556,.079,.087),point(a,1.488,.105,.113),point(a,1.41,.107,.105),point(a,end,.091,.081)];
      const j=Math.min(9,Math.max(0,Math.round((a-.78)/(Math.PI*2-1.56)*9)));
      tube(out,points,t=>[Math.max(.00005,.0041*(1-smooth((t-.79)/.21))),.0012],(_,t)=>t<.27?fixed():chainInfluence(chains[j],(t-.27)/.73),40,6,i*.193);
    }
    if(selection.hair==='low-pigtails'){
      const bows=surface();
      for(const sign of [-1,1]){
        const curve=new CatmullRomCurve3([[sign*.101,1.438,-.061],[sign*.117,1.393,-.064],[sign*.127,1.347,-.058],[sign*.132,1.306,-.044]].map(vector));
        const chain=m.addChain(`HairLowPigtail${sign}`,Array.from({length:4},(_,i)=>curve.getPoint(i/3).toArray()),head,2,.60,.27,[2,3]);
        for(const joint of g.extensions.VRMC_springBone.springs.at(-1).joints)joint.gravityPower=.30;
        for(let i=0;i<26;i++){
          const points=Array.from({length:33},(_,k)=>{const t=k/32,p=curve.getPoint(t),a=i/26*Math.PI*2,r=.016*Math.sin(Math.PI*t*.95);return p.add(new Vector3(Math.cos(a)*r,0,Math.sin(a)*r*.7)).toArray();});
          tube(out,points,t=>[.0027*(1-smooth((t-.70)/.30))+.00004,.0012],(_,t)=>chainInfluence(chain,t),32,6,i*.13);
        }
        ribbon(bows,curve,chain,.03,.46);
      }
      m.addMesh('Ari short pigtail bows',bows,m.addMaterial('Ari black ties',[.028,.031,.033]),2);
    }
    g.meshes[2]={name:`Ari ${selection.hair}`,primitives:[{...m.geometry(out),material:p.material}]};return;
  }
  if(selection.hair==='braid'||selection.hair==='twintails'){
    const twin=selection.hair==='twintails',bows=surface(),sides=twin?[-1,1]:[1];
    for(const side of sides){
      const guide=twin?[[side*.088,1.44,-.035],[side*.105,1.35,.015],[side*.126,1.285,.095],[side*.143,1.22,.165],[side*.145,1.145,.193],[side*.143,1.04,.195],[side*.14,.895,.192]]:[[.088,1.44,-.042],[.114,1.35,.015],[.135,1.285,.098],[.15,1.22,.17],[.152,1.145,.199],[.148,1.065,.201],[.145,.99,.199]];
      const total=twin?60:110;
      for(let i=0;i<total;i++){
        const a=twin?(i/(total-1)*Math.PI)*side:i/(total-1)*Math.PI*2;
        const root=[Math.sin(a)*.029,1.592,Math.cos(a)*.030-.019],mid=[Math.sin(a)*.089,1.544+Math.abs(Math.cos(a))*.015,Math.cos(a)*.099-.019];
        tube(out,[root,mid,[(mid[0]+guide[0][0])*.5,1.508,(mid[2]+guide[0][2])*.5],guide[0]],t=>[.0032*(1-.65*t),.001],fixed,30,6,i*.173);
      }
      braidedTail(m,out,bows,guide,twin?`HairTwin${side<0?'Right':'Left'}`:'HairBraid',head);
    }
    g.meshes[2]={name:`Ari ${selection.hair} with tapered loose ends`,primitives:[{...m.geometry(out),material:p.material}]};
    m.addMesh('Ari hair ribbons',bows,m.addMaterial('Ari sage ribbon',[.36,.58,.52],'gingham'),2);
    return;
  }
  const guide=[[0,1.565,-.105],[.035,1.565,-.16],[.085,1.49,-.19],[.14,1.37,-.205],[.18,1.23,-.20],[.17,1.08,-.18]];
  const guideCurve=new CatmullRomCurve3(guide.map(vector)),points=Array.from({length:8},(_,i)=>guideCurve.getPoint(i/7).toArray());
  const chain=m.addChain('HairTie',points,head,2,.42,.19,[2,3,10]);
  // Fine swept crown sections meet at the actual attachment instead of a bald cap.
  for(let i=0;i<38;i++){
    const a=i/37*Math.PI*1.8+.1,root=[Math.sin(a)*.08,1.565+Math.cos(a)*.023,Math.cos(a)*.065-.026];
    tube(out,[root,[root[0]*.8,1.585,-.062],[root[0]*.35,1.566,-.10],guide[0]],t=>[.0035*(1-.5*t),.0008],fixed,22,5,i*.137);
  }
  const strands=54;
  for(let i=0;i<strands;i++){
    const strandPoints=Array.from({length:65},(_,k)=>{
      const t=k/64,c=guideCurve.getPoint(t),tangent=guideCurve.getTangent(Math.max(.00001,Math.min(.99999,t))),axis=new Vector3(1,0,0).addScaledVector(tangent,-tangent.x).normalize(),normal=tangent.clone().cross(axis).normalize();
      const angle=i/strands*Math.PI*2,r=.032*Math.sin(Math.PI*t*.85)*(.8+.15*Math.sin(i*3));
      return c.addScaledVector(axis,Math.cos(angle)*r).addScaledVector(normal,Math.sin(angle)*r*.65).toArray();
    });
    tube(out,strandPoints,t=>[Math.max(.00006,.0031*Math.pow(1-smooth((t-.78)/.22),.6)),.001],(_,t)=>chainInfluence(chain,t),64,6,i*.173);
  }
  g.meshes[2]={name:`Ari ${selection.hair} fine locks`,primitives:[{...m.geometry(out),material:p.material}]};
}

function outfit(m:Model,selection:WardrobeSelection) {
  if(selection.outfit==='original')return;
  if(!['tailored','streetwear','linen'].includes(selection.outfit)){screenshotOutfit(m,selection.outfit);return;}
  const g=m.g,body=g.meshes[1],find=(word:string)=>body.primitives.find((p:JSONNode)=>g.materials[p.material].name.includes(word));
  const coat=find('Tops_01'),inner=find('Onepiece_00'),pants=find('Bottoms_01'),shoes=find('Shoes_01');
  if(!coat||!inner||!pants||!shoes)throw new Error('Ari garment binding is missing.');
  const colors=selection.outfit==='tailored'?[[.43,.36,.30],[.88,.84,.76],[.065,.071,.075],[.035,.027,.023]]:selection.outfit==='streetwear'?[[.23,.29,.20],[.84,.82,.74],[.052,.066,.055],[.73,.73,.68]]:[[.68,.62,.50],[.27,.34,.23],[.27,.34,.23],[.09,.047,.027]];
  for(const [i,p]of [coat,inner,pants,shoes].entries())g.materials[p.material].extras={...g.materials[p.material].extras,ariGarmentColor:colors[i]};
  m.warp(g.meshes[3].primitives[0],([x,y,z])=>[x,1.309+(y-1.323)*.18,z]);
  if(selection.outfit!=='linen')m.warp(pants,([x,y,z])=>{
    const loose=smooth((.80-y)/.55)*(selection.outfit==='tailored'?.42:.27),center=Math.sign(x)*.087;
    return [center+(x-center)*(1+loose),y,z*(1+loose*.45)];
  });
  // Each garment retains the original cutout alpha and rig weights.
  const bomber=selection.outfit==='streetwear';
  m.warp(coat,([x,y,z])=>{
    const lower=smooth((1.12-y)/.35),arm=smooth((Math.abs(x)-.16)/.13);
    return [x*(1+(bomber?.06:.025)*(1-arm)),y+(bomber?.16:selection.outfit==='tailored'?.05:.025)*lower+(y-1.275)*(bomber?.18:.04)*arm,z*(bomber?1.13:1.04)];
  });
  const influence=m.influence(coat),detail=surface();
  const coatGeometry=new BufferGeometry();coatGeometry.setAttribute('position',new Float32BufferAttribute(m.values(coat.attributes.POSITION).flat(),3));coatGeometry.setIndex(m.values(coat.indices).flat());
  const coatMaterial=new MeshBasicMaterial({side:DoubleSide}),coatSurface=new Mesh(coatGeometry,coatMaterial),ray=new Raycaster();coatSurface.updateMatrixWorld();
  const panel=(points:number[][],binding=influence,project=true)=>{const base=detail.positions.length;points.forEach((p,i)=>{
    if(project){ray.set(new Vector3(p[0],p[1],1),new Vector3(0,0,-1));const hit=ray.intersectObject(coatSurface)[0];if(hit)p=[p[0],p[1],hit.point.z+.004];}
    detail.positions.push(p);detail.uv.push([i%2,Math.floor(i/2)]);detail.influences.push(binding(p));
  });for(let i=1;i<points.length-1;i++)detail.indices.push(base,base+i,base+i+1);};
  if(selection.outfit==='tailored'){
    for(const sign of [-1,1]){
      panel([[sign*.057,1.318,.085],[sign*.14,1.258,.163],[sign*.102,1.207,.186],[sign*.065,1.13,.17],[sign*.04,1.25,.155]]);
      panel([[sign*.089,.934,.158],[sign*.15,.933,.133],[sign*.15,.922,.135],[sign*.089,.923,.16]]);
    }
    m.addMesh('Ari blazer lapels and welt pockets',detail,m.addMaterial('Ari taupe tailoring',colors[0].map(v=>v*.88)),1);
  } else if(bomber) {
    const trouserInfluence=m.influence(pants);
    for(const sign of [-1,1])panel([[sign*.14,.76,.035],[sign*.172,.75,-.03],[sign*.172,.635,-.03],[sign*.14,.64,.035]],trouserInfluence,false);
    m.addMesh('Ari utility cargo pockets',detail,m.addMaterial('Ari utility pocket fabric',[.07,.085,.065]),1);
  } else {
    // A continuous weighted skirt, anchored at the waist, with sparse guide
    // bones rather than thousands of independently simulated cloth vertices.
    const hips=g.extensions.VRMC_vrm.humanoid.humanBones.hips.node,chains:number[][]=[];
    const spring=g.extensions.VRMC_springBone;
    const colliders:number[]=[];
    for(const side of ['left','right']){
      const bone=g.extensions.VRMC_vrm.humanoid.humanBones[`${side}UpperLeg`].node;
      const knee=g.extensions.VRMC_vrm.humanoid.humanBones[`${side}LowerLeg`].node;
      const tail=new Vector3().setFromMatrixPosition(m.worlds[knee]).applyMatrix4(m.worlds[bone].clone().invert()).toArray();
      colliders.push(spring.colliders.push({node:bone,shape:{capsule:{offset:[0,0,0],tail,radius:.065}}})-1);
    }
    const group=spring.colliderGroups.push({name:'Ari skirt leg clearance',colliders})-1;
    for(let i=0;i<12;i++){
      const a=i/12*Math.PI*2;
      chains.push(m.addChain(`Skirt${i}`,[[Math.sin(a)*.14,.985,Math.cos(a)*.115],[Math.sin(a)*.19,.69,Math.cos(a)*.15],[Math.sin(a)*.255,.34,Math.cos(a)*.19]],hips,1,.82,.32,[group]));
    }
    for(const under of [true,false]){
      const s=surface(),rows=24,sides=96;
      for(let row=0;row<=rows;row++)for(let side=0;side<=sides;side++){
        const t=row/rows,a=side/sides*Math.PI*2,ripple=Math.sin(a*16+.1*Math.sin(t*8))*.013*Math.pow(t,.8),rx=.141+.114*t+(under?-.005:0)+ripple,rz=.117+.073*t+(under?-.004:0)+ripple;
        const p=[Math.sin(a)*rx,.985-t*(under?.675:.645)+.005*Math.sin(a*8)*t*t,Math.cos(a)*rz+.004],angular=(side%sides)/sides*chains.length,lo=Math.floor(angular),f=angular-lo,vertical=Math.min(t*2,1.99999),j=Math.floor(vertical),v=vertical-j;
        s.positions.push(p);s.uv.push([side/sides,t]);s.influences.push({joints:[chains[lo][j],chains[lo][j+1],chains[(lo+1)%chains.length][j],chains[(lo+1)%chains.length][j+1]],weights:[(1-f)*(1-v),(1-f)*v,f*(1-v),f*v]});
      }
      for(let row=0;row<rows;row++)for(let side=0;side<sides;side++){const a=row*(sides+1)+side,b=a+sides+1;s.indices.push(a,b,a+1,a+1,b,b+1);}
      m.addMesh(under?'Ari linen under-hem':'Ari sage linen skirt',s,m.addMaterial(under?'Ari oatmeal under-hem':'Ari sage linen',under?[.76,.71,.61]:colors[1],'pleat'),1);
    }
    // Covered trousers are omitted, while the original skin mask still covers
    // the concealed torso. The visible legs and shoes remain from Ari's base.
    body.primitives=body.primitives.filter((p:JSONNode)=>p!==pants);
  }
  closedFootwear(m,colors[3],selection.outfit==='linen',bomber);
  coatGeometry.dispose();coatMaterial.dispose();
}

function torsoBinding(m:Model) {
  const g=m.g,skin=g.skins[1],bones=g.extensions.VRMC_vrm.humanoid.humanBones;
  const joint=(name:string)=>skin.joints.indexOf(bones[name].node);
  return ([,y]:number[]):Influence=>{
    const anchors:[[number,string],[number,string],[number,string],[number,string]]=[[.98,'hips'],[1.10,'spine'],[1.22,'chest'],[1.30,'upperChest']];
    let i=0;while(i<2&&y>anchors[i+1][0])i++;
    const f=smooth((y-anchors[i][0])/(anchors[i+1][0]-anchors[i][0]));
    return {joints:[joint(anchors[i][1]),joint(anchors[i+1][1]),0,0],weights:[1-f,f,0,0]};
  };
}

function skirt(m:Model,color:number[],hem:number,waist=1.02,ruffle=false,trim?:number[]) {
  const g=m.g,hips=g.extensions.VRMC_vrm.humanoid.humanBones.hips.node,spring=g.extensions.VRMC_springBone,colliders:number[]=[];
  const rootRx=waist<.9?.184:.14,rootRz=waist<.9?.137:.113;
  for(const side of ['left','right']){
    const bone=g.extensions.VRMC_vrm.humanoid.humanBones[`${side}UpperLeg`].node,knee=g.extensions.VRMC_vrm.humanoid.humanBones[`${side}LowerLeg`].node;
    const tail=new Vector3().setFromMatrixPosition(m.worlds[knee]).applyMatrix4(m.worlds[bone].clone().invert()).toArray();
    colliders.push(spring.colliders.push({node:bone,shape:{capsule:{offset:[0,0,0],tail,radius:.063}}})-1);
  }
  const group=spring.colliderGroups.push({name:'Ari pleated skirt leg clearance',colliders})-1;
  const chains=Array.from({length:12},(_,i)=>{
    const a=i/12*Math.PI*2;
    return m.addChain(`SkirtPleat${i}`,[[Math.sin(a)*rootRx,waist,Math.cos(a)*rootRz+.016],[Math.sin(a)*(rootRx+.235)/2,(waist+hem)/2,Math.cos(a)*(rootRz+.170)/2+.016],[Math.sin(a)*.235,hem,Math.cos(a)*.170+.016]],hips,1,1.05,.33,[group]);
  });
  const binding=(a:number,t:number):Influence=>{
    const angular=(a/(Math.PI*2)%1+1)%1*12,lo=Math.floor(angular),f=angular-lo,v=Math.max(0,Math.min(t*2,1.9999)),j=Math.floor(v),blend=v-j;
    return {joints:[chains[lo][j],chains[lo][j+1],chains[(lo+1)%12][j],chains[(lo+1)%12][j+1]],weights:[(1-f)*(1-blend),(1-f)*blend,f*(1-blend),f*blend]};
  };
  const point=(a:number,t:number,offset=0)=>{
    // Folds start at the waist seam, deepen under gravity, and soften at the hem.
    const phase=a*24+.22*Math.sin(a*3)*t;
    const pleat=(Math.cos(phase)+.20*Math.sin(phase*2))*(.002+.010*Math.pow(Math.max(t,0),.75));
    const drape=.0022*Math.sin(a*7+t*8)*Math.sin(Math.PI*t);
    const frill=ruffle?Math.sin(a*36)*.007*smooth((t-.88)/.12):0;
    const flare=Math.pow(Math.max(0,t),.80),rx=rootRx+(.235-rootRx)*flare+pleat+frill+drape+offset,rz=rootRz+(.170-rootRz)*flare+pleat*.72+frill+drape+offset;
    return [Math.sin(a)*rx,waist-(waist-hem)*t+.0025*Math.sin(a*7)*t*t,Math.cos(a)*rz+.016];
  };
  const layer=(name:string,start:number,end:number,tint:number[],offset=0)=>{
    const s=surface(),rows=name.includes('hem')?4:40,sides=192;
    for(let row=0;row<=rows;row++)for(let side=0;side<=sides;side++){
      const t=start+(end-start)*row/rows,a=side/sides*Math.PI*2;
      s.positions.push(point(a,t,offset));s.uv.push([side/sides,t]);s.influences.push(binding(a,t));
    }
    for(let row=0;row<rows;row++)for(let side=0;side<sides;side++){const a=row*(sides+1)+side,b=a+sides+1;s.indices.push(a,b,a+1,a+1,b,b+1);}
    // Turn the last 3 mm inward so the hem has a visible fabric edge.
    const edge=s.positions.length;
    for(let side=0;side<=sides;side++){
      const a=side/sides*Math.PI*2,p=point(a,end,offset-.0025);p[1]+=.003;
      s.positions.push(p);s.uv.push([side/sides,end]);s.influences.push(binding(a,end));
    }
    for(let side=0;side<sides;side++){const a=rows*(sides+1)+side,b=edge+side;s.indices.push(a,b,a+1,a+1,b,b+1);}
    m.addMesh(name,s,m.addFabric(name,tint,'twill'),1);
  };
  layer('Ari pleated skirt',0,1,color);
  if(trim)layer('Ari skirt ribbon hem',.92,.98,trim,.002);
  if(ruffle)layer('Ari ruffled underskirt',.89,1.055,trim??[.09,.085,.08],-.002);
  const band=surface();
  for(let row=0;row<=8;row++)for(let side=0;side<=192;side++){
    const a=side/192*Math.PI*2,t=row/8*.085,p=point(a,t,.003);band.positions.push(p);band.uv.push([side/192,row/8*.08]);band.influences.push(binding(a,t));
  }
  for(let row=0;row<8;row++)for(let side=0;side<192;side++){const a=row*193+side;band.indices.push(a,a+193,a+1,a+1,a+193,a+194);}
  m.addMesh('Ari tailored waistband',band,m.addFabric('Ari waistband',color.map(v=>v*.87),'twill'),1);
}

/** Screenshot-inspired garments share Ari's body but have separate silhouettes. */
function screenshotOutfit(m:Model,id:WardrobeSelection['outfit']) {
  const g=m.g,body=g.meshes[1],find=(word:string)=>body.primitives.find((p:JSONNode)=>g.materials[p.material].name.includes(word));
  const coat=find('Tops_01'),inner=find('Onepiece_00'),pants=find('Bottoms_01');
  const sweatshirt=id.endsWith('sweater'),cardigan=id.startsWith('school-cardigan'),vest=id==='school-vest',navy=id==='navy-blouse',sailor=id==='pink-sailor',ruffle=id==='ruffle-dress',jacket=id==='black-jacket',butter=id==='butter-coat',forest=id==='forest-pinafore';
  const brown=id==='school-cardigan-brown',blue=id==='cloud-sweater';
  const outer=sweatshirt?(blue?[.54,.68,.76]:[.82,.72,.35]):cardigan?(brown?[.32,.21,.14]:[.23,.25,.26]):vest?[.77,.66,.47]:jacket?[.026,.031,.033]:butter?[.82,.66,.29]:ruffle?[.045,.039,.04]:[.86,.85,.78];
  const bottom=cardigan?(brown?[.14,.09,.065]:[.095,.105,.11]):vest?[.085,.089,.09]:navy?[.035,.07,.09]:sailor?[.69,.33,.42]:ruffle?[.58,.44,.34]:forest?[.18,.25,.095]:butter?outer:[.13,.16,.20];
  const bind=torsoBinding(m);
  const recolor=(primitive:JSONNode,color:number[])=>{g.materials[primitive.material].extras={...g.materials[primitive.material].extras,ariGarmentColor:color};};
  recolor(coat,outer);recolor(inner,jacket?[.85,.84,.78]:outer);recolor(pants,[.035,.039,.043]);
  const puff=sweatshirt?1.18:ruffle||sailor||navy||forest?1.10:vest?.85:1;
  m.warp(coat,([x,y,z])=>{
    const arm=smooth((Math.abs(x)-.16)/.10),lower=smooth((1.13-y)/.36),shorten=cardigan?.03:jacket?.06:sweatshirt?-.08:butter?-.10:.14;
    return [x*(1+(sweatshirt?.06:cardigan?.15*lower:0)*(1-arm)),y+shorten*lower+(y-1.275)*(puff-1)*arm,z*(1+(puff-1)*(.3+.7*arm)+(cardigan?.38*lower*(1-arm):0))];
  });
  m.warp(g.meshes[3].primitives[0],([x,y,z])=>[x,1.302+(y-1.323)*.16,z]);
  if(!cardigan&&!jacket){
    // Reuse the fitted sleeve weights; replace the open cardigan torso.
    const positions=m.values(coat.attributes.POSITION),indices=m.values(coat.indices).flat(),sleeves:number[]=[];
    for(let i=0;i<indices.length;i+=3){const tri=indices.slice(i,i+3);if(tri.every(v=>positions[v][1]>1.245||(Math.abs(positions[v][0])>.162&&positions[v][1]>1.12)))sleeves.push(...tri);}
    coat.indices=m.append(sleeves.map(i=>[i]),'SCALAR',5125);
    recolor(coat,sweatshirt||butter?outer:ruffle?[.045,.039,.04]:[.84,.83,.77]);
    body.primitives=body.primitives.filter((p:JSONNode)=>p!==inner);
    // A blouse/sweatshirt has its own neckline; the collapsed v9 turtleneck
    // would otherwise remain as an unrelated band above it.
    g.scenes[g.scene??0].nodes=g.scenes[g.scene??0].nodes.filter((i:number)=>g.nodes[i].mesh!==3);
  }
  const profile=[[.89,.163,.118,.016],[.99,.135,.090,.024],[1.10,.121,.084,.041],[1.18,.145,.125,.025],[1.25,.170,.094,-.006],[1.315,.17,.065,-.015]];
  const shape=(y:number,extra=0)=>{
    let i=0;while(i<profile.length-2&&y>profile[i+1][0])i++;
    const a=profile[i],b=profile[i+1],f=smooth((y-a[0])/(b[0]-a[0]));return [a[1]+(b[1]-a[1])*f+extra,a[2]+(b[2]-a[2])*f+extra,a[3]+(b[3]-a[3])*f];
  };
  const closedTop=(name:string,tint:number[],bottomY:number,extra:number,vNeck=false,kind='cotton')=>{
    const s=surface(),rows=64,sides=144;
    const point=(a:number,t:number)=>{
      const top=1.317-(vNeck?.105*Math.max(0,Math.cos(a))**6:0),y=bottomY+(top-bottomY)*t,[rx,rz,z]=shape(y,extra);
      const neck=smooth((t-.80)/.20);
      const tuck=!sweatshirt&&!butter&&!vNeck?1-.13*(1-smooth(t/.22)):1;
      const x=Math.sin(a)*(rx*(1-neck)+.048*neck)*tuck;
      const gather=(Math.sin(a*16+.6*Math.sin(a*3))+.35*Math.sin(a*27))*.0028*Math.exp(-(((t-.10)/.14)**2));
      const crease=(center:number,width:number)=>Math.exp(-(((y-center)/width)**2))-.38*Math.exp(-(((y-center-.013)/(width*1.5))**2));
      const folds=(crease(1.07+.16*Math.abs(x)+.018*Math.sin(a*2),.010)*.0034+crease(1.115-.25*Math.abs(x),.014)*.0023)*Math.sin(a)**2;
      const drape=.0018*Math.sin(a*9+t*5)*Math.sin(Math.PI*t);
      const radial=(gather+folds+drape)*(1-neck);
      return [x+Math.sin(a)*radial,y,Math.cos(a)*((rz*(1-neck)+.062*neck)*tuck+radial)+z];
    };
    for(let row=0;row<=rows;row++)for(let side=0;side<=sides;side++){
      const a=side/sides*Math.PI*2,t=row/rows,p=point(a,t);s.positions.push(p);s.uv.push([side/sides,t]);s.influences.push(bind(p));
    }
    for(let row=0;row<rows;row++)for(let side=0;side<sides;side++){const a=row*(sides+1)+side,b=a+sides+1;s.indices.push(a,a+1,b,a+1,b+1,b);}
    m.addMesh(name,s,m.addFabric(name,tint,kind),1);
    const edging=surface();
    for(const [start,end] of [[0,sweatshirt?.06:.020],[.980,1]]){
      const base=edging.positions.length;
      for(let row=0;row<=6;row++)for(let side=0;side<=sides;side++){
        const t=start+(end-start)*row/6,a=side/sides*Math.PI*2,p=point(a,t),r=.0015+.001*Math.sin(row/6*Math.PI);
        p[0]+=Math.sin(a)*r;p[2]+=Math.cos(a)*r;edging.positions.push(p);edging.uv.push([side/sides,t]);edging.influences.push(bind(p));
      }
      for(let row=0;row<6;row++)for(let side=0;side<sides;side++){const a=base+row*(sides+1)+side,b=a+sides+1;edging.indices.push(a,a+1,b,a+1,b+1,b);}
    }
    m.addMesh(name+' turned seams',edging,m.addFabric(name+' seam fabric',tint.map(v=>v*.86),sweatshirt||vest?'rib':kind),1);
  };
  if(sweatshirt)closedTop('Ari oversized sweatshirt body',outer,.83,.026,false,'cotton');
  else if(butter)closedTop('Ari shaped wool coat',outer,.79,.026,false,'wool');
  else if(!cardigan&&!jacket){
    closedTop('Ari closed blouse',[.73,.72,.67],.985,.010);
    if(vest||forest||ruffle)closedTop('Ari fitted vest bodice',vest?outer:bottom,1.045,.018,true,vest?'wool':'twill');
  }
  const decoration=surface();
  const front=(x:number,y:number,extra=.020)=>{
    const closed=!cardigan&&!jacket,bottomY=sweatshirt?.83:butter?.79:.985;
    const [rx,rz,z]=shape(y,closed?(sweatshirt||butter?.026:.010):.020);
    const neck=closed?smooth(((y-bottomY)/(1.317-bottomY)-.80)/.20):0;
    const nx=rx*(1-neck)+.048*neck,nz=rz*(1-neck)+.062*neck;
    return z+nz*Math.sqrt(Math.max(.02,1-(x/nx)**2))+(closed?extra-.018:.003);
  };
  const panel=(points:number[][])=>{
    clothPatch(decoration,(u,v)=>{
      const p=points[0].map((_,k)=>(1-v)*(points[0][k]*(1-u)+points[1][k]*u)+v*(points[3][k]*(1-u)+points[2][k]*u));
      p[2]+=.006*Math.sin(Math.PI*u)*Math.sin(Math.PI*v)+.003*Math.sin(Math.PI*u)*v;
      return p;
    },bind);
  };
  const collar=(wide=false)=>{for(const sign of [-1,1])panel([[sign*.018,1.318,front(sign*.018,1.318)],[sign*(wide?.133:.073),wide?1.24:1.282,front(sign*(wide?.133:.073),wide?1.24:1.282)],[sign*.046,1.255,front(sign*.046,1.255)],[0,1.30,front(0,1.30)]]);};
  if(!sweatshirt)collar(sailor);
  if(decoration.positions.length)m.addMesh('Ari collar details',decoration,m.addFabric('Ari collar',sailor?[.72,.35,.43]:jacket?[.045,.048,.052]:[.88,.87,.81]),1);
  const accents=surface();
  const chestBow=(color:number[],y=1.266,scale=.8)=>{
    const s=surface(),z=front(0,y,.026),center=[0,y,z];
    for(const sign of [-1,1]){
      // Each loop travels out and back to the knot. Width grows at the folded
      // outer edge; the return half lies behind the front half.
      clothPatch(s,(u,v)=>{
        const q=u*Math.PI,w=(v-.5)*.043*scale*Math.pow(Math.sin(q),.5);
        return [sign*(.003+.060*scale*Math.sin(q)),y+w+.004*Math.sin(q),z+.007+.012*scale*Math.sin(u*Math.PI*2)+.005*Math.sin(q)];
      },bind,10,36,.0012);
      clothPatch(s,(u,v)=>{
        const x=sign*(.007+.035*v)*scale+(u-.5)*.018*scale,yy=y-(.014+.086*v)*scale;
        return [x,yy+(v>.84?.003*Math.abs(u-.5):0),front(x,yy,.026)+.012+.004*Math.sin(v*5+u*2)];
      },bind,24,10,.0012);
    }
    tube(s,[[0,y+.009,z+.007],center,[0,y-.009,z+.012]],()=>[.009*scale,.006],bind,8,8);
    m.addMesh('Ari chest ribbon',s,m.addFabric('Ari chest ribbon',color,'satin'),1);
  };
  if(vest||navy||sailor||ruffle||forest||butter)chestBow(sailor?[.72,.36,.44]:forest?[.16,.23,.08]:[.033,.039,.042],sailor?1.253:1.29,sailor?1:.65);
  else if(cardigan){
    const y=1.285,z=front(0,y);tube(accents,[[0,y,z],[0,1.22,front(0,1.22)],[0,1.145,front(0,1.145)]],t=>[t<.7?.013:.018,.0025],bind,24,6);m.addMesh('Ari school tie',accents,m.addFabric('Ari school tie',[.055,.06,.065],'twill'),1);
  }
  if(ruffle){
    const bib=surface();for(let row=0;row<=18;row++)for(let col=0;col<=20;col++){const y=1.29-row/18*.115,x=(col/20-.5)*.13*(.8+.2*row/18),p=[x,y,front(x,y,.035)+.002*Math.sin(col*3)];bib.positions.push(p);bib.uv.push([col/20,row/18]);bib.influences.push(bind(p));}
    for(let row=0;row<18;row++)for(let col=0;col<20;col++){const a=row*21+col;bib.indices.push(a,a+1,a+21,a+1,a+22,a+21);}m.addMesh('Ari ruffled blouse bib',bib,m.addFabric('Ari ivory bib',[.78,.76,.71],'cotton'),1);
  }
  const sewing=surface(),buttons=surface();
  if(!sweatshirt){
    const stop=butter?.81:cardigan||jacket?1.07:1.055;
    clothPatch(sewing,(u,v)=>{const x=(u-.5)*.014,y=1.276-(1.276-stop)*v;return [x,y,front(x,y)+.003+.001*Math.sin(Math.PI*u)];},bind,36,8);
    for(let y=1.245;y>stop+.02;y-=butter?.068:.046){
      const z=front(0,y)+.006;
      tube(buttons,[[0,y,z-.001],[0,y,z+.001],[0,y,z+.0025]],t=>[.004*(1-.3*t),.004*(1-.3*t)],bind,4,16);
    }
    if(butter||jacket)for(const sign of [-1,1]){
      clothPatch(sewing,(u,v)=>{const x=sign*(.065+.059*u),y=(butter?.965:1.035)-v*.054;return [x,y,front(x,y,.022)+.002+.003*Math.sin(Math.PI*u)*Math.sin(Math.PI*v)];},bind,16,20);
      // Pocket flap stands off the patch, with a rolled edge rather than paint.
      clothPatch(sewing,(u,v)=>{const x=sign*(.064+.062*u),y=(butter?.971:1.041)-v*.014;return [x,y,front(x,y,.026)+.002*Math.sin(Math.PI*v)];},bind,6,20);
    }
    m.addMesh('Ari sewn placket and pockets',sewing,m.addFabric('Ari seam fabric',(butter||jacket||cardigan?outer:[.73,.72,.67]).map(v=>v*.90),butter?'wool':'cotton'),1);
    m.addMesh('Ari domed garment buttons',buttons,m.addFabric('Ari garment buttons',butter?[.14,.11,.075]:navy||forest||vest?[.37,.28,.12]:[.20,.18,.15],'leather'),1);
  }
  if(!jacket){
    body.primitives=body.primitives.filter((p:JSONNode)=>p!==pants);
    skirt(m,bottom,sweatshirt?.64:butter?.61:.62,sweatshirt?.85:butter?.81:vest||navy||forest||ruffle?1.05:1.01,ruffle||sailor,sailor?[.88,.83,.78]:ruffle?[.035,.029,.033]:undefined);
  }
  closedFootwear(m,sweatshirt?[.75,.77,.73]:[.033,.029,.027],false,sweatshirt);
  // Fit the stockings to Ari's actual calves, with their original leg weights.
  // A generic cylinder exposed the back of the leg and looked like shin guards.
  if(!jacket&&!butter){
    const socks=fittedStockings(m);
    m.addMesh('Ari knee socks',socks,m.addFabric('Ari ivory socks',[.72,.71,.66],'rib'),1);
  }
}

function fittedStockings(m:Model):Surface {
  const p=m.g.meshes[1].primitives.find((p:JSONNode)=>p.material===8),s=surface(),cache=new Map<string,number>();
  const positions=m.values(p.attributes.POSITION),normals=m.values(p.attributes.NORMAL),joints=m.values(p.attributes.JOINTS_0),weights=m.values(p.attributes.WEIGHTS_0);
  type Vertex={p:number[];n:number[];influence:Influence};
  const blend=(a:Vertex,b:Vertex,t:number):Vertex=>{
    const weighted=new Map<number,number>();
    for(const [v,f]of [[a,1-t],[b,t]] as const)v.influence.joints.forEach((j,i)=>weighted.set(j,(weighted.get(j)??0)+v.influence.weights[i]*f));
    const pairs=[...weighted.entries()].sort((a,b)=>b[1]-a[1]).slice(0,4);while(pairs.length<4)pairs.push([0,0]);const sum=pairs.reduce((v,p)=>v+p[1],0);
    return {p:a.p.map((v,i)=>v+(b.p[i]-v)*t),n:a.n.map((v,i)=>v+(b.n[i]-v)*t),influence:{joints:pairs.map(p=>p[0]),weights:pairs.map(p=>p[1]/sum)}};
  };
  const clip=(vertices:Vertex[],height:number,above:boolean)=>{
    const result:Vertex[]=[];for(let i=0;i<vertices.length;i++){
      const a=vertices[i],b=vertices[(i+1)%vertices.length],inside=(v:Vertex)=>above?v.p[1]>=height:v.p[1]<=height;
      if(inside(a))result.push(a);if(inside(a)!==inside(b))result.push(blend(a,b,(height-a.p[1])/(b.p[1]-a.p[1])));
    }return result;
  };
  const ids=m.values(p.indices).flat();
  for(let i=0;i<ids.length;i+=3){
    const tri=ids.slice(i,i+3);if(tri.every(j=>positions[j][1]>.485)||tri.every(j=>positions[j][1]<.095))continue;
    const polygon=clip(clip(tri.map(j=>({p:positions[j],n:normals[j],influence:{joints:joints[j],weights:weights[j]}})),.095,true),.485,false);
    const added=polygon.map(v=>{
      const point=v.p.map((x,k)=>x+v.n[k]*.0023),key=point.map(x=>x.toFixed(6)).join(',');let id=cache.get(key);
      if(id===undefined){id=s.positions.length;cache.set(key,id);s.positions.push(point);s.uv.push([Math.atan2(point[0]-Math.sign(point[0])*.077,point[2]+.025)/(Math.PI*2),point[1]*2]);s.influences.push(v.influence);}return id;
    });
    for(let j=1;j<added.length-1;j++)s.indices.push(added[0],added[j],added[j+1]);
  }
  return s;
}

function closedFootwear(m:Model,color:number[],boots=false,sneakers=false) {
  const g=m.g,body=g.meshes[1],shoes=body.primitives.find((p:JSONNode)=>g.materials[p.material].name.includes('Shoes_01'));
  // The v9 sandals have actual openings, so changing alpha is insufficient.
  // Closed footwear uses a shared foot last and keeps the original sole height.
  body.primitives=body.primitives.filter((p:JSONNode)=>p!==shoes);
  const footwear=surface(),laces=surface();
  for(const side of ['left','right']){
    const foot=g.extensions.VRMC_vrm.humanoid.humanBones[`${side}Foot`].node,leg=g.extensions.VRMC_vrm.humanoid.humanBones[`${side}LowerLeg`].node;
    const x=new Vector3().setFromMatrixPosition(m.worlds[foot]).x,skin=g.skins[1],base=footwear.positions.length;
    const rings=[[-.001,.052,.110,.022],[.016,.054,.111,.022],[.036,.053,.109,.021],[.061,.047,.093,.01],[.086,.037,.063,-.012],[.112,.033,.036,-.032],...(boots?[[.15,.036,.036,-.032],[.225,.041,.04,-.03]]:[])];
    for(const [row,[y,rx,rz,z]]of rings.entries())for(let col=0;col<=40;col++){
      const a=col/40*Math.PI*2,blend=boots?smooth((y-.11)/.09):0;
      footwear.positions.push([x+Math.sin(a)*rx,y,z+Math.cos(a)*rz]);footwear.uv.push([col/40,row/(rings.length-1)]);footwear.influences.push({joints:[skin.joints.indexOf(foot),skin.joints.indexOf(leg),0,0],weights:[1-blend,blend,0,0]});
    }
    for(let row=0;row<rings.length-1;row++)for(let col=0;col<40;col++){const a=base+row*41+col,b=a+41;footwear.indices.push(a,a+1,b,a+1,b+1,b);}
    const fixed=()=>({joints:[skin.joints.indexOf(foot),0,0,0],weights:[1,0,0,0]});
    if(sneakers)for(let i=0;i<4;i++)tube(laces,[[x-.03,.087-i*.005,-.01+i*.014],[x,.092-i*.005,-.01+i*.014],[x+.03,.087-i*.005,-.01+i*.014]],()=>[.0018,.0018],fixed,8,5);
  }
  m.addMesh('Ari closed footwear',footwear,m.addFabric('Ari footwear',color,'leather'),1);
  if(laces.positions.length)m.addMesh('Ari sneaker laces',laces,m.addMaterial('Ari ivory laces',[.86,.85,.78]),1);
}

export function buildWardrobeModel(buffer:ArrayBuffer,value:WardrobeSelection):ArrayBuffer {
  const selection=parseWardrobe(value);
  if(selection.outfit==='original'&&selection.hair==='original')return buffer;
  const model=new Model(buffer);
  outfit(model,selection);hair(model,selection);
  model.g.asset.extras={...model.g.asset.extras,ariWardrobe:selection};
  return model.finish();
}
