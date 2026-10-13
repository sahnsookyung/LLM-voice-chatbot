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
function hair(m:Model,selection:WardrobeSelection) {
  if(selection.hair==='original')return;
  const g=m.g,p=g.meshes[2].primitives[0],springs=g.extensions.VRMC_springBone.springs;
  // Keep the authored fringe. Replace the long locks with a dedicated tied rig.
  const old=m.values(p.attributes.POSITION),uv=m.values(p.attributes.TEXCOORD_0),js=m.values(p.attributes.JOINTS_0),ws=m.values(p.attributes.WEIGHTS_0),indices=m.values(p.indices).flat();
  const adj=new Map<number,Set<number>>();for(let i=0;i<indices.length;i+=3){const a=indices.slice(i,i+3);for(const n of a){if(!adj.has(n))adj.set(n,new Set());for(const k of a)adj.get(n)!.add(k);}}
  const seen=new Set<number>(),keep=new Set<number>();for(const start of adj.keys()){if(seen.has(start))continue;const queue=[start],group:number[]=[];seen.add(start);while(queue.length){const i=queue.pop()!;group.push(i);for(const n of adj.get(i)!)if(!seen.has(n)){seen.add(n);queue.push(n);}}if(group.every(i=>old[i][1]>1.38))for(const i of group)keep.add(i);}
  const out=surface(),map=new Map<number,number>();for(const i of keep){map.set(i,out.positions.length);out.positions.push(old[i]);out.uv.push(uv[i]);out.influences.push({joints:js[i],weights:ws[i]});}
  for(let i=0;i<indices.length;i+=3)if(keep.has(indices[i]))out.indices.push(...indices.slice(i,i+3).map(v=>map.get(v)!));
  g.extensions.VRMC_springBone.springs=springs.filter((s:JSONNode)=>s.name!=='Hair'||s.joints[0].node<33);
  const head=g.extensions.VRMC_vrm.humanoid.humanBones.head.node,headJoint=g.skins[2].joints.indexOf(head);
  const fixed=()=>({joints:[headJoint,0,0,0],weights:[1,0,0,0]});
  if(selection.hair==='bob'){
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
    g.meshes[2]={name:'Ari soft tapered bob',primitives:[{...m.geometry(out),material:p.material}]};return;
  }
  const braid=selection.hair==='braid';
  const guide=braid?[[0,1.55,-.105],[.085,1.48,-.11],[.135,1.39,-.04],[.152,1.31,.09],[.148,1.20,.175],[.135,1.08,.195],[.12,.99,.185]]:[[0,1.565,-.105],[.035,1.565,-.16],[.085,1.49,-.19],[.14,1.37,-.205],[.18,1.23,-.20],[.17,1.08,-.18]];
  const guideCurve=new CatmullRomCurve3(guide.map(vector)),points=Array.from({length:8},(_,i)=>guideCurve.getPoint(i/7).toArray());
  const chain=m.addChain('HairTie',points,head,2,braid?.60:.42,braid?.30:.19,[2,3,10]);
  // Fine swept crown sections meet at the actual attachment instead of a bald cap.
  for(let i=0;i<38;i++){
    const a=i/37*Math.PI*1.8+.1,root=[Math.sin(a)*.08,1.565+Math.cos(a)*.023,Math.cos(a)*.065-.026];
    tube(out,[root,[root[0]*.8,1.585,-.062],[root[0]*.35,1.566,-.10],guide[0]],t=>[.0035*(1-.5*t),.0008],fixed,22,5,i*.137);
  }
  const strands=braid?36:54;
  for(let i=0;i<strands;i++){
    const strandPoints=Array.from({length:65},(_,k)=>{
      const t=k/64,c=guideCurve.getPoint(t),tangent=guideCurve.getTangent(Math.max(.00001,Math.min(.99999,t))),axis=new Vector3(1,0,0).addScaledVector(tangent,-tangent.x).normalize(),normal=tangent.clone().cross(axis).normalize();
      const bundle=braid?Math.floor(i/12):0,angle=braid?bundle/3*Math.PI*2+t*Math.PI*15:i/strands*Math.PI*2;
      const r=braid?.017*(.95-.45*t):.032*Math.sin(Math.PI*t*.85)*(.8+.15*Math.sin(i*3));
      const offset=braid?((i%12)/11-.5)*.014:0;
      return c.addScaledVector(axis,Math.cos(angle)*r+offset).addScaledVector(normal,Math.sin(angle)*r*.65).toArray();
    });
    tube(out,strandPoints,t=>[Math.max(.00006,(braid?.0034:.0031)*Math.pow(1-smooth((t-.78)/.22),.6)),braid?.0015:.001],(_,t)=>chainInfluence(chain,t),64,6,i*.173);
  }
  g.meshes[2]={name:`Ari ${selection.hair} fine locks`,primitives:[{...m.geometry(out),material:p.material}]};
}

function outfit(m:Model,selection:WardrobeSelection) {
  if(selection.outfit==='original')return;
  const g=m.g,body=g.meshes[1],find=(word:string)=>body.primitives.find((p:JSONNode)=>g.materials[p.material].name.includes(word));
  const coat=find('Tops_01'),inner=find('Onepiece_00'),pants=find('Bottoms_01'),shoes=find('Shoes_01');
  if(!coat||!inner||!pants||!shoes)throw new Error('Ari garment binding is missing.');
  const colors=selection.outfit==='tailored'?[[.43,.36,.30],[.88,.84,.76],[.065,.071,.075],[.035,.027,.023]]:selection.outfit==='streetwear'?[[.23,.29,.20],[.84,.82,.74],[.052,.066,.055],[.73,.73,.68]]:[[.68,.62,.50],[.27,.34,.23],[.27,.34,.23],[.09,.047,.027]];
  for(const [i,p]of [coat,inner,pants,shoes].entries())g.materials[p.material].extras={...g.materials[p.material].extras,ariGarmentColor:colors[i]};
  // Use the shoe's closed base surface for loafers, sneakers and boots instead
  // of retaining the original sandals' alpha cutouts.
  g.materials[shoes.material].alphaMode='OPAQUE';delete g.materials[shoes.material].alphaCutoff;
  m.warp(g.meshes[3].primitives[0],([x,y,z])=>[x,1.323+(y-1.323)*.18,z]);
  if(selection.outfit!=='linen')m.warp(pants,([x,y,z])=>{
    const loose=smooth((.80-y)/.55)*(selection.outfit==='tailored'?.42:.27),center=Math.sign(x)*.087;
    return [center+(x-center)*(1+loose),y,z*(1+loose*.45)];
  });
  if(selection.outfit==='streetwear')m.warp(shoes,([x,y,z])=>{const center=Math.sign(x)*.087;return [center+(x-center)*1.10,y*1.13,z*1.06];});
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
  // The v9 sandals have actual openings, so changing alpha is insufficient.
  // Closed footwear uses a shared foot last and keeps the original sole height.
  body.primitives=body.primitives.filter((p:JSONNode)=>p!==shoes);
  const footwear=surface(),laces=surface();
  for(const side of ['left','right']){
    const foot=g.extensions.VRMC_vrm.humanoid.humanBones[`${side}Foot`].node,leg=g.extensions.VRMC_vrm.humanoid.humanBones[`${side}LowerLeg`].node;
    const x=new Vector3().setFromMatrixPosition(m.worlds[foot]).x,skin=g.skins[1],base=footwear.positions.length;
    const rings=[[-.001,.052,.110,.022],[.016,.054,.111,.022],[.036,.053,.109,.021],[.061,.047,.093,.01],[.086,.037,.063,-.012],[.112,.033,.036,-.032],...(selection.outfit==='linen'?[[.15,.036,.036,-.032],[.225,.041,.04,-.03]]:[])];
    for(const [row,[y,rx,rz,z]]of rings.entries())for(let col=0;col<=40;col++){
      const a=col/40*Math.PI*2,blend=selection.outfit==='linen'?smooth((y-.11)/.09):0;
      footwear.positions.push([x+Math.sin(a)*rx,y,z+Math.cos(a)*rz]);footwear.uv.push([col/40,row/(rings.length-1)]);footwear.influences.push({joints:[skin.joints.indexOf(foot),skin.joints.indexOf(leg),0,0],weights:[1-blend,blend,0,0]});
    }
    for(let row=0;row<rings.length-1;row++)for(let col=0;col<40;col++){const a=base+row*41+col,b=a+41;footwear.indices.push(a,a+1,b,a+1,b+1,b);}
    const fixed=()=>({joints:[skin.joints.indexOf(foot),0,0,0],weights:[1,0,0,0]});
    if(bomber)for(let i=0;i<4;i++)tube(laces,[[x-.03,.087-i*.005,-.01+i*.014],[x,.092-i*.005,-.01+i*.014],[x+.03,.087-i*.005,-.01+i*.014]],()=>[.0018,.0018],fixed,8,5);
  }
  m.addMesh('Ari closed footwear',footwear,m.addMaterial('Ari footwear',colors[3],'leather'),1);
  if(laces.positions.length)m.addMesh('Ari sneaker laces',laces,m.addMaterial('Ari ivory laces',[.86,.85,.78]),1);
  coatGeometry.dispose();coatMaterial.dispose();
}

export function buildWardrobeModel(buffer:ArrayBuffer,value:WardrobeSelection):ArrayBuffer {
  const selection=parseWardrobe(value);
  if(selection.outfit==='original'&&selection.hair==='original')return buffer;
  const model=new Model(buffer);
  outfit(model,selection);hair(model,selection);
  model.g.asset.extras={...model.g.asset.extras,ariWardrobe:selection};
  return model.finish();
}
