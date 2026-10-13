import {readFile,writeFile} from 'node:fs/promises';
import {createHash} from 'node:crypto';
import assert from 'node:assert/strict';
import {Vector3,Matrix3,Matrix4,Quaternion,CatmullRomCurve3,BufferGeometry,Float32BufferAttribute,Mesh,MeshBasicMaterial,DoubleSide,Raycaster} from '../renderer/node_modules/three/build/three.module.js';

const root=new URL('../characters/ari/',import.meta.url),hash=b=>createHash('sha256').update(b).digest('hex');
const raw=await readFile(new URL('model/ari-vroid-v7.vrm',root));
assert.equal(hash(raw),'2a6cc18768e2d0e6922ea6714bf4577000832d6b7da1f2c2dba44c014149f449');
const size=raw.readUInt32LE(12),g=JSON.parse(raw.toString('utf8',20,20+size)),bin=raw.subarray(28+size);
const views=g.bufferViews.map(v=>Buffer.from(bin.subarray(v.byteOffset??0,(v.byteOffset??0)+v.byteLength)));
const widths={SCALAR:1,VEC2:2,VEC3:3,VEC4:4,MAT4:16};
function accessor(index){const a=g.accessors[index],v=g.bufferViews[a.bufferView],data=views[a.bufferView],width=widths[a.type];
  const [bytes,read,write]={5123:[2,'readUInt16LE','writeUInt16LE'],5125:[4,'readUInt32LE','writeUInt32LE'],5126:[4,'readFloatLE','writeFloatLE']}[a.componentType];
  const offset=i=>(a.byteOffset??0)+i*(v.byteStride??width*bytes);
  return {a,get:i=>Array.from({length:width},(_,k)=>data[read](offset(i)+k*bytes)),set:(i,p)=>p.forEach((v,k)=>{assert.ok(Number.isFinite(v));data[write](v,offset(i)+k*bytes);})};}
const all=index=>{const a=accessor(index);return Array.from({length:a.a.count},(_,i)=>a.get(i));};
function bounds(index){const a=accessor(index),p=all(index);a.a.min=[0,1,2].map(k=>Math.min(...p.map(v=>v[k])));a.a.max=[0,1,2].map(k=>Math.max(...p.map(v=>v[k])));}
function append(data,type,componentType=5126,target=34962){const bytes=componentType===5123?2:4,b=Buffer.alloc(data.length*widths[type]*bytes);
  data.flat().forEach((v,i)=>componentType===5123?b.writeUInt16LE(v,i*bytes):componentType===5125?b.writeUInt32LE(v,i*bytes):b.writeFloatLE(v,i*bytes));
  const bufferView=g.bufferViews.push({buffer:0,byteLength:b.length,target})-1;views.push(b);
  const index=g.accessors.push({bufferView,componentType,count:data.length,type})-1;if(type==='VEC3')bounds(index);return index;}
function components(indices){const adj=new Map();for(let i=0;i<indices.length;i+=3){const t=indices.slice(i,i+3);for(const a of t){if(!adj.has(a))adj.set(a,new Set());for(const b of t)adj.get(a).add(b);}}
  const left=new Set(adj.keys()),groups=[];while(left.size){const q=[left.values().next().value],group=[];left.delete(q[0]);while(q.length){const i=q.pop();group.push(i);for(const j of adj.get(i))if(left.delete(j))q.push(j);}groups.push(group);}
  return groups.sort((a,b)=>Math.min(...a)-Math.min(...b));}
const smooth=t=>{t=Math.max(0,Math.min(1,t));return t*t*(3-2*t);};
const bell=(v,c,r)=>Math.abs(v-c)>=r?0:(1-((v-c)/r)**2)**2;
const face=g.meshes[0],fp=all(0),fn=all(1);
const ears=new Set(components(all(face.primitives.find(p=>p.material===3).indices).flat()).filter(c=>c.length<200&&c.every(i=>Math.abs(fp[i][0])>.055&&fp[i][1]>1.39&&fp[i][1]<1.47)).flat());
assert.equal(ears.size,540);
function earWarp([x,y,z]){const a=Math.abs(x),root=smooth((a-.058)/.012),rim=bell(y,1.458,.018);
  return [Math.sign(x)*(.058+(a-.058)*(1-.17*root-.06*rim*root)),1.436+(y-1.436)*(1-.24*root),z-.005*root+.0015*bell(y,1.45,.016)*bell(a,.08,.008)];}
function normalAt(p,n,warp=earWarp){const columns=p.map((_,axis)=>{const a=p.slice(),b=p.slice();a[axis]-=1e-5;b[axis]+=1e-5;const lo=warp(a),hi=warp(b);return hi.map((v,k)=>(v-lo[k])/2e-5);});
  const j=new Matrix3().set(...[0,1,2].flatMap(k=>columns.map(c=>c[k])));assert.ok(j.determinant()>.1);return new Vector3(...n).applyMatrix3(j.invert().transpose()).normalize().toArray();}
const pa=accessor(0),na=accessor(1);for(const i of ears){pa.set(i,earWarp(fp[i]));na.set(i,normalAt(fp[i],fn[i]));}
for(const target of face.primitives[0].targets){const p=accessor(target.POSITION),n=accessor(target.NORMAL);
  for(const i of ears){const posed=fp[i].map((v,k)=>v+p.get(i)[k]),normal=fn[i].map((v,k)=>v+n.get(i)[k]);p.set(i,earWarp(posed).map((v,k)=>v-pa.get(i)[k]));n.set(i,normalAt(posed,normal).map((v,k)=>v-na.get(i)[k]));}bounds(target.POSITION);}
bounds(0);
const gold=g.meshes.find(m=>m.name==='Ari small gold hoops').primitives[0],goldPos=accessor(gold.attributes.POSITION);
for(let i=0;i<goldPos.a.count;i++){const p=goldPos.get(i),sign=Math.sign(p[0]),attachment=[sign*.0733262085141611,1.407303855,.013881654972252433];const moved=earWarp(attachment);
  goldPos.set(i,p.map((v,k)=>moved[k]+(v-attachment[k])*.88));}bounds(gold.attributes.POSITION);

// A full turn exposes the lower-back shirt/cardigan contact missed by the
// earlier front-only fit. Preserve the seam and expand only this back region.
const coat=g.meshes[1].primitives.find(p=>p.material===13),coatPos=accessor(coat.attributes.POSITION),coatNormal=accessor(coat.attributes.NORMAL);
const backWarp=([x,y,z])=>[x,y,z-.030*bell(y,.965,.20)*bell(x,0,.28)*smooth((-.02-z)/.04)];
for(const i of new Set(all(coat.indices).flat())){const p=coatPos.get(i),n=coatNormal.get(i);coatPos.set(i,backWarp(p));coatNormal.set(i,normalAt(p,n,backWarp));}bounds(coat.attributes.POSITION);

// Each authored lock consists of two surface islands and two cap islands.
// Reconstruct its centreline, then split its width into finer tapered sections.
const oldHair=g.meshes[2].primitives[0],hp=all(oldHair.attributes.POSITION),hu=all(oldHair.attributes.TEXCOORD_0),hj=all(oldHair.attributes.JOINTS_0),hw=all(oldHair.attributes.WEIGHTS_0);
const groups=components(all(oldHair.indices).flat());assert.equal(groups.length,256);
const bodyPrimitive=g.meshes[1].primitives[0],bodyGeometry=new BufferGeometry();
bodyGeometry.setAttribute('position',new Float32BufferAttribute(all(bodyPrimitive.attributes.POSITION).flat(),3));
bodyGeometry.setIndex(g.meshes[1].primitives.flatMap(p=>all(p.indices).flat()));
const bodySurface=new Mesh(bodyGeometry,new MeshBasicMaterial({side:DoubleSide}));bodySurface.updateMatrixWorld();const ray=new Raycaster();
const pos=[],norm=[],uv=[],joints=[],weights=[],indices=[],lockStats=[];
function influences(rows,t){const index=t*(rows.length-1),a=Math.floor(index),b=Math.min(a+1,rows.length-1),f=index-a,map=new Map();
  for(const [row,scale] of [[rows[a],1-f],[rows[b],f]])for(const [joint,weight] of row.weights)map.set(joint,(map.get(joint)??0)+weight*scale);
  const top=[...map].filter(([,w])=>w>1e-7).sort((a,b)=>b[1]-a[1]).slice(0,4),sum=top.reduce((s,[,w])=>s+w,0);while(top.length<4)top.push([0,0]);
  return [top.map(([j])=>j),top.map(([,w])=>w/sum)];}
for(let group=0;group<groups.length;group+=4){
  assert.ok(groups[group].length>30&&groups[group+1].length>30&&groups[group+2].length<=4&&groups[group+3].length<=4);
  const rowMap=new Map();for(const i of [...groups[group],...groups[group+1]]){const key=hu[i][1].toFixed(5);if(!rowMap.has(key))rowMap.set(key,[]);rowMap.get(key).push(i);}
  const rows=[...rowMap].sort((a,b)=>Number(a[0])-Number(b[0])).map(([,ids])=>{
    const unique=[...new Map(ids.map(i=>[hp[i].map(v=>v.toFixed(7)).join(','),hp[i]])).values()];
    const center=new Vector3(...[0,1,2].map(k=>unique.reduce((sum,p)=>sum+p[k],0)/unique.length));let width=0;
    for(const a of unique)for(const b of unique)width=Math.max(width,Math.hypot(...a.map((v,k)=>v-b[k])));
    const inf=new Map();for(const i of ids)hj[i].forEach((j,k)=>inf.set(j,(inf.get(j)??0)+hw[i][k]/ids.length));return {center,width,weights:inf};});
  const long=Math.min(...rows.map(r=>r.center.y))<1.3,front=rows.at(-1).center.z>0,maxWidth=Math.max(...rows.map(r=>r.width));
  const sections=Math.min(4,Math.max(1,Math.ceil(maxWidth/.014))),arc=[0];
  rows.forEach((r,i)=>{if(i)arc.push(arc[i-1]+r.center.distanceTo(rows[i-1].center));});
  const centers=rows.map((r,i)=>{const sigma=long?.042:.009,average=new Vector3();let total=0;
    rows.forEach((s,k)=>{const w=Math.exp(-.5*((arc[k]-arc[i])/sigma)**2);average.addScaledVector(s.center,w);total+=w;});
    return r.center.clone().lerp(average.divideScalar(total),smooth((1.545-r.center.y)/.075)*.85);});
  const curve=new CatmullRomCurve3(centers,false,'centripetal'),segments=long?56:36,sides=6;
  for(let section=0;section<sections;section++){
    const base=pos.length,end=1-(section/(sections||1))*(long?.065:.035),offset=(section+.5)/sections-.5;
    for(let row=0;row<=segments;row++){
      const t=row/segments,s=t*end,c=curve.getPoint(s),tangent=curve.getTangent(Math.min(.99999,Math.max(.00001,s))).normalize();
      const radial=new Vector3(c.x,Math.max(0,c.y-1.46),c.z+.027).normalize(),axis=radial.clone().cross(tangent).normalize(),normal=tangent.clone().cross(axis).normalize();
      const index=s*(rows.length-1),a=Math.floor(index),b=Math.min(a+1,rows.length-1),f=index-a;
      const width=rows[a].width*(1-f)+rows[b].width*f;
      c.addScaledVector(axis,offset*width*.97);
      c.addScaledVector(normal,Math.sin(t*Math.PI)*(.00035+.0002*(section%2)));
      const tip=Math.pow(smooth((1-t)/.18),.7),radius=Math.max(.000035,width/sections*.49*tip),depth=Math.min(.0012,Math.max(.00006,radius*(long?.13:.2)));
      // Project front locks outside the existing fitted clothing in the rest pose.
      if(long&&front){c.z=Math.max(c.z,.075*smooth((1.46-c.y)/.13));
        if(c.y<1.4){ray.set(new Vector3(c.x,c.y,1),new Vector3(0,0,-1));const hit=ray.intersectObject(bodySurface)[0];if(hit)c.z=Math.max(c.z,hit.point.z+depth+.004);}}
      const [js,ws]=influences(rows,s);
      for(let side=0;side<=sides;side++){
        const angle=side/sides*Math.PI*2,cs=Math.cos(angle),sn=Math.sin(angle);
        pos.push(c.clone().addScaledVector(axis,cs*radius).addScaledVector(normal,sn*depth).toArray());
        norm.push(axis.clone().multiplyScalar(cs/radius).addScaledVector(normal,sn/depth).normalize().toArray());
        uv.push([side/sides*.12+(group*.137+section*.193)%1,t]);joints.push(js);weights.push(ws);
      }
    }
    for(let row=0;row<segments;row++)for(let side=0;side<sides;side++){const a=base+row*(sides+1)+side,b=a+sides+1;indices.push([a,a+1,b],[a+1,b+1,b]);}
  }
  lockStats.push({source:group/4,sections,long,maxWidth});
}
bodyGeometry.dispose();bodySurface.material.dispose();
oldHair.attributes={POSITION:append(pos,'VEC3'),NORMAL:append(norm,'VEC3'),TEXCOORD_0:append(uv,'VEC2'),JOINTS_0:append(joints,'VEC4',5123),WEIGHTS_0:append(weights,'VEC4')};
oldHair.indices=append(indices.flat().map(i=>[i]),'SCALAR',pos.length>65535?5125:5123,34963);
g.meshes[2].name='Ari fine tapered silk locks';
for(const spring of g.extensions.VRMC_springBone.springs.filter(s=>s.name==='Hair')){
  // World-space inertia lets hair lag behind a whole-body turn. A rotating
  // model-root centre would rotate yesterday's tail positions along with Ari.
  delete spring.center;
  spring.joints.forEach((joint,i)=>{if(joint.stiffness===undefined)return;const t=i/Math.max(1,spring.joints.length-2);
    joint.stiffness=.85-.40*t;joint.dragForce=.34-.10*t;joint.hitRadius=Math.min(joint.hitRadius,.006);});
}
// Refit the exported spring colliders to the current body and cardigan. The
// original VRoid spheres ended several centimetres inside the fitted chest.
const parents=new Map(),worlds=new Map();g.nodes.forEach((n,i)=>(n.children??[]).forEach(c=>parents.set(c,i)));
function world(i){if(worlds.has(i))return worlds.get(i);const n=g.nodes[i],m=new Matrix4().compose(new Vector3(...n.translation??[0,0,0]),new Quaternion(...n.rotation??[0,0,0,1]),new Vector3(...n.scale??[1,1,1]));if(parents.has(i))m.premultiply(world(parents.get(i)));worlds.set(i,m);return m;}
const springBone=g.extensions.VRMC_springBone,bodyColliders=[];
function capsule(bone,a,b,radius){const node=g.extensions.VRMC_vrm.humanoid.humanBones[bone].node,inverse=world(node).clone().invert();
  bodyColliders.push(springBone.colliders.push({node,shape:{capsule:{offset:new Vector3(...a).applyMatrix4(inverse).toArray(),tail:new Vector3(...b).applyMatrix4(inverse).toArray(),radius}}})-1);}
capsule('upperChest',[-.045,1.16,.025],[.045,1.16,.025],.149);
capsule('spine',[0,1.005,.014],[0,1.10,.014],.139);
capsule('upperChest',[-.115,1.278,-.018],[.115,1.278,-.018],.073);
for(const side of [-1,1]){const name=side===1?'left':'right';
  capsule(`${name}UpperArm`,[side*.12,1.272,-.025],[side*.32,1.272,-.025],.085);
  capsule(`${name}LowerArm`,[side*.33,1.272,-.025],[side*.51,1.272,-.025],.079);}
const group=springBone.colliderGroups.push({name:'Ari fitted clothing contact',colliders:bodyColliders})-1;
for(const spring of springBone.springs.filter(s=>s.name==='Hair'))spring.colliderGroups=[3,2,group];
for(const [texture,file] of [[3,'iris-v8-jade-silver.png'],[19,'hair-v8-silk.png']]){const image=g.images[g.textures[texture].source];views[image.bufferView]=await readFile(new URL(`design/textures/${file}`,root));}
const hair=g.materials[14];hair.emissiveFactor=[0,0,0];delete hair.emissiveTexture;delete hair.normalTexture;
hair.extras={...hair.extras,ariHairContact:true};
const toon=hair.extensions.VRMC_materials_mtoon;delete toon.matcapTexture;toon.parametricRimColorFactor=[.003,.004,.005];toon.parametricRimFresnelPowerFactor=5;toon.parametricRimLiftFactor=0;toon.shadingToonyFactor=.3;toon.shadeColorFactor=[.72,.75,.78];
// Fine painted fibers must use mipmaps when reduced to a portrait. The source
// VRoid sampler uses linear filtering without mipmaps, which shimmers badly.
for(const texture of [3,19,g.materials[13].extensions.VRMC_materials_mtoon.shadeMultiplyTexture.index])g.samplers[g.textures[texture].sampler].minFilter=9987;
g.extensions.VRMC_vrm.meta.version='0.8';g.asset.generator+=' + Ari v8 small ears and silk hair';
let offset=0;const parts=[];views.forEach((b,i)=>{const pad=(4-offset%4)%4;parts.push(Buffer.alloc(pad),b);offset+=pad;g.bufferViews[i].byteOffset=offset;g.bufferViews[i].byteLength=b.length;offset+=b.length;});
g.buffers[0].byteLength=offset;parts.push(Buffer.alloc((4-offset%4)%4));const packed=Buffer.concat(parts),j=Buffer.from(JSON.stringify(g)),json=Buffer.alloc(Math.ceil(j.length/4)*4,0x20);j.copy(json);
const output=Buffer.alloc(28+json.length+packed.length);output.write('glTF');output.writeUInt32LE(2,4);output.writeUInt32LE(output.length,8);output.writeUInt32LE(json.length,12);output.writeUInt32LE(0x4e4f534a,16);json.copy(output,20);output.writeUInt32LE(packed.length,20+json.length);output.writeUInt32LE(0x004e4942,24+json.length);packed.copy(output,28+json.length);
await writeFile(new URL('model/ari-vroid-v8.vrm',root),output);
console.log(JSON.stringify({sha256:hash(output),bytes:output.length,hairVertices:pos.length,hairTriangles:indices.length,sourceLocks:lockStats.length,fineLocks:lockStats.reduce((s,l)=>s+l.sections,0)},null,2));
