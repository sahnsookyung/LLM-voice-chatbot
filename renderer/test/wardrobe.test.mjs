import test from 'node:test';
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import {Matrix4,Quaternion,Vector3} from 'three';
import {buildWardrobeModel} from '../.test-build/wardrobe-model.js';
import {parseWardrobe,restoreWardrobe,wardrobeCombinations,wardrobeKey} from '../.test-build/wardrobe.js';
const raw=readFileSync(new URL('../../characters/ari/model/ari-vroid-v9.vrm',import.meta.url));
const original=raw.buffer.slice(raw.byteOffset,raw.byteOffset+raw.byteLength);
function read(buffer){const v=new DataView(buffer),size=v.getUint32(12,true),g=JSON.parse(new TextDecoder().decode(new Uint8Array(buffer,20,size))),bin=new Uint8Array(buffer,28+size);
  const bytes=index=>{const a=g.accessors[index],b=g.bufferViews[a.bufferView];return bin.subarray(b.byteOffset??0,(b.byteOffset??0)+b.byteLength);};
  const values=index=>{const a=g.accessors[index],b=g.bufferViews[a.bufferView],w={SCALAR:1,VEC2:2,VEC3:3,VEC4:4,MAT4:16}[a.type],width=a.componentType===5123?2:4,data=new DataView(bin.buffer,bin.byteOffset);return Array.from({length:a.count},(_,i)=>Array.from({length:w},(_,k)=>{const o=(b.byteOffset??0)+(a.byteOffset??0)+i*(b.byteStride??w*width)+k*width;return a.componentType===5126?data.getFloat32(o,true):a.componentType===5123?data.getUint16(o,true):data.getUint32(o,true);}));};
  return {g,bytes,values};}
const source=read(original);
function worldMatrices(g){const worlds=[];
  function walk(i,parent){const n=g.nodes[i],world=parent.clone().multiply(n.matrix?new Matrix4().fromArray(n.matrix):new Matrix4().compose(new Vector3(...(n.translation??[0,0,0])),new Quaternion(...(n.rotation??[0,0,0,1])),new Vector3(...(n.scale??[1,1,1]))));worlds[i]=world;for(const c of n.children??[])walk(c,world);}
  for(const n of g.scenes[g.scene??0].nodes)walk(n,new Matrix4());return worlds;}
const originalWorlds=worldMatrices(source.g);
test('stockings wrap both calf surfaces and short skirts have a longer hem',()=>{
  const {g,values}=read(buildWardrobeModel(original,{outfit:'navy-blouse',hair:'original'}));
  const socks=g.meshes.find(m=>m.name==='Ari knee socks'),points=values(socks.primitives[0].attributes.POSITION);
  const body=source.g.meshes[1].primitives.find(p=>p.material===8),bodyPoints=source.values(body.attributes.POSITION),normals=source.values(body.attributes.NORMAL);
  const actual=new Set(points.map(p=>p.map(v=>v.toFixed(5)).join(',')));
  for(const i of new Set(source.values(body.indices).flat())){
    const p=bodyPoints[i];if(p[1]>.12&&p[1]<.46)assert.ok(actual.has(p.map((v,k)=>(v+normals[i][k]*.0023).toFixed(5)).join(',')),'sock retains the fitted front AND back calf vertices');
  }
  assert.ok(points.some(p=>p[2]<-.06)&&points.some(p=>p[2]>.02));
  const skirt=g.meshes.find(m=>m.name==='Ari pleated skirt'),hem=Math.min(...values(skirt.primitives[0].attributes.POSITION).map(p=>p[1]));
  assert.ok(hem>.60&&hem<.63,'6 cm lower hem stays above the knees');
});
test('independent wardrobe state covers the screenshot styles and survives persistence',()=>{
  const pairs=wardrobeCombinations();assert.equal(pairs.length,135);assert.equal(new Set(pairs.map(wardrobeKey)).size,135);
  for(const value of pairs)assert.deepEqual(restoreWardrobe(JSON.stringify(value)),value);
  assert.deepEqual(restoreWardrobe('broken'),{outfit:'original',hair:'original'});
  assert.throws(()=>parseWardrobe({outfit:'bob',hair:'linen'}));
  assert.throws(()=>buildWardrobeModel(original,{outfit:'unknown',hair:'bob'}));
  assert.equal(buildWardrobeModel(original,{outfit:'original',hair:'original'}),original);
});
for(const selection of wardrobeCombinations())test(`3D ${wardrobeKey(selection)} preserves Ari's face and has valid skinned parts`,()=>{
  const {g,bytes,values}=read(buildWardrobeModel(original,selection));
  assert.deepEqual(g.meshes[0],source.g.meshes[0]);
  for(const p of g.meshes[0].primitives)for(const index of [...Object.values(p.attributes),p.indices,...(p.targets??[]).flatMap(t=>Object.values(t))])assert.deepEqual(bytes(index),source.bytes(index),'face and expression data must be byte-identical');
  assert.deepEqual(g.extensions.VRMC_vrm,source.g.extensions.VRMC_vrm);
  assert.deepEqual(g.materials.slice(0,9),source.g.materials.slice(0,9),'face, eyes and skin materials stay unchanged');
  const skin=g.meshes[1].primitives.find(p=>p.material===8),oldSkin=source.g.meshes[1].primitives.find(p=>p.material===8),positions=values(skin.attributes.POSITION),oldPositions=source.values(oldSkin.attributes.POSITION);
  for(const [i]of values(skin.indices))assert.deepEqual(positions[i],oldPositions[i],'the actual body geometry stays unchanged');
  const human=Object.values(g.extensions.VRMC_vrm.humanoid.humanBones).map(b=>b.node);
  for(const index of human){const before=source.g.nodes[index],after=g.nodes[index];assert.deepEqual(after.translation,before.translation);assert.deepEqual(after.rotation,before.rotation);}
  const worlds=worldMatrices(g);
  for(const [s,skin]of g.skins.entries()){const inverse=values(skin.inverseBindMatrices),oldInverse=source.values(source.g.skins[s].inverseBindMatrices);assert.equal(inverse.length,skin.joints.length);skin.joints.forEach((node,i)=>{
    const actual=worlds[node].clone().multiply(new Matrix4().fromArray(inverse[i]));
    // Existing export bindings contain authored offsets. Preserve those exactly;
    // newly added joints must be identity at rest.
    const expected=i<oldInverse.length?originalWorlds[node].clone().multiply(new Matrix4().fromArray(oldInverse[i])):new Matrix4();
    assert.ok(actual.elements.every((v,k)=>Math.abs(v-expected.elements[k])<3e-5),`bad bind ${g.nodes[node].name}`);
  });}
  for(const node of g.nodes.filter(n=>n.mesh!==undefined))for(const p of g.meshes[node.mesh].primitives){
    const count=g.accessors[p.attributes.POSITION].count,skin=g.skins[node.skin];
    assert.ok(values(p.indices).every(([i])=>i<count));
    assert.ok(values(p.attributes.POSITION).every(p=>p.every(Number.isFinite)));
    assert.ok(values(p.attributes.JOINTS_0).every(p=>p.every(i=>i<skin.joints.length)));
    assert.ok(values(p.attributes.WEIGHTS_0).every(p=>p.every(v=>v>=0&&v<=1)&&Math.abs(p.reduce((a,b)=>a+b,0)-1)<1e-5));
  }
  const spring=g.extensions.VRMC_springBone,used=new Set();
  for(const s of spring.springs){for(const j of s.joints){assert.ok(g.nodes[j.node]);assert.ok(!used.has(j.node),'duplicate spring joint');used.add(j.node);}for(const i of s.colliderGroups??[])assert.ok(spring.colliderGroups[i]);}
  if(selection.outfit==='linen')assert.ok(g.meshes.some(m=>m.name==='Ari sage linen skirt'));
  if(selection.hair==='bob')assert.ok(Math.min(...values(g.meshes[2].primitives[0].attributes.POSITION).map(p=>p[1]))>1.30);
  if(selection.hair==='ponytail')assert.ok(spring.springs.some(s=>s.name==='HairTie'));
  if(['braid','twintails'].includes(selection.hair)){
    const ribbon=g.meshes.find(m=>m.name==='Ari hair ribbons');assert.ok(ribbon,'braids must have a fastening ribbon');
    const rp=ribbon.primitives[0],ribbonWeights=values(rp.attributes.WEIGHTS_0),bowPositions=values(rp.attributes.POSITION),hairPositions=values(g.meshes[2].primitives[0].attributes.POSITION);
    assert.ok(Math.min(...hairPositions.map(p=>p[1]))<Math.min(...bowPositions.map(p=>p[1]))-.025,'loose tapered hair must extend past the ribbon');
    assert.ok(values(rp.attributes.JOINTS_0).every((j,i)=>j.some((v,k)=>ribbonWeights[i][k]>0&&g.skins[2].joints[v]>=source.g.nodes.length)),'ribbon follows the new braid rig');
    if(selection.hair==='twintails'){
      const left=spring.springs.find(s=>s.name==='HairTwinLeft'),right=spring.springs.find(s=>s.name==='HairTwinRight');assert.ok(left&&right);
      assert.ok(left.joints.every(j=>!right.joints.some(r=>r.node===j.node)),'two tails have independent guides');
      assert.ok(bowPositions.some(p=>p[0]>.07)&&bowPositions.some(p=>p[0]<-.07),'each tail has a ribbon');
    }else assert.ok(spring.springs.some(s=>s.name==='HairBraid'));
    for(const chain of spring.springs.filter(s=>s.name==='HairBraid'||s.name.startsWith('HairTwin'))){
      const first=chain.joints[0],last=chain.joints.at(-2);
      assert.ok(last.gravityPower>last.stiffness&&first.stiffness>last.stiffness,'braid lengths hang under gravity with firmer roots');
      const bottom=new Vector3().setFromMatrixPosition(worlds[chain.joints.at(-1).node]);
      // Above the bust the braid follows the shoulder/clothing contour. Below
      // it, the free length should hang rather than continue flaring outward.
      for(const upper of chain.joints.map(j=>new Vector3().setFromMatrixPosition(worlds[j.node])).filter(p=>p.y<1.18))
        assert.ok(Math.abs(bottom.x-upper.x)<.025&&Math.abs(bottom.z-upper.z)<.04,'the lower rest guide hangs vertically instead of flaring sideways');
    }
  }
});
