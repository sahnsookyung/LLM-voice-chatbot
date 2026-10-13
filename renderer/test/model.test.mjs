import test from 'node:test';
import assert from 'node:assert/strict';
import {readFile} from 'node:fs/promises';
import {BufferGeometry,Float32BufferAttribute,Mesh,MeshBasicMaterial,DoubleSide,Raycaster,Vector3,Matrix4,Quaternion} from 'three';

const raw = await readFile(new URL('../../characters/ari/model/ari-vroid-v6.vrm', import.meta.url));
assert.equal(raw.toString('ascii', 0, 4), 'glTF');
const jsonSize = raw.readUInt32LE(12);
const gltf = JSON.parse(raw.toString('utf8', 20, 20 + jsonSize));
const bin = raw.subarray(28 + jsonSize);
const vrm = gltf.extensions.VRMC_vrm;
function values(index) {
  const a = gltf.accessors[index], view = gltf.bufferViews[a.bufferView];
  const widths = {SCALAR:1, VEC2:2, VEC3:3, VEC4:4, MAT4:16};
  const types = {5121:[1,'readUInt8'],5123:[2,'readUInt16LE'],5125:[4,'readUInt32LE'],5126:[4,'readFloatLE']};
  const [bytes, read] = types[a.componentType], width = widths[a.type];
  return Array.from({length:a.count}, (_, i) => Array.from({length:width}, (_, j) =>
    bin[read]((view.byteOffset ?? 0) + (a.byteOffset ?? 0) + i * (view.byteStride ?? bytes * width) + j * bytes)));
}

test('VRoid asset is self-contained, textured and within the geometry budget', () => {
  assert.equal(vrm.specVersion, '1.0');
  assert.ok(gltf.asset.generator.startsWith('VRoid Studio'));
  assert.ok(gltf.buffers.every(b => !b.uri));
  assert.ok(gltf.images.every(i => i.bufferView !== undefined && !i.uri));
  assert.ok(gltf.materials.every(m => m.extensions.VRMC_materials_mtoon));
  let triangles = 0;
  for (const mesh of gltf.meshes) for (const p of mesh.primitives) {
    triangles += gltf.accessors[p.indices].count / 3;
    for (const point of values(p.attributes.POSITION)) assert.ok(point.every(Number.isFinite));
  }
  assert.ok(triangles > 30000 && triangles < 70000);
  assert.ok(raw.length < 20 * 1024 * 1024);
});

test('humanoid includes distinct articulated fingers and valid expression bindings', () => {
  const bones = vrm.humanoid.humanBones;
  assert.equal(Object.keys(bones).length, 54);
  for (const side of ['left', 'right']) {
    const hand = bones[`${side}Hand`].node;
    for (const finger of ['ThumbMetacarpal','IndexProximal','MiddleProximal','RingProximal','LittleProximal']) {
      assert.ok(gltf.nodes[hand].children.includes(bones[`${side}${finger}`].node));
    }
  }
  const presets = vrm.expressions.preset;
  for (const name of ['happy','angry','sad','surprised','aa','ou','blinkLeft','blinkRight']) {
    assert.ok(presets[name].morphTargetBinds.length > 0, name);
    for (const bind of presets[name].morphTargetBinds) {
      const mesh = gltf.meshes[gltf.nodes[bind.node].mesh];
      assert.ok(mesh.primitives.every(p => p.targets[bind.index]));
      assert.ok(bind.weight > 0 && bind.weight <= 1);
    }
  }
});

test('skin weights are normalized and joint indices reference the exported skeleton', () => {
  for (const node of gltf.nodes.filter(n => n.skin !== undefined)) {
    const joints = gltf.skins[node.skin].joints;
    assert.ok(joints.every(j => !!gltf.nodes[j]));
    for (const p of gltf.meshes[node.mesh].primitives) {
      for (const w of values(p.attributes.WEIGHTS_0)) {
        assert.ok(w.every(v => Number.isFinite(v) && v >= 0));
        assert.ok(Math.abs(w.reduce((a,b) => a+b,0)-1) < .0001);
      }
      for (const j of values(p.attributes.JOINTS_0)) assert.ok(j.every(v => v < joints.length));
    }
  }
});

test('hair springs contain valid joints and collision references', () => {
  const springs = gltf.extensions.VRMC_springBone;
  assert.ok(springs.springs.length > 0);
  for (const spring of springs.springs) {
    assert.ok(spring.joints.length >= 2);
    assert.ok(spring.joints.every(j => !!gltf.nodes[j.node]));
    for (const group of spring.colliderGroups ?? []) assert.ok(springs.colliderGroups[group]);
  }
});

test('offline tailoring preserves topology, face morphs, textures and rig data', async () => {
  const source = await readFile(new URL('../../characters/ari/model/ari-vroid-v6-raw.vrm', import.meta.url));
  const size = source.readUInt32LE(12);
  const original = JSON.parse(source.toString('utf8',20,20+size));
  const sourceBin = source.subarray(28+size);
  for (const key of ['images','textures','skins']) assert.deepEqual(gltf[key],original[key]);
  assert.deepEqual(gltf.meshes.slice(0,3),original.meshes,'authored topology and face morph bindings');
  assert.deepEqual(gltf.extensions.VRMC_vrm,original.extensions.VRMC_vrm,'facial expressions and humanoid metadata');
  const hairNodes=new Set(original.extensions.VRMC_springBone.springs.filter(s=>s.name==='Hair').flatMap(s=>s.joints.map(j=>j.node)));
  original.nodes.forEach((node,i)=>{if(!hairNodes.has(i))assert.deepEqual(gltf.nodes[i],node,`non-hair bone ${i}`);});
  assert.deepEqual(gltf.materials.slice(0,14),original.materials.slice(0,14));
  const attributes = gltf.meshes[1].primitives[0].attributes;
  const editedViews = new Set([attributes.POSITION,attributes.NORMAL,138,139,...gltf.skins.map(s=>s.inverseBindMatrices)].map(i=>gltf.accessors[i].bufferView));
  original.bufferViews.forEach((view,i)=>{
    if (editedViews.has(i)) return;
    const start=view.byteOffset??0,end=start+view.byteLength;
    assert.ok(bin.subarray(start,end).equals(sourceBin.subarray(start,end)),`unexpected change in buffer ${i}`);
  });
  const a=gltf.accessors[attributes.POSITION],view=gltf.bufferViews[a.bufferView];
  let changes=0;
  for (const [i,point] of values(attributes.POSITION).entries()) {
    const offset=(view.byteOffset??0)+(a.byteOffset??0)+i*(view.byteStride??12);
    const before=[0,4,8].map(j=>sourceBin.readFloatLE(offset+j));
    const distance=Math.hypot(...point.map((v,j)=>v-before[j]));
    assert.ok(distance<.035,'bounded local tailoring');
    assert.equal(point[1],before[1],'height and limb length are preserved');
    if(distance>1e-8)changes++;
  }
  assert.ok(changes>2000,'tailoring must actually be present');
  for (const n of values(attributes.NORMAL)) assert.ok(Math.abs(Math.hypot(...n)-1)<.001);
});

test('neutral cardigan clears the inner knit over both chest and pocket seams', () => {
  const surface = name => {
    const p=gltf.meshes[1].primitives.find(p=>gltf.materials[p.material].name.includes(name));
    const geometry=new BufferGeometry();
    geometry.setAttribute('position',new Float32BufferAttribute(values(p.attributes.POSITION).flat(),3));
    geometry.setIndex(values(p.indices).flat());
    const mesh=new Mesh(geometry,new MeshBasicMaterial({side:DoubleSide}));
    mesh.updateMatrixWorld();return mesh;
  };
  const inner=surface('Onepiece_00_CLOTH'),outer=surface('Tops_01_CLOTH');
  const ray=new Raycaster();let samples=0;
  try {
    for(const side of [-1,1])for(let x=.055;x<=.125;x+=.005)for(let y=1.01;y<=1.25;y+=.005){
      ray.set(new Vector3(x*side,y,1),new Vector3(0,0,-1));
      const shirt=ray.intersectObject(inner)[0],coat=ray.intersectObject(outer)[0];
      if(!shirt||!coat)continue;
      assert.ok(coat.point.z-shirt.point.z>.005,`garment overlap at ${x*side}, ${y}`);
      samples++;
    }
    assert.ok(samples>1000,'cover both sides of the chest and upper pocket region');
  } finally {
    for(const mesh of [inner,outer]){mesh.geometry.dispose();mesh.material.dispose();}
  }
});


test('longer hair retains valid rest skinning and preserves roots and face', async () => {
  const source=await readFile(new URL('../../characters/ari/model/ari-vroid-v6-raw.vrm',import.meta.url));
  const size=source.readUInt32LE(12),original=JSON.parse(source.toString('utf8',20,20+size)),sourceBin=source.subarray(28+size);
  const pos=values(138),normal=values(139),view=original.bufferViews[original.accessors[138].bufferView];
  let moved=0,roots=0;
  pos.forEach((point,i)=>{
    const before=[0,4,8].map(k=>sourceBin.readFloatLE(view.byteOffset+i*12+k));
    if(before[1]>=1.405){assert.deepEqual(point,before);roots++;}
    else if(Math.hypot(...point.map((v,k)=>v-before[k]))>.00001)moved++;
    assert.ok(Math.abs(Math.hypot(...normal[i])-1)<.001);
  });
  assert.ok(moved>5000 && roots>5000);
  assert.ok(Math.min(...pos.map(p=>p[1]))<1.08,'hair reaches the lower bust');
  const parents=new Map();gltf.nodes.forEach((n,i)=>(n.children??[]).forEach(c=>parents.set(c,i)));
  const cache=new Map();
  const world=(i,nodes=gltf.nodes,store=cache)=>{
    if(store.has(i))return store.get(i);
    const n=nodes[i],m=new Matrix4().compose(new Vector3(...n.translation??[0,0,0]),new Quaternion(...n.rotation??[0,0,0,1]),new Vector3(...n.scale??[1,1,1]));
    if(parents.has(i))m.premultiply(world(parents.get(i),nodes,store));store.set(i,m);return m;
  };
  const originalCache=new Map();
  for(const skin of gltf.skins){
    const binds=values(skin.inverseBindMatrices);
    const a=original.accessors[skin.inverseBindMatrices],v=original.bufferViews[a.bufferView];
    skin.joints.forEach((node,i)=>{
      const actual=world(node).clone().multiply(new Matrix4().fromArray(binds[i]));
      const oldBind=Array.from({length:16},(_,k)=>sourceBin.readFloatLE((v.byteOffset??0)+(a.byteOffset??0)+i*64+k*4));
      const expected=world(node,original.nodes,originalCache).clone().multiply(new Matrix4().fromArray(oldBind));
      actual.elements.forEach((value,k)=>assert.ok(Math.abs(value-expected.elements[k])<.000002,`rest skin matrix ${node}`));
    });
  }
});

test('mock-neck is skinned, outward-facing, and overlaps the shirt neckline', () => {
  const mesh=gltf.meshes.find(m=>m.name==='Ari fitted mock-neck');assert.ok(mesh);
  const p=mesh.primitives[0],pos=values(p.attributes.POSITION),norm=values(p.attributes.NORMAL);
  const original=values(127),joints=values(p.attributes.JOINTS_0),weights=values(p.attributes.WEIGHTS_0);
  assert.equal(gltf.materials[p.material].name,gltf.materials[11].name,'reuse the knit material');
  assert.equal(pos.length,97*7);assert.equal(values(p.indices).length,1152*3);
  for(let i=0;i<97;i+=4)assert.ok(original.some(v=>Math.abs(v[1]-.012-pos[i][1])<1e-6 && Math.hypot(v[0]-pos[i][0],v[2]-pos[i][2])<.025),'bounded overlap below shirt edge');
  for(let i=97*6;i<pos.length;i++){
    assert.ok(Math.abs(pos[i][1]-1.342)<1e-6);
    assert.equal(joints[i][0],17);assert.equal(weights[i][0],1,'collar follows neck');
    assert.ok(norm[i][0]*pos[i][0]+norm[i][2]*(pos[i][2]+.027)>0,'outward normal');
  }
  const skin=gltf.meshes[1].primitives.find(p=>p.material===8),geometry=new BufferGeometry();
  geometry.setAttribute('position',new Float32BufferAttribute(values(skin.attributes.POSITION).flat(),3));
  geometry.setIndex(values(skin.indices).flat());
  const body=new Mesh(geometry,new MeshBasicMaterial({side:DoubleSide}));body.updateMatrixWorld();
  const ray=new Raycaster();let samples=0;
  try {
    for(const [x,y,z] of pos){
      const direction=new Vector3(x,0,z+.027),radius=direction.length();
      ray.set(new Vector3(0,y,-.027),direction.normalize());
      const hit=ray.intersectObject(body)[0];
      if(hit){assert.ok(radius-hit.distance>.001,`collar clearance ${radius-hit.distance} at ${x}, ${y}, ${z}`);samples++;}
    }
    assert.ok(samples>600);
  } finally { geometry.dispose();body.material.dispose(); }
});
