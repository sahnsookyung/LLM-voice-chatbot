import {readFile, writeFile} from 'node:fs/promises';
import {createHash} from 'node:crypto';
import {Matrix3, Vector3} from '../renderer/node_modules/three/build/three.module.js';

// Offline tailoring only. The raw VRoid export remains the reproducible source;
// no per-frame deformation or additional geometry is needed in the companion.
const root = new URL('../characters/ari/model/', import.meta.url);
const raw = await readFile(new URL('ari-vroid-v5-raw.vrm', root));
if (createHash('sha256').update(raw).digest('hex') !== '62f35a6c3be2fdb26069f2e2ebdcd00910f6720e7967793bc3dfb60e704a5694') {
  throw new Error('Raw export changed: inspect the new mesh and revise the tailoring profile before rebuilding.');
}
const jsonLength = raw.readUInt32LE(12);
const gltf = JSON.parse(raw.toString('utf8', 20, 20 + jsonLength));
const bin = Buffer.from(raw.subarray(28 + jsonLength));
if (gltf.extensions.VRMC_vrm.meta.version !== '0.5' || gltf.meshes[1].name !== 'Body (merged)') {
  throw new Error('This tailoring profile requires Ari’s original v5 export.');
}
function accessor(index) {
  const a = gltf.accessors[index], v = gltf.bufferViews[a.bufferView];
  const components = {SCALAR:1,VEC3:3}[a.type];
  const bytes = {5123:2,5125:4,5126:4}[a.componentType];
  if (!components || !bytes || a.sparse) throw new Error('Unsupported accessor');
  const offset = i => (v.byteOffset ?? 0) + (a.byteOffset ?? 0) + i * (v.byteStride ?? bytes * components);
  const read = {5123:'readUInt16LE',5125:'readUInt32LE',5126:'readFloatLE'}[a.componentType];
  return {a, get: i => Array.from({length:components}, (_,j) => bin[read](offset(i)+j*bytes)),
    set: (i, values) => values.forEach((value,j) => bin.writeFloatLE(value,offset(i)+j*bytes))};
}
const body = gltf.meshes[1];
const position = accessor(body.primitives[0].attributes.POSITION);
const normal = accessor(body.primitives[0].attributes.NORMAL);
const materialVertices = name => {
  const primitive = body.primitives.find(p => gltf.materials[p.material].name.includes(name));
  if (!primitive) throw new Error(`Missing ${name}`);
  const indices = accessor(primitive.indices);
  return new Set(Array.from({length:indices.a.count},(_,i)=>indices.get(i)[0]));
};
const cardigan = materialVertices('Tops_01_CLOTH');
const torso = new Set([...materialVertices('Body_00_SKIN'),...materialVertices('Bottoms_01_CLOTH'),
  ...materialVertices('Onepiece_00_CLOTH'),...cardigan]);
const bell = (value, center, radius) => {
  const t = Math.abs((value-center)/radius);
  return t < 1 ? (1-t*t)**3 : 0; // Smooth value and slope at the support boundary.
};
function deform([x,y,z], outer) {
  const waist = bell(y,1.055,.17), hip = bell(y,.875,.17);
  const central = bell(x,0,.28);
  const width = 1 + central * (-.09 * waist + .045 * hip);
  let nx = x*width, nz = .02 + (z-.02)*(1-central*.035*waist);
  if (outer) {
    nx *= 1 - .055 * waist * central;
    // Separate the cardigan from the fitted knit over chest and pocket seams.
    // Back/sleeves are excluded; the envelope is smooth across sewn UV seams.
    const front = Math.max(0, Math.min(1,(z+.005)/.055));
    nz += front * bell(x,0,.25) * (.033*bell(y,1.17,.16)+.021*bell(y,1.015,.16));
  }
  return [nx,y,nz];
}
let edited = 0, maximumMove = 0;
const epsilon = 1e-5;
for (const i of torso) {
  const p = position.get(i), next = deform(p,cardigan.has(i));
  const distance = Math.hypot(...p.map((v,j)=>v-next[j]));
  if (distance < 1e-9) continue;
  // Inverse-transpose Jacobian preserves authored smoothing and split normals.
  const columns = p.map((_,axis) => {
    const lo=p.slice(), hi=p.slice();lo[axis]-=epsilon;hi[axis]+=epsilon;
    const a=deform(lo,cardigan.has(i)),b=deform(hi,cardigan.has(i));
    return b.map((v,j)=>(v-a[j])/(2*epsilon));
  });
  const jacobian = new Matrix3().set(...[0,1,2].flatMap(row=>columns.map(c=>c[row])));
  if (jacobian.determinant() <= 0) throw new Error('Tailoring inverted geometry');
  const n = new Vector3(...normal.get(i)).applyMatrix3(jacobian.invert().transpose()).normalize();
  position.set(i,next);normal.set(i,n.toArray());edited++;maximumMove=Math.max(maximumMove,distance);
}
position.a.min = [0,1,2].map(axis=>Math.min(...Array.from({length:position.a.count},(_,i)=>position.get(i)[axis])));
position.a.max = [0,1,2].map(axis=>Math.max(...Array.from({length:position.a.count},(_,i)=>position.get(i)[axis])));
gltf.asset.generator += ' + Ari v5 offline garment tailoring';
const text = Buffer.from(JSON.stringify(gltf));
const json = Buffer.alloc(Math.ceil(text.length/4)*4,0x20);text.copy(json);
const output = Buffer.alloc(28+json.length+bin.length);
output.write('glTF');output.writeUInt32LE(2,4);output.writeUInt32LE(output.length,8);
output.writeUInt32LE(json.length,12);output.writeUInt32LE(0x4e4f534a,16);json.copy(output,20);
output.writeUInt32LE(bin.length,20+json.length);output.writeUInt32LE(0x004e4942,24+json.length);bin.copy(output,28+json.length);
await writeFile(new URL('ari-vroid-v5.vrm',root),output);
console.log(JSON.stringify({editedVertices:edited,maximumMoveMetres:maximumMove,bytes:output.length,
  sourceSha256:createHash('sha256').update(raw).digest('hex'),sha256:createHash('sha256').update(output).digest('hex')},null,2));
