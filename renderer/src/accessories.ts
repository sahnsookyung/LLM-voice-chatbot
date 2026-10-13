import {BufferGeometry,CatmullRomCurve3, CircleGeometry, Color, Float32BufferAttribute, Group, Mesh, MeshStandardMaterial, TorusGeometry, TubeGeometry, Vector3} from 'three';
import {mergeGeometries} from 'three/addons/utils/BufferGeometryUtils.js';
import type {VRM} from '@pixiv/three-vrm';
import {finishFabric} from './fabric.js';
import type {Hairstyle,Outfit} from './wardrobe.js';
import {beretPalette,type FabricColor} from './wardrobe-palette.js';

export type Accessories = {glasses:'none'|'round';headwear:'none'|'beret'};
export const defaultAccessories:Accessories={glasses:'none',headwear:'none'};
export function parseAccessories(value:unknown):Accessories {
  if(!value||typeof value!=='object')throw new Error('Invalid accessory selection.');
  const v=value as Accessories;
  if(!['none','round'].includes(v.glasses)||!['none','beret'].includes(v.headwear))throw new Error('Unknown accessory.');
  return {glasses:v.glasses,headwear:v.headwear};
}
export function compatibleAccessories(value:Accessories,hair:Hairstyle):Accessories {
  const parsed=parseAccessories(value);
  return hair==='double-buns'?{...parsed,headwear:'none'}:parsed;
}

/** Rigid accessories retain the head's bind transform and need no model reload. */
export class AccessoryRig {
  private root=new Group();
  private selection:Accessories={...defaultAccessories};
  constructor(vrm:VRM,private outfit:Outfit='original'){
    const head=vrm.humanoid.getRawBoneNode('head')!;
    this.root.name='Ari selectable accessories';this.root.matrix.copy(head.matrixWorld).invert();this.root.matrixAutoUpdate=false;head.add(this.root);
  }
  set(value:Accessories){
    const next=parseAccessories(value);
    if(next.glasses===this.selection.glasses&&next.headwear===this.selection.headwear)return;
    this.clear();this.selection=next;
    if(next.glasses==='round')this.glasses();
    if(next.headwear==='beret')this.beret();
  }
  palette(){return beretPalette(this.outfit);}
  private mesh(geometry:BufferGeometry,color:number|FabricColor,roughness=.65){
    const tint=Array.isArray(color)?new Color(...color):color;
    const mesh=new Mesh(geometry,new MeshStandardMaterial({color:tint,roughness,metalness:roughness<.5?.35:0}));mesh.frustumCulled=false;this.root.add(mesh);return mesh;
  }
  private glasses(){
    const parts:BufferGeometry[]=[];
    const pipe=(points:number[][],radius=.0015)=>new TubeGeometry(new CatmullRomCurve3(points.map(p=>new Vector3(...p as [number,number,number]))),32,radius,6,false);
    for(const sign of [-1,1]){
      const ring=new TorusGeometry(.026,.0017,6,64);ring.scale(1.04,.87,1);ring.translate(sign*.038,1.447,.079);parts.push(ring);
      parts.push(pipe([[sign*.065,1.452,.078],[sign*.075,1.452,.051],[sign*.082,1.448,-.017],[sign*.078,1.438,-.031]]));
      const lens=new CircleGeometry(.025,48);lens.scale(1.04,.87,1);lens.translate(sign*.038,1.447,.079);
      const glass=this.mesh(lens,0xc1e5dc,.25);const material=glass.material as MeshStandardMaterial;material.transparent=true;material.opacity=.055;material.depthWrite=false;
    }
    parts.push(pipe([[-.011,1.447,.079],[0,1.453,.084],[.011,1.447,.079]],.0014));
    const joined=mergeGeometries(parts)!;parts.forEach(p=>p.dispose());this.mesh(joined,0x30271f,.35);
  }
  private beret(){
    const palette=this.palette();
    // The opening wraps around the hairline, below the scalp apex. A folded
    // underside and asymmetric crown replace the old floating flattened sphere.
    const tilt=.38,cos=Math.cos(tilt),sin=Math.sin(tilt);
    const place=(x:number,h:number,z:number)=>new Vector3(x,1.521+h*cos+z*sin,-.024-h*sin+z*cos);
    // Cross-section in the opening's tilted plane: band, turned-under lip,
    // relaxed overhang, crown. Tilting the opening makes its front edge arch
    // over the hairline and lets the back wrap down around the head.
    const profile=new CatmullRomCurve3([[.085,0,.105],[.091,.007,.115],[.110,.020,.138],[.110,.048,.132],[.085,.078,.104],[.046,.092,.058],[0,.098,0]].map(p=>new Vector3(...p as [number,number,number])));
    const positions:number[]=[],uv:number[]=[],indices:number[]=[],rows=48,sides=128;
    for(let row=0;row<=rows;row++)for(let side=0;side<=sides;side++){
      const t=row/rows,a=side/sides*Math.PI*2,c=profile.getPoint(t);
      // Broad, uneven folds gather into a centered band. Avoid a regular
      // scallop or a laterally shifted dome, both of which read as a mushroom.
      const envelope=Math.sin(Math.PI*t)*Math.exp(-(((t-.35)/.45)**2));
      let fold=.0018*Math.sin(a*3+.7+t)*envelope;
      for(const [phase,width,depth]of [[.45,.24,.010],[1.8,.20,.007],[-2.5,.27,.009],[-.8,.22,.011]]){
        const distance=Math.atan2(Math.sin(a-phase-t*.12),Math.cos(a-phase-t*.12));
        fold-=depth*Math.exp(-((distance/width)**2))*envelope;
      }
      const x=Math.sin(a)*(Math.max(0,c.x)+fold),z=Math.cos(a)*(Math.max(0,c.z)+fold);
      positions.push(...place(x,c.y+fold*.65+.002*Math.cos(a*3+.4)*envelope,z).toArray());uv.push(side/sides,t);
    }
    for(let row=0;row<rows;row++)for(let side=0;side<sides;side++){const a=row*(sides+1)+side,b=a+sides+1;indices.push(a,a+1,b,a+1,b+1,b);}
    const cap=new BufferGeometry();cap.setAttribute('position',new Float32BufferAttribute(positions,3));cap.setAttribute('uv',new Float32BufferAttribute(uv,2));cap.setIndex(indices);cap.computeVertexNormals();
    for(let row=0;row<=rows;row++){
      const start=row*(sides+1),end=start+sides,n=new Vector3().fromBufferAttribute(cap.attributes.normal,start).add(new Vector3().fromBufferAttribute(cap.attributes.normal,end)).normalize();
      cap.attributes.normal.setXYZ(start,n.x,n.y,n.z);cap.attributes.normal.setXYZ(end,n.x,n.y,n.z);
    }
    for(let side=0;side<=sides;side++)cap.attributes.normal.setXYZ(rows*(sides+1)+side,0,cos,-sin);
    const crown=this.mesh(cap,palette.crown);crown.name='Ari beret crown';finishFabric(crown.material as MeshStandardMaterial,'wool');
    const ring=(h:number,rx:number,rz:number)=>new CatmullRomCurve3(Array.from({length:65},(_,i)=>{const a=i/64*Math.PI*2;return place(Math.sin(a)*rx,h,Math.cos(a)*rz);}));
    const rim=new TubeGeometry(ring(.001,.085,.105),128,.0025,8,false);const band=this.mesh(rim,palette.band);band.name='Ari beret band';finishFabric(band.material as MeshStandardMaterial,'wool');
    const seam=new TubeGeometry(ring(.007,.091,.115),128,.0005,5,false);this.mesh(seam,palette.seam).name='Ari beret seam';
    const stem=new TubeGeometry(new CatmullRomCurve3([place(0,.098,0),place(.001,.107,0),place(.005,.109,0)]),16,.0018,8,false);finishFabric(this.mesh(stem,palette.stem).material as MeshStandardMaterial,'wool');
    const bowCenter=place(.080,.005,.045),bowPositions:number[]=[],bowUV:number[]=[],bowIndices:number[]=[];
    const ribbon=(sample:(u:number,v:number)=>Vector3)=>{
      const base=bowPositions.length/3;
      for(let row=0;row<=24;row++)for(let col=0;col<=8;col++){bowPositions.push(...sample(row/24,col/8).add(bowCenter).toArray());bowUV.push(row/24,col/8);}
      for(let row=0;row<24;row++)for(let col=0;col<8;col++){const a=base+row*9+col;bowIndices.push(a,a+1,a+9,a+1,a+10,a+9);}
    };
    for(const sign of [-1,1]){
      ribbon((u,v)=>new Vector3(sign*.019*Math.sin(Math.PI*u),(v-.5)*.014*Math.pow(Math.sin(Math.PI*u),.6)+sign*.003*Math.sin(Math.PI*u),.004+.006*Math.sin(u*Math.PI*2)));
      ribbon((u,v)=>new Vector3(sign*(.004+.014*u)+(v-.5)*.009,-.005-.035*u,.002+.003*Math.sin(u*4)));
    }
    const bow=new BufferGeometry();bow.setAttribute('position',new Float32BufferAttribute(bowPositions,3));bow.setAttribute('uv',new Float32BufferAttribute(bowUV,2));bow.setIndex(bowIndices);bow.computeVertexNormals();const ribbonMesh=this.mesh(bow,palette.ribbon);ribbonMesh.name='Ari beret ribbon';ribbonMesh.material.side=2;finishFabric(ribbonMesh.material as MeshStandardMaterial,'satin');
  }
  private clear(){
    this.root.traverse(node=>{if(node instanceof Mesh){node.geometry.dispose();for(const material of Array.isArray(node.material)?node.material:[node.material])material.dispose();}});this.root.clear();
  }
  dispose(){this.clear();this.root.removeFromParent();}
}
