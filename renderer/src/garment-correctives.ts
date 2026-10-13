import {BufferGeometry,Float32BufferAttribute,Mesh, type Object3D} from 'three';

const smooth=(v:number)=>{const t=Math.max(0,Math.min(1,v));return t*t*(3-2*t);};

/** Bind-space fold corrections shared by every coincident garment seam.
 * This preserves seam continuity even when panels use different materials. */
export function garmentDelta(x:number,y:number,z:number,side:number):[number,number,number] {
  const across=x*side,sideWeight=smooth((across+.025)/.08);
  const underarm=sideWeight*smooth((across-.045)/.10)*(1-smooth((across-.25)/.055))*smooth((y-1.10)/.10)*(1-smooth((y-1.285)/.05));
  // Fitted bodices stay anchored at the waist; only loose lower hems lift.
  const hem=sideWeight*smooth((y-.77)/.18)*(1-smooth((y-.965)/.075))*(1-smooth((across-.16)/.08));
  const front=(z+.018)/Math.sqrt((z+.018)**2+.0001);
  const gather=Math.sin((y-1.11)*105+across*34)*.0025*underarm;
  return [side*.005*underarm,.016*underarm+.012*hem,front*(.005*underarm+gather)];
}

/** Two GPU morph weights per upper garment; no cloth solver or vertex updates
 * in the frame loop. Retired targets are owned/disposed with model geometry. */
export class GarmentCorrectives {
  private targets:{mesh:Mesh;indices:number[]}[]=[];
  readonly weights=[0,0];
  constructor(root:Object3D){
    const prepared=new Map<BufferGeometry,number[]>();
    root.traverse(node=>{
      if(!(node instanceof Mesh))return;
      const materials=Array.isArray(node.material)?node.material:[node.material];
      if(!materials.some(m=>m.userData.ariFabric||/Tops_01/.test(m.name)))return;
      const geometry=node.geometry,position=geometry.attributes.position;
      let high=0;for(let i=0;i<position.count;i++)high=Math.max(high,position.getY(i));
      if(high<1.13)return; // Footwear, socks and skirts keep their own rigs.
      let indices=prepared.get(geometry);
      if(!indices){
        if(geometry.morphAttributes.position?.length&&!geometry.morphTargetsRelative)throw new Error('Garment corrections require relative morph targets.');
        geometry.morphTargetsRelative=true;
        const positions=geometry.morphAttributes.position??=[],normals=geometry.morphAttributes.normal??=[];
        // glTF normally exports both. Pad an existing position-only target so
        // the normal target indices remain aligned with the position targets.
        while(normals.length<positions.length)normals.push(new Float32BufferAttribute(new Float32Array(position.count*3),3));
        const rest=new BufferGeometry();rest.setAttribute('position',position);rest.setIndex(geometry.index);rest.computeVertexNormals();
        indices=[];
        for(const side of [1,-1]){
          const delta=new Float32Array(position.count*3),posed=new Float32Array(position.count*3);
          for(let i=0;i<position.count;i++){
            const x=position.getX(i),y=position.getY(i),z=position.getZ(i),d=garmentDelta(x,y,z,side);
            delta.set(d,i*3);posed.set([x+d[0],y+d[1],z+d[2]],i*3);
          }
          const target=new BufferGeometry();target.setAttribute('position',new Float32BufferAttribute(posed,3));target.setIndex(geometry.index);target.computeVertexNormals();
          const normalDelta=new Float32Array(position.count*3),base=rest.attributes.normal,n=target.attributes.normal;
          for(let i=0;i<position.count;i++)normalDelta.set([n.getX(i)-base.getX(i),n.getY(i)-base.getY(i),n.getZ(i)-base.getZ(i)],i*3);
          target.dispose();
          const attribute=new Float32BufferAttribute(delta,3);attribute.name=side>0?'ariSleeveLiftLeft':'ariSleeveLiftRight';
          indices.push(positions.length);positions.push(attribute);normals.push(new Float32BufferAttribute(normalDelta,3));
        }
        rest.dispose();
        geometry.morphAttributes.position=positions;geometry.morphAttributes.normal=normals;prepared.set(geometry,indices);
      }
      const previous=node.morphTargetInfluences?.slice();node.updateMorphTargets();
      previous?.forEach((v,i)=>node.morphTargetInfluences![i]=v);
      this.targets.push({mesh:node,indices});
    });
  }
  update(pose:Record<string,number>){
    this.weights[0]=smooth(((pose.leftArm??0)-.5)/1.5);
    this.weights[1]=smooth(((pose.upperArm??0)-.5)/1.5);
    for(const {mesh,indices}of this.targets)for(let i=0;i<2;i++)mesh.morphTargetInfluences![indices[i]]=this.weights[i];
  }
  get meshes(){return this.targets.length;}
}
