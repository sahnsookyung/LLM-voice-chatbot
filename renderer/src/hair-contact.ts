import {Matrix4,Quaternion,Vector3,SkinnedMesh,type Object3D} from 'three';
import {MToonMaterial,type VRM,type VRMHumanBoneName} from '@pixiv/three-vrm';
import type {Outfit} from './wardrobe';

// Padded clothing volumes in Ari's authored rest coordinates (metres). The
// spring colliders drive the locks; these volumes also catch vertices between
// sparse spring joints. No CPU traversal of the 66k hair vertices per frame.
const volumes: {bone:VRMHumanBoneName; center:number[]; radii:number[]}[] = [
  {bone:'head',center:[0,1.488,-.010],radii:[.085,.113,.108]},
  {bone:'neck',center:[0,1.338,-.025],radii:[.05,.072,.054]},
  {bone:'upperChest',center:[0,1.278,-.018],radii:[.19,.067,.091]},
  {bone:'leftUpperArm',center:[.208,1.272,-.025],radii:[.155,.09,.09]},
  {bone:'rightUpperArm',center:[-.208,1.272,-.025],radii:[.155,.09,.09]},
  {bone:'leftLowerArm',center:[.414,1.272,-.025],radii:[.142,.085,.085]},
  {bone:'rightLowerArm',center:[-.414,1.272,-.025],radii:[.142,.085,.085]},
  {bone:'spine',center:[0,1.025,.014],radii:[.167,.155,.14]},
  {bone:'upperChest',center:[0,1.159,.025],radii:[.195,.174,.163]},
];

/** Last-stage geometric contact, after skinning and before projection. */
export class HairContact {
  readonly count=volumes.length;
  private mesh!:SkinnedMesh;
  private bindings:{bone:Object3D;local:Matrix4}[];
  private to=volumes.map(()=>new Matrix4());
  private from=volumes.map(()=>new Matrix4());
  private world=new Matrix4();
  constructor(vrm:VRM, outfit:Outfit='original') {
    vrm.scene.traverse(node=>{
      if(!(node instanceof SkinnedMesh))return;
      for(const material of Array.isArray(node.material)?node.material:[node.material]){
        if(!(material instanceof MToonMaterial)||!material.userData.ariHairContact)return;
        this.mesh=node;
        const compile=material.onBeforeCompile.bind(material),key=material.customProgramCacheKey.bind(material);
        material.customProgramCacheKey=()=>`${key()}:ari-hair-contact-v2`;
        material.onBeforeCompile=(shader,renderer)=>{
          compile(shader,renderer);
          const anchor='#include <skinning_vertex>';
          if(!shader.vertexShader.includes(anchor))throw new Error('Hair contact shader hook missing');
          shader.uniforms.ariHairTo={value:this.to};shader.uniforms.ariHairFrom={value:this.from};
          shader.vertexShader=`uniform mat4 ariHairTo[${this.count}];\nuniform mat4 ariHairFrom[${this.count}];\n${shader.vertexShader}`;
          shader.vertexShader=shader.vertexShader.replace(anchor,`${anchor}
            // Revisit overlaps so resolving an arm does not push into the torso.
            for (int pass = 0; pass < 2; pass++) {
              for (int i = 0; i < ${this.count}; i++) {
                vec3 q = (ariHairTo[i] * vec4(transformed, 1.0)).xyz;
                float radius = length(q);
                if (radius < 1.08) {
                  // C1 soft contact shell: always outside the solid, but no
                  // abrupt change of velocity at the old hard clamp boundary.
                  float contact = 1.0 + pow(max(0.0, radius - 0.92), 2.0) / 0.32;
                  q = radius > 0.00001 ? q * (contact / radius) : vec3(0.0, 0.0, 1.0);
                  transformed = (ariHairFrom[i] * vec4(q, 1.0)).xyz;
                }
              }
            }`);
        };
        material.needsUpdate=true;
      }
    });
    if(!this.mesh)throw new Error('Ari hair contact material missing');
    this.bindings=volumes.map(volume=>{
      const bone=vrm.humanoid.getRawBoneNode(volume.bone);
      if(!bone)throw new Error(`Hair collision bone missing: ${volume.bone}`);
      const radii=new Vector3(...volume.radii);
      if(outfit==='streetwear' && !['head','neck'].includes(volume.bone))radii.multiplyScalar(1.10);
      else if(outfit==='tailored' && !['head','neck'].includes(volume.bone))radii.multiplyScalar(1.035);
      const authored=new Matrix4().compose(new Vector3(...volume.center),new Quaternion(),radii);
      return {bone,local:bone.matrixWorld.clone().invert().multiply(authored)};
    });
    this.update();
  }
  update() {
    // The caller updates scene matrices once, shared with the render pass.
    this.bindings.forEach(({bone,local},i)=>{
      this.world.multiplyMatrices(bone.matrixWorld,local);
      this.to[i].copy(this.world).invert().multiply(this.mesh.matrixWorld);
      this.from[i].copy(this.to[i]).invert();
    });
  }
}
