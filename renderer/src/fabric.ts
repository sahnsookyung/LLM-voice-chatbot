import {MeshStandardMaterial} from 'three';

export type Fabric = 'cotton'|'wool'|'twill'|'rib'|'satin'|'leather';

/** Small weave detail is shaded, distance filtered, and attached to the cloth UVs.
 * Large folds belong in the mesh; this only supplies yarn-scale roughness. */
export function finishFabric(material:MeshStandardMaterial,kind:Fabric) {
  material.roughness=kind==='satin'?.52:kind==='leather'?.58:kind==='wool'?.96:.86;
  material.metalness=0;
  const compile=material.onBeforeCompile.bind(material),key=material.customProgramCacheKey.bind(material);
  material.customProgramCacheKey=()=>`${key()}:ari-fabric-v1:${kind}`;
  material.onBeforeCompile=(shader,renderer)=>{
    compile(shader,renderer);
    shader.vertexShader=`varying vec2 ariFabricUV;\n${shader.vertexShader}`.replace('#include <begin_vertex>','#include <begin_vertex>\nariFabricUV=uv;');
    shader.fragmentShader=`varying vec2 ariFabricUV;\n${shader.fragmentShader}`;
    const rib=kind==='rib',wool=kind==='wool',leather=kind==='leather';
    shader.fragmentShader=shader.fragmentShader.replace('#include <normal_fragment_maps>',`#include <normal_fragment_maps>
      vec2 yarnUV=ariFabricUV*vec2(${rib?'160.0,90.0':wool?'330.0,200.0':'440.0,280.0'});
      vec2 yarnFilter=exp(-2.0*pow(fwidth(yarnUV),vec2(2.0)));
      float yarnX=sin(yarnUV.x*6.283185+${wool?'0.7*sin(yarnUV.y*6.283185)':'0.0'})*yarnFilter.x;
      float yarnY=cos(yarnUV.y*6.283185)*yarnFilter.y;
      float weave=${rib?'yarnX*.7+yarnY*.12':leather?'yarnX*yarnY*.12':'yarnX*.35+yarnY*.35'};
      // Screen derivatives form a cotangent frame even on deformed/skinned fabric.
      vec3 dpX=dFdx(-vViewPosition),dpY=dFdy(-vViewPosition);
      vec3 r1=cross(dpY,normal),r2=cross(normal,dpX);
      float det=dot(dpX,r1);
      vec3 grad=sign(det)*(dFdx(weave)*r1+dFdy(weave)*r2);
      normal=normalize(max(abs(det),1.0e-12)*normal-${rib?'0.00018':'0.000045'}*grad);
      diffuseColor.rgb*=0.97+0.03*weave;
    `);
  };
  material.needsUpdate=true;
}
