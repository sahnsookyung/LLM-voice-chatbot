import {Color, Mesh, MeshStandardMaterial} from 'three';
import {MToonMaterial, type VRM} from '@pixiv/three-vrm';
import {finishFabric,type Fabric} from './fabric';

/** Keep VRoid's cutout alpha while using the painted knit in light and shade. */
export function customizeClothing(vrm: VRM) {
  const materials = new Set<MToonMaterial>();
  const fabrics=new Set<MeshStandardMaterial>();
  vrm.scene.traverse(node => {
    if (!(node instanceof Mesh)) return;
    for (const material of Array.isArray(node.material) ? node.material : [node.material]) {
      if (material instanceof MToonMaterial && (material.userData.ariColorFromShade || material.userData.ariGarmentColor || material.userData.ariSurfaceDetail)) materials.add(material);
      if(material instanceof MeshStandardMaterial&&material.userData.ariFabric)fabrics.add(material);
    }
  });
  if (!materials.size) throw new Error('Ari knit detail material missing');
  for(const material of fabrics)finishFabric(material,material.userData.ariFabric as Fabric);
  for (const material of materials) {
    const compile = material.onBeforeCompile.bind(material), cacheKey = material.customProgramCacheKey.bind(material);
    const color=material.userData.ariGarmentColor as number[] | undefined;
    const detail=material.userData.ariSurfaceDetail as string|undefined;
    material.customProgramCacheKey = () => `${cacheKey()}:ari-knit-v4:${color?.join(',')??'original'}:${detail??''}`;
    if(color){
      material.shadingToonyFactor=.35;
      material.matcapFactor.setRGB(0,0,0);
      material.parametricRimColorFactor.setRGB(0,0,0);
    }
    material.onBeforeCompile = (shader, renderer) => {
      compile(shader,renderer);
      const anchor = 'material.shadingShift = shadingShiftFactor;';
      if (!shader.fragmentShader.includes(anchor)) throw new Error('Ari knit shader hook missing');
      // Reuse the shade sample already read by MToon; no extra sampler or texture fetch.
      if(detail){
        shader.vertexShader=`varying vec2 ariSurfaceUV;\n${shader.vertexShader}`.replace('#include <begin_vertex>','#include <begin_vertex>\nariSurfaceUV = uv;');
        shader.fragmentShader=`varying vec2 ariSurfaceUV;\n${shader.fragmentShader}`;
        shader.fragmentShader=shader.fragmentShader.replace(anchor,`
          float surfaceDetail = ${detail==='gingham'?'0.55 + 0.45 * (0.5 + 0.5 * tanh(cos(ariSurfaceUV.x * 100.530965) * 3.0)) * (0.5 + 0.5 * tanh(cos(ariSurfaceUV.y * 25.132741) * 3.0))':detail==='pleat'?'0.89 + 0.11 * cos(ariSurfaceUV.x * 100.530965)':'0.94 + 0.06 * cos(ariSurfaceUV.x * 6.283185)'};
          material.diffuseColor *= surfaceDetail; material.shadeColor *= surfaceDetail;
          ${anchor}`);
      } else if(color){
        shader.uniforms.ariFabricColor={value:new Color(color[0],color[1],color[2])};
        shader.fragmentShader=`uniform vec3 ariFabricColor;\n${shader.fragmentShader}`;
        shader.fragmentShader=shader.fragmentShader.replace(anchor,`
          float fabricLuminance = clamp(dot(${material.userData.ariColorFromShade?'material.shadeColor / max(shadeColorFactor,vec3(0.001))':'material.diffuseColor'},vec3(0.333)),0.0,1.0);
          float fabricDetail = ${material.userData.ariColorFromShade?'max(0.035,fabricLuminance * fabricLuminance / 0.78)':'0.30 + 0.70 * sqrt(clamp(fabricLuminance * 6.0,0.0,1.0))'};
          material.diffuseColor = ariFabricColor * fabricDetail;
          material.shadeColor = ariFabricColor * fabricDetail * 0.55;
          ${anchor}`);
      } else shader.fragmentShader = shader.fragmentShader.replace(anchor, `
        material.diffuseColor = material.shadeColor * litFactor / max(shadeColorFactor,vec3(0.001));
        ${anchor}`);
    };
    material.needsUpdate = true;
  }
}
