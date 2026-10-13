import {Mesh} from 'three';
import {MToonMaterial, VRMExpression, VRMExpressionMorphTargetBind, type VRM} from '@pixiv/three-vrm';

/** Separate facial regions keep speech independent of the emotional mouth corners. */
const expressions: Record<string, Record<string, number>> = {
  ariHappy: {Fcl_BRW_Fun:.25,Fcl_EYE_Fun:.18,Fcl_MTH_Fun:.55},
  ariCurious: {Fcl_BRW_Surprised:.5,Fcl_EYE_Surprised:.15,Fcl_MTH_Up:.1},
  ariConcerned: {Fcl_BRW_Sorrow:.75,Fcl_EYE_Sorrow:.2,Fcl_MTH_Down:.2},
  ariAnnoyed: {Fcl_BRW_Angry:.7,Fcl_EYE_Angry:.25,Fcl_MTH_Angry:.2},
  ariEmbarrassed: {Fcl_BRW_Joy:.3,Fcl_EYE_Sorrow:.2,Fcl_MTH_Fun:.3},
};
export function customizeFace(vrm: VRM) {
  const meshes: Mesh[] = [];
  vrm.scene.traverse(node => {if(node instanceof Mesh && node.morphTargetDictionary) meshes.push(node);});
  for (const [name, targets] of Object.entries(expressions)) {
    const expression = new VRMExpression(name);
    for (const [target, weight] of Object.entries(targets)) {
      let found = false;
      for (const mesh of meshes) {
        const index = mesh.morphTargetDictionary?.[target];
        if (index === undefined) continue;
        expression.addBind(new VRMExpressionMorphTargetBind({primitives:[mesh],index,weight}));
        found = true;
      }
      if (!found) throw new Error(`Ari is missing facial target ${target}`);
    }
    vrm.expressionManager!.registerExpression(expression);
  }
  // An analytic UV mask follows the skinned face. No overlay mesh or texture uploads per frame.
  const blush = {value:0};
  const materials = new Set<MToonMaterial>();
  vrm.scene.traverse(node => {
    if (!(node instanceof Mesh)) return;
    for (const material of Array.isArray(node.material) ? node.material : [node.material]) {
      if (material instanceof MToonMaterial && material.name.includes('Face') && material.name.includes('SKIN')) materials.add(material);
    }
  });
  if (!materials.size) throw new Error('Ari face skin material missing');
  for (const material of materials) {
    const compile = material.onBeforeCompile.bind(material), cacheKey = material.customProgramCacheKey.bind(material);
    material.customProgramCacheKey = () => `${cacheKey()}:ari-cheeks-v1`;
    material.onBeforeCompile = (shader, renderer) => {
      compile(shader,renderer);
      shader.uniforms.ariBlush = blush;
      shader.fragmentShader = 'uniform float ariBlush;\n' + shader.fragmentShader.replace('material.shadingShift = shadingShiftFactor;', `
        vec2 cheekL = (uv - vec2(.30,.63)) / vec2(.09,.045);
        vec2 cheekR = (uv - vec2(.70,.63)) / vec2(.09,.045);
        float cheeks = min(1.0, exp(-dot(cheekL,cheekL)*2.0) + exp(-dot(cheekR,cheekR)*2.0));
        vec3 warmth = vec3(1.0,.30,.36);
        material.diffuseColor *= mix(vec3(1.0),warmth,cheeks*ariBlush);
        material.shadeColor *= mix(vec3(1.0),warmth,cheeks*ariBlush);
        material.shadingShift = shadingShiftFactor;`);
    };
    material.needsUpdate = true;
  }
  return blush;
}
