import {
  Scene, WebGLRenderer, OrthographicCamera, HemisphereLight, DirectionalLight,
  Group, Mesh, Box3, Euler, Vector3, SRGBColorSpace, NoToneMapping, type Object3D, type Quaternion,
} from 'three';
import {GLTFLoader} from 'three/addons/loaders/GLTFLoader.js';
import {VRMLoaderPlugin, VRMUtils, type VRM, type VRMHumanBoneName} from '@pixiv/three-vrm';
import modelData from '../../characters/ari/model/ari-vroid-v9.vrm?inline';
import {customizeFace} from './face';
import {customizeClothing} from './clothing';
import {PlantedLeg} from './leg-ik';
import {HairContact} from './hair-contact';
import type {Framing} from './types';
import {defaultWardrobe, type WardrobeSelection} from './wardrobe';
import {buildWardrobeModel} from './wardrobe-model';

let baseBytes:ArrayBuffer|undefined;

/** VRoid's authored MToon materials, expressions and normalized humanoid rig. */
export class Avatar3D {
  private scene = new Scene();
  private camera = new OrthographicCamera();
  private renderer: WebGLRenderer;
  private root = new Group();
  private vrm?: VRM;
  private turn = 0;
  private targetTurn = 0;
  private euler = new Euler();
  private lost = false;
  private disposed = false;
  private bounds = new Box3();
  private morphMeshes = 0;
  private blush = {value:0};
  private legs: PlantedLeg[] = [];
  private hipsRest = new Vector3();
  private hairRest: {bone:Object3D; rotation:Quaternion; previous:Quaternion}[] = [];
  private hairSpeed = 0;
  private hairContact?: HairContact;
  private hairTips: Object3D[] = [];
  private garmentRest:{bone:Object3D;rotation:Quaternion}[]=[];
  private reduced = false;
  private simulationTimes = new Float32Array(240);
  private drawTimes = new Float32Array(240);
  private simulationCount = 0;
  private drawCount = 0;
  // Opt-in cumulative CPU timings for the native benchmark, never GPU timings.
  private profile = new URLSearchParams(location.search).has('profile');
  private cpuTotals = {steps:0,draws:0,pose:0,vrm:0,contact:0,simulation:0,draw:0};
  private wardrobe:WardrobeSelection={...defaultWardrobe};

  private constructor(private host: HTMLElement, private onError: (message: string) => void) {
    this.renderer = new WebGLRenderer({alpha: true, antialias: true, powerPreference: 'low-power'});
    this.renderer.debug.onShaderError = () => this.onError('Ari’s material shader could not compile.');
    this.renderer.setPixelRatio(Math.min(devicePixelRatio, 1.5));
    this.renderer.outputColorSpace = SRGBColorSpace;
    // MToon already defines the illustration's colors; filmic mapping washes them out.
    this.renderer.toneMapping = NoToneMapping;
    this.renderer.domElement.setAttribute('aria-label', 'Ari full-body VRoid character');
    this.renderer.domElement.addEventListener('webglcontextlost', this.onContextLost);
    this.scene.add(this.root, new HemisphereLight(0xffffff, 0x8892a0, 1.3));
    const key = new DirectionalLight(0xffffff, 1.6);
    key.position.set(-2, 3, 4);
    this.scene.add(key);
    // draw() updates matrices before contact uniforms; render must reuse them.
    this.scene.matrixWorldAutoUpdate = false;
  }

  private onContextLost = (event: Event) => {
    event.preventDefault();
    this.lost = true;
    this.onError('The 3D preview lost its graphics context. Switch to Illustration, then try 3D again.');
  };

  static async create(host: HTMLElement, onError: (message: string) => void, wardrobe:WardrobeSelection=defaultWardrobe) {
    const avatar = new Avatar3D(host, onError);
    avatar.wardrobe={...wardrobe};
    try {
      // Bundled bytes avoid file-origin network requests in the native WKWebView.
      if(!baseBytes){
        const encoded = modelData.slice(modelData.indexOf(',') + 1),bytes = atob(encoded),buffer = new Uint8Array(bytes.length);
        for (let i = 0; i < bytes.length; i++) buffer[i] = bytes.charCodeAt(i);
        baseBytes=buffer.buffer;
      }
      const loader = new GLTFLoader();
      loader.register(parser => new VRMLoaderPlugin(parser));
      const gltf = await loader.parseAsync(buildWardrobeModel(baseBytes,wardrobe), '');
      avatar.root.add(gltf.scene); // Owned immediately, including failure cleanup.
      const vrm = gltf.userData.vrm as VRM | undefined;
      if (!vrm) throw new Error('The bundled character does not contain a VRM rig.');
      avatar.vrm = vrm;
      VRMUtils.rotateVRM0(vrm);
      VRMUtils.removeUnnecessaryVertices(vrm.scene);
      avatar.blush = customizeFace(vrm);
      customizeClothing(vrm);
      VRMUtils.combineMorphs(vrm);
      VRMUtils.combineSkeletons(vrm.scene);
      vrm.scene.traverse(node => {
        if (node instanceof Mesh) {
          if (node.morphTargetDictionary) avatar.morphMeshes++;
          node.frustumCulled = false; // Gestures extend beyond rest-pose bounds.
        }
      });
      if (vrm.lookAt) vrm.lookAt.autoUpdate = false;
      vrm.scene.updateMatrixWorld(true);
      avatar.hairContact = new HairContact(vrm,wardrobe.outfit);
      avatar.hipsRest.copy(vrm.humanoid.getNormalizedBoneNode('hips')!.position);
      for (const side of ['left','right'] as const) {
        avatar.legs.push(new PlantedLeg(vrm.scene,vrm.humanoid.getNormalizedBoneNode(`${side}UpperLeg`)!,vrm.humanoid.getNormalizedBoneNode(`${side}LowerLeg`)!,vrm.humanoid.getNormalizedBoneNode(`${side}Foot`)!));
      }
      avatar.pose({});
      vrm.update(0);
      vrm.scene.updateMatrixWorld(true);
      avatar.bounds.setFromObject(vrm.scene);
      vrm.springBoneManager?.reset();
      avatar.hairRest = [...(vrm.springBoneManager?.joints ?? [])]
        .filter(joint => joint.bone.name.includes('Hair'))
        .map(joint => ({bone:joint.bone,rotation:joint.bone.quaternion.clone(),previous:joint.bone.quaternion.clone()}));
      avatar.hairTips = [...(vrm.springBoneManager?.joints ?? [])]
        .filter(joint=>joint.bone.name.includes('Hair')&&joint.child&&!joint.child.children.some(child=>child.name.includes('Hair')))
        .map(joint=>joint.child!);
      avatar.garmentRest=[...(vrm.springBoneManager?.joints??[])].filter(joint=>joint.bone.name.startsWith('Skirt')).map(joint=>({bone:joint.bone,rotation:joint.bone.quaternion.clone()}));
      host.appendChild(avatar.renderer.domElement);
      return avatar;
    } catch (error) {
      avatar.dispose();
      throw error;
    }
  }

  resize(framing: Framing) {
    const width = Math.max(1, this.host.clientWidth);
    const height = Math.max(1, this.host.clientHeight);
    const modelHeight = this.bounds.max.y - this.bounds.min.y;
    const span = framing === 'full'
      ? Math.max(modelHeight * 1.38, 1.4 * height / width)
      : Math.max(.57, .48 * height / width);
    const center = framing === 'full'
      ? (this.bounds.max.y + this.bounds.min.y) / 2
      : this.bounds.max.y - .27;
    this.camera.left = -span * width / height / 2;
    this.camera.right = -this.camera.left;
    this.camera.top = span / 2;
    this.camera.bottom = -span / 2;
    this.camera.near = .01;
    this.camera.far = 10;
    this.camera.position.set(0, center, 4);
    this.camera.lookAt(0, center, 0);
    this.camera.updateProjectionMatrix();
    this.renderer.setSize(width, height);
  }

  setAngle(degrees: number, immediate=false) {
    if (Number.isFinite(degrees)) this.targetTurn = Math.max(-180, Math.min(180, degrees)) * Math.PI / 180;
    if(immediate)this.turn=this.targetTurn;
  }

  prime(w:Record<string,number>,reduced:boolean) {
    this.update(w,0,reduced);
    this.vrm?.scene.updateMatrixWorld(true);
    this.vrm?.springBoneManager?.reset();
    for(const joint of this.hairRest)joint.previous.copy(joint.bone.quaternion);
    this.draw();
  }

  private bone(name: VRMHumanBoneName, x = 0, y = 0, z = 0) {
    this.vrm?.humanoid.getNormalizedBoneNode(name)?.quaternion.setFromEuler(this.euler.set(x, y, z));
  }

  private pose(w: Record<string, number>) {
    this.bone('head', w.headPitch ?? 0, w.headYaw ?? 0, w.headTilt ?? 0);
    this.bone('chest', w.chest ?? 0, w.chestYaw??0, w.bodyLean ?? 0);
    const hips = this.vrm?.humanoid.getNormalizedBoneNode('hips');
    if (hips) {hips.position.copy(this.hipsRest);hips.position.x += w.hipX ?? 0;hips.position.y -= w.hipDrop ?? 0;}
    // VRM's normalized rest pose is a T-pose. Lower arms symmetrically first.
    this.bone('leftShoulder',0,0,w.shoulderLift??0);
    this.bone('rightShoulder',0,0,-(w.rightShoulderLift??0));
    this.bone('leftUpperArm', 0, -(w.leftArmForward??0), -1.25 + (w.leftArm ?? 0));
    this.bone('rightUpperArm', 0, w.rightArmForward??0, 1.25 - (w.upperArm ?? 0));
    this.bone('leftLowerArm', 0, -.08 - (w.leftForearm ?? 0), w.leftForearmLift??0);
    this.bone('leftHand', -(w.leftWristRoll??0), -(w.leftWristYaw??0), w.leftHand ?? 0);
    this.bone('rightLowerArm', 0, (w.forearm ?? 0) + .08, -(w.forearmLift ?? 0));
    this.bone('rightHand', -(w.rightWristRoll??0), w.rightWristYaw??0, w.hand ?? 0);
    this.legs[0]?.solve(w.leftFootLift ?? 0,w.leftFootForward ?? 0);
    this.legs[1]?.solve(w.rightFootLift??0);
    // A relaxed finger cascade, not four equal curls. Opening the hand retains
    // some bend; small, slow variation is suppressed during a deliberate pose.
    for (const side of ['left', 'right'] as const) {
      const sign = side === 'left' ? -1 : 1;
      const open = Math.max(0,Math.min(1,w[`${side}Open`]??0));
      const curl = 1-open*.82,grip=(w[`${side}Grip`]??0)*(1-open);
      const point = side === 'right' ? w.point??0 : 0;
      for (const [index,finger] of (['Index', 'Middle', 'Ring', 'Little'] as const).entries()) {
        const extended = finger==='Index'?1-point:1;
        const fist = finger==='Index'?0:point;
        const spread=[.045,.012,-.018,-.06][index]*(.3+open*.7)*(1-point);
        this.bone(`${side}${finger}Proximal`, 0, sign*spread, sign * (([.12,.18,.24,.30][index]*curl+grip)*extended + .75*fist));
        this.bone(`${side}${finger}Intermediate`, 0, 0, sign * (([.18,.24,.30,.36][index]*curl+grip)*extended + .90*fist));
        this.bone(`${side}${finger}Distal`, 0, 0, sign * ((.08+index*.025)*curl*extended + .45*fist));
      }
      this.bone(`${side}ThumbMetacarpal`,0,sign*(.08+open*.08),sign*.04);
      this.bone(`${side}ThumbProximal`,0,sign*(.07+point*.28),sign*(.08*curl+point*.12));
      this.bone(`${side}ThumbDistal`,0,0,sign*(.10*curl+point*.25));
    }
  }

  update(w: Record<string, number>, dt: number, reduced: boolean) {
    if (this.lost || this.disposed || !this.vrm) return;
    const start=performance.now();
    this.turn += (this.targetTurn - this.turn) * (reduced ? 1 : 1 - Math.exp(-dt / .25));
    this.root.rotation.y = this.turn+(w.bodyYaw??0);
    const expressions: Record<string, number> = {
      ariHappy: w.Happy ?? 0,
      ariCurious: w.Curious ?? 0,
      ariConcerned: w.Concerned ?? 0,
      ariAnnoyed: w.Annoyed ?? 0,
      ariEmbarrassed: w.Embarrassed ?? 0,
      surprised: (w.Surprised ?? 0) * .5,
      aa: w.JawOpen ?? 0,
      ou: w.MouthRound ?? 0,
      blinkLeft: w.BlinkL ?? 0,
      blinkRight: w.BlinkR ?? 0,
    };
    for (const [name, value] of Object.entries(expressions)) {
      this.vrm.expressionManager?.setValue(name, Math.max(0, Math.min(1, value)));
    }
    if (this.vrm.lookAt) {
      this.vrm.lookAt.yaw = ((w.LookRight ?? 0) - (w.LookLeft ?? 0)) * 20;
      this.vrm.lookAt.pitch = ((w.LookDown ?? 0) - (w.LookUp ?? 0)) * 15;
    }
    this.blush.value = w.Blush ?? 0;
    this.pose(w);
    const posed = this.profile ? performance.now() : 0;
    this.vrm.update(reduced ? 0 : dt);
    const rigged = this.profile ? performance.now() : 0;
    if (reduced && !this.reduced) this.vrm.springBoneManager?.reset();
    this.reduced = reduced;
    let speed=0;
    for(const joint of this.hairRest){
      // angleTo assumes unit quaternions. A copied float32 rest rotation can
      // otherwise report nonzero speed even when it is exactly unchanged.
      const current=joint.bone.quaternion;
      if(!joint.previous.equals(current)){
        const cosine=Math.abs(joint.previous.dot(current))/(joint.previous.length()*current.length());
        speed+=(2*Math.acos(Math.min(1,cosine)))**2;
      }
      joint.previous.copy(current);
    }
    this.hairSpeed=Math.sqrt(speed/Math.max(1,this.hairRest.length))/Math.max(dt,.001);
    const elapsed=performance.now()-start;
    this.simulationTimes[this.simulationCount++%240]=elapsed;
    if(this.profile){this.cpuTotals.steps++;this.cpuTotals.pose+=posed-start;this.cpuTotals.vrm+=rigged-posed;this.cpuTotals.simulation+=elapsed;}
  }

  draw() {
    if(this.lost||this.disposed||!this.vrm)return;
    const start=performance.now();
    this.scene.updateMatrixWorld(true);
    this.hairContact?.update();
    const contacted=this.profile?performance.now():0;
    this.renderer.render(this.scene, this.camera);
    const elapsed=performance.now()-start;
    this.drawTimes[this.drawCount++%240]=elapsed;
    if(this.profile){this.cpuTotals.draws++;this.cpuTotals.contact+=contacted-start;this.cpuTotals.draw+=elapsed;}
  }

  stats() {
    const humanoid=this.vrm?.humanoid;
    const p95=(values:Float32Array,count:number)=>{const sorted=[...values.slice(0,Math.min(count,values.length))].sort((a,b)=>a-b);return sorted[Math.floor((sorted.length-1)*.95)]??0;};
    const limbExtent=Math.max(0,...(['leftHand','rightHand','leftIndexDistal','rightIndexDistal','leftFoot','rightFoot'] as VRMHumanBoneName[]).map(name=>{
      const point=humanoid?.getNormalizedBoneNode(name)?.getWorldPosition(new Vector3()).project(this.camera);
      return point?Math.max(Math.abs(point.x),Math.abs(point.y)):0;
    }));
    return {
      format: 'VRM 1.0',
      modelVersion: 'v9',
      wardrobe:{...this.wardrobe},
      geometries:this.renderer.info.memory.geometries,
      physicsHz:60,
      ...(this.profile?{cpuTotals:{...this.cpuTotals}}:{}),
      simulationMsP95:p95(this.simulationTimes,this.simulationCount),
      drawSubmissionMsP95:p95(this.drawTimes,this.drawCount),
      hairTipRadius:Math.max(0,...this.hairTips.map(bone=>{const p=this.root.worldToLocal(bone.getWorldPosition(new Vector3()));return Math.hypot(p.x,p.z);})),
      hairTips:this.hairTips.length,
      blush: this.blush.value,
      hairJoints: this.hairRest.length,
      garmentJoints:this.garmentRest.length,
      garmentDeflection:Math.sqrt(this.garmentRest.reduce((sum,{bone,rotation})=>sum+rotation.angleTo(bone.quaternion)**2,0)/Math.max(1,this.garmentRest.length)),
      hairSpeed: this.hairSpeed,
      hairContactVolumes: this.hairContact?.count??0,
      bodyYaw:this.root.rotation.y,
      rightHandHeight:humanoid?.getNormalizedBoneNode('rightHand')?.getWorldPosition(new Vector3()).y??0,
      rightIndexCurl:humanoid?.getNormalizedBoneNode('rightIndexProximal')?.rotation.z??0,
      rightMiddleCurl:humanoid?.getNormalizedBoneNode('rightMiddleProximal')?.rotation.z??0,
      shoulderAngle:humanoid?.getNormalizedBoneNode('leftShoulder')?.rotation.z??0,
      limbExtent,
      hairDeflection: Math.sqrt(this.hairRest.reduce((sum, {bone,rotation}) => sum + rotation.angleTo(bone.quaternion) ** 2, 0) / Math.max(1,this.hairRest.length)),
      footError: Math.max(0,...this.legs.map(leg => leg.error())),
      drawCalls: this.renderer.info.render.calls,
      triangles: this.renderer.info.render.triangles,
      bones: Object.keys(this.vrm?.humanoid.rawHumanBones ?? {}).length,
      morphMeshes: this.morphMeshes,
      expressions: this.vrm?.expressionManager?.expressions.length ?? 0,
      expressionWeights: Object.fromEntries((this.vrm?.expressionManager?.expressions ?? []).map(expression => [expression.expressionName, expression.weight])),
      textures: this.renderer.info.memory.textures,
    };
  }

  dispose() {
    if (this.disposed) return;
    this.disposed = true;
    VRMUtils.deepDispose(this.scene); // Includes MToon textures and skeleton buffers.
    this.vrm = undefined;
    this.hairRest = [];
    this.hairTips = [];
    this.garmentRest=[];
    this.hairContact = undefined;
    this.scene.clear();
    this.renderer.domElement.removeEventListener('webglcontextlost', this.onContextLost);
    this.renderer.dispose();
    this.renderer.forceContextLoss();
    this.renderer.domElement.remove();
  }
}
