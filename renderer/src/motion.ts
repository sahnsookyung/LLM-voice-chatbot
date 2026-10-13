import type {Bridge} from './types';
export const emotions = ['neutral','happy','curious','concerned','surprised','annoyed','embarrassed','sleepy'] as const;
export type Emotion = typeof emotions[number];
export const clamp = (x:number,min=0,max=1)=>Math.max(min,Math.min(max,x));
const approach = (current:number,target:number,dt:number,tau:number)=>current+(target-current)*(1-Math.exp(-dt/tau));
const ease = (t:number)=>{t=clamp(t);return t*t*t*(t*(t*6-15)+10);};
// A reach, a readable hold, then a slower release. Each segment has zero
// velocity and acceleration at its endpoints; distal joints start later.
const phrase = (age:number,duration:number,delay=0)=>ease((age-delay-.12)/1.15)*(1-ease((age-delay-duration*.57)/(duration*.43)));
const durations:Record<string,number>={wave:3.2,nod:3.2,explain:3.2,shift:3.2,step:3.2,'hair-sway':4.8,twirl:4.6,'raise-arms':4.5,shrug:3.8,point:4.5};

/** A single bounded controller. New targets blend from the current pose, including interruptions. */
export class MotionController {
  expression:Emotion='neutral'; intensity=0.8; state='idle'; reduced=false;
  private time=0; private audioTime=-10; private targetLevel=0; private level=0;
  private blinkAt=2; private blinkStart=-10; private gazeAt=0; private gx=0; private gy=0;
  private gazeX=0; private gazeY=0; private gesture='';
  private weights:Record<string,number>={};
  private velocities:Record<string,number>={};
  private spinBase=0; private spinTarget=0;
  private gestureAge=10; private gestureSpeed=1;
  get moving(){return !this.reduced&&this.gestureAge<(durations[this.gesture]??0)+1.5;}
  constructor(private random:()=>number=Math.random) { this.blinkAt=1+random()*3; }
  update(e:Bridge) {
    if(e.expression && (emotions as readonly string[]).includes(e.expression)) this.expression=e.expression as Emotion;
    if(e.state) this.state=e.state.toLowerCase();
    if(e.reducedMotion!==undefined) this.reduced=e.reducedMotion;
    if(e.intensity!==undefined && Number.isFinite(e.intensity))this.intensity=clamp(e.intensity);
    if(e.gestureSpeed!==undefined&&Number.isFinite(e.gestureSpeed))this.gestureSpeed=clamp(e.gestureSpeed,.65,1.5);
    if(e.level!==undefined && Number.isFinite(e.level)){this.targetLevel=clamp(e.level);this.audioTime=this.time;}
    if(e.gesture==='cancel' || (e.gesture && ['wave','nod','explain','shift','step','hair-sway','twirl','raise-arms','shrug','point'].includes(e.gesture))){
      this.spinBase=this.weights.bodyYaw??0;
      this.spinTarget=Math.round(this.spinBase/(Math.PI*2))*Math.PI*2;
      this.gesture=e.gesture==='cancel'?'':e.gesture;this.gestureAge=0;
    }
  }
  step(delta:number) {
    const dt=clamp(Number.isFinite(delta)?delta:0,0,.1);this.time+=dt;
    if(this.time-this.audioTime>.20)this.targetLevel=0;
    this.level=approach(this.level,this.targetLevel,dt,this.targetLevel>this.level?.045:.10);
    if(this.time>=this.blinkAt){this.blinkStart=this.time;this.blinkAt=this.time+2.4+this.random()*4.8;}
    const blinkAge=this.time-this.blinkStart;
    const blink=this.reduced?0:blinkAge<.22?Math.sin(clamp(blinkAge/.22)*Math.PI)**2:0;
    if(this.time>=this.gazeAt){this.gx=(this.random()-.5)*.65;this.gy=(this.random()-.5)*.35;this.gazeAt=this.time+2.5+this.random()*4;}
    this.gazeX=approach(this.gazeX,this.reduced?0:this.gx,dt,.20);
    this.gazeY=approach(this.gazeY,this.reduced?0:this.gy,dt,.24);
    const targets:Record<string,number>={Happy:0,Curious:0,Concerned:0,Surprised:0,Annoyed:0,Embarrassed:0,Sleepy:0,Blush:0,EyeWide:0,JawOpen:clamp(this.level*1.6,0,.85),MouthRound:0};
    const mood=this.expression;
    const key:Record<string,string>={happy:'Happy',curious:'Curious',concerned:'Concerned',surprised:'Surprised',annoyed:'Annoyed',embarrassed:'Embarrassed',sleepy:'Sleepy'};
    if(key[mood])targets[key[mood]]=this.intensity;
    if(mood==='surprised'){targets.EyeWide=.6*this.intensity;targets.JawOpen=Math.max(targets.JawOpen,.22*this.intensity);}
    // Eye directions are exclusive; blink compression is applied after emotional openness.
    targets.LookLeft=Math.max(0,-this.gazeX);targets.LookRight=Math.max(0,this.gazeX);
    targets.LookUp=Math.max(0,this.gazeY);targets.LookDown=Math.max(0,-this.gazeY);
    targets.Blush=this.intensity*(mood==='embarrassed'?.7:mood==='happy'?.18:mood==='annoyed'?.1:0);
    for(const [name,target] of Object.entries(targets)) {
      if(name==='JawOpen' || name.startsWith('Look')) {this.weights[name]=approach(this.weights[name]??0,target,dt,name==='JawOpen'?.06:.2);continue;}
      // Critically damped response preserves velocity when an expression is interrupted.
      const rate=name==='Blush'?5:9, offset=(this.weights[name]??0)-target;
      const velocity=this.velocities[name]??0, c=velocity+rate*offset, decay=Math.exp(-rate*dt);
      const next=target+(offset+c*dt)*decay;
      this.weights[name]=clamp(next);
      this.velocities[name]=next<0||next>1?0:(velocity-rate*c*dt)*decay;
    }
    this.weights.BlinkL=this.weights.BlinkR=Math.max(blink,(this.weights.Sleepy??0)*.48);
    this.weights.EyeWide*=1-blink;
    this.gestureAge+=dt*this.gestureSpeed;
    const age=this.gestureAge;
    const envelope=age<3.2?Math.sin(clamp(age/3.2)*Math.PI)**2:0;
    const motion=this.reduced?0:1;
    const cue=(name:string,delay=0)=>this.gesture===name?phrase(age,durations[name],delay)*motion:0;
    const wave=cue('wave');
    const explain=cue('explain');
    const shift=this.gesture==='shift'?envelope*motion:0;
    const step=this.gesture==='step'?envelope*motion:0;
    const idleShift=Math.sin(this.time*.32)*.008*motion;
    const nod=this.gesture==='nod'?envelope*Math.sin(age*5.6)*.09*motion:0;
    const hairSway=this.gesture==='hair-sway'&&age<4.8?Math.sin(age/4.8*Math.PI)**2*Math.sin(age*2.8)*.30*motion:0;
    const raise=cue('raise-arms'),raiseLeft=cue('raise-arms',.13);
    const shrug=cue('shrug'),shrugLeft=cue('shrug',.09);
    const point=cue('point'),pointHand=cue('point',.20);
    const rightOpen=cue('wave',.20)+cue('raise-arms',.24)*.55+cue('shrug',.20)*.8+cue('explain',.22)*.6;
    const leftOpen=cue('raise-arms',.35)*.55+cue('shrug',.28)*.75+cue('explain',.35)*.7;
    const hold=(duration:number)=>Math.sin(clamp(age/duration)*Math.PI)**2*motion;
    const twirl=this.gesture==='twirl'?hold(4.6):0;
    const t=clamp(age/4.6),turn=t*t*t*(t*(t*6-15)+10);
    if(this.reduced)this.spinTarget=Math.round((this.weights.bodyYaw??0)/(Math.PI*2))*Math.PI*2;
    const poseTargets:Record<string,number>={
      headYaw:this.gazeX*.19*motion+hairSway-cue('point',-.12)*.24,
      headPitch:(-this.gazeY*.15+nod+(this.state==='listening'?.035:0))*motion+raise*.035-shrug*.025,
      headTilt:((mood==='curious'?.055:mood==='embarrassed'?-.035:0)+Math.sin(this.time*.39)*.01)*motion+shrug*.08+twirl*.045,
      chest:Math.sin(this.time*1.3)*.007*motion-raise*.025,
      chestYaw:-cue('point',-.08)*.055+explain*.025,
      hair:Math.sin(this.time*.9)*.007*motion,
      bodyYaw:this.gesture==='twirl'&&!this.reduced?this.spinBase+Math.PI*2*turn:this.spinTarget,
      upperArm:wave*.82+explain*.28+raise*2.35+shrug*.4+point*1.48+twirl*.82,
      forearm:cue('wave',.12)*.18+cue('explain',.13)*.95+cue('shrug',.12)*1.1-point*.055,
      forearmLift:cue('wave',.12)*1.65+cue('raise-arms',.12)*.24+cue('shrug',.12)*.8+twirl*.31,
      wave,
      hand:cue('wave',.24)*Math.sin(age*8)*.18+cue('point',.27)*.055-cue('raise-arms',.24)*.12,
      rightWristRoll:cue('wave',.20)*1.05+cue('shrug',.22)*1.38+cue('explain',.25)*.25,
      leftWristRoll:cue('shrug',.31)*1.34+cue('explain',.32)*.45,
      rightWristYaw:cue('wave',.26)*Math.sin(age*5.7+.5)*.09+cue('point',.24)*.07,
      leftWristYaw:cue('explain',.30)*.07,
      rightArmForward:point*.12+raise*.05+explain*.09,
      leftArmForward:raiseLeft*.045+explain*.06,
      leftArm:explain*.32+step*.1+raiseLeft*2.3+shrugLeft*.37+twirl*.72,
      leftForearm:cue('explain',.21)*.7+cue('shrug',.21)*1.05,
      leftForearmLift:cue('raise-arms',.24)*.26+cue('shrug',.21)*.78+twirl*.37,
      shoulderLift:shrug*.16+raise*.07,
      rightShoulderLift:cue('shrug',-.06)*.165+raise*.067,
      palmUp:shrug*1.45,
      point:pointHand,
      rightOpen:clamp(rightOpen),leftOpen:clamp(leftOpen),
      rightGrip:(Math.sin(this.time*.83)+Math.sin(this.time*.37+.4))*.012*motion,
      leftGrip:(Math.sin(this.time*.71+.8)+Math.sin(this.time*.41))*.012*motion,
      leftHand:cue('explain',.3)*.12+cue('raise-arms',.35)*.1,
      hipX:idleShift+shift*.024-step*.025,
      hipDrop:(.004+Math.abs(idleShift)*.3+shift*.007+step*.01)*motion,
      bodyLean:-(idleShift+shift*.024-step*.025)*.6,
      leftFootLift:step*.035+twirl*Math.max(0,Math.sin(age*5))**2*.028,
      leftFootForward:step*.025,
      rightFootLift:twirl*Math.max(0,-Math.sin(age*5))**2*.028,
    };
    for(const [name,target]of Object.entries(poseTargets)){
      // Preserve pose velocity on interruption, including a mid-turn cancel.
      // Wrists soften the action, while the torso starts to orient first.
      const rate=name==='bodyYaw'?14:/hand|wrist|open|grip/i.test(name)?12:11;
      const offset=(this.weights[name]??0)-target,velocity=this.velocities[name]??0;
      const c=velocity+rate*offset,decay=Math.exp(-rate*dt);
      this.weights[name]=target+(offset+c*dt)*decay;
      this.velocities[name]=(velocity-rate*c*dt)*decay;
    }
    return this.weights;
  }
}
