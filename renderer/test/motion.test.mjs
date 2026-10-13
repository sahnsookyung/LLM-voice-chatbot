import test from 'node:test';
import assert from 'node:assert/strict';
import {MotionController,emotions} from '../.test-build/motion.js';
const settle=(motion,seconds=2,fps=30)=>{let result;for(let i=0;i<seconds*fps;i++)result={...motion.step(1/fps)};return result;};
test('interruptions blend from the current expression instead of snapping',()=>{
  const motion=new MotionController(()=>.5);motion.update({expression:'happy'});
  const happy=settle(motion);assert.ok(happy.Happy>.79);
  motion.update({expression:'concerned'});const first={...motion.step(1/30)};
  assert.ok(first.Happy>0 && first.Happy<happy.Happy);assert.ok(first.Concerned>0&&first.Concerned<.2);
  const concerned=settle(motion);assert.ok(concerned.Happy<.01);assert.ok(concerned.Concerned>.79);
});
test('emotion smoothing is independent of frame rate',()=>{
  const a=new MotionController(()=>.5),b=new MotionController(()=>.5);a.update({expression:'happy'});b.update({expression:'happy'});
  assert.ok(Math.abs(settle(a,1,30).Happy-settle(b,1,60).Happy)<.00001);
});
test('stale or invalid audio cannot leave a mouth open',()=>{
  const m=new MotionController(()=>.5);m.update({level:1});assert.ok(m.step(.1).JawOpen>0);
  m.update({level:NaN});assert.ok(settle(m).JawOpen<.001);
});
test('every mood, gaze and repeated gesture produces bounded finite morph weights',()=>{
  const m=new MotionController(()=>.2);
  for(const expression of emotions){m.update({expression,gesture:'wave',intensity:1,level:1});for(let i=0;i<300;i++){const w=m.step(1/30);for(const [key,value] of Object.entries(w)){assert.ok(Number.isFinite(value),key);if(/^[A-Z]/.test(key))assert.ok(value>=0&&value<=1,`${key}: ${value}`);}}}
});
test('reduced motion settles all secondary motion while keeping expression and speech',()=>{
  const m=new MotionController(()=>.1);m.update({gesture:'wave',expression:'curious'});settle(m,1);
  m.update({reducedMotion:true});const w=settle(m,3);for(const key of ['headYaw','headTilt','chest','hair','upperArm','forearm','forearmLift','hand'])assert.ok(Math.abs(w[key])<.0001,key);
  assert.ok(w.Curious>.7);m.update({level:.5});assert.ok(m.step(.1).JawOpen>.2);
});

test('hair sway alternates smoothly, settles, and supports interruption',()=>{
  const m=new MotionController(()=>.5);m.update({gesture:'hair-sway'});
  const yaw=Array.from({length:240},()=>m.step(1/30).headYaw);
  assert.ok(Math.min(...yaw)<-.15&&Math.max(...yaw)>.15);
  assert.ok(yaw.every((v,i)=>Math.abs(v)<=.31&&(!i||Math.abs(v-yaw[i-1])<.035)));
  assert.ok(Math.abs(yaw.at(-1))<.001);
  m.update({gesture:'hair-sway'});settle(m,2);m.update({gesture:'cancel'});
  assert.ok(Math.abs(settle(m,3).headYaw)<.001);
});

test('a twirl completes one turn with alternate foot lifts and no rewind',()=>{
  const m=new MotionController(()=>.5);m.update({gesture:'twirl'});
  const poses=Array.from({length:240},()=>({...m.step(1/30)}));
  assert.ok(Math.abs(poses.at(-1).bodyYaw-Math.PI*2)<.001);
  assert.ok(poses.every((p,i)=>!i||p.bodyYaw>=poses[i-1].bodyYaw-1e-10));
  assert.ok(Math.max(...poses.map(p=>p.leftFootLift))>.015&&Math.max(...poses.map(p=>p.rightFootLift))>.015);
  assert.ok(Math.abs(poses.at(-1).upperArm)<.001);
  m.update({gesture:'point'});assert.ok(Math.abs(settle(m,.5).bodyYaw-Math.PI*2)<.001);
});

test('new gestures are bilateral where needed, bounded, and smoothly interruptible',()=>{
  const m=new MotionController(()=>.5);
  for(const gesture of ['raise-arms','shrug','point','twirl']){
    m.update({gesture});let previous={...m.step(1/60)},peak={};
    for(let i=0;i<330;i++){
      const w={...m.step(1/60)};
      for(const [key,value] of Object.entries(w)){
        assert.ok(Number.isFinite(value),key);
        if(['upperArm','leftArm','point','shoulderLift','bodyYaw'].includes(key))assert.ok(Math.abs(value-previous[key])<.06,`${gesture} ${key} discontinuity`);
        peak[key]=Math.max(peak[key]??0,value);
      }
      previous=w;
    }
    if(gesture==='raise-arms'){assert.ok(peak.upperArm>2.2&&peak.leftArm>2.2);}
    if(gesture==='shrug'){assert.ok(peak.shoulderLift>.14&&peak.palmUp>.85);}
    if(gesture==='point'){assert.ok(peak.point>.9&&peak.upperArm>1.35);}
  }
  m.update({reducedMotion:true});const w=settle(m,4);
  for(const key of ['upperArm','leftArm','leftForearmLift','shoulderLift','palmUp','point','rightFootLift'])assert.ok(Math.abs(w[key])<.0001,key);
  assert.ok(Math.abs(Math.sin(w.bodyYaw))<.001);
});

test('gesture reach leads the wrists and fingers, with staggered bilateral arms',()=>{
  const m=new MotionController(()=>.5);m.update({gesture:'raise-arms'});
  const w=settle(m,.75,60);
  assert.ok(w.upperArm/2.35>w.leftArm/2.3+.04,'right arm starts first');
  assert.ok(w.upperArm/2.35>w.rightOpen/.55+.08,'hand opens after reaching begins');
  const pose=settle(m,1.5,60);assert.ok(pose.rightOpen>.5&&pose.leftOpen>.5);
});
test('interrupted arm motion preserves velocity instead of abruptly reversing',()=>{
  const m=new MotionController(()=>.5);m.update({gesture:'raise-arms'});settle(m,.75,60);
  const dt=1/6000,a=m.step(dt).upperArm,b=m.step(dt).upperArm;
  m.update({gesture:'cancel'});const c=m.step(dt).upperArm;
  assert.ok((b-a)/dt>1);assert.ok(Math.abs((c-b)/dt-(b-a)/dt)<.03);
  const end=settle(m,3);assert.ok(Math.abs(end.upperArm)<.001&&!m.moving);
});
test('tempo changes gesture speed without skipping orientation or increasing its amplitude',()=>{
  const turn=speed=>{const m=new MotionController(()=>.5);m.update({gesture:'twirl',gestureSpeed:speed});const halfway=settle(m,2);return {m,halfway};};
  const slow=turn(.65),fast=turn(1.5);assert.ok(fast.halfway.bodyYaw>slow.halfway.bodyYaw+2);
  for(const {m} of [slow,fast])assert.ok(Math.abs(settle(m,9).bodyYaw-Math.PI*2)<.001);
});
