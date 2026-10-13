import test from 'node:test';
import assert from 'node:assert/strict';
import {allExpressionPairs,makeSequence,SequencePlayer} from '../.test-build/sequence.js';
import {emotions,MotionController} from '../.test-build/motion.js';
const seeded=()=>{let seed=42;return()=>{seed=(Math.imul(seed,1664525)+1013904223)>>>0;return seed/2**32;};};
test('pair rehearsal covers all 56 directed changes without duplicates or neutral resets',()=>{
  const order=allExpressionPairs(seeded());
  assert.equal(order.length,57);
  const pairs=new Set(order.slice(1).map((to,i)=>`${order[i]}→${to}`));
  assert.equal(pairs.size,56);
  for(const a of emotions)for(const b of emotions)if(a!==b)assert.ok(pairs.has(`${a}→${b}`));
});
test('shuffle includes every emotion once and custom order preserves repetitions',()=>{
  assert.deepEqual(new Set(makeSequence('shuffle',4,'',seeded()).map(c=>c.emotion)),new Set(emotions));
  assert.deepEqual(makeSequence('custom',4,'happy, sleepy, happy').map(c=>c.emotion),['happy','sleepy','happy']);
  assert.throws(()=>makeSequence('custom',4,'happy, typo'));
  assert.throws(()=>makeSequence('custom',4,'happy'));
});
test('paused and suspended rehearsal cannot accumulate timers or skip expressions',()=>{
  const p=new SequencePlayer(makeSequence('custom',2,'happy, concerned'));
  p.step(100);assert.equal(p.elapsed,0);
  p.playing=true;p.step(100);assert.equal(p.elapsed,.25);assert.equal(p.index,0);
  p.playing=false;p.step(100);assert.equal(p.elapsed,.25);
  p.next();assert.equal(p.current.emotion,'concerned');assert.equal(p.visited.size,1);
  p.playing=true;for(let i=0;i<10;i++)p.step(.25);
  assert.ok(p.done);assert.ok(!p.playing);assert.equal(p.index,1);
});
test('all expression pairs and rapid interruptions stay finite, smooth, and settle cleanly',()=>{
  const motion=new MotionController(seeded());
  const emotional=['Happy','Curious','Concerned','Surprised','Annoyed','Embarrassed','Sleepy','Blush'];
  let last={...motion.step(1/60)};
  for(const expression of allExpressionPairs(seeded())) {
    motion.update({expression,intensity:1});
    for(let i=0;i<90;i++) {
      const w={...motion.step(1/60)};
      for(const key of emotional) {assert.ok(w[key]>=0&&w[key]<=1,key);assert.ok(Math.abs(w[key]-last[key])<.09,`${expression} ${key} snapped`);}
      last=w;
    }
  }
  for(let i=0;i<240;i++) {motion.update({expression:emotions[i%8]});last={...motion.step(1/60)};assert.ok(Object.values(last).every(Number.isFinite));}
  motion.update({expression:'neutral'});for(let i=0;i<300;i++)last=motion.step(1/60);
  for(const key of emotional)assert.ok(last[key]<.0001,key);
});
test('sleepiness and blush blend out, and speech continues through changing emotions',()=>{
  const m=new MotionController(()=>.5);m.update({expression:'sleepy'});for(let i=0;i<60;i++)m.step(1/30);
  m.update({expression:'embarrassed'});assert.ok(m.step(1/60).BlinkL>.3);
  for(let i=0;i<120;i++){m.update({level:.35});m.step(1/60);}
  const w={...m.step(1/60)};assert.ok(w.Blush>.5);assert.ok(w.JawOpen>.4);assert.ok(w.Embarrassed>.75);assert.ok(w.Sleepy<.001);
  m.update({expression:'neutral',level:0});assert.ok(m.step(1/60).Blush>.4);
});

test('movement rehearsal gives all new gestures time to complete and returns to neutral',()=>{
  const cues=makeSequence('movement');
  assert.deepEqual(cues.map(c=>c.gesture),['hair-sway','twirl','raise-arms','shrug','point','cancel']);
  assert.ok(cues.slice(0,-1).every(c=>c.duration>=5));
  assert.equal(cues.at(-1).emotion,'neutral');
});
