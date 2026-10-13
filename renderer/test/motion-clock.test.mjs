import test from 'node:test';
import assert from 'node:assert/strict';
import {MotionClock} from '../.test-build/motion-clock.js';
import {MotionController} from '../.test-build/motion.js';

test('fixed simulation is identical across 30, 60, 120 Hz and uneven callbacks',()=>{
  function simulate(deltas){const clock=new MotionClock(),motion=new MotionController(()=>.5);motion.update({gesture:'twirl'});let w,steps=0;
    for(const dt of deltas){const tick=clock.advance(dt,true);steps+=tick.steps;for(let i=0;i<tick.steps;i++)w={...motion.step(clock.step)};}
    return {steps,w};}
  const baseline=simulate(Array(300).fill(1/30));
  for(const deltas of [Array(600).fill(1/60),Array(1200).fill(1/120),Array.from({length:200},()=>[.008,.025,.017]).flat()])assert.deepEqual(simulate(deltas),baseline);
  assert.equal(baseline.steps,600);
});
test('drawing uses 60 Hz during gestures and 30 Hz when idle without losing fractional time',()=>{
  for(const moving of [true,false]){const clock=new MotionClock();let draws=0,steps=0;
    for(let i=0;i<1200;i++){const tick=clock.advance(1/120,moving);draws+=Number(tick.draw);steps+=tick.steps;}
    assert.equal(steps,600);assert.equal(draws,moving?600:300);
  }
});
test('stalls, invalid timing and visibility resets cannot create unbounded catch-up',()=>{
  const clock=new MotionClock();assert.equal(clock.advance(120,true).steps,6);
  assert.deepEqual(clock.advance(NaN,true),{steps:0,draw:false});
  clock.advance(.008,true);clock.reset();assert.equal(clock.advance(.01,true).steps,0);
  assert.equal(clock.advance(-1,true).steps,0);
});
test('millisecond callback jitter does not skip otherwise available gesture frames',()=>{
  for(const moving of [true,false]){
    const clock=new MotionClock();let draws=0,steps=0;
    for(let i=0;i<600;i++){
      const tick=clock.advance([.016,.017,.017][i%3],moving);
      draws+=Number(tick.draw);steps+=tick.steps;
    }
    assert.equal(steps,600,'presentation tolerance must not change physics time');
    assert.equal(draws,moving?600:300);
  }
});
test('presentation jitter allowance retains debt and does not increase the frame cap',()=>{
  const clock=new MotionClock();let draws=0,steps=0,time=0;
  for(let i=0;i<10000;i++){
    const dt=.008+(.0001*(i%11));time+=dt;
    const tick=clock.advance(dt,true);draws+=Number(tick.draw);steps+=tick.steps;
    assert.ok(draws<=Math.floor((time+.001)*60+1e-7));
  }
  assert.equal(steps,Math.floor(time*60+1e-7));
});
