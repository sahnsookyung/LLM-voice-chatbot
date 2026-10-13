import {asset} from './builtin';
import {emotions, type Emotion} from './motion';
import {makeSequence, SequencePlayer, type Cue, type SequenceMode} from './sequence';
import type {Bridge} from './types';
import {mountLookLibrary} from './look-library';
import {lookURL, type Look} from './looks';
import {mountWardrobe} from './wardrobe-controls';

export function mountStudio(host: HTMLElement, send: (event: Bridge) => void) {
  document.body.classList.add('studio');
  const layout = document.createElement('main');
  layout.innerHTML = `<header><div><span class="eyebrow">ARI / CHARACTER STUDIO</span><h1>From illustration to motion.</h1><p>Compare the likeness. Rehearse expressions and movement.</p></div><span class="pill">Local preview</span></header>
  <section class="workspace"><div class="preview"><div class="preview-label">Live character <span id="view-label">Full-body illustration</span></div><div id="stage-slot"><div id="reference-frame" hidden><img id="reference-target" src="${asset('full-body-neutral-v1.png')}" alt="Approved Ari illustration for side-by-side likeness comparison" hidden></div></div><div class="floor"></div></div>
  <aside><h2>View & movement</h2><label for="mode">Appearance</label><select id="mode"><option value="illustration">Illustration · approved likeness</option><option value="model">VRoid · refined face</option></select>
  <div class="row"><div class="field"><label for="framing">Framing</label><select id="framing"><option value="full">Entire body</option><option value="portrait">Portrait</option></select></div><button id="reset">Reset</button></div>
  <section id="wardrobe-controls"></section>
  <label for="reference-look">Comparison reference</label><select id="reference-look" disabled><option value="original">Original Ari</option></select>
  <div id="model-controls" hidden><label for="angle">Turn character <output id="angle-value">0°</output></label><input id="angle" type="range" min="-180" max="180" value="0" step="1">
  <div class="row gestures"><button data-gesture="wave">Wave</button><button data-gesture="nod">Nod</button><button data-gesture="explain">Explain</button><button data-gesture="shift">Shift</button><button data-gesture="step">Step</button><button data-gesture="hair-sway">Hair sway</button><button data-gesture="twirl">Cute twirl</button><button data-gesture="raise-arms">Raise arms</button><button data-gesture="shrug">Shrug</button><button data-gesture="point">Point</button></div>
  <div class="row gestures"><button data-gesture="double-peace">Double peace</button><button data-gesture="peace-eye">Peace by eye</button><button data-gesture="thinking">Thoughtful</button><button data-gesture="adjust-glasses">Adjust glasses</button></div>
  <label>Expression details</label><div class="row gestures"><button data-detail="soft-smile">Gentle smile</button><button data-detail="playful">Playful wink</button><button data-detail="pensive">Pensive</button><button data-detail="none">Clear detail</button></div>
  <label for="gesture-speed">Movement speed <output id="gesture-speed-value">1.00×</output></label><input id="gesture-speed" type="range" min=".65" max="1.5" step=".05" value="1"><p class="hint">Try a slow and a fast twirl to compare the hair’s swing and settling.</p>
  <label for="intensity">Expression intensity <output id="intensity-value">80%</output></label><input id="intensity" type="range" min="0" max="1" step=".05" value=".8"><button id="speech">Test mouth movement</button></div>
  <label class="check"><input id="reduced" type="checkbox"> Reduce motion</label>
  <section class="testbench" aria-labelledby="test-title"><h2 id="test-title">Expression rehearsal</h2>
  <label for="sequence-mode">Sequence</label><select id="sequence-mode"><option value="shuffle">Shuffle all eight</option><option value="conversation">Conversation arc</option><option value="pairs">Every transition · 56 pairs</option><option value="interruptions">Rapid interruptions</option><option value="custom">Custom order</option><option value="movement">Movement & hair rehearsal</option></select>
  <div id="custom-controls" hidden><label for="custom-order">Emotion names, separated by commas</label><textarea id="custom-order" rows="2">neutral, curious, happy, embarrassed, concerned, surprised, annoyed, sleepy</textarea></div>
  <label for="pace">Time per expression <output id="pace-value">4 s</output></label><input id="pace" type="range" min="2" max="8" value="4" step=".5">
  <div class="row"><label class="check"><input id="sequence-gestures" type="checkbox" checked> Gestures</label><label class="check"><input id="sequence-speech" type="checkbox"> Mouth test</label></div>
  <div class="row"><button id="sequence-play">Play</button><button id="sequence-next">Next</button><button id="sequence-stop">Stop</button></div>
  <p id="sequence-status" role="status" aria-live="polite">Ready. Shuffles include every expression without repeats.</p><progress id="sequence-progress" value="0" max="1" aria-label="Sequence progress"></progress>
  <p class="hint">Select any emotion below to interrupt a sequence. Rehearsals pause when this window is hidden.</p></section>
  <p id="status" role="status">Select VRoid for continuous expressions and full-body gestures.</p>
  <details class="note"><summary>Likeness notes</summary><p>Ari v9 tucks a little hair behind the ears, staggers gesture timing, and adds relaxed wrists and individual finger curls. Compare the portrait from several angles; the reference’s asymmetric fringe and sculpted clothing folds remain further work.</p><p>The new wardrobe options are initial 3D adaptations. Tailoring, braid detail and cloth drape remain simpler than the illustrations. Hair and skirt use guide bones and approximate contact volumes.</p><p>Studio saves its outfit and hairstyle separately from the desktop overlay. Both use the v9 face and body. The latest editable VRoid Studio project is v6; later mesh, texture and animation changes were made after export and do not appear in that project.</p></details></aside></section>
  <section id="look-library" class="look-library" aria-label="Reference look library"></section>
  <section class="expressions"><div class="section-title"><h2>Eight expressions. One Ari.</h2><p>Select any emotion, even during a transition.</p></div><div class="emotion-grid"></div></section>
  <section class="reference"><div><span class="eyebrow">MODEL SHEET</span><h2>Front, side, and back.</h2><p>The illustration remains the likeness target.</p></div><img src="${asset('turnaround-v1.png')}" alt="Ari character turnaround: front, side and back" loading="lazy"></section>`;
  document.body.appendChild(layout);
  layout.querySelector('#stage-slot')!.appendChild(host);
  const get = <T extends HTMLElement>(id: string) => layout.querySelector<T>(`#${id}`)!;
  const mode = get<HTMLSelectElement>('mode'), framing = get<HTMLSelectElement>('framing');
  const target = get<HTMLImageElement>('reference-target');
  const angle = get<HTMLInputElement>('angle'), intensity = get<HTMLInputElement>('intensity');
  const speed = get<HTMLInputElement>('gesture-speed');
  const reduced = get<HTMLInputElement>('reduced');
  const sequenceMode = get<HTMLSelectElement>('sequence-mode'), custom = get<HTMLTextAreaElement>('custom-order');
  const pace = get<HTMLInputElement>('pace'), gestures = get<HTMLInputElement>('sequence-gestures'), mouth = get<HTMLInputElement>('sequence-speech');
  const play = get<HTMLButtonElement>('sequence-play'), progress = get<HTMLProgressElement>('sequence-progress');
  const status = get<HTMLParagraphElement>('sequence-status');
  let emotion: Emotion = 'neutral', player: SequencePlayer | undefined;
  let referenceLook: Look | null = null;
  let referenceRequest = '', referenceGeneration = 0;
  const setReference = (url: string) => {
    if (referenceRequest === url) return;
    referenceRequest=url;
    const generation=++referenceGeneration;
    if (target.src === url) return;
    const next = new Image(); next.src=url;
    next.decode().then(() => {if (generation === referenceGeneration) target.src=url;}).catch(() => {
      if (generation === referenceGeneration) {referenceRequest='';get('status').textContent='The comparison image could not reload. The previous reference is retained; check the image file.';}
    });
  };
  let timer: ReturnType<typeof setInterval> | undefined, speechTimer: ReturnType<typeof setInterval> | undefined;
  let last = 0, speechTime = 0, loadingSince = 0;
  const grid = layout.querySelector('.emotion-grid')!;
  const portraitReference = () => emotion === 'neutral' ? 'anime-reference-v2.png' : `emotions/${emotion}.png`;
  const showEmotion = (value: Emotion) => {
    emotion = value; send({expression:value});
    for (const button of grid.children) button.setAttribute('aria-pressed',String((button as HTMLElement).dataset.emotion === value));
    if (framing.value === 'portrait' && !referenceLook) setReference(asset(portraitReference()));
  };
  const updateView = () => {
    send({visualMode:mode.value as 'model'|'illustration',framing:framing.value as 'full'|'portrait'});
    const comparing=mode.value==='model'&&(framing.value==='portrait'||!!referenceLook);
    get('model-controls').hidden = mode.value !== 'model'; target.hidden = !comparing;
    get('reference-frame').hidden = !comparing;
    get('reference-frame').dataset.crop=String(!!referenceLook && framing.value==='portrait');
    get('status').textContent = mode.value === 'model' ? 'Ari v9 · flowing gestures, relaxed hands, and responsive hair. Try the movement rehearsal.' : 'Select VRoid for continuous expressions and full-body gestures.';
    setReference(referenceLook ? lookURL(referenceLook) : asset(framing.value === 'full' ? 'full-body-neutral-v1.png' : portraitReference()));
    target.alt=referenceLook ? `Ari reference: ${referenceLook.name}` : 'Approved Ari illustration for likeness comparison';
    host.style.width = comparing ? '50%' : '100%';
    get('view-label').textContent = `${framing.value === 'full' ? 'Full body' : 'Portrait'} · ${mode.value === 'model' ? 'VRoid character' : referenceLook?.name ?? 'illustration'}`;
    if (referenceLook) get('status').textContent = mode.value === 'model' ? `Comparing ${referenceLook.name} with your selected 3D outfit and hairstyle.` : `${referenceLook.name} · static illustration. Outfit and hairstyle combinations are available in 3D.`;
  };
  const stopSpeech = () => {clearInterval(speechTimer);speechTimer=undefined;send({level:0,state:'idle'});};
  const describe = (prefix = '') => {
    if (!player) return;
    const next = player.cues[player.index+1]?.emotion;
    status.textContent = `${prefix}${player.current.emotion}${next ? ` → ${next}` : ''} · ${player.index+1}/${player.cues.length} · ${player.visited.size}/56 different transitions`;
    progress.max=player.cues.length;progress.value=player.index+(player.done?1:player.elapsed/player.current.duration);
    play.textContent=player.playing?'Pause':player.done?'Replay':'Resume';
  };
  const pause = (reason='Paused · ') => {
    clearInterval(timer);timer=undefined;
    if(player)player.playing=false;
    stopSpeech();describe(reason);
  };
  const apply = (cue: Cue) => {
    showEmotion(cue.emotion);
    send({state:mouth.checked?'speaking':'idle',level:0,...(gestures.checked && cue.gesture ? {gesture:cue.gesture}: {})});
    describe();
  };
  const prepare = () => {
    try {const cues=makeSequence(sequenceMode.value as SequenceMode,Number(pace.value),custom.value);
      if(sequenceMode.value==='movement')for(const cue of cues)cue.duration/=Math.min(1,Number(speed.value));
      player=new SequencePlayer(cues);return true;}
    catch(error){status.textContent=(error as Error).message;return false;}
  };
  const tick = () => {
    const character=window.characterStatus(false);
    if(document.hidden || character.visible === false){pause('Paused while hidden · ');return;}
    const now=performance.now(), dt=Math.min((now-last)/1000,.25);last=now;
    if(!character.modelReady){
      if(now-loadingSince>15000){pause();status.textContent='Character could not load. Switch to Illustration, then try VRoid again.';}
      else status.textContent='Loading character before rehearsal…';
      return;
    }
    const cue=player?.step(dt);
    if(status.textContent==='Loading character before rehearsal…')describe();
    if(cue)apply(cue);
    if(player?.done){pause('Finished · ');return;}
    if(player?.playing && mouth.checked){speechTime+=dt;send({state:'speaking',level:Math.max(0,Math.sin(speechTime*3))*Math.abs(Math.sin(speechTime*9))*.4});}
    if(player)progress.value=player.index+player.elapsed/player.current.duration;
  };
  play.addEventListener('click',()=>{
    if(player?.playing){pause();return;}
    stopSpeech();
    if(!player || player.done){if(!prepare())return;apply(player!.current);}
    mode.value='model';if(sequenceMode.value==='movement')framing.value='full';updateView();player!.playing=true;describe();last=loadingSince=performance.now();
    clearInterval(timer);timer=setInterval(tick,50);
  });
  get('sequence-next').addEventListener('click',()=>{
    pause();
    if(!player || player.done){if(!prepare())return;apply(player!.current);}
    else {const cue=player.next();if(cue)apply(cue);}
    mode.value='model';updateView();describe(player!.done?'Finished · ':'Paused · ');
  });
  const stop = () => {pause();send({gesture:'cancel'});player=undefined;play.textContent='Play';progress.value=0;status.textContent='Stopped. Select Play to start a new sequence.';};
  get('sequence-stop').addEventListener('click',stop);
  sequenceMode.addEventListener('change',()=>{stop();get('custom-controls').hidden=sequenceMode.value!=='custom';pace.disabled=['interruptions','movement'].includes(sequenceMode.value);get('pace-value').textContent=sequenceMode.value==='movement'?'5–8 s':pace.disabled?'0.4 s':`${pace.value} s`;});
  custom.addEventListener('input',stop);
  pace.addEventListener('input',()=>{stop();get('pace-value').textContent=`${pace.value} s`;});
  mouth.addEventListener('change',()=>{if(!mouth.checked)stopSpeech();});
  mode.addEventListener('change',()=>{pause();updateView();});framing.addEventListener('change',updateView);
  angle.addEventListener('input',()=>{send({angle:Number(angle.value)});get('angle-value').textContent=`${angle.value}°`;});
  get('reset').addEventListener('click',()=>{stop();angle.value='0';angle.dispatchEvent(new Event('input'));showEmotion('neutral');send({gesture:'cancel',state:'idle',level:0});});
  intensity.addEventListener('input',()=>{send({intensity:Number(intensity.value)});get('intensity-value').textContent=`${Math.round(Number(intensity.value)*100)}%`;});
  speed.addEventListener('input',()=>{stop();send({gestureSpeed:Number(speed.value)});get('gesture-speed-value').textContent=`${Number(speed.value).toFixed(2)}×`;});
  for(const button of layout.querySelectorAll<HTMLButtonElement>('[data-gesture]'))button.addEventListener('click',()=>{
    pause();const gesture=button.dataset.gesture;
    if(['twirl','raise-arms','shrug','point'].includes(gesture??'')){framing.value='full';updateView();}
    send({gesture});
  });
  for(const button of layout.querySelectorAll<HTMLButtonElement>('[data-detail]'))button.addEventListener('click',()=>{
    pause();const detail=button.dataset.detail as NonNullable<Bridge['expressionDetail']>;
    showEmotion(detail==='pensive'?'curious':detail==='none'?'neutral':'happy');send({expressionDetail:detail});
    get('status').textContent=`Expression detail: ${button.textContent}. Speech still blends with the face.`;
  });
  reduced.checked=matchMedia('(prefers-reduced-motion: reduce)').matches;
  reduced.addEventListener('change',()=>send({reducedMotion:reduced.checked}));send({reducedMotion:reduced.checked});
  get('speech').addEventListener('click',()=>{pause();let ticks=0;send({state:'speaking'});speechTimer=setInterval(()=>{if(++ticks>80||document.hidden||window.characterStatus(false).visible===false){stopSpeech();return;}send({level:Math.abs(Math.sin(ticks*.38))*.45});},40);});
  document.addEventListener('visibilitychange',()=>{if(document.hidden)pause('Paused while hidden · ');});
  window.addEventListener('pagehide',()=>{clearInterval(timer);clearInterval(speechTimer);});
  for(const value of emotions){
    const button=document.createElement('button');button.className='emotion';button.dataset.emotion=value;button.setAttribute('aria-pressed',String(value==='neutral'));
    const img=new Image();img.src=asset(`emotions/${value}.png`);img.loading='lazy';img.alt=`Ari: ${value}`;
    const label=document.createElement('span');label.textContent=value;button.append(img,label);
    button.addEventListener('click',()=>{pause();if(mode.value==='illustration'){framing.value='portrait';updateView();}showEmotion(value);player=undefined;progress.value=0;play.textContent='Play';status.textContent=`Manual · ${value}. Select another emotion to test an interruption.`;});
    grid.appendChild(button);
  }
  mountLookLibrary(get('look-library'), get<HTMLSelectElement>('reference-look'), look => {
    referenceLook=look;
    send({referenceLook:look});
    updateView();
  });
  mountWardrobe(get('wardrobe-controls'), wardrobe=>{
    pause();mode.value='model';send({wardrobe});updateView();
  },accessories=>{
    pause();mode.value='model';send({accessories});updateView();
  });
}
