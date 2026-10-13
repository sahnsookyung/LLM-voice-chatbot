import {BuiltinAvatar} from './builtin';
import {createLegacy} from './legacy';
import {mountStudio} from './studio';
import type {Bridge} from './types';
const report=(message:string)=>{
  window.webkit?.messageHandlers.character.postMessage({type:'error',message});
  const status=document.getElementById('status');if(status)status.textContent=message;
};
let host=document.createElement('div');host.id='avatar';document.body.appendChild(host);
let builtin:BuiltinAvatar|undefined=new BuiltinAvatar(host,report);
let legacy:Awaited<ReturnType<typeof createLegacy>>|undefined;
let generation=0;let latest:Bridge={framing:'full',visualMode:'illustration'};
const studio=document.body.dataset.studio==='true'||new URLSearchParams(location.search).has('studio');
async function selectCharacter(event:Bridge){
  const id=++generation;
  if(event.character?.id==='ari'){
    legacy?.dispose();legacy=undefined;host.hidden=false;
    builtin??=new BuiltinAvatar(host,report);builtin.event(latest);return;
  }
  builtin?.dispose();builtin=undefined;host.hidden=true;
  const renderer=legacy??await createLegacy();
  if(id!==generation){if(renderer!==legacy)renderer.dispose();return;}
  legacy=renderer;legacy.event(latest);
}
window.companion=(event:Bridge)=>{
  latest={...latest,...event};
  if(event.character){void selectCharacter(event).catch(()=>report('The selected character could not load.'));return;}
  builtin?.event(event);legacy?.event(event);
};
window.characterStatus=(diagnostics=true)=>builtin?.status(diagnostics)??{kind:'imported',loaded:!!legacy};
if(studio)mountStudio(host,event=>window.companion(event));
window.webkit?.messageHandlers.character.postMessage({type:'ready'});
