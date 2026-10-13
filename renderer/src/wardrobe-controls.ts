import {hairstyles,outfits,parseWardrobe,restoreWardrobe,wardrobeCombinations,wardrobeKey,wardrobeLabels,type WardrobeSelection} from './wardrobe';
import {defaultAccessories,parseAccessories,compatibleAccessories,type Accessories} from './accessories';

export function mountWardrobe(container:HTMLElement,apply:(value:WardrobeSelection)=>void,applyAccessories:(value:Accessories)=>void) {
  const combinations=wardrobeCombinations();
  container.classList.add('testbench');
  container.innerHTML=`<h2>3D wardrobe</h2><p class="hint">Mix an outfit with any hairstyle. These controls change animated Ari; illustrations below remain design references.</p>
    <label for="outfit">Outfit</label><select id="outfit"></select><label for="hairstyle">Hairstyle</label><select id="hairstyle"></select>
    <div class="row"><div class="field"><label for="glasses">Glasses</label><select id="glasses"><option value="none">None</option><option value="round">Round frames</option></select></div><div class="field"><label for="headwear">Headwear</label><select id="headwear"><option value="none">None</option><option value="beret">Butter beret</option></select></div></div>
    <p id="accessory-fit" class="hint" role="status"></p>
    <div class="row"><button id="wardrobe-cycle">Cycle all ${combinations.length} looks</button><button id="wardrobe-stop" disabled>Stop cycle</button></div>
    <p id="wardrobe-status" role="status" aria-live="polite">Original cardigan · long layered hair</p>`;
  const outfit=container.querySelector<HTMLSelectElement>('#outfit')!,hair=container.querySelector<HTMLSelectElement>('#hairstyle')!;
  const start=container.querySelector<HTMLButtonElement>('#wardrobe-cycle')!,stop=container.querySelector<HTMLButtonElement>('#wardrobe-stop')!,status=container.querySelector<HTMLElement>('#wardrobe-status')!;
  const glasses=container.querySelector<HTMLSelectElement>('#glasses')!,headwear=container.querySelector<HTMLSelectElement>('#headwear')!;
  const fit=container.querySelector<HTMLElement>('#accessory-fit')!;
  let accessories:Accessories={...defaultAccessories};
  try{accessories=parseAccessories(JSON.parse(localStorage.getItem('ari-accessories-v1')??'null'));}catch{/* Previous versions had no accessories. */}
  glasses.value=accessories.glasses;headwear.value=accessories.headwear;
  const fitAccessories=()=>{
    const removed=accessories.headwear==='beret'&&selected.hair==='double-buns';
    accessories=compatibleAccessories(accessories,selected.hair);headwear.value=accessories.headwear;
    headwear.querySelector<HTMLOptionElement>('option[value="beret"]')!.disabled=selected.hair==='double-buns';
    fit.textContent=selected.hair==='double-buns'?(removed?'Beret removed: double buns need room above the head.':'Beret unavailable with double buns. Choose another hairstyle to wear it.'):'';
    try{localStorage.setItem('ari-accessories-v1',JSON.stringify(accessories));}catch{/* Preserve the in-session selection. */}
    applyAccessories(accessories);
  };
  const selectAccessories=()=>{accessories=parseAccessories({glasses:glasses.value,headwear:headwear.value});fitAccessories();};
  glasses.addEventListener('change',selectAccessories);headwear.addEventListener('change',selectAccessories);
  for(const id of outfits)outfit.add(new Option(id==='original'?'Original cardigan':wardrobeLabels[id],id));
  for(const id of hairstyles)hair.add(new Option(id==='original'?'Long layered hair':wardrobeLabels[id],id));
  let selected:WardrobeSelection=restoreWardrobe(null);
  try{selected=restoreWardrobe(localStorage.getItem('ari-wardrobe-v1'));}catch{/* Preview still works when storage is unavailable. */}
  let cycle:WardrobeSelection[]|undefined,index=0,loadedAt=0,requestedAt=0,notice='',timer:ReturnType<typeof setInterval>|undefined;
  const label=()=>`${outfit.selectedOptions[0].text} · ${hair.selectedOptions[0].text}`;
  const end=(message='')=>{cycle=undefined;notice=message;start.disabled=false;stop.disabled=true;};
  const publish=()=>{
    outfit.value=selected.outfit;hair.value=selected.hair;requestedAt=performance.now();loadedAt=0;
    fitAccessories();
    try{localStorage.setItem('ari-wardrobe-v1',JSON.stringify(selected));}catch{/* Keep the in-session selection. */}
    status.textContent=`Loading ${label()}…`;apply({...selected});
  };
  const choose=()=>{end();selected=parseWardrobe({outfit:outfit.value,hair:hair.value});publish();};
  outfit.addEventListener('change',choose);hair.addEventListener('change',choose);
  start.addEventListener('click',()=>{cycle=combinations;index=0;notice='';selected=cycle[0];start.disabled=true;stop.disabled=false;publish();});
  stop.addEventListener('click',()=>{end('Cycle stopped');status.textContent=`Cycle stopped · ${label()}`;});
  timer=setInterval(()=>{
    if(document.hidden){if(cycle){end('Cycle stopped while hidden');status.textContent=`${notice} · ${label()}`;}return;}
    const state=window.characterStatus(false);
    if(state.visible===false){if(cycle){end('Cycle stopped while hidden');status.textContent=`${notice} · ${label()}`;}return;}
    if(state.modelError){end();status.textContent='This look could not load. Your previous preview is retained; reselect a look to retry.';return;}
    if(state.mode!=='model'){if(cycle)end();status.textContent=`Selected for 3D · ${label()}`;return;}
    if(state.modelLoading||state.appliedWardrobe!==wardrobeKey(selected)){
      if(performance.now()-requestedAt>20000){end();status.textContent='This look is taking longer to load. You can select another look.';}
      return;
    }
    if(!loadedAt)loadedAt=performance.now();
    const text=cycle?`Look ${index+1}/${combinations.length} · ${label()}`:`${notice||'Ready'} · ${label()}`;
    if(status.textContent!==text)status.textContent=text;
    if(cycle&&performance.now()-loadedAt>=2500){if(++index>=cycle.length){end(`Finished all ${combinations.length} combinations`);status.textContent=`${notice} · ${label()}`;}else{selected=cycle[index];publish();}}
  },200);
  window.addEventListener('pagehide',()=>clearInterval(timer),{once:true});
  outfit.value=selected.outfit;hair.value=selected.hair;
  if(wardrobeKey(selected)!=='original/original')publish();
  else if(accessories.glasses!=='none'||accessories.headwear!=='none')fitAccessories();
}
