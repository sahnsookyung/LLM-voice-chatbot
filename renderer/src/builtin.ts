import {Avatar3D} from './avatar3d';
import {MotionController} from './motion';
import {MotionClock} from './motion-clock';
import type {Bridge, Framing, VisualMode} from './types';
import {lookURL, type Look} from './looks';
import {defaultWardrobe, parseWardrobe, wardrobeKey, type WardrobeSelection} from './wardrobe';
export const asset=(path:string)=>new URL(`./ari/${path}`,document.baseURI).href;

export class BuiltinAvatar {
  private controller=new MotionController();
  private clock=new MotionClock();
  private model?:Avatar3D; private active=true; private visible=true; private generation=0;
  private frame=0; private last=0; private mode:VisualMode='illustration'; private framing:Framing='full';
  private image=new Image(); private imagePath=''; private requestedPath=''; private angle=0; private frames=0;
  private observer:ResizeObserver;
  private referenceLook:Look|null=null;
  private wardrobe:WardrobeSelection={...defaultWardrobe};
  private appliedWardrobe='';
  private modelLoading=false;
  private modelError='';
  constructor(private host:HTMLElement,private error:(message:string)=>void) {
    host.classList.add('avatar-stage');this.image.className='ari-art';this.image.alt='Ari, black-haired anime character';host.appendChild(this.image);
    this.observer=new ResizeObserver(()=>this.model?.resize(this.framing));this.observer.observe(host);
    document.addEventListener('visibilitychange',this.visibility);
    this.updateArt();this.schedule();
  }
  private visibility=()=>{this.last=0;this.clock.reset();if(document.hidden)cancelAnimationFrame(this.frame);else this.schedule();};
  private schedule(){cancelAnimationFrame(this.frame);if(this.active&&this.visible&&this.model&&!document.hidden)this.frame=requestAnimationFrame(this.tick);}
  private tick=(now:number)=>{
    if(!this.active||!this.visible||document.hidden)return;
    const elapsed=this.last?(now-this.last)/1000:this.clock.step;this.last=now;
    const {steps,draw}=this.clock.advance(elapsed,this.controller.moving);
    for(let i=0;i<steps;i++)this.model?.update(this.controller.step(this.clock.step),this.clock.step,this.controller.reduced);
    if(draw&&this.model){this.model.draw();this.frames++;}
    this.frame=requestAnimationFrame(this.tick);
  };
  private updateArt(){
    const path=this.referenceLook ? lookURL(this.referenceLook) : asset(this.framing==='full'?'full-body-neutral-v1.png':`emotions/${this.controller.expression}.png`);
    if(path===this.requestedPath)return;
    this.requestedPath=path;
    const generation=++this.generation;
    if(path===this.imagePath)return;
    const next=new Image();next.className='ari-art';next.alt=`Ari — ${this.framing==='full'?'full body':this.controller.expression}`;
    next.alt=this.referenceLook ? `Ari — ${this.referenceLook.name}` : next.alt;
    next.src=path;
    next.decode().then(()=>{
      if(!this.active||generation!==this.generation)return;
      this.image.replaceWith(next);this.image=next;this.imagePath=path;
      this.image.hidden=this.mode==='model';
      if(!this.controller.reduced && this.mode==='illustration')next.animate([{opacity:.3},{opacity:1}],{duration:220});
    }).catch(()=>{if(this.active&&generation===this.generation){this.requestedPath='';this.error('Ari’s illustration could not load. The previous image is retained; check the file and try Reload artwork.');}});
  }
  async setMode(mode:VisualMode){
    this.mode=mode;
    if(mode==='illustration'){this.model?.dispose();this.model=undefined;this.image.hidden=false;delete this.host.dataset.loading;this.schedule();return;}
    await this.loadModel();
  }
  private async loadModel(){
    if(this.modelLoading||!this.active||this.mode!=='model'||(this.model&&this.appliedWardrobe===wardrobeKey(this.wardrobe)))return;
    this.modelLoading=true;this.modelError='';
    this.host.dataset.loading='true';
    try {
      // At most one candidate is loading. Rapid input replaces the requested
      // selection; stale candidates are disposed before the newest one starts.
      while(this.active&&this.mode==='model'){
        const selection={...this.wardrobe},key=wardrobeKey(selection);
        const model=await Avatar3D.create(this.host,this.error,selection);
        if(!this.active||this.mode!=='model'){model.dispose();return;}
        if(key!==wardrobeKey(this.wardrobe)){model.dispose();continue;}
        try{model.resize(this.framing);model.setAngle(this.angle,true);model.prime(this.controller.step(0),this.controller.reduced);}
        catch(error){model.dispose();throw error;}
        const previous=this.model;this.model=model;this.appliedWardrobe=key;previous?.dispose();
        this.image.hidden=true;this.last=0;this.clock.reset();this.schedule();break;
      }
    }catch(error){
      if(this.active){this.modelError=(error as Error).message;this.error(`The selected 3D look could not load. The previous preview is retained. ${this.modelError}`);}
    }finally{this.modelLoading=false;delete this.host.dataset.loading;}
  }
  event(e:Bridge){
    this.controller.update(e);
    if(e.wardrobe){try{this.wardrobe=parseWardrobe(e.wardrobe);}catch{this.error('Unknown outfit or hairstyle. The current look is retained.');return;}}
    if(e.referenceLook!==undefined)this.referenceLook=e.referenceLook;
    if(e.visualMode && (e.visualMode==='model'||e.visualMode==='illustration') && e.visualMode!==this.mode)void this.setMode(e.visualMode);
    if(e.framing && (e.framing==='full'||e.framing==='portrait')){this.framing=e.framing;this.model?.resize(this.framing);}
    if(e.angle!==undefined&&Number.isFinite(e.angle)){this.angle=e.angle;this.model?.setAngle(e.angle);}
    if(e.expression||e.framing||e.referenceLook!==undefined)this.updateArt();
    this.host.dataset.framing=this.framing;
    this.host.dataset.reference=String(!!this.referenceLook);
    this.host.dataset.reduced=String(this.controller.reduced);
    if(e.visible!==undefined){this.visible=e.visible;this.last=0;this.clock.reset();this.schedule();}
    if(e.wardrobe&&this.mode==='model')void this.loadModel();
  }
  status(diagnostics=true){return {kind:'ari',mode:this.mode,framing:this.framing,expression:this.controller.expression,referenceLook:this.referenceLook?.id??'original',imageURL:this.imagePath,loaded:!!this.imagePath,modelReady:!!this.model,modelLoading:this.modelLoading,modelError:this.modelError,requestedWardrobe:{...this.wardrobe},appliedWardrobe:this.appliedWardrobe,visible:this.visible,frames:this.frames,...(diagnostics?this.model?.stats():{})};}
  dispose(){this.active=false;this.generation++;cancelAnimationFrame(this.frame);this.observer.disconnect();document.removeEventListener('visibilitychange',this.visibility);this.model?.dispose();this.host.replaceChildren();}
}
