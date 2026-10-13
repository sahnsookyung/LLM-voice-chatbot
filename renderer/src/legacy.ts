import 'pixi.js/unsafe-eval'; // Pixi's CSP-safe shader polyfill; does not use eval.
import { Application, Container, Graphics, Sprite, Texture } from 'pixi.js';

import type {Layer, Character, Bridge} from './types';
export async function createLegacy() {
const app = new Application();

await app.init({resizeTo:window,backgroundAlpha:0,antialias:true,resolution:Math.min(devicePixelRatio,2),autoDensity:true,preference:'webgl'});
document.body.appendChild(app.canvas);
app.ticker.maxFPS=30;
const root=new Container();app.stage.addChild(root);
let head=new Container(),eyes=new Container(),mouth=new Graphics(),brows=new Graphics(),blush=new Graphics();
let expression='neutral',targetLevel=0,level=0,state='idle',visible=true,reduced=false,blink=1,time=0,lastLevel=0;
let layers:{sprite:Sprite;layer:Layer}[]=[],generation=0;
let palette='teal';
const palettes:Record<string,Record<number,number>>={teal:{},plum:{[0x273b47]:0x3f2d44,[0x304d58]:0x64425e,[0x466a70]:0x875f7b,[0x3d616a]:0x79516e,[0x405b63]:0x684c65,[0x376b65]:0x886140},copper:{[0x273b47]:0x53382e,[0x304d58]:0x85563c,[0x466a70]:0xaa7851,[0x3d616a]:0x996648,[0x405b63]:0x5e6854,[0x376b65]:0x5c7856},midnight:{[0x273b47]:0x242936,[0x304d58]:0x363e50,[0x466a70]:0x56627d,[0x3d616a]:0x48566b,[0x405b63]:0x485873,[0x376b65]:0x526c99}};
function path(draw:(g:Graphics)=>void,color:number,parent:Container=root){const g=new Graphics();draw(g);g.fill(palettes[palette]?.[color]??color);parent.addChild(g);return g;}
function ellipse(x:number,y:number,w:number,h:number,c:number,parent:Container=root){return path(g=>g.ellipse(x,y,w,h),c,parent);}
function line(draw:(g:Graphics)=>void,color:number,width:number,parent:Container){const g=new Graphics();draw(g);g.stroke({color,width,cap:'round',join:'round'});parent.addChild(g);return g;}
function original(){
 root.removeChildren().forEach(c=>c.destroy({children:true}));layers=[];
 // Back hair, shoulders, and clothing stay distinct from the face rig.
 path(g=>g.moveTo(196,276).bezierCurveTo(170,45,650,23,626,325).lineTo(655,740).bezierCurveTo(552,846,241,826,149,704).closePath(),0x273b47);
 path(g=>g.moveTo(245,593).bezierCurveTo(154,609,97,670,82,985).lineTo(718,985).bezierCurveTo(704,677,650,615,550,593).closePath(),0xeae9df);
 path(g=>g.moveTo(311,573).lineTo(318,688).quadraticCurveTo(400,777,482,688).lineTo(487,573).closePath(),0xf2c9b1);
 path(g=>g.moveTo(290,637).quadraticCurveTo(400,760,508,637).lineTo(537,985).lineTo(260,985).closePath(),0x405b63);
 path(g=>g.moveTo(281,624).lineTo(389,735).lineTo(284,823).lineTo(244,708).closePath(),0xfffbef);
 path(g=>g.moveTo(516,624).lineTo(413,735).lineTo(514,823).lineTo(556,708).closePath(),0xfffbef);
 line(g=>g.moveTo(201,778).lineTo(174,974),0xb6bdb4,4,root);line(g=>g.moveTo(601,778).lineTo(628,974),0xb6bdb4,4,root);
 ellipse(400,788,12,12,0xd0b875);ellipse(400,865,7,7,0xd0b875);
 head=new Container();head.pivot.set(400,585);head.position.set(400,585);root.addChild(head);
 ellipse(222,369,34,52,0xf2c9b1,head);ellipse(578,369,34,52,0xf2c9b1,head);
 path(g=>g.moveTo(223,234).bezierCurveTo(237,113,566,111,578,246).lineTo(559,447).bezierCurveTo(541,517,454,584,402,590).bezierCurveTo(345,584,257,517,240,447).closePath(),0xffdfc6,head);
 path(g=>g.moveTo(239,382).bezierCurveTo(253,505,349,558,402,573).bezierCurveTo(454,558,547,508,560,384).lineTo(554,456).bezierCurveTo(530,523,451,583,402,590).bezierCurveTo(337,578,255,518,242,449).closePath(),0xf1c4ad,head);
 blush=new Graphics();head.addChild(blush);
 eyes=new Container();head.addChild(eyes);
 for(const x of [318,481]){
  const eye=new Container();eye.position.set(x,356);eyes.addChild(eye);
  path(g=>g.moveTo(-50,0).quadraticCurveTo(0,-38,48,-1).quadraticCurveTo(0,40,-48,9).closePath(),0xfff8ef,eye);
  ellipse(0,3,25,30,0x376b65,eye);ellipse(0,5,13,26,0x203e41,eye);ellipse(6,-10,8,10,0xfffbed,eye);ellipse(-11,16,4,5,0x91cdc1,eye);
  line(g=>g.moveTo(-53,-2).quadraticCurveTo(0,-39,49,-1),0x293740,7,eye);
  line(g=>g.moveTo(-50,-2).lineTo(-60,-14),0x293740,5,eye);
 }
 brows=new Graphics();head.addChild(brows);
 line(g=>g.moveTo(401,396).lineTo(393,434).quadraticCurveTo(400,440,410,436),0xd59c8c,3,head);
 mouth=new Graphics();head.addChild(mouth);
 // Swept fringe and side locks frame the independent eyes and mouth.
 path(g=>g.moveTo(206,305).bezierCurveTo(162,153,241,75,383,82).bezierCurveTo(534,57,646,163,595,341).lineTo(564,421).lineTo(548,252).bezierCurveTo(480,263,413,214,381,170).bezierCurveTo(365,245,291,283,240,295).lineTo(238,440).lineTo(211,376).closePath(),0x304d58,head);
 path(g=>g.moveTo(248,253).bezierCurveTo(270,156,319,123,388,116).bezierCurveTo(363,203,309,242,248,253).closePath(),0x466a70,head);
 path(g=>g.moveTo(406,120).bezierCurveTo(533,108,582,191,578,255).bezierCurveTo(532,249,439,200,406,120).closePath(),0x3d616a,head);
 line(g=>g.moveTo(393,104).bezierCurveTo(366,191,320,238,272,255),0x72928e,3,head);
 path(g=>g.moveTo(227,331).quadraticCurveTo(189,551,250,683).lineTo(275,603).quadraticCurveTo(241,457,250,282).closePath(),0x304d58,head);
 path(g=>g.moveTo(560,289).quadraticCurveTo(610,481,544,694).lineTo(519,616).quadraticCurveTo(561,462,542,303).closePath(),0x304d58,head);
 line(g=>g.moveTo(224,413).quadraticCurveTo(213,555,250,636),0x587b7e,4,head);
 line(g=>g.moveTo(581,394).quadraticCurveTo(589,544,544,641),0x587b7e,4,head);
 const pin=new Container();pin.position.set(546,250);pin.rotation=.4;head.addChild(pin);
 path(g=>g.roundRect(-8,-30,16,60,8),0xd7bd78,pin);ellipse(0,-12,10,10,0xf8e4a5,pin);
 for(const x of [222,578]){line(g=>g.moveTo(x,421).lineTo(x,459),0xd8b878,3,head);ellipse(x,465,7,13,0xf3d797,head);}
}
function fit(){const s=Math.min(innerWidth/800,innerHeight/1000);root.scale.set(s);root.position.set((innerWidth-800*s)/2,innerHeight-1000*s);}
original();fit();window.addEventListener('resize',fit);
async function character(c:Character){
 const id=++generation;
 if(c.procedural){palette=c.palette??'teal';original();fit();return;}
 const loaded=await Promise.all(c.layers.map(async l=>{const img=new Image();const src=c.assets?.[l.image];if(!src?.startsWith('data:image/png;base64,'))throw new Error('Missing image');img.src=src;await img.decode();return {layer:l,texture:Texture.from(img)};}));
 if(id!==generation)return;
 root.removeChildren().forEach(n=>n.destroy({children:true,texture:true,textureSource:true}));layers=[];
 for(const {layer,texture} of loaded.sort((a,b)=>a.layer.z-b.layer.z)){
  const sprite=new Sprite(texture);sprite.position.set(layer.x,layer.y);sprite.pivot.set(layer.pivot_x,layer.pivot_y);root.addChild(sprite);layers.push({sprite,layer});
 }
 const s=Math.min(innerWidth/c.width,innerHeight/c.height);root.scale.set(s);root.position.set((innerWidth-c.width*s)/2,innerHeight-c.height*s);
}
const event=(e:Bridge)=>{
 if(e.expression)expression=e.expression;
 if(e.state)state=e.state;
 if(e.level!==undefined){targetLevel=Math.min(1,Math.max(0,e.level));lastLevel=performance.now();}
 if(e.reducedMotion!==undefined)reduced=e.reducedMotion;
 if(e.visible!==undefined){visible=e.visible;if(visible)app.ticker.start();else app.ticker.stop();}
 if(e.character)character(e.character).catch(()=>window.webkit?.messageHandlers.character.postMessage({type:'error',message:'Character textures could not be loaded.'}));
};
app.ticker.add(t=>{
 if(!visible)return;
 time+=t.deltaMS/1000;
 if(performance.now()-lastLevel>180)targetLevel=0;
 level+=(targetLevel-level)*Math.min(1,t.deltaMS/(targetLevel>level?45:110));
 const blinkTime=time%4.9;blink=blinkTime>4.65?Math.max(.04,Math.abs(blinkTime-4.775)/.125):1;
 if(layers.length){
  for(const {sprite,layer} of layers){sprite.visible=!layer.expression||layer.expression===expression;
   if(layer.role==='eyes')sprite.scale.y=blink;
   if(layer.role==='mouth'&&layer.openness!==undefined){const values=layers.filter(n=>n.layer.role==='mouth'&&(!n.layer.expression||n.layer.expression===expression));const closest=values.reduce((a,b)=>Math.abs((a.layer.openness??0)-level)<Math.abs((b.layer.openness??0)-level)?a:b,values[0]);sprite.visible=sprite.visible&&sprite===closest?.sprite;}
  }return;
 }
 head.rotation=reduced?0:Math.sin(time*.85)*.012;
 head.position.y=585+(reduced?0:Math.sin(time*1.6)*2);
 const sleepy=expression==='sleepy',happy=expression==='happy',surprised=expression==='surprised';
 for(const eye of eyes.children){eye.scale.y=blink*(sleepy?.5:happy?.82:surprised?1.15:1);}
 brows.clear();
 const tilt=expression==='concerned'?-12:expression==='annoyed'?12:expression==='curious'?-7:0;
 for(const [x,side] of [[318,1],[481,-1]]){brows.moveTo(x-38,304-tilt*side).quadraticCurveTo(x,295+(surprised?-18:0),x+36,304+tilt*side).stroke({color:0x34525b,width:5,cap:'round'});}
 blush.clear();if(expression==='embarrassed'||happy){blush.ellipse(293,422,29,12).ellipse(511,422,29,12).fill({color:0xe8948d,alpha:expression==='embarrassed'?.55:.25});}
 mouth.clear();
 const opening=level>0.035?3+level*31:surprised?11:0;
 if(opening){mouth.ellipse(402,487,17+level*8,opening).fill(0x703e46);mouth.ellipse(402,489+opening*.45,10+level*6,opening*.3).fill(0xd48d8c);}
 else{mouth.moveTo(381,483).quadraticCurveTo(402,happy?500:expression==='concerned'?477:491,423,483).stroke({color:0xa36265,width:3,cap:'round'});}
 if(state==='listening'&&!reduced)head.rotation-=.02;
});
const visibility=()=>{if(document.hidden)app.ticker.stop();else if(visible)app.ticker.start();};
document.addEventListener('visibilitychange',visibility);
return {event, dispose:()=>{generation++;window.removeEventListener('resize',fit);document.removeEventListener('visibilitychange',visibility);app.destroy(true,{children:true,texture:true,textureSource:true});}};
}
