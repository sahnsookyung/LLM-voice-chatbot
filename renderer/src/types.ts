export type Layer = {id:string;image:string;x:number;y:number;z:number;pivot_x:number;pivot_y:number;role:string;expression?:string;openness?:number};
export type Character = {id?:string;width:number;height:number;procedural:boolean;palette?:string;layers:Layer[];assets?:Record<string,string>};
export type Framing = 'portrait' | 'full';
export type VisualMode = 'illustration' | 'model';
export type Bridge = {expression?:string;level?:number;state?:string;visible?:boolean;character?:Character;reducedMotion?:boolean;framing?:Framing;visualMode?:VisualMode;gesture?:string;angle?:number;intensity?:number;gestureSpeed?:number;referenceLook?:import('./looks').Look|null;wardrobe?:import('./wardrobe').WardrobeSelection};
declare global {interface Window {
  companion:(event:Bridge)=>void;
  characterStatus:(diagnostics?:boolean)=>Record<string,unknown>;
  ariLookBase?:string;
  webkit?:{messageHandlers:{character:{postMessage:(event:unknown)=>void}}};
}}
