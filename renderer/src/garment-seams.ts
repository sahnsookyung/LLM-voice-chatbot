export type Influence={joints:number[];weights:number[]};
export type Surface={positions:number[][];uv:number[][];influences:Influence[];indices:number[]};
export const emptySurface=():Surface=>({positions:[],uv:[],influences:[],indices:[]});

export function blendInfluences(a:Influence,b:Influence,t:number):Influence {
  const weights=new Map<number,number>();
  for(const [v,f]of [[a,1-t],[b,t]] as const)v.joints.forEach((j,i)=>weights.set(j,(weights.get(j)??0)+v.weights[i]*f));
  const pairs=[...weights].sort((a,b)=>b[1]-a[1]).slice(0,4);
  while(pairs.length<4)pairs.push([0,0]);
  const sum=pairs.reduce((v,p)=>v+p[1],0);
  return {joints:pairs.map(p=>p[0]),weights:pairs.map(p=>p[1]/sum)};
}
const mix=(a:number[],b:number[],t:number)=>a.map((v,i)=>v+(b[i]-v)*t);

/** Cut a complete sleeve ring, interpolating UVs and skin weights at the cut. */
export function cutSleeve(source:Surface,sign:number,cut=.24) {
  const result=emptySurface(),cache=new Map<string,number>(),boundary=new Map<string,number>();
  type Vertex={p:number[];uv:number[];skin:Influence};
  for(let i=0;i<source.indices.length;i+=3){
    const tri=source.indices.slice(i,i+3).map(j=>({p:source.positions[j],uv:source.uv[j],skin:source.influences[j]}));
    const polygon:Vertex[]=[];
    for(let j=0;j<3;j++){
      const a=tri[j],b=tri[(j+1)%3],inside=a.p[0]*sign>=cut;
      if(inside)polygon.push(a);
      if(inside!==(b.p[0]*sign>=cut)){
        const t=(sign*cut-a.p[0])/(b.p[0]-a.p[0]),p=mix(a.p,b.p,t);p[0]=sign*cut;
        polygon.push({p,uv:mix(a.uv,b.uv,t),skin:blendInfluences(a.skin,b.skin,t)});
      }
    }
    const ids=polygon.map(v=>{
      const key=[...v.p,...v.uv].map(n=>n.toFixed(7)).join(',');let id=cache.get(key);
      if(id===undefined){id=result.positions.length;cache.set(key,id);result.positions.push(v.p);result.uv.push(v.uv);result.influences.push(v.skin);}
      if(Math.abs(v.p[0]-sign*cut)<1e-7)boundary.set(v.p.map(n=>n.toFixed(6)).join(','),id);
      return id;
    });
    for(let j=1;j<ids.length-1;j++)if(new Set([ids[0],ids[j],ids[j+1]]).size===3)result.indices.push(ids[0],ids[j],ids[j+1]);
  }
  const ring=[...boundary.values()];
  if(ring.length<8)throw new Error('Ari sleeve cut did not produce an armhole ring.');
  const center=[1,2].map(k=>ring.reduce((sum,i)=>sum+result.positions[i][k],0)/ring.length);
  ring.sort((a,b)=>Math.atan2(result.positions[a][1]-center[0],result.positions[a][2]-center[1])-Math.atan2(result.positions[b][1]-center[0],result.positions[b][2]-center[1]));
  return {surface:result,ring};
}

/** A raglan shoulder joins the torso opening to the retained detailed sleeve.
 * Both ends reuse exact positions and weights, so the seam cannot pull apart. */
export function stitchSleeve(body:Surface,opening:number[],sleeve:Surface,end:number[]) {
  const center=(surface:Surface,ids:number[])=>[1,2].map(k=>ids.reduce((sum,i)=>sum+surface.positions[i][k],0)/ids.length);
  const area=(surface:Surface,ids:number[])=>ids.reduce((sum,id,i)=>{const a=surface.positions[id],b=surface.positions[ids[(i+1)%ids.length]];return sum+a[1]*b[2]-b[1]*a[2];},0);
  const ring=end.slice();if(area(body,opening)*area(sleeve,ring)<0)ring.reverse();
  const c=center(body,opening),d=center(sleeve,ring),p=body.positions[opening[0]],angle=Math.atan2(p[1]-c[0],p[2]-c[1]);
  let start=0,best=Infinity;
  ring.forEach((id,i)=>{const p=sleeve.positions[id],a=Math.atan2(p[1]-d[0],p[2]-d[1]),distance=1-Math.cos(a-angle);if(distance<best){best=distance;start=i;}});
  const ordered=[...ring.slice(start),...ring.slice(0,start)],count=opening.length;
  const sample=(t:number)=>{const u=t*ordered.length,i=Math.floor(u)%ordered.length,f=u-Math.floor(u),a=ordered[i],b=ordered[(i+1)%ordered.length];return {p:mix(sleeve.positions[a],sleeve.positions[b],f),skin:blendInfluences(sleeve.influences[a],sleeve.influences[b],f)};};
  // Normalize winding from the opening boundary, then connect unequal rings
  // without dropping an edge. This also handles UV-split sleeve vertices.
  const join=(a:number[],b:number[])=>{
    let i=0,j=0;
    while(i<a.length||j<b.length){
      if((i+1)/a.length<=(j+1)/b.length){body.indices.push(a[i%a.length],a[(i+1)%a.length],b[j%b.length]);i++;}
      else{body.indices.push(a[i%a.length],b[(j+1)%b.length],b[j%b.length]);j++;}
    }
  };
  let previous=opening;
  for(let row=1;row<12;row++){
    const t=row/12,ids:number[]=[];
    for(let i=0;i<count;i++){
      const a=opening[i],target=sample(i/count),p=mix(body.positions[a],target.p,t),radial=p[2]-(c[1]*(1-t)+d[1]*t);
      p[2]+=Math.sign(radial)*.003*Math.sin(t*Math.PI)*Math.sin(i/count*Math.PI*8+t*2);
      ids.push(body.positions.length);body.positions.push(p);body.uv.push([i/count,t*.35]);body.influences.push(blendInfluences(body.influences[a],target.skin,t*t*(3-2*t)));
    }
    join(previous,ids);previous=ids;
  }
  const final=ordered.map(id=>{
    const index=body.positions.length;body.positions.push(sleeve.positions[id].slice());body.uv.push(sleeve.uv[id].slice());body.influences.push(sleeve.influences[id]);return index;
  });
  join(previous,final);
  return final.map((id,i)=>[id,ordered[i]]);
}
