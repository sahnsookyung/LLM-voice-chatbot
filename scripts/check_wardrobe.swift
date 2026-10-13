import AppKit
import WebKit

// Native GPU/UI integration, with an isolated data store and no user settings.
final class WardrobeCheck: NSObject, WKNavigationDelegate {
  let store=WKWebsiteDataStore.nonPersistent()
  var web:WKWebView!
  var window:NSWindow!
  let path=URL(fileURLWithPath:CommandLine.arguments[1])
  let output=URL(fileURLWithPath:CommandLine.arguments[2])
  let outfits=["original","tailored","streetwear","linen"]
  let hair=["original","bob","ponytail","braid"]
  var phase=0,index=0,frames=0,started=Date(),readyAt=Date(),peak=0.0
  var reloaded=false,capturing=false
  func fail(_ message:String)->Never {print("FAIL:",message);exit(1)}
  func command(_ text:String){web.evaluateJavaScript(text){_,error in if let error{self.fail(error.localizedDescription)}}}
  func run(){
    let config=WKWebViewConfiguration();config.websiteDataStore=store
    web=WKWebView(frame:NSRect(x:0,y:0,width:1280,height:1100),configuration:config);web.navigationDelegate=self
    window=NSWindow(contentRect:web.frame,styleMask:[.titled],backing:.buffered,defer:false);window.title="Ari wardrobe checks";window.contentView=web;window.orderFrontRegardless()
    try! FileManager.default.createDirectory(at:output,withIntermediateDirectories:true)
    web.loadFileURL(path,allowingReadAccessTo:path.deletingLastPathComponent())
    DispatchQueue.main.asyncAfter(deadline:.now()+240){self.fail("timed out: phase \(self.phase), combination \(self.index)")}
  }
  func pick(_ outfit:String,_ hair:String){
    command("document.querySelector('#outfit').value='\(outfit)';document.querySelector('#outfit').dispatchEvent(new Event('change'));document.querySelector('#hairstyle').value='\(hair)';document.querySelector('#hairstyle').dispatchEvent(new Event('change'))")
  }
  func capture(_ name:String,_ completion:@escaping()->Void){
    capturing=true
    web.evaluateJavaScript("JSON.stringify((()=>{const r=document.querySelector('#stage-slot').getBoundingClientRect();return {x:r.x,y:r.y,width:r.width,height:r.height}})())"){value,error in
      guard error==nil,let s=value as? String,let d=s.data(using:.utf8),let r=try? JSONSerialization.jsonObject(with:d) as? [String:Double] else{self.fail("snapshot bounds unavailable")}
      let config=WKSnapshotConfiguration();config.rect=NSRect(x:r["x"]!,y:r["y"]!,width:r["width"]!,height:r["height"]!);config.snapshotWidth=640
      self.web.takeSnapshot(with:config){image,error in
        guard error==nil,let tiff=image?.tiffRepresentation,let png=NSBitmapImageRep(data:tiff)?.representation(using:.png,properties:[:]) else{self.fail("snapshot unavailable")}
        do{try png.write(to:self.output.appendingPathComponent("\(name).png"))}catch{self.fail(error.localizedDescription)}
        self.capturing=false;completion()
      }
    }
  }
  func webView(_ webView:WKWebView,didFinish navigation:WKNavigation!){
    if !reloaded {command("window.companion({reducedMotion:true,expression:'happy',level:.25,framing:'full'});document.querySelector('#reduced').checked=true")}
    poll()
  }
  func poll(){
    if capturing{return}
    web.evaluateJavaScript("JSON.stringify({s:window.characterStatus(),outfit:document.querySelector('#outfit')?.value,hair:document.querySelector('#hairstyle')?.value,canvases:document.querySelectorAll('canvas').length})"){value,error in
      guard error==nil,let s=value as? String,let d=s.data(using:.utf8),let result=try? JSONSerialization.jsonObject(with:d) as? [String:Any],let state=result["s"] as? [String:Any] else{self.fail("studio status unavailable")}
      if let error=state["modelError"] as? String,!error.isEmpty{self.fail(error)}
      let key=state["appliedWardrobe"] as? String ?? "",loading=state["modelLoading"] as? Bool ?? true,count=state["frames"] as? Int ?? 0
      let expected="\(self.outfits[min(self.index/4,3)])/\(self.hair[self.index%4])"
      switch self.phase {
      case 0:
        if state["loaded"] as? Bool==true{self.phase=1;self.pick("original","original")}
      case 1:
        if !loading,key==expected,count>self.frames+3{
          guard result["canvases"] as? Int==1,state["bones"] as? Int==54,(state["footError"] as? Double ?? 1)<0.001 else{self.fail("rig/canvas/foot contract: \(key)")}
          guard let weights=state["expressionWeights"] as? [String:Double],(weights["ariHappy"] ?? 0)>0 else{self.fail("swap lost expression")}
          let selected="\(result["outfit"] ?? "")/\(result["hair"] ?? "")"
          guard selected==expected else{self.fail("independent selectors reset another part")}
          print("PASS: \(key), \(state["drawCalls"] ?? "?") draws, \(state["triangles"] ?? "?") triangles; one canvas, face expression and feet retained")
          self.capture(key.replacingOccurrences(of:"/",with:"-")){
            self.frames=count;self.index+=1
            if self.index<16{self.pick(self.outfits[self.index/4],self.hair[self.index%4])}
            else{self.phase=2;self.started=Date();self.command("for(let i=0;i<40;i++){const o=document.querySelector('#outfit'),h=document.querySelector('#hairstyle');o.value=['original','tailored','streetwear','linen'][i%4];o.dispatchEvent(new Event('change'));h.value=['original','bob','ponytail','braid'][(i+1)%4];h.dispatchEvent(new Event('change'));}document.querySelector('#outfit').value='streetwear';document.querySelector('#outfit').dispatchEvent(new Event('change'));document.querySelector('#hairstyle').value='ponytail';document.querySelector('#hairstyle').dispatchEvent(new Event('change'))")}
            self.poll()
          };return
        }
      case 2:
        if !loading,key=="streetwear/ponytail"{
          guard result["canvases"] as? Int==1 else{self.fail("rapid swaps leaked canvases")}
          print("PASS: 80 rapid independent selector changes settle on the final pair with one canvas")
          self.phase=3;self.reloaded=true;self.web.reload();return
        }
      case 3:
        if !loading,key=="streetwear/ponytail"{
          guard result["outfit"] as? String=="streetwear",result["hair"] as? String=="ponytail" else{self.fail("saved pair not restored")}
          print("PASS: outfit and hairstyle restored independently after document reload")
          self.phase=4;self.started=Date();self.command("window.companion({reducedMotion:false,gesture:'twirl',gestureSpeed:1.5,expression:'embarrassed',level:.3})")
        }
      case 4:
        guard let speed=state["hairSpeed"] as? Double,speed.isFinite,let deflection=state["hairDeflection"] as? Double,deflection.isFinite,deflection<3 else{self.fail("unstable ponytail")}
        self.peak=max(self.peak,speed)
        if Date().timeIntervalSince(self.started)>7{
          guard self.peak>0.02,speed<self.peak * 0.7 else{self.fail("ponytail did not move and settle")}
          print("PASS: ponytail moves during fast twirl and settles; peak \(self.peak), final \(speed) rad/s")
          self.phase=5;self.pick("linen","braid")
        }
      case 5:
        if !loading,key=="linen/braid"{self.phase=6;self.started=Date();self.command("window.companion({gesture:'raise-arms',gestureSpeed:1})")}
      case 6:
        guard (state["footError"] as? Double ?? 1)<0.001 else{self.fail("linen pose lost planted feet")}
        if Date().timeIntervalSince(self.started)>2.0{
          self.phase=7;self.capture("linen-braid-raise-arms"){
            self.command("window.companion({wardrobe:{outfit:'invalid',hair:'bob'}})");self.started=Date();self.poll()
          };return
        }
      case 7:
        if Date().timeIntervalSince(self.started)>0.3{
          guard key=="linen/braid",!loading else{self.fail("invalid part replaced the good model")}
          print("PASS: invalid combination retains the last valid animated model")
          self.phase=8;self.started=Date();self.command("window.companion({gesture:'twirl',gestureSpeed:1.5,angle:35})")
        }
      case 8:
        guard let deflection=state["garmentDeflection"] as? Double,deflection.isFinite,deflection<2.5,(state["garmentJoints"] as? Int ?? 0)>0 else{self.fail("unstable skirt rig")}
        if Date().timeIntervalSince(self.started)>2{
          self.phase=9;self.capture("linen-braid-twirl"){self.started=Date();self.poll()};return
        }
      case 9:
        if Date().timeIntervalSince(self.started)>5{
          guard (state["garmentDeflection"] as? Double ?? 9)<0.7 else{self.fail("skirt did not settle")}
          print("PASS: linen skirt rig remains bounded during twirl and settles")
          self.phase=10;self.command("document.querySelector('#mode').value='illustration';document.querySelector('#mode').dispatchEvent(new Event('change'))")
        }
      case 10:
        if state["mode"] as? String=="illustration"{
          guard result["canvases"] as? Int==0 else{self.fail("leaked renderer after leaving 3D")}
          print("PASS: leaving 3D disposes the active renderer")
          print("PASS: wardrobe integration completed");exit(0)
        }
      default:break
      }
      DispatchQueue.main.asyncAfter(deadline:.now()+0.2){self.poll()}
    }
  }
}
let app=NSApplication.shared;app.setActivationPolicy(.accessory)
let check=WardrobeCheck();check.run();app.run()
