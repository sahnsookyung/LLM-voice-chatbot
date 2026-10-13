import AppKit
import WebKit

final class StyleBoardCheck:NSObject,WKNavigationDelegate,WKScriptMessageHandler {
  var web:WKWebView!,window:NSWindow!
  let path=URL(fileURLWithPath:CommandLine.arguments[1]),output=URL(fileURLWithPath:CommandLine.arguments[2])
  let looks:[(String,String,String,String)]=[
    ("butter-coat","original","none","beret"),("cloud-sweater","double-buns","round","none"),("sunny-sweater","soft-waves","none","none"),
    ("school-cardigan","low-pigtails","none","none"),("school-cardigan-brown","soft-waves","round","none"),
    ("school-vest","half-up","none","none"),("navy-blouse","soft-waves","round","none"),
    ("pink-sailor","low-pigtails","none","none"),("ruffle-dress","half-up","none","none"),
    ("black-jacket","soft-waves","none","none"),
    ("forest-pinafore","twintails","none","none"),("original","double-buns","round","beret")]
  let poses=["double-peace","peace-eye","thinking","adjust-glasses"]
  var phase=0,index=0,pose=0,started=Date(),ready=false,capturing=false,angle=0,reloading=false
  func fail(_ s:String)->Never{print("FAIL:",s);exit(1)}
  func run(){
    let config=WKWebViewConfiguration();config.websiteDataStore = .nonPersistent();config.userContentController.add(self,name:"character")
    web=WKWebView(frame:NSRect(x:0,y:0,width:1280,height:1100),configuration:config);web.navigationDelegate=self
    window=NSWindow(contentRect:web.frame,styleMask:[.titled],backing:.buffered,defer:false);window.title="Ari screenshot-style checks";window.contentView=web;window.orderFrontRegardless()
    try! FileManager.default.createDirectory(at:output,withIntermediateDirectories:true)
    web.loadFileURL(path,allowingReadAccessTo:path.deletingLastPathComponent())
    DispatchQueue.main.asyncAfter(deadline:.now()+420){self.fail("timed out at phase \(self.phase), look \(self.index), pose \(self.pose)")}
  }
  func userContentController(_ userContentController:WKUserContentController,didReceive message:WKScriptMessage){
    if let value=message.body as? [String:Any],value["type"] as? String=="error"{fail(value["message"] as? String ?? "renderer error")}
  }
  func command(_ js:String){web.evaluateJavaScript(js){_,error in if let error{self.fail(error.localizedDescription)}}}
  func pick(){let look=looks[index];ready=false
    command("for(const [id,value] of [['outfit','\(look.0)'],['hairstyle','\(look.1)'],['glasses','\(look.2)'],['headwear','\(look.3)']]){const e=document.getElementById(id);e.value=value;e.dispatchEvent(new Event('change'));}window.companion({reducedMotion:false,gesture:'cancel',expressionDetail:'soft-smile',angle:0})")
  }
  func capture(_ name:String,_ then:@escaping()->Void){capturing=true
    web.evaluateJavaScript("JSON.stringify((()=>{const r=document.querySelector('#stage-slot').getBoundingClientRect();return {x:r.x,y:r.y,width:r.width,height:r.height}})())"){value,error in
      guard let s=value as? String,let d=s.data(using:.utf8),let r=try? JSONSerialization.jsonObject(with:d) as? [String:Double] else{self.fail("snapshot bounds unavailable")}
      let c=WKSnapshotConfiguration();c.rect=NSRect(x:r["x"]!,y:r["y"]!,width:r["width"]!,height:r["height"]!);c.snapshotWidth=640
      self.web.takeSnapshot(with:c){image,error in
        guard error==nil,let tiff=image?.tiffRepresentation,let png=NSBitmapImageRep(data:tiff)?.representation(using:.png,properties:[:]) else{self.fail("snapshot failed")}
        try! png.write(to:self.output.appendingPathComponent(name+".png"));self.capturing=false;then();self.poll()
      }
    }
  }
  func webView(_ webView:WKWebView,didFinish navigation:WKNavigation!){reloading=false;poll()}
  func poll(){if capturing || reloading{return}
    web.evaluateJavaScript("JSON.stringify({s:window.characterStatus(),canvases:document.querySelectorAll('canvas').length,outfit:document.querySelector('#outfit').value,hair:document.querySelector('#hairstyle').value,glasses:document.querySelector('#glasses').value,headwear:document.querySelector('#headwear').value,beretDisabled:document.querySelector('#headwear option[value=beret]').disabled})"){value,error in
      guard let s=value as? String,let data=s.data(using:.utf8),let r=try? JSONSerialization.jsonObject(with:data) as? [String:Any],let state=r["s"] as? [String:Any] else{self.fail("status unavailable at phase \(self.phase): \(String(describing: error)), value \(String(describing: value))")}
      if let error=state["modelError"] as? String,!error.isEmpty{self.fail(error)}
      switch self.phase {
      case 0:
        if state["loaded"] as? Bool==true{
          if CommandLine.arguments.contains("--poses") {self.phase=2;self.command("window.companion({visualMode:'model',wardrobe:{outfit:'navy-blouse',hair:'low-pigtails'},accessories:{glasses:'round',headwear:'none'},expressionDetail:'playful',angle:0})")}
          else {self.phase=1;self.pick()}
        }
      case 1:
        let look=self.looks[self.index]
        if state["appliedWardrobe"] as? String=="\(look.0)/\(look.1)",state["modelLoading"] as? Bool==false{
          if !self.ready{self.ready=true;self.started=Date()}
          if Date().timeIntervalSince(self.started)>1.5{
            guard r["canvases"] as? Int==1,(state["footError"] as? Double ?? 1)<0.001,r["outfit"] as? String==look.0,r["hair"] as? String==look.1 else{self.fail("selection/rig contract")}
            guard let accessories=state["accessories"] as? [String:String],accessories["glasses"]==look.2,accessories["headwear"]==(look.1=="double-buns" ? "none" : look.3) else{self.fail("accessories reset during a model swap")}
            if look.1=="double-buns"{guard r["headwear"] as? String=="none",r["beretDisabled"] as? Bool==true else{self.fail("incompatible hat still available with buns")}}
            print("PASS: \(look.0)/\(look.1), \(look.2) glasses, \(accessories["headwear"] ?? "?") headwear; \(state["drawCalls"] ?? "?") draws")
            let suffix=self.angle==0 ? "" : self.angle==1 ? "-quarter" : "-back"
            self.capture(look.0+"-"+look.1+suffix){
              if self.angle<2 {
                self.angle+=1;self.ready=false
                self.command("window.companion({angle:\(self.angle==1 ? 45 : 180)})")
              } else {
                self.angle=0;self.index+=1
                if self.index<self.looks.count{self.pick()}else{self.phase=2;self.command("window.companion({wardrobe:{outfit:'navy-blouse',hair:'low-pigtails'},accessories:{glasses:'round',headwear:'none'},expressionDetail:'playful',angle:0})")}
              }
            };return
          }
        }
      case 2:
        if state["appliedWardrobe"] as? String=="navy-blouse/low-pigtails",state["modelLoading"] as? Bool==false{self.phase=3;self.started=Date();self.command("window.companion({gesture:'\(self.poses[self.pose])'})")}
      case 3:
        if Date().timeIntervalSince(self.started)>2.2{
          guard (state["footError"] as? Double ?? 1)<0.001,(state["limbExtent"] as? Double ?? 2)<1.05 else{self.fail("pose feet or framing")}
          print("PASS: posed \(self.poses[self.pose]), feet and framing retained")
          self.capture(self.poses[self.pose]){
            self.pose+=1
            if self.pose<self.poses.count{self.phase=2}else{self.phase=4;self.started=Date();self.command("window.companion({gesture:'cancel',expressionDetail:'pensive',level:.35})")}
          };return
        }
      case 4:
        if Date().timeIntervalSince(self.started)>0.15{
          guard let weights=state["expressionWeights"] as? [String:Double],(weights["aa"] ?? 0)>0 else{self.fail("expression detail blocked speech")}
          print("PASS: expression detail blends with speech")
          self.phase=5;self.started=Date();self.command("window.companion({level:0,expressionDetail:'soft-smile',angle:90})")
        }
      case 5:
        if Date().timeIntervalSince(self.started)>1{
          self.phase=6;self.capture("glasses-profile"){
            self.reloading=true
            self.web.evaluateJavaScript("for(const [id,value] of [['outfit','navy-blouse'],['hairstyle','low-pigtails'],['glasses','round'],['headwear','none']]){const e=document.getElementById(id);e.value=value;e.dispatchEvent(new Event('change'));}"){_,error in
              if let error{self.fail(error.localizedDescription)};self.web.reload()
            }
          };return
        }
      case 6:
        if state["modelLoading"] as? Bool==false,state["appliedWardrobe"] as? String=="navy-blouse/low-pigtails" {
          guard r["glasses"] as? String=="round",r["headwear"] as? String=="none",let a=state["accessories"] as? [String:String],a["glasses"]=="round" else{self.fail("accessories did not persist across reload")}
          print("PASS: independent accessories and wardrobe survive document reload")
          self.phase=7;self.started=Date();self.command("window.companion({accessories:{glasses:'round',headwear:'beret'},framing:'portrait',angle:35})")
        }
      case 7:
        if Date().timeIntervalSince(self.started)>1 {
          guard state["framing"] as? String=="portrait",let a=state["accessories"] as? [String:String],a["headwear"]=="beret" else{self.fail("portrait hat setup not applied")}
          self.phase=8;self.capture("beret-portrait"){self.command("document.querySelector('#mode').value='illustration';document.querySelector('#mode').dispatchEvent(new Event('change'))")};return
        }
      case 8:
        if state["mode"] as? String=="illustration"{guard r["canvases"] as? Int==0 else{self.fail("accessories or renderer retained a canvas")};print("PASS: style-board native checks complete");exit(0)}
      default:break
      }
      DispatchQueue.main.asyncAfter(deadline:.now()+0.2){self.poll()}
    }
  }
}
let app=NSApplication.shared;app.setActivationPolicy(.accessory);let check=StyleBoardCheck();check.run();app.run()
