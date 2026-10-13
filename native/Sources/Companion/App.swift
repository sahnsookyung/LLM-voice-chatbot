import AppKit
import Carbon
import SwiftUI
import WebKit

@main struct CompanionApp: App {
  @NSApplicationDelegateAdaptor(AppDelegate.self) var delegate
  var body: some Scene {
    Settings { EmptyView() }
      .commands {
        CommandMenu("Character") {
          Button("Open character studio") { delegate.openStudio() }
            .keyboardShortcut("a", modifiers: [.command, .shift])
          Button("Show / hide avatar") { delegate.toggleCharacter() }
            .keyboardShortcut("h", modifiers: [.command, .shift])
        }
      }
  }
}
final class CompanionPanel: NSPanel {
  override var canBecomeKey: Bool { false }
  override var canBecomeMain: Bool { false }
}
final class CharacterWebView: WKWebView { override var mouseDownCanMoveWindow: Bool { true } }
@MainActor
final class AppDelegate: NSObject, NSApplicationDelegate, WKScriptMessageHandler, NSWindowDelegate, WKNavigationDelegate {
  var model: AppModel!
  var chat: NSWindow!
  var character: CompanionPanel!
  var studio: NSWindow?
  var status: NSStatusItem!
  var hotkey: EventHotKeyRef?
  var eventHandler: EventHandlerRef?
  var characterReady = false
  private var characterPlaced = false
  private var visibilityMenuItem: NSMenuItem?
  func applicationDidFinishLaunching(_ notification: Notification) {
    NSApp.setActivationPolicy(.regular)
    model = AppModel()
    chat = NSWindow(
      contentRect: NSRect(x: 0, y: 0, width: 960, height: 720),
      styleMask: [.titled, .closable, .miniaturizable, .resizable, .fullSizeContentView],
      backing: .buffered, defer: false)
    chat.title = "Local Companion"
    chat.titlebarAppearsTransparent = true
    chat.isReleasedWhenClosed = false
    chat.minSize = NSSize(width: 760, height: 580)
    chat.center()
    chat.contentView = NSHostingView(rootView: CompanionView(model: model))
    chat.makeKeyAndOrderFront(nil)
    character = CompanionPanel(
      contentRect: NSRect(x: 80, y: 80, width: 300, height: 375),
      styleMask: [.borderless, .nonactivatingPanel], backing: .buffered, defer: false)
    character.isOpaque = false
    character.title = "Companion character"
    character.backgroundColor = .clear
    character.hasShadow = false
    character.level = .floating
    character.isMovableByWindowBackground = true
    character.collectionBehavior = [.canJoinAllSpaces, .fullScreenAuxiliary]
    character.hidesOnDeactivate = false
    character.delegate = self
    let controller = WKUserContentController()
    controller.add(self, name: "character")
    let config = WKWebViewConfiguration()
    config.userContentController = controller
    let web = CharacterWebView(frame: character.contentView!.bounds, configuration: config)
    web.autoresizingMask = [.width, .height]
    web.setValue(false, forKey: "drawsBackground")
    character.contentView = web
    model.webView = web
    let url = Bundle.main.resourceURL!.appendingPathComponent("renderer/index.html")
    web.loadFileURL(url, allowingReadAccessTo: url.deletingLastPathComponent())
    model.resizeCharacter = { [weak self] in self?.resizeCharacter() }
    model.openCharacterStudio = { [weak self] in self?.openStudio() }
    model.setOverlayVisibility = { [weak self] visible in self?.applyOverlayVisibility(visible) }
    resizeCharacter()
    applyOverlayVisibility(model.overlayVisible)
    model.showCharacter = { [weak self] in
      self?.model.overlayVisible = true
    }
    DispatchQueue.main.asyncAfter(deadline: .now() + 8) { [weak self] in
      guard let self, !self.characterReady else { return }
      self.model.notice = "The character could not load. Please restart Local Companion."
    }
    status = NSStatusBar.system.statusItem(withLength: NSStatusItem.squareLength)
    status.button?.image = NSImage(
      systemSymbolName: "sparkle", accessibilityDescription: "Local Companion")
    let menu = NSMenu()
    for (title, action, key) in [
      ("Open conversation", #selector(openChat), ""),
      (model.overlayVisible ? "Hide avatar" : "Show avatar", #selector(toggleCharacter), ""),
      ("Character studio", #selector(openStudio), ""),
      ("Toggle click-through", #selector(toggleClickThrough), ""),
      ("Mute microphone and stop", #selector(mute), ""),
      ("Quit Local Companion", #selector(quit), "q"),
    ] {
      let item = NSMenuItem(title: title, action: action, keyEquivalent: key)
      item.target = self
      if action == #selector(toggleCharacter) { visibilityMenuItem = item }
      menu.addItem(item)
    }
    status.menu = menu
    var types = [
      EventTypeSpec(
        eventClass: OSType(kEventClassKeyboard), eventKind: UInt32(kEventHotKeyPressed)),
      EventTypeSpec(
        eventClass: OSType(kEventClassKeyboard), eventKind: UInt32(kEventHotKeyReleased)),
    ]
    InstallEventHandler(
      GetApplicationEventTarget(),
      { _, event, context in
        guard let event, let context else { return noErr }
        let app = Unmanaged<AppDelegate>.fromOpaque(context).takeUnretainedValue()
        let kind = GetEventKind(event)
        Task { @MainActor in
          if kind == UInt32(kEventHotKeyPressed) {
            app.model.beginPTT()
          } else {
            app.model.endPTT()
          }
        }
        return noErr
      }, 2, &types, Unmanaged.passUnretained(self).toOpaque(), &eventHandler)
    let result = RegisterEventHotKey(
      UInt32(kVK_Space), UInt32(controlKey | optionKey),
      EventHotKeyID(signature: 0x4c56_4331, id: 1), GetApplicationEventTarget(), 0, &hotkey)
    if result != noErr {
      model.notice = "The global Talk shortcut is unavailable. Use the Talk button."
    }
    NSApp.activate(ignoringOtherApps: true)
  }
  func applicationShouldTerminateAfterLastWindowClosed(_ sender: NSApplication) -> Bool { false }
  func applicationShouldHandleReopen(_ sender: NSApplication, hasVisibleWindows flag: Bool) -> Bool
  {
    openChat()
    return true
  }
  func applicationWillTerminate(_ notification: Notification) {
    if let hotkey { UnregisterEventHotKey(hotkey) }
    if let eventHandler { RemoveEventHandler(eventHandler) }
    model.shutdown()
  }
  func userContentController(
    _ userContentController: WKUserContentController, didReceive message: WKScriptMessage
  ) {
    guard let event = message.body as? [String: String] else { return }
    if event["type"] == "ready" {
      characterReady = true
      var appearance = model.overlayAppearanceEvent
      appearance["visible"] = model.overlayVisible && character.occlusionState.contains(.visible)
      model.animate(appearance)
      Task { await model.loadCharacter() }
    }
    if event["type"] == "error" { model.notice = event["message"] ?? "Character renderer failed." }
  }
  func windowDidChangeOcclusionState(_ notification: Notification) {
    guard let window = notification.object as? NSWindow else { return }
    if window === studio, let web = window.contentView as? WKWebView {
      let visible = window.occlusionState.contains(.visible)
      web.evaluateJavaScript("window.companion?.({visible:\(visible)})")
    } else if window === character {
      model.animate(["visible": model.overlayVisible && character.occlusionState.contains(.visible)])
    }
  }
  func windowWillClose(_ notification: Notification) {
    if let window = notification.object as? NSWindow, window === studio,
       let web = window.contentView as? WKWebView {
      web.evaluateJavaScript("window.companion?.({visible:false})")
    }
  }
  @objc func openChat() {
    chat.makeKeyAndOrderFront(nil)
    NSApp.activate(ignoringOtherApps: true)
  }
  @objc func showCharacter() { model.showCharacter?() }
  @objc func toggleCharacter() { model.overlayVisible.toggle() }
  private func resizeCharacter() {
    let height: CGFloat = model.characterFraming == "full" ? 540 : 375
    character.setContentSize(NSSize(width: 300, height: height))
    if let area = (chat.screen ?? NSScreen.main)?.visibleFrame {
      character.setFrameOrigin(NSPoint(
        x: min(max(character.frame.minX, area.minX), area.maxX - character.frame.width),
        y: min(max(character.frame.minY, area.minY), area.maxY - character.frame.height)))
    }
  }
  @objc func openStudio() {
    if let studio {
      NSApp.activate(ignoringOtherApps: true)
      studio.makeKeyAndOrderFront(nil)
      studio.orderFrontRegardless()
      (studio.contentView as? WKWebView)?.evaluateJavaScript("window.companion?.({visible:true})")
      return
    }
    let window = NSWindow(contentRect: NSRect(x: 0, y: 0, width: 1160, height: 850),
      styleMask: [.titled, .closable, .resizable, .miniaturizable], backing: .buffered, defer: false)
    window.title = "Ari Character Studio"
    window.minSize = NSSize(width: 760, height: 600)
    window.isReleasedWhenClosed = false
    let web = WKWebView(frame: window.contentView!.bounds, configuration: DesignAssets.studioConfiguration())
    web.navigationDelegate = self
    web.autoresizingMask = [.width, .height]
    window.contentView = web
    let url = Bundle.main.resourceURL!.appendingPathComponent("renderer/studio.html")
    web.loadFileURL(url, allowingReadAccessTo: url.deletingLastPathComponent())
    window.center()
    window.makeKeyAndOrderFront(nil)
    studio = window
    window.delegate = self
    NSApp.addWindowsItem(window, title: window.title, filename: false)
    NSApp.activate(ignoringOtherApps: true)
    window.orderFrontRegardless()
  }
  func webView(_ webView: WKWebView, decidePolicyFor navigationAction: WKNavigationAction,
               decisionHandler: @escaping (WKNavigationActionPolicy) -> Void) {
    if navigationAction.navigationType == .linkActivated {
      if let url = navigationAction.request.url, url.scheme == "https" { NSWorkspace.shared.open(url) }
      decisionHandler(.cancel)
    } else { decisionHandler(.allow) }
  }
  private func presentCharacter() {
    if !characterPlaced, let screen = chat.screen ?? NSScreen.main {
      let area = screen.visibleFrame
      character.setFrameOrigin(
        NSPoint(
          x: max(area.minX, area.maxX - character.frame.width - 24),
          y: area.minY + 24))
      characterPlaced = true
    }
    character.ignoresMouseEvents = false
    character.orderFrontRegardless()
    model.animate(["visible": true])
  }
  private func applyOverlayVisibility(_ visible: Bool) {
    if visible { presentCharacter() }
    else { character.orderOut(nil); model.animate(["visible": false]) }
    visibilityMenuItem?.title = visible ? "Hide avatar" : "Show avatar"
  }
  @objc func toggleClickThrough() { character.ignoresMouseEvents.toggle() }
  @objc func mute() { model.mute() }
  @objc func quit() { NSApp.terminate(nil) }
}

private let ink = Color(red: 0.15, green: 0.22, blue: 0.23)
private let accent = Color(red: 0.21, green: 0.43, blue: 0.39)
private let paper = Color(red: 0.97, green: 0.96, blue: 0.93)
struct CompanionView: View {
  @ObservedObject var model: AppModel
  @State private var draft = ""
  @State private var page = "Conversation"
  @State private var memory = ""
  @State private var editingMemory: String?
  @State private var creatingCharacter = false
  @State private var newName = ""
  @State private var newPersona =
    "A calm, curious adult character who speaks naturally and enjoys thoughtful conversation."
  @State private var newPalette = "teal"
  @State private var useAriVoice = true
  var body: some View {
    HStack(spacing: 0) {
      VStack(alignment: .leading, spacing: 24) {
        HStack(spacing: 10) {
          Image(systemName: "sparkle").font(.title2).foregroundStyle(accent)
          VStack(alignment: .leading, spacing: 2) {
            Text("COMPANION").font(.system(size: 10, weight: .semibold, design: .rounded)).tracking(
              2)
            Text("A little company.").font(.system(size: 12)).foregroundStyle(.secondary)
          }
        }.padding(.top, 12)
        VStack(alignment: .leading, spacing: 6) {
          ForEach(["Conversation", "History", "Character", "Memory", "Settings"], id: \.self) {
            item in
            Button {
              page = item
              Task { await model.refresh() }
            } label: {
              Label(item, systemImage: icon(item)).lineLimit(1).font(
                .system(size: 13, weight: page == item ? .semibold : .regular)
              ).frame(maxWidth: .infinity, alignment: .leading).padding(10).background(
                page == item ? accent.opacity(0.10) : .clear, in: RoundedRectangle(cornerRadius: 9))
            }.buttonStyle(.plain)
          }
        }
        Spacer()
        VStack(alignment: .leading, spacing: 8) {
          HStack {
            Circle().fill(model.connected ? accent : .orange).frame(width: 6, height: 6)
            Text(model.demo ? "Development preview" : "On your Mac").font(
              .system(size: 11, weight: .medium))
          }
          Text("English · 한국어 · 日本語").font(.system(size: 10)).foregroundStyle(.secondary)
          Text(
            model.memoryGB > 0
              ? String(format: "%.1f GB · %@", model.memoryGB, model.profile.capitalized)
              : "Private by default"
          ).font(.system(size: 10)).foregroundStyle(.secondary)
        }
      }.padding(20).frame(width: 194).frame(maxHeight: .infinity).background(paper)
      Rectangle().fill(ink.opacity(0.08)).frame(width: 1)
      VStack(spacing: 0) {
        HStack {
          VStack(alignment: .leading, spacing: 5) {
            Text(page == "Conversation" ? model.characterName : page).font(
              .system(size: 27, weight: .medium, design: .serif))
            Text(subtitle).font(.system(size: 12)).foregroundStyle(.secondary)
          }
          Spacer()
          if page == "Conversation" {
            Button {
              Task { await model.openConversation(nil) }
            } label: {
              Image(systemName: "square.and.pencil")
            }.help("New conversation").buttonStyle(.borderless)
          }
          Button { model.overlayVisible.toggle() } label: {
            Label(model.overlayVisible ? "Hide avatar" : "Show avatar",
                  systemImage: model.overlayVisible ? "eye.slash" : "eye")
          }.help("Show or hide the desktop avatar (Command–Shift–H)").buttonStyle(.borderless)
        }.padding(.horizontal, 30).padding(.top, 32).padding(.bottom, 22)
        Divider().opacity(0.5)
        if !model.notice.isEmpty {
          HStack(alignment: .top) {
            Image(systemName: "info.circle")
            Text(model.notice).font(.system(size: 12)).textSelection(.enabled)
            Spacer()
            Button {
              model.notice = ""
            } label: {
              Image(systemName: "xmark")
            }.buttonStyle(.plain)
          }.padding(12).background(Color.orange.opacity(0.09)).padding(.horizontal, 24).padding(
            .top, 12)
        }
        content
      }.background(Color(red: 0.995, green: 0.989, blue: 0.974))
    }.foregroundStyle(ink).tint(accent).preferredColorScheme(.light)
      .sheet(isPresented: $creatingCharacter) {
        VStack(alignment: .leading, spacing: 18) {
          Text("Create a character").font(.title2)
          Text(
            "Use the built-in animated rig, then give it a palette and a personality. Import a bundle for entirely different artwork."
          ).foregroundStyle(.secondary)
          TextField("Character name", text: $newName).textFieldStyle(.roundedBorder)
          Picker("Palette", selection: $newPalette) {
            ForEach(["teal", "plum", "copper", "midnight"], id: \.self) {
              Text($0.capitalized).tag($0)
            }
          }
          TextEditor(text: $newPersona).frame(height: 140).border(.secondary.opacity(0.2))
          Toggle("Use Ari's original generated voice", isOn: $useAriVoice)
          HStack {
            Button("Cancel") { creatingCharacter = false }.keyboardShortcut(.cancelAction)
            Spacer()
            Button("Create") {
              Task {
                if await model.createCharacter(
                  name: newName, persona: newPersona, palette: newPalette, useVoice: useAriVoice)
                {
                  creatingCharacter = false
                  newName = ""
                }
              }
            }.buttonStyle(.borderedProminent).disabled(
              newName.trimmingCharacters(in: .whitespaces).isEmpty || newPersona.isEmpty)
          }
        }.padding(28).frame(width: 460)
      }
  }
  var subtitle: String {
    switch page {
    case "Conversation":
      return model.listening
        ? "Listening · microphone on"
        : model.state == "Ready" ? "Here when you want to talk. Microphone off." : model.state
    case "Character": return "A personality, a voice, a little expression."
    case "Memory": return "Only the things you explicitly choose to remember."
    case "History": return "Your conversations stay on this Mac."
    default: return "Keep room for everything else you're doing."
    }
  }
  func icon(_ title: String) -> String {
    [
      "Conversation": "bubble.left.and.bubble.right", "History": "clock",
      "Character": "face.smiling", "Memory": "bookmark", "Settings": "slider.horizontal.3",
    ][title]!
  }
  @ViewBuilder var content: some View {
    switch page {
    case "Conversation": conversation
    case "Character":
      ScrollView {
        VStack(alignment: .leading, spacing: 20) {
          Toggle("Show desktop avatar", isOn: $model.overlayVisible)
          Picker("Character", selection: $model.characterID) {
            ForEach(model.characters.compactMap { $0["id"] as? String }, id: \.self) {
              id in
              Text(model.characters.first { $0["id"] as? String == id }?["name"] as? String ?? id)
                .tag(id)
            }
          }.onChange(of: model.characterID) { _, id in
            model.command(["type": "character", "character_id": id])
          }
          if model.characterID == "ari" {
            HStack {
              Picker("Appearance", selection: $model.visualMode) {
                Text("Illustration").tag("illustration")
                Text("VRoid 3D").tag("model")
              }
              Picker("Framing", selection: $model.characterFraming) {
                Text("Full body").tag("full")
                Text("Portrait").tag("portrait")
              }
            }
            overlayWardrobeControls
            Button("Open character studio", systemImage: "person.crop.rectangle") {
              model.openCharacterStudio?()
            }
            Text(model.visualMode == "model"
              ? "Ari v9 — the same finished model as Character Studio, with blinking, gaze, speech and gestures. Likeness is still in progress."
              : "Full body shows the neutral reference. Portrait shows eight emotion illustrations. Explore both in the studio.")
              .font(.callout).foregroundStyle(.secondary)
            if model.visualMode == "model" {
              LazyVGrid(columns: [GridItem(.adaptive(minimum: 110))], alignment: .leading) {
                ForEach(["wave", "nod", "explain", "shift", "step", "twirl", "raise-arms", "shrug", "point"], id: \.self) { gesture in
                  Button(gesture == "twirl" ? "Cute twirl" : gesture.replacingOccurrences(of: "-", with: " ").capitalized) {
                    model.showCharacter?(); model.animate(["gesture": gesture])
                  }
                }
              }
            }
          }
          TextField("Name", text: $model.characterName).textFieldStyle(.roundedBorder)
          HStack {
            Picker(
              "Voice",
              selection: Binding(
                get: { model.voiceID }, set: { id in Task { await model.selectVoice(id) } })
            ) {
              if !model.voices.contains(where: { $0["id"] as? String == "character" }) {
                Text("No voice reference").tag("character")
              }
              ForEach(model.voices.compactMap { $0["id"] as? String }, id: \.self) { id in
                Text(model.voices.first { $0["id"] as? String == id }?["name"] as? String ?? id)
                  .tag(id)
              }
            }.disabled(model.voiceBusy || model.voices.isEmpty)
            Button("Preview voice", systemImage: "play.circle") {
              Task { await model.previewVoice() }
            }
            .disabled(
              model.voiceBusy
                || !model.voices.contains(where: { $0["id"] as? String == model.voiceID }))
          }
          Text(
            model.voices.first { $0["id"] as? String == model.voiceID }?["description"] as? String
              ?? "Choose a voice for this character."
          )
          .font(.callout).foregroundStyle(.secondary)
          Text("PERSONALITY").font(.caption).tracking(1.4)
          TextEditor(text: $model.persona).font(.body).frame(minHeight: 180).padding(8).background(
            .white, in: RoundedRectangle(cornerRadius: 12))
          HStack {
            Button("Save personality") { Task { await model.savePersona() } }.buttonStyle(
              .borderedProminent)
            Button("Import character…") { model.importCharacter() }
            Button("Create character…") { creatingCharacter = true }
          }
          Text(
            "Ari’s illustrated design and 3D study share the same reference. Imported bundles can supply layered PNG art and a consented voice reference."
          ).font(.callout).foregroundStyle(.secondary)
          Text("Expression preview").font(.headline)
          LazyVGrid(columns: [GridItem(.adaptive(minimum: 110))]) {
            ForEach(
              [
                "neutral", "happy", "curious", "concerned", "surprised", "annoyed", "embarrassed",
                "sleepy",
              ], id: \.self
            ) { mood in
              Button(mood.capitalized) {
                model.showCharacter?()
                model.animate(["expression": mood])
              }.buttonStyle(.bordered)
            }
          }
        }.padding(30)
      }
    case "Memory":
      ScrollView {
        VStack(alignment: .leading, spacing: 18) {
          Text("Nothing is remembered automatically.").font(.headline)
          HStack {
            TextField("A fact you'd like to keep", text: $memory).textFieldStyle(.roundedBorder)
            Button(editingMemory == nil ? "Save with my approval" : "Approve changes") {
              let text = memory
              let identifier = editingMemory
              memory = ""
              editingMemory = nil
              Task {
                if let identifier {
                  await model.editMemory(identifier, text: text)
                } else {
                  await model.saveMemory(text)
                }
              }
            }.disabled(memory.trimmingCharacters(in: .whitespaces).isEmpty)
          }
          ForEach(model.memories.compactMap { $0["id"] as? String }, id: \.self) { id in
            HStack {
              Text(model.memories.first { $0["id"] as? String == id }?["text"] as? String ?? "")
                .textSelection(.enabled)
              Spacer()
              Button("Edit") {
                editingMemory = id
                memory =
                  model.memories.first { $0["id"] as? String == id }?["text"] as? String ?? ""
              }.buttonStyle(.borderless)
              Button(role: .destructive) {
                Task { await model.deleteMemory(id) }
              } label: {
                Image(systemName: "trash")
              }.buttonStyle(.borderless)
            }
          }
          if model.memories.isEmpty {
            Text("Your memory collection is empty.").foregroundStyle(.secondary).padding(.top, 20)
          }
        }.padding(30)
      }
    case "History":
      ScrollView {
        VStack(alignment: .leading, spacing: 14) {
          HStack {
            Button("New conversation") {
              page = "Conversation"
              Task { await model.openConversation(nil) }
            }
            Spacer()
            Button("Export history…") { model.exportHistory() }
          }
          ForEach(model.conversations.compactMap { $0["id"] as? String }, id: \.self) { id in
            HStack {
              Button {
                page = "Conversation"
                Task { await model.openConversation(id) }
              } label: {
                VStack(alignment: .leading, spacing: 5) {
                  Text(
                    model.conversations.first { $0["id"] as? String == id }?["title"] as? String
                      ?? "Conversation"
                  ).lineLimit(2)
                  if id == model.conversationID {
                    Text("Current conversation").font(.caption).foregroundStyle(.secondary)
                  }
                }
              }.buttonStyle(.plain)
              Spacer()
              Button(role: .destructive) {
                Task { await model.deleteConversation(id) }
              } label: {
                Image(systemName: "trash")
              }.disabled(id == model.conversationID).buttonStyle(.borderless)
            }.padding(14).background(.white, in: RoundedRectangle(cornerRadius: 10))
          }
        }.padding(30)
      }
    default:
      ScrollView {
        VStack(alignment: .leading, spacing: 24) {
          VStack(alignment: .leading, spacing: 10) {
            Text("Performance").font(.headline)
            Picker("Mode", selection: $model.profile) {
              Text("Everyday · 9B").tag("everyday")
              Text("Quiet · 4B").tag("quiet")
            }.pickerStyle(.segmented).onChange(of: model.profile) { _, value in
              model.command(["type": "profile", "profile": value])
            }
            Text(
              "Quiet leaves more room for your browser, editor, and other apps. Models unload after ten minutes without a conversation."
            ).font(.callout).foregroundStyle(.secondary)
          }
          Picker("Language", selection: $model.language) {
            Text("Automatic").tag("auto")
            Text("English").tag("english")
            Text("한국어").tag("korean")
            Text("日本語").tag("japanese")
          }.onChange(of: model.language) { _, value in
            model.mute()
            model.command(["type": "language", "language": value])
          }
          Toggle("Speak replies", isOn: $model.speak)
          VStack(alignment: .leading, spacing: 8) {
            Text("Microphone").font(.headline)
            Text(
              "Hold Control–Option–Space from any app to talk, or use the Talk button. Hands-free stays on until you mute it. The microphone starts off and stays off after sleep."
            ).font(.callout).foregroundStyle(.secondary)
          }
          VStack(alignment: .leading, spacing: 8) {
            Text("Local diagnostics").font(.headline)
            Text(String(format: "Worker memory: %.1f GB", model.memoryGB))
            if let latency = model.latency {
              Text(String(format: "Last first-audio generation: %.2f s", latency))
            }
            Button("Open runtime folder") {
              NSWorkspace.shared.open(model.root.appendingPathComponent(".runtime"))
            }
            Button("Export conversations and memories…") { model.exportHistory() }
          }
        }.padding(30)
      }
    }
  }
  var conversation: some View {
    VStack(spacing: 0) {
      ScrollViewReader { proxy in
        ScrollView {
          LazyVStack(alignment: .leading, spacing: 23) {
            if model.messages.isEmpty {
              VStack(alignment: .leading, spacing: 16) {
                Image(systemName: "sun.horizon").font(.system(size: 32, weight: .ultraLight))
                  .foregroundStyle(accent)
                Text("Make yourself at home.").font(
                  .system(size: 27, weight: .regular, design: .serif))
                Text(
                  "Tell me what's on your mind, practice a language,\nor simply take a small break."
                ).font(.system(size: 14)).foregroundStyle(.secondary).lineSpacing(5)
                HStack {
                  suggestion("오늘 하루 어땠어?")
                  suggestion("日本語で話そう")
                  suggestion("Let's chat")
                }
              }.padding(.top, 80).padding(.bottom, 40)
            }
            ForEach(model.messages) { message in
              VStack(alignment: .leading, spacing: 8) {
                Text(message.role == "user" ? "YOU" : model.characterName.uppercased()).font(
                  .system(size: 10, weight: .semibold)
                ).tracking(1.4).foregroundStyle(.secondary)
                Text(message.text.isEmpty ? "…" : message.text).font(.system(size: 15)).lineSpacing(
                  5
                ).textSelection(.enabled)
                if message.interrupted {
                  Text("Interrupted · some text may not have been spoken").font(.caption)
                    .foregroundStyle(.secondary)
                }
              }.frame(maxWidth: .infinity, alignment: .leading).id(message.id)
            }
            Color.clear.frame(height: 1).id("bottom")
          }.padding(.horizontal, 32).padding(.bottom, 24)
        }.onChange(of: model.messages.last?.text) { _, _ in
          proxy.scrollTo("bottom", anchor: .bottom)
        }
      }
      if !model.transcript.isEmpty {
        Text(model.transcript).font(.callout).foregroundStyle(.secondary).frame(
          maxWidth: .infinity, alignment: .leading
        ).padding(.horizontal, 32).padding(.bottom, 10)
      }
      VStack(spacing: 12) {
        if model.microphoneRequested {
          Text(model.inputStatus).font(.caption).foregroundStyle(.secondary)
            .frame(maxWidth: .infinity, alignment: .leading)
        }
        HStack(alignment: .bottom) {
          TextField("Say something…", text: $draft, axis: .vertical).textFieldStyle(.plain).font(
            .system(size: 14)
          ).lineLimit(1...5).onSubmit { submit() }
          Button(action: submit) {
            Image(systemName: "arrow.up").font(.system(size: 14, weight: .semibold)).frame(
              width: 30, height: 30
            ).background(accent, in: Circle()).foregroundStyle(.white)
          }.buttonStyle(.plain).disabled(
            draft.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty || !model.connected)
        }.padding(16).background(.white, in: RoundedRectangle(cornerRadius: 16)).overlay(
          RoundedRectangle(cornerRadius: 16).stroke(ink.opacity(0.1), lineWidth: 1))
        HStack(spacing: 16) {
          Button {
            model.toggleTalk()
          } label: {
            Label(
              model.microphoneRequested && !model.handsFree ? "Send voice" : "Talk",
              systemImage: "mic.fill"
            )
            .font(.system(size: 12, weight: .medium)).padding(.horizontal, 14).padding(.vertical, 8)
            .background(model.listening ? accent.opacity(0.2) : ink.opacity(0.055), in: Capsule())
          }.buttonStyle(.plain).disabled(!model.connected)
            .help("Click to start, then click Send voice. Or hold Control–Option–Space.")
          Button {
            model.toggleHandsFree()
          } label: {
            Label(
              model.handsFree && model.microphoneRequested ? "Mute" : "Hands-free",
              systemImage: model.handsFree && model.microphoneRequested ? "mic.slash" : "waveform")
          }.buttonStyle(.plain).font(.system(size: 12)).disabled(!model.connected)
          if model.microphoneRequested {
            ProgressView(value: Double(model.inputLevel)).frame(width: 65)
              .accessibilityLabel("Microphone input level")
              .accessibilityValue("\(Int(model.inputLevel * 100)) percent")
          }
          Spacer()
          Button {
            model.stop()
          } label: {
            Label("Stop", systemImage: "stop.fill")
          }.buttonStyle(.plain).font(.system(size: 12)).keyboardShortcut(.escape, modifiers: [])
        }
      }.padding(.horizontal, 26).padding(.bottom, 22)
    }
  }
  private func lookBinding(_ key: WritableKeyPath<OverlayLook, String>) -> Binding<String> {
    Binding(get: { model.overlayLook[keyPath: key] }, set: { model.selectOverlayLook(key, $0) })
  }
  private var overlayWardrobeControls: some View {
    VStack(alignment: .leading, spacing: 12) {
      Text("Desktop wardrobe").font(.headline)
      Toggle("Randomize outfit on startup", isOn: $model.randomizeOutfitOnStartup)
      Text("Picks a different outfit next launch. Hair and accessories stay as selected. Turn off to keep your current outfit.")
        .font(.caption).foregroundStyle(.secondary)
      Picker("Outfit", selection: lookBinding(\.outfit)) {
        ForEach(model.wardrobeCatalog.outfits) { Text($0.label).tag($0.id) }
      }
      Picker("Hairstyle", selection: lookBinding(\.hair)) {
        ForEach(model.wardrobeCatalog.hairstyles) { Text($0.label).tag($0.id) }
      }
      HStack {
        Picker("Glasses", selection: lookBinding(\.glasses)) {
          Text("None").tag("none")
          Text("Round frames").tag("round")
        }
        Picker("Headwear", selection: lookBinding(\.headwear)) {
          Text("None").tag("none")
          if model.overlayLook.hair != "double-buns" { Text("Butter beret").tag("beret") }
        }
      }
      if model.overlayLook.hair == "double-buns" {
        Text("Beret unavailable with double buns. Choose another hairstyle to wear it.")
          .font(.caption).foregroundStyle(.secondary)
      }
      Text("Changes apply immediately to the 3D desktop avatar and are saved. Studio keeps its own preview selections.")
        .font(.caption).foregroundStyle(.secondary)
    }
  }
  func suggestion(_ text: String) -> some View {
    Button(text) { model.sendText(text) }.buttonStyle(.bordered).font(.system(size: 11)).disabled(
      !model.connected)
  }
  func submit() {
    let text = draft.trimmingCharacters(in: .whitespacesAndNewlines)
    guard !text.isEmpty, model.connected else { return }
    draft = ""
    model.sendText(text)
  }
}
