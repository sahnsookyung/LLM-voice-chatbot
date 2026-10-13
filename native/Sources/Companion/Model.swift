import AVFoundation
import AppKit
import SwiftUI
import WebKit

struct ChatMessage: Identifiable {
  var id: String
  var role: String
  var text: String
  var interrupted = false
}
@MainActor final class AppModel: ObservableObject {
  @Published var connected = false
  @Published var state = "Starting"
  @Published var notice = ""
  @Published var messages: [ChatMessage] = []
  @Published var transcript = ""
  @Published var profile = "everyday"
  @Published var language = "auto"
  @Published var listening = false
  @Published var handsFree = false
  @Published var speak = true
  @Published var demo = false
  @Published var memoryGB = 0.0
  @Published var latency: Double?
  @Published var characterName = "Ari"
  @Published var characterID = "ari"
  @Published var visualMode = UserDefaults.standard.string(forKey: "ariVisualMode") ?? "illustration" {
    didSet { UserDefaults.standard.set(visualMode, forKey: "ariVisualMode"); animate(overlayAppearanceEvent) }
  }
  @Published var overlayVisible = UserDefaults.standard.object(forKey: "ariOverlayVisible") as? Bool ?? true {
    didSet {
      UserDefaults.standard.set(overlayVisible, forKey: "ariOverlayVisible")
      setOverlayVisibility?(overlayVisible)
    }
  }
  let wardrobeCatalog: WardrobeCatalog
  @Published private(set) var overlayLook: OverlayLook
  @Published var randomizeOutfitOnStartup = OverlayLook.randomizesOnStartup(in: .standard) {
    didSet { UserDefaults.standard.set(randomizeOutfitOnStartup, forKey: OverlayLook.randomizeOnStartupKey) }
  }
  var setOverlayVisibility: ((Bool) -> Void)?
  @Published var characterFraming = UserDefaults.standard.string(forKey: "ariFraming") ?? "full" {
    didSet {
      UserDefaults.standard.set(characterFraming, forKey: "ariFraming")
      animate(["framing": characterFraming])
      resizeCharacter?()
    }
  }
  var resizeCharacter: (() -> Void)?
  var openCharacterStudio: (() -> Void)?
  @Published var persona = ""
  @Published var characters: [[String: Any]] = []
  @Published var memories: [[String: Any]] = []
  @Published var conversations: [[String: Any]] = []
  @Published var voices: [[String: Any]] = []
  @Published var voiceID = "character"
  @Published var voiceBusy = false
  @Published var inputLevel: Float = 0
  @Published var inputStatus = ""
  var conversationID = ""
  let audio = AudioController()
  weak var webView: WKWebView?
  var showCharacter: (() -> Void)?
  private var process: Process?
  private var socket: URLSessionWebSocketTask?
  private var receiveTask: Task<Void, Never>?
  private var outbound: Task<Void, Never>?
  @Published private(set) var microphoneRequested = false
  private var microphoneRequest = UUID()
  private var voicePreview: AVAudioPlayer?
  private var previewGeneration = UUID()
  private var releasePending = false
  private var warmAudio = Data()
  private var port = 0
  private let token = UUID().uuidString + UUID().uuidString
  private var pressureSource: DispatchSourceMemoryPressure?
  private var shuttingDown = false
  let root: URL
  init() {
    let catalog = WardrobeCatalog.load(from: Bundle.main.resourceURL?.appendingPathComponent("renderer/wardrobe-catalog.json"))
    wardrobeCatalog = catalog
    var random = SystemRandomNumberGenerator()
    overlayLook = OverlayLook.loadForStartup(defaults: .standard, catalog: catalog, using: &random)
    let configured =
      ProcessInfo.processInfo.environment["LVC_ROOT"]
      ?? ((try? String(
        contentsOf: Bundle.main.resourceURL!.appendingPathComponent("workspace.txt"),
        encoding: .utf8))?.trimmingCharacters(in: .whitespacesAndNewlines))
      ?? FileManager.default.currentDirectoryPath
    root = URL(fileURLWithPath: configured)
    audio.onPCM = { [weak self] data in
      guard let self else { return }
      if self.listening {
        self.sendData(data)
      } else if self.microphoneRequested {
        if self.warmAudio.count + data.count <= 320000 {
          self.warmAudio.append(data)
        } else {
          self.mute()
          self.notice = "Speech is still warming up. Please try Talk again."
        }
      }
    }
    audio.onAck = { [weak self] turn, samples in
      self?.command(["type": "ack", "turn_id": turn, "samples": samples])
    }
    audio.onLevel = { [weak self] level in self?.animate(["level": level]) }
    audio.onInputLevel = { [weak self] level in self?.inputLevel = level }
    audio.onExpression = { [weak self] expression in self?.animate(["expression": expression]) }
    audio.onGesture = { [weak self] gesture in self?.animate(["gesture": gesture]) }
    audio.onFailure = { [weak self] message in
      self?.mute()
      self?.notice = message
    }
    pressureSource = DispatchSource.makeMemoryPressureSource(
      eventMask: [.normal, .warning, .critical], queue: .main)
    pressureSource?.setEventHandler { [weak self] in
      guard let self, let data = self.pressureSource?.data else { return }
      self.command([
        "type": "pressure",
        "pressure": data.contains(.critical)
          ? "critical" : data.contains(.warning) ? "warning" : "normal",
      ])
    }
    pressureSource?.resume()
    NSWorkspace.shared.notificationCenter.addObserver(
      forName: NSWorkspace.willSleepNotification, object: nil, queue: .main
    ) { [weak self] _ in Task { @MainActor in self?.mute() } }
    NSWorkspace.shared.notificationCenter.addObserver(
      forName: NSWorkspace.didWakeNotification, object: nil, queue: .main
    ) { [weak self] _ in
      Task { @MainActor in self?.notice = "Welcome back. The microphone is off." }
    }
    Task { await launch() }
  }
  func launch() async {
    guard process == nil else { return }
    let executable = root.appendingPathComponent(".venv/bin/python")
    guard FileManager.default.isExecutableFile(atPath: executable.path) else {
      state = "Setup needed"
      notice = "Run scripts/setup.sh in the project, then reopen the app."
      return
    }
    let p = Process()
    let pipe = Pipe()
    p.executableURL = executable
    p.arguments = [
      "-m", "companion", "--parent", String(ProcessInfo.processInfo.processIdentifier),
    ]
    if ProcessInfo.processInfo.arguments.contains("--demo") { p.arguments?.append("--demo") }
    var env = ProcessInfo.processInfo.environment
    env["LVC_ROOT"] = root.path
    env["LVC_TOKEN"] = token
    env["PYTHONPATH"] = root.appendingPathComponent("backend").path
    env["PYTHONUNBUFFERED"] = "1"
    p.environment = env
    p.currentDirectoryURL = root
    p.standardOutput = pipe
    let logs = root.appendingPathComponent(".runtime/app.log")
    FileManager.default.createFile(
      atPath: logs.path, contents: nil, attributes: [.posixPermissions: 0o600])
    p.standardError = try? FileHandle(forWritingTo: logs)
    p.terminationHandler = { [weak self] _ in
      Task { @MainActor in
        guard let self, !self.shuttingDown else { return }
        self.connected = false
        self.state = "Disconnected"
        self.audio.close()
        self.notice =
          "The local service stopped. Reopen the app; diagnostics are in .runtime/app.log."
      }
    }
    do {
      try p.run()
      process = p
      let data = try await Task.detached {
        var line = Data()
        while line.count < 8192, let byte = try pipe.fileHandleForReading.read(upToCount: 1),
          !byte.isEmpty
        {
          if byte[0] == 10 { break }
          line.append(byte)
        }
        return line
      }.value
      guard let first = String(data: data, encoding: .utf8)?.split(separator: "\n").first,
        let bytes = String(first).data(using: .utf8),
        let info = try JSONSerialization.jsonObject(with: bytes) as? [String: Any],
        let found = info["port"] as? Int
      else { throw URLError(.cannotConnectToHost) }
      port = found
      for _ in 0..<60 {
        if (try? await request("health")) != nil { break }
        try await Task.sleep(for: .milliseconds(100))
      }
      await connect()
      await refresh()
    } catch {
      notice = "Unable to start the local service: \(error.localizedDescription)"
      state = "Setup needed"
    }
  }
  func request(_ path: String, method: String = "GET", body: Any? = nil) async throws -> Any {
    var req = URLRequest(url: URL(string: "http://127.0.0.1:\(port)/\(path)")!)
    req.httpMethod = method
    req.setValue("Bearer \(token)", forHTTPHeaderField: "Authorization")
    if let body {
      req.httpBody = try JSONSerialization.data(withJSONObject: body)
      req.setValue("application/json", forHTTPHeaderField: "Content-Type")
    }
    let (data, response) = try await URLSession.shared.data(for: req)
    guard let http = response as? HTTPURLResponse, (200..<300).contains(http.statusCode) else {
      let detail =
        (try? JSONSerialization.jsonObject(with: data) as? [String: Any])?["detail"] as? String
      throw NSError(
        domain: "Companion", code: 1,
        userInfo: [NSLocalizedDescriptionKey: detail ?? "The local service rejected the request."])
    }
    return try JSONSerialization.jsonObject(with: data)
  }
  func connect(conversation: String? = nil) async {
    receiveTask?.cancel()
    socket?.cancel(with: .normalClosure, reason: nil)
    audio.stopCapture()
    audio.stopPlayback()
    listening = false
    // Wait for actual cleanup instead of assuming process teardown fits a fixed delay.
    connected = false
    for _ in 0..<80 {
      let health = try? await request("health") as? [String: Any]
      if health?["session_active"] as? Bool == false { break }
      try? await Task.sleep(for: .milliseconds(100))
    }
    var components = URLComponents(string: "ws://127.0.0.1:\(port)/session")!
    if let conversation {
      components.queryItems = [URLQueryItem(name: "conversation", value: conversation)]
    }
    var req = URLRequest(url: components.url!)
    req.setValue("Bearer \(token)", forHTTPHeaderField: "Authorization")
    let ws = URLSession.shared.webSocketTask(with: req)
    socket = ws
    ws.resume()
    receiveTask = Task { [weak self] in
      do {
        while !Task.isCancelled {
          let msg = try await ws.receive()
          guard let self else { return }
          switch msg {
          case .data(let data): try self.audio.accept(data)
          case .string(let text):
            if let data = text.data(using: .utf8),
              let event = try JSONSerialization.jsonObject(with: data) as? [String: Any]
            {
              await self.handle(event)
            }
          @unknown default: break
          }
        }
      } catch {
        guard !Task.isCancelled else { return }
        self?.connected = false
        self?.audio.stopCapture()
        self?.audio.stopPlayback()
        self?.notice = "Connection lost. Reopen the app to reconnect."
      }
    }
  }
  func command(_ value: [String: Any]) {
    guard let data = try? JSONSerialization.data(withJSONObject: value),
      let text = String(data: data, encoding: .utf8)
    else { return }
    enqueue(.string(text))
  }
  private func sendData(_ data: Data) { enqueue(.data(data)) }
  private func enqueue(_ message: URLSessionWebSocketTask.Message) {
    guard let ws = socket else { return }
    let previous = outbound
    outbound = Task { [weak self] in
      await previous?.value
      do { try await ws.send(message) } catch {
        if self?.connected == true { self?.notice = "The local connection could not send data." }
      }
    }
  }
  func handle(_ event: [String: Any]) async {
    let type = event["type"] as? String ?? ""
    let turn = event["turn_id"] as? String ?? ""
    switch type {
    case "ready":
      connected = true
      state = "Ready"
      profile = event["profile"] as? String ?? "everyday"
      language = event["language"] as? String ?? "auto"
      conversationID = event["conversation_id"] as? String ?? ""
      demo = event["demo"] as? Bool ?? false
      characterID = event["character_id"] as? String ?? "ari"
      await loadCharacter()
    case "state":
      state = (event["state"] as? String ?? "idle").capitalized
      animate(["state": (event["state"] as? String ?? "idle")])
    case "notice", "error":
      notice = event["message"] as? String ?? "Unknown error"
      if notice.hasPrefix("No speech was recognized") {
        if audio.capturedFrames == 0 {
          notice =
            "The microphone did not deliver audio. Check System Settings → Sound → Input, then try Talk again."
        } else if audio.capturedPeak < 0.005 {
          notice =
            "The microphone delivered silence. Check the input volume and selected device in System Settings → Sound → Input."
        }
      }
      if type == "error" {
        state = "Needs attention"
        microphoneRequested = false
        audio.stopCapture()
        listening = false
        handsFree = false
        microphoneRequest = UUID()
        warmAudio.removeAll()
      }
    case "listening":
      listening = event["enabled"] as? Bool ?? false
      if listening {
        guard microphoneRequested else {
          mute()
          return
        }
        do {
          if !releasePending { try audio.startCapture(voiceProcessing: handsFree) }
          inputStatus = audio.captureFormat
          for start in stride(from: 0, to: warmAudio.count, by: 2048) {
            sendData(warmAudio.subdata(in: start..<min(start + 2048, warmAudio.count)))
          }
          warmAudio.removeAll(keepingCapacity: true)
          if releasePending { endPTT() }
        } catch {
          mute()
          notice = error.localizedDescription
        }
      } else {
        audio.stopCapture()
        microphoneRequested = false
        handsFree = false
        inputLevel = 0
      }
    case "transcript": transcript = event["text"] as? String ?? ""
    case "turn":
      transcript = ""
      audio.beginTurn(turn)
      messages.append(
        ChatMessage(id: UUID().uuidString, role: "user", text: event["user_text"] as? String ?? ""))
      messages.append(ChatMessage(id: turn, role: "assistant", text: ""))
    case "text":
      if let i = messages.firstIndex(where: { $0.id == turn }) {
        messages[i].text += event["text"] as? String ?? ""
      }
    case "cancel":
      audio.stopPlayback()
      if let i = messages.firstIndex(where: { $0.id == turn }) { messages[i].interrupted = true }
    case "done": latency = event["first_audio_seconds"] as? Double
    case "gesture":
      audio.applyGesture(
        turn: turn, gesture: event["gesture"] as? String ?? "",
        waitForAudio: event["wait_for_audio"] as? Bool ?? false)
    case "expression":
      audio.queueExpression(
        turn: turn, offset: (event["sample_offset"] as? NSNumber)?.uint64Value ?? 0,
        expression: event["expression"] as? String ?? "neutral")
    case "metrics": memoryGB = (event["process_rss_bytes"] as? Double ?? 0) / 1_000_000_000
    case "profile": profile = event["profile"] as? String ?? profile
    case "character":
      characterID = event["character_id"] as? String ?? "ari"
      await loadCharacter()
    default: break
    }
  }
  func sendText(_ text: String) {
    notice = ""
    command(["type": "text", "text": text, "speak": speak])
  }
  func beginPTT() {
    guard connected, !microphoneRequested, !listening else { return }
    handsFree = false
    releasePending = false
    startMic(mode: "ptt")
  }
  func toggleTalk() {
    if microphoneRequested && !handsFree {
      endPTT()
    } else {
      if microphoneRequested || listening { mute() }
      beginPTT()
    }
  }
  func endPTT() {
    guard microphoneRequested, !handsFree else { return }
    guard listening else {
      releasePending = true
      audio.stopCapture()
      return
    }
    audio.stopCapture()
    listening = false
    state = "Processing speech"
    command(["type": "commit"])
    microphoneRequested = false
    releasePending = false
  }
  func startMic(mode: String) {
    voicePreview?.stop()
    previewGeneration = UUID()
    audio.stopPlayback()
    microphoneRequested = true
    let request = UUID()
    microphoneRequest = request
    warmAudio.removeAll(keepingCapacity: true)
    notice = ""
    state = "Warming"
    Task {
      let granted = await AVCaptureDevice.requestAccess(for: .audio)
      guard microphoneRequested, microphoneRequest == request else { return }
      guard granted else {
        microphoneRequested = false
        handsFree = false
        notice =
          "Microphone access is off. Enable it in System Settings → Privacy & Security → Microphone."
        state = "Ready"
        return
      }
      if releasePending && warmAudio.isEmpty {
        microphoneRequested = false
        releasePending = false
        state = "Ready"
        notice = "Microphone access is ready. Click Talk, speak, then click Send voice."
        return
      }
      do {
        // Push-to-talk does not play a reply while recording, so it needs no echo canceller.
        if !releasePending { try audio.startCapture(voiceProcessing: mode == "handsfree") }
        inputStatus = audio.captureFormat
        command(["type": "start", "mode": mode])
      } catch {
        mute()
        notice = error.localizedDescription
      }
    }
  }
  func toggleHandsFree() {
    if listening || microphoneRequested {
      mute()
    } else {
      handsFree = true
      releasePending = false
      startMic(mode: "handsfree")
    }
  }
  func mute() {
    microphoneRequest = UUID()
    microphoneRequested = false
    releasePending = false
    listening = false
    handsFree = false
    inputLevel = 0
    state = "Ready"
    warmAudio.removeAll()
    audio.stopCapture()
    stop()
    command(["type": "mute"])
  }
  func stop() {
    previewGeneration = UUID()
    voicePreview?.stop()
    audio.stopPlayback()
    command(["type": "stop"])
  }
  func animate(_ data: [String: Any]) {
    guard let raw = try? JSONSerialization.data(withJSONObject: data),
      let json = String(data: raw, encoding: .utf8)
    else { return }
    webView?.evaluateJavaScript(
      "window.companion && window.companion(\(json))", completionHandler: nil)
  }
  var overlayAppearanceEvent: [String: Any] {
    var event = overlayLook.bridge
    event["visualMode"] = visualMode
    event["framing"] = characterFraming
    return event
  }
  func selectOverlayLook(_ keyPath: WritableKeyPath<OverlayLook, String>, _ value: String) {
    var next = overlayLook
    next[keyPath: keyPath] = value
    overlayLook = next.normalized(in: wardrobeCatalog)
    if let data = try? JSONEncoder().encode(overlayLook) {
      UserDefaults.standard.set(data, forKey: OverlayLook.savedLookKey)
    }
    // Send the new look and 3D mode together, without briefly loading the old look.
    visualMode = "model"
  }
  func refresh() async {
    do {
      memories = try await request("memories") as? [[String: Any]] ?? []
      conversations = try await request("conversations") as? [[String: Any]] ?? []
      characters = try await request("characters") as? [[String: Any]] ?? []
    } catch { notice = error.localizedDescription }
  }
  func loadCharacter() async {
    guard port > 0, connected else { return }
    do {
      guard var c = try await request("characters/\(characterID)") as? [String: Any] else { return }
      characterName = c["name"] as? String ?? "Ari"
      persona = c["persona"] as? String ?? ""
      await loadVoices()
      var assets: [String: String] = [:]
      for layer in c["layers"] as? [[String: Any]] ?? [] {
        guard let name = layer["image"] as? String, assets[name] == nil else { continue }
        var components = URLComponents(string: "http://127.0.0.1:\(port)")!
        components.path = "/characters/\(characterID)/assets/\(name)"
        var req = URLRequest(url: components.url!)
        req.setValue("Bearer \(token)", forHTTPHeaderField: "Authorization")
        let (data, response) = try await URLSession.shared.data(for: req)
        guard (response as? HTTPURLResponse)?.statusCode == 200 else { continue }
        assets[name] = "data:image/png;base64," + data.base64EncodedString()
      }
      c["assets"] = assets
      var appearance = overlayAppearanceEvent
      appearance["character"] = c
      appearance["reducedMotion"] = NSWorkspace.shared.accessibilityDisplayShouldReduceMotion
      appearance["visible"] = overlayVisible && webView?.window?.occlusionState.contains(.visible) == true
      animate(appearance)
    } catch { notice = error.localizedDescription }
  }
  func loadVoices() async {
    let character = characterID
    do {
      let result = try await request("characters/\(character)/voices") as? [String: Any]
      guard character == characterID else { return }
      voices = result?["options"] as? [[String: Any]] ?? []
      voiceID = result?["selected"] as? String ?? "character"
    } catch { notice = error.localizedDescription }
  }
  func selectVoice(_ id: String) async {
    guard !voiceBusy else { return }
    voiceBusy = true
    defer { voiceBusy = false }
    stop()
    do {
      _ = try await request(
        "characters/\(characterID)/voice", method: "PUT", body: ["voice_id": id])
      voiceID = id
    } catch { notice = error.localizedDescription }
  }
  func previewVoice() async {
    mute()
    let generation = UUID()
    previewGeneration = generation
    let path = "characters/\(characterID)/voices/\(voiceID)/audio"
    do {
      var req = URLRequest(url: URL(string: "http://127.0.0.1:\(port)/\(path)")!)
      req.setValue("Bearer \(token)", forHTTPHeaderField: "Authorization")
      let (data, response) = try await URLSession.shared.data(for: req)
      guard previewGeneration == generation, !microphoneRequested else { return }
      guard (response as? HTTPURLResponse)?.statusCode == 200 else {
        throw NSError(
          domain: "Voice", code: 1,
          userInfo: [NSLocalizedDescriptionKey: "This voice preview is unavailable."])
      }
      voicePreview = try AVAudioPlayer(data: data)
      voicePreview?.play()
    } catch { notice = error.localizedDescription }
  }
  func savePersona() async {
    do {
      _ = try await request(
        "characters/\(characterID)/persona", method: "PUT",
        body: ["name": characterName, "persona": persona])
      await refresh()
      notice = "Character updated."
    } catch { notice = error.localizedDescription }
  }
  func createCharacter(name: String, persona: String, palette: String, useVoice: Bool) async -> Bool
  {
    do {
      let value =
        try await request(
          "characters", method: "POST",
          body: [
            "name": name, "persona": persona, "palette": palette, "use_ari_voice": useVoice,
          ]) as? [String: Any]
      if let identifier = value?["id"] as? String {
        command(["type": "character", "character_id": identifier])
        await refresh()
        return true
      }
    } catch { notice = error.localizedDescription }
    return false
  }
  func saveMemory(_ text: String) async {
    do {
      _ = try await request("memories", method: "POST", body: ["text": text, "approved": true])
      await refresh()
    } catch { notice = error.localizedDescription }
  }
  func editMemory(_ id: String, text: String) async {
    do {
      _ = try await request(
        "memories/\(id)", method: "PUT", body: ["text": text, "approved": true])
      await refresh()
    } catch { notice = error.localizedDescription }
  }
  func deleteMemory(_ id: String) async {
    do {
      _ = try await request("memories/\(id)", method: "DELETE")
      await refresh()
    } catch { notice = error.localizedDescription }
  }
  func openConversation(_ id: String?) async {
    mute()
    messages = []
    if let id {
      do {
        let rows = try await request("conversations/\(id)") as? [[String: Any]] ?? []
        messages = rows.map {
          ChatMessage(
            id: $0["id"] as? String ?? UUID().uuidString, role: $0["role"] as? String ?? "user",
            text: $0["generated"] as? String ?? "",
            interrupted: ($0["status"] as? String) == "interrupted")
        }
      } catch { notice = error.localizedDescription }
    }
    await connect(conversation: id)
  }
  func deleteConversation(_ id: String) async {
    do {
      _ = try await request("conversations/\(id)", method: "DELETE")
      await refresh()
    } catch { notice = error.localizedDescription }
  }
  func exportHistory() {
    let panel = NSSavePanel()
    panel.nameFieldStringValue = "companion-history.json"
    panel.allowedContentTypes = [.json]
    guard panel.runModal() == .OK, let url = panel.url else { return }
    Task {
      do {
        let object = try await request("export")
        try JSONSerialization.data(withJSONObject: object, options: [.prettyPrinted, .sortedKeys])
          .write(to: url, options: .atomic)
      } catch { notice = error.localizedDescription }
    }
  }
  func importCharacter() {
    let panel = NSOpenPanel()
    panel.allowedContentTypes = [.zip]
    panel.allowsMultipleSelection = false
    guard panel.runModal() == .OK, let url = panel.url else { return }
    Task {
      do {
        let data = try Data(contentsOf: url)
        guard data.count <= 64 * 1024 * 1024 else {
          throw NSError(
            domain: "Import", code: 1,
            userInfo: [NSLocalizedDescriptionKey: "Character archive exceeds 64 MiB."])
        }
        let boundary = UUID().uuidString
        var body = Data(
          "--\(boundary)\r\nContent-Disposition: form-data; name=\"file\"; filename=\"character.zip\"\r\nContent-Type: application/zip\r\n\r\n"
            .utf8)
        body.append(data)
        body.append(Data("\r\n--\(boundary)--\r\n".utf8))
        var req = URLRequest(url: URL(string: "http://127.0.0.1:\(port)/characters/import")!)
        req.httpMethod = "POST"
        req.httpBody = body
        req.setValue("Bearer \(token)", forHTTPHeaderField: "Authorization")
        req.setValue(
          "multipart/form-data; boundary=\(boundary)", forHTTPHeaderField: "Content-Type")
        let (result, response) = try await URLSession.shared.data(for: req)
        let object = try JSONSerialization.jsonObject(with: result) as? [String: Any]
        guard (response as? HTTPURLResponse)?.statusCode == 200, let id = object?["id"] as? String
        else {
          throw NSError(
            domain: "Import", code: 1,
            userInfo: [
              NSLocalizedDescriptionKey: object?["detail"] as? String
                ?? "Invalid character archive."
            ])
        }
        command(["type": "character", "character_id": id])
        await refresh()
      } catch { notice = error.localizedDescription }
    }
  }
  func shutdown() {
    shuttingDown = true
    receiveTask?.cancel()
    outbound?.cancel()
    audio.close()
    voicePreview?.stop()
    socket?.cancel(with: .goingAway, reason: nil)
    pressureSource?.cancel()
    process?.terminate()
  }
}
