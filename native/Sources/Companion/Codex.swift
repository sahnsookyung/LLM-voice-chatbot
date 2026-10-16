import AppKit
import SwiftUI

struct CodexActivityView: View {
  let messages: [ChatMessage]
  @State private var expanded = false

  var body: some View {
    DisclosureGroup(isExpanded: $expanded) {
      // DisclosureGroup still evaluates its content when collapsed. Avoid
      // constructing thousands of hidden history rows on long task threads.
      if expanded {
        ScrollView {
          LazyVStack(alignment: .leading, spacing: 12) {
            ForEach(messages) { message in
              VStack(alignment: .leading, spacing: 4) {
                Text(message.role == "tool" ? "Tool activity" : "Progress update")
                  .font(.caption).foregroundStyle(.secondary)
                Text(message.text.isEmpty ? "Working…" : message.text)
                  .font(.callout).textSelection(.enabled)
              }.frame(maxWidth: .infinity, alignment: .leading)
            }
          }.padding(.top, 8)
        }.frame(height: 180).defaultScrollAnchor(.bottom)
      }
    } label: {
      Text("Task activity · \(messages.count)").font(.callout).foregroundStyle(.secondary)
    }
  }
}

struct CodexPresentationState {
  var voice = "off"
  var voiceUnavailableReason = ""
  var task = "idle"
  var threadID = ""
  var hostID = ""
  var uncertain = false
  var readOnly = false
  var requests: [String: [String: Any]] = [:]
  var historyMore = false
  var historyTruncated = false
  var tasksMore = false
  var archived = false
  var settings: [String: Any] = [:]
  var capabilities: [String: [String: Any]] = [:]
  var canSubmit: Bool { !threadID.isEmpty && !uncertain && !readOnly && !["unknown", "in_desktop", "starting"].contains(voice) }
  var canStartVoice: Bool { canSubmit && voiceUnavailableReason.isEmpty }

  mutating func apply(_ event: [String: Any]) {
    switch event["type"] as? String {
    case "codex_state":
      voice = event["voice"] as? String ?? "off"
      voiceUnavailableReason = event["voice_unavailable_reason"] as? String ?? ""
      task = event["task"] as? String ?? "idle"
      threadID = event["thread_id"] as? String ?? ""
      hostID = event["host_id"] as? String ?? ""
      uncertain = event["submission_uncertain"] as? Bool ?? true
      readOnly = event["read_only"] as? Bool ?? false
    case "codex_selected":
      threadID = (event["thread"] as? [String: Any])?["id"] as? String ?? ""
      settings = event["settings"] as? [String: Any] ?? [:]
    case "codex_settings": settings = event["settings"] as? [String: Any] ?? [:]
    case "codex_request":
      if let id = event["request_id"] as? String { requests[id] = event }
    case "codex_request_resolved":
      if let id = event["request_id"] as? String { requests.removeValue(forKey: id) }
    case "codex_requests_clear": requests.removeAll()
    case "codex_disconnected":
      voice = "off"; uncertain = true; requests.removeAll()
    case "codex_history":
      historyMore = event["more"] as? Bool ?? false
      historyTruncated = event["truncated"] as? Bool ?? false
    case "codex_catalog":
      tasksMore = event["next_cursor"] as? String != nil
      archived = event["archived"] as? Bool ?? false
    case "codex_capability":
      if let name = event["name"] as? String { capabilities[name] = event }
    default: break
    }
  }
}

extension AppModel {
  var isCodex: Bool { backend == "codex" }
  var conversationMessages: [ChatMessage] { messages.filter { !$0.isTaskActivity } }
  var codexActivity: [ChatMessage] { messages.filter(\.isTaskActivity) }
  var canSend: Bool { connected && (!isCodex || codex.canSubmit) }
  var codexAvailable: Bool { codexReadiness["available"] as? Bool == true }

  func loadBackend() async {
    do {
      let value = try await request("backend") as? [String: Any] ?? [:]
      backend = value["selected"] as? String ?? "local"
      codexValidation = (value["validation"] as? [String: Any])?["active"] as? Bool == true
      codexReadiness = value["codex"] as? [String: Any] ?? [:]
      if codexValidation {
        notice = "Validation session: only the designated test task is available. This is not a verified Codex release."
      } else if isCodex && !codexAvailable {
        notice = "Codex is unavailable. Check readiness or explicitly switch to Local."
      }
    } catch { notice = error.localizedDescription }
  }

  func checkCodex() async {
    guard !backendBusy else { return }
    backendBusy = true
    defer { backendBusy = false }
    do { codexReadiness = try await request("codex/status") as? [String: Any] ?? [:] }
    catch { notice = error.localizedDescription }
  }

  func chooseBackend(_ value: String) async {
    guard !backendBusy, value != backend else { return }
    backendBusy = true
    defer { backendBusy = false }
    do {
      _ = try await request("backend", method: "PUT", body: ["backend": value])
      // Close only Ari's voice connection. The shared host and task stay alive.
      if isCodex { endCodexVoice() } else { mute() }
      backend = value
      codex = CodexPresentationState()
      messages = []; transcript = ""; recognitionStatus = ""
      memoryGB = 0; latency = nil
      setAutomationActive?(false)
      await connect()
    } catch {
      notice = error.localizedDescription
      codexReadiness = (try? await request("codex/status") as? [String: Any]) ?? codexReadiness
    }
  }

  func codexCommand(_ action: String, _ values: [String: Any] = [:]) {
    guard isCodex else { return }
    var payload = values
    payload["action"] = action
    command(["type": "codex", "codex": payload])
  }

  func toggleCodexVoice() {
    if microphoneRequested || listening { mute(); return }
    guard codex.voiceUnavailableReason.isEmpty else { notice = codex.voiceUnavailableReason; return }
    guard canSend else { notice = "Choose a task and refresh its history before starting voice."; return }
    handsFree = true
    startMic(mode: "handsfree")
  }

  func endCodexVoice() {
    mute()
    audio.stopPlayback()
    animate(["state": "idle", "expression": "neutral", "level": 0])
    codexCommand("voice_end")
  }

  func openInCodex() {
    guard !codex.threadID.isEmpty,
      let url = URL(string: "codex://threads/" + codex.threadID.addingPercentEncoding(withAllowedCharacters: .urlPathAllowed)!) else { return }
    NSWorkspace.shared.open(url)
  }

  func newCodexConversation() {
    guard !codexValidation else { notice = "Validation is restricted to its designated test task."; return }
    let panel = NSOpenPanel()
    panel.title = "Choose this Codex task's working folder"
    panel.canChooseDirectories = true; panel.canChooseFiles = false
    panel.allowsMultipleSelection = false
    guard panel.runModal() == .OK, let url = panel.url else { return }
    codexCommand("new", ["cwd": url.path])
  }

  func handleCodex(_ event: [String: Any]) -> Bool {
    let type = event["type"] as? String ?? ""
    guard type.hasPrefix("codex_") else { return false }
    codex.apply(event)
    switch type {
    case "codex_state":
      state = "Task: \(codex.task) · Voice: \(codex.voice.replacingOccurrences(of: "_", with: " "))"
      animate(["state": codex.task == "working" ? "thinking" : codex.voice == "listening" ? "listening" : "idle"])
    case "codex_catalog":
      let rows = event["threads"] as? [[String: Any]] ?? []
      if event["append"] as? Bool == true {
        let existing = Set(codexTasks.compactMap { $0["id"] as? String })
        codexTasks += rows.filter { !existing.contains($0["id"] as? String ?? "") }
      } else { codexTasks = rows }
      codexVoices = event["voices"] as? [String] ?? []
      codexVoice = event["voice"] as? String ?? ""
    case "codex_history":
      messages = Self.mergeCodexHistory(messages, event: event)
    case "codex_item":
      if let row = event["item"] as? [String: Any], let item = Self.codexMessage(row) {
        if let i = messages.firstIndex(where: { $0.id == item.id }) { messages[i] = item }
        else { messages.append(item) }
        if messages.count > 2000 { messages.removeFirst(messages.count - 2000) }
      }
    case "codex_audio_begin":
      if let id = event["turn_id"] as? String { audio.beginTurn(id) }
    case "codex_automation": setAutomationActive?(event["active"] as? Bool ?? false)
    case "codex_disconnected", "codex_audio_failed":
      mute(); audio.stopPlayback(); connected = false; setAutomationActive?(false)
      notice = event["message"] as? String ?? "Codex disconnected. Reconnect before continuing."
    default: break
    }
    return true
  }

  static func codexMessage(_ row: [String: Any]) -> ChatMessage? {
    guard let id = row["id"] as? String else { return nil }
    return ChatMessage(id: id, role: row["role"] as? String ?? "tool", text: row["text"] as? String ?? "",
                       interrupted: row["status"] as? String == "interrupted",
                       isTaskActivity: row["presentation"] as? String == "activity" || row["role"] as? String == "tool")
  }

  static func mergeCodexHistory(_ current: [ChatMessage], event: [String: Any]) -> [ChatMessage] {
    var result = event["append"] as? Bool == true ? current : []
    let incoming = (event["messages"] as? [[String: Any]] ?? []).compactMap(codexMessage)
    var positions = Dictionary(uniqueKeysWithValues: result.enumerated().map { ($0.element.id, $0.offset) })
    for row in incoming {
      if let index = positions[row.id] { result[index] = row }
      else { positions[row.id] = result.count; result.append(row) }
    }
    return Array(result.suffix(2000))
  }
}

struct CodexSettingsView: View {
  @ObservedObject var model: AppModel
  var body: some View {
    VStack(alignment: .leading, spacing: 12) {
      Text("Conversation backend").font(.headline)
      if model.codexValidation {
        Text("Validation session · one test task only. Uses OpenAI services. Saved backend, task and voice preferences remain unchanged; reopen Ari normally to exit.").font(.callout).foregroundStyle(.orange)
      }
      Picker("Backend", selection: Binding(get: { model.backend }, set: { value in Task { await model.chooseBackend(value) } })) {
        Text("Local").tag("local"); Text("Codex").tag("codex")
      }.pickerStyle(.segmented).disabled(model.backendBusy || model.codexValidation)
      Text("Codex uses OpenAI services and account limits. Ari handles the microphone, speaker and avatar; Codex owns task history and tools.").font(.callout).foregroundStyle(.secondary)
      DisclosureGroup("Codex readiness · \(model.codexAvailable ? "verified" : "unavailable")") {
        VStack(alignment: .leading, spacing: 8) {
          ForEach(model.codexReadiness["reasons"] as? [String] ?? [], id: \.self) { Text($0).textSelection(.enabled) }
          if let fingerprint = model.codexReadiness["fingerprint"] as? [String: String] {
            Text(fingerprint["host_version"] ?? "").font(.caption)
          }
          let launcherBlocked = (model.codexReadiness["launcher"] as? [String: Any])?["available"] as? Bool == false
          Text(launcherBlocked ? "No restart needed: this launch route is incompatible with the installed desktop build." : "Finish desktop work before using the shared-host launcher. Ari never quits Codex or changes your global CLI.").font(.caption)
          Button("Check compatibility again") { Task { await model.checkCodex() } }.disabled(model.backendBusy)
          if !launcherBlocked {
            Button("Show launcher") { NSWorkspace.shared.activateFileViewerSelecting([model.root.appendingPathComponent("scripts/launch_codex.py")]) }
          }
        }.padding(.vertical, 8)
      }
      if model.isCodex {
        Picker("Voice", selection: Binding(get: { model.codexVoice }, set: { model.codexVoice = $0; model.codexCommand("voice_select", ["voice": $0]) })) {
          Text("Codex default").tag("")
          ForEach(model.codexVoices, id: \.self) { Text($0.capitalized).tag($0) }
        }.disabled(model.codex.voice != "off")
        Text("Listed voices are candidates from the host; availability is confirmed when voice starts.").font(.caption)
        Toggle("Use Control–Option–Space to start / mute voice", isOn: $model.codexShortcut).disabled(model.codexValidation)
        Text("Release does not submit a recording. End voice stops audio; Stop task interrupts work.").font(.caption)
        CodexTaskSettingsView(model: model)
        Button("Export Codex timings (no audio or text)…") { model.exportHistory(diagnostics: true) }
        Button("Reconnect and reload canonical history") { Task { await model.connect() } }
      }
    }
  }
}

struct CodexTaskSettingsView: View {
  @ObservedObject var model: AppModel
  private var models: [[String: Any]] { (model.codex.capabilities["models"]?["value"] as? [String: Any])?["data"] as? [[String: Any]] ?? [] }
  var body: some View {
    VStack(alignment: .leading, spacing: 10) {
      Text("Task settings · inherited from Codex").font(.headline)
      Picker("Model", selection: Binding(get: { model.codex.settings["model"] as? String ?? "" }, set: { model.codexCommand("settings", ["model": $0]) })) {
        ForEach(models.compactMap { $0["model"] as? String }, id: \.self) { Text($0).tag($0) }
      }.disabled(model.codex.threadID.isEmpty || model.codexValidation)
      let selected = models.first { $0["model"] as? String == model.codex.settings["model"] as? String }
      let efforts = (selected?["supportedReasoningEfforts"] as? [[String: Any]] ?? []).compactMap { $0["reasoningEffort"] as? String }
      Picker("Reasoning", selection: Binding(get: { model.codex.settings["reasoningEffort"] as? String ?? "" }, set: { model.codexCommand("settings", ["effort": $0]) })) {
        ForEach(efforts, id: \.self) { Text($0.capitalized).tag($0) }
      }.disabled(model.codex.threadID.isEmpty || model.codexValidation)
      ForEach(["account", "limits", "tools", "plugins"], id: \.self) { name in
        DisclosureGroup(name.capitalized) {
          VStack(alignment: .leading, spacing: 6) {
            ForEach(Self.capabilityLines(name, event: model.codex.capabilities[name] ?? [:]), id: \.self) { Text($0).font(.caption).textSelection(.enabled) }
          }
        }
      }
      DisclosureGroup("Effective task permissions") {
        Text(Self.pretty(model.codex.settings.filter { ["approvalPolicy", "approvalsReviewer", "activePermissionProfile", "cwd"].contains($0.key) }))
          .font(.caption.monospaced()).textSelection(.enabled)
      }
      Text("Computer Use requires the shared desktop host and its macOS permissions. Plugin availability alone is not a successful readiness test. Manage permissions and plugins in Codex.").font(.caption)
      Button("Open task in Codex") { model.openInCodex() }.disabled(model.codex.threadID.isEmpty)
    }
  }
  static func capabilityLines(_ name: String, event: [String: Any]) -> [String] {
    guard event["unavailable"] as? Bool != true, let value = event["value"] as? [String: Any] else { return ["Unavailable from this host"] }
    switch name {
    case "account":
      return [value["signed_in"] as? Bool == true ? "Signed in · " + (value["plan"] as? String ?? value["type"] as? String ?? "Codex account") : "Sign in through Codex desktop"]
    case "tools":
      let tools = value["data"] as? [[String: Any]] ?? []
      return tools.isEmpty ? ["No tool catalog returned"] : tools.map { row in
        let count = row["toolCount"] as? Int ?? (row["tools"] as? [String: Any])?.count ?? 0
        return (row["name"] as? String ?? "Tool server") + " · \(count) catalogued tools · " + (row["toolsError"] as? String ?? row["runtimeStatus"] as? String ?? "runtime status unavailable")
      }
    case "plugins":
      let plugins = (value["marketplaces"] as? [[String: Any]] ?? []).flatMap { $0["plugins"] as? [[String: Any]] ?? [] }.filter { $0["installed"] as? Bool == true }
      return plugins.isEmpty ? ["No installed plugins returned"] : plugins.map { row in
        (row["name"] as? String ?? "Plugin") + " · " + (row["enabled"] as? Bool == true ? "Enabled" : "Disabled") + ((row["disabledReason"] as? String).map { " · " + $0 } ?? "")
      }
    case "limits":
      let buckets = value["rateLimitsByLimitId"] as? [String: [String: Any]] ?? ["Codex": value["rateLimits"] as? [String: Any] ?? [:]]
      var lines: [String] = []
      for key in buckets.keys.sorted() {
        for windowName in ["primary", "secondary"] {
          guard let window = buckets[key]?[windowName] as? [String: Any], let used = window["usedPercent"] as? Double else { continue }
          var line = "\(key) · \(windowName): \(Int(max(0, min(100, 100 - used))))% remaining"
          if let stamp = window["resetsAt"] as? Double { line += " · resets " + Date(timeIntervalSince1970: stamp).formatted(date: .abbreviated, time: .shortened) }
          lines.append(line)
        }
      }
      if value["ordinaryUsageAllowed"] as? Bool == false { lines.insert("Included usage is currently unavailable", at: 0) }
      return lines.isEmpty ? ["Usage limits unavailable"] : lines
    default: return ["Unavailable"]
    }
  }
  static func pretty(_ value: Any) -> String {
    guard let data = try? JSONSerialization.data(withJSONObject: value, options: [.prettyPrinted, .sortedKeys]), let text = String(data: data, encoding: .utf8) else { return "Unavailable" }
    return text
  }
}

struct CodexTaskPicker: View {
  @ObservedObject var model: AppModel
  @State private var search = ""
  var body: some View {
    VStack(alignment: .leading, spacing: 12) {
      HStack {
        Text("Codex tasks").font(.headline)
        Button("Refresh") { model.codexCommand("refresh") }
        Button("New conversation…") { model.newCodexConversation() }.disabled(model.codexValidation)
      }
      Menu("New task in a Codex project") {
        let projects = (model.codex.capabilities["projects"]?["value"] as? [String: Any])?["data"] as? [[String: Any]] ?? []
        ForEach(projects.compactMap { $0["id"] as? String }, id: \.self) { id in
          let project = projects.first { $0["id"] as? String == id } ?? [:]
          let roots = project["roots"] as? [[String: Any]] ?? []
          ForEach(roots.compactMap { $0["path"] as? String }, id: \.self) { path in
            Button((project["name"] as? String ?? "Project") + " · " + path) {
              model.codexCommand("new", ["project_id": id, "cwd": path])
            }
          }
        }
      }
      .disabled(model.codexValidation)
      TextField("Search tasks or project folders", text: $search)
      Button(model.codex.archived ? "Show active tasks" : "Show archived tasks") { model.codexCommand("archived") }.disabled(model.codexValidation)
      ForEach(model.codexTasks.compactMap { $0["id"] as? String }, id: \.self) { id in
        let row = model.codexTasks.first { $0["id"] as? String == id } ?? [:]
        let title = row["name"] as? String ?? "Untitled task"
        let cwd = row["cwd"] as? String ?? ""
        if search.isEmpty || (title + cwd).localizedCaseInsensitiveContains(search) {
          Button { model.codexCommand("select", ["thread_id": id]) } label: {
            VStack(alignment: .leading, spacing: 4) {
              Text(title).lineLimit(2)
              Text(cwd).font(.caption).foregroundStyle(.secondary)
            }.frame(maxWidth: .infinity, alignment: .leading)
          }.buttonStyle(.borderless)
        }
      }
      if model.codex.tasksMore { Button("Load more tasks") { model.codexCommand("tasks_more") } }
    }
  }
}

struct CodexVoiceControls: View {
  @ObservedObject var model: AppModel
  var body: some View {
    VStack(alignment: .leading, spacing: 8) {
    if !model.codex.voiceUnavailableReason.isEmpty {
      Text(model.codex.voiceUnavailableReason).font(.caption).foregroundStyle(.secondary)
    }
    HStack {
      Button(model.microphoneRequested ? "Mute microphone" : !model.codex.voiceUnavailableReason.isEmpty ? "Voice unavailable" : model.codex.voice == "muted" ? "Unmute microphone" : "Start voice") { model.toggleCodexVoice() }
        .disabled((!model.connected || !model.codex.canStartVoice) && !model.microphoneRequested)
        .help("Click once to keep voice on; click Mute microphone to stop listening.")
      Button("End voice") { model.endCodexVoice() }.disabled(model.codex.voice == "off" || model.codex.voice == "in_desktop")
      if model.microphoneRequested {
        ProgressView(value: Double(model.inputLevel)).frame(width: 60).accessibilityLabel("Microphone input level")
      }
      Spacer()
      Button("Stop task") { model.stop() }.keyboardShortcut(.escape, modifiers: [])
    }.font(.callout)
    }
  }
}

struct CodexRequestCard: View {
  @ObservedObject var model: AppModel
  let id: String
  let value: [String: Any]
  @State private var answers: [String: String] = [:]
  private var params: [String: Any] { value["params"] as? [String: Any] ?? [:] }
  var body: some View {
    VStack(alignment: .leading, spacing: 10) {
      Text("Codex needs your decision").font(.headline)
      let method = value["method"] as? String ?? ""
      if method == "item/tool/requestUserInput" {
        ForEach((params["questions"] as? [[String: Any]] ?? []).compactMap { $0["id"] as? String }, id: \.self) { key in
          let question = (params["questions"] as? [[String: Any]] ?? []).first { $0["id"] as? String == key } ?? [:]
          Text(question["question"] as? String ?? key)
          TextField("Your answer", text: Binding(get: { answers[key] ?? "" }, set: { answers[key] = $0 }))
        }
        Button("Send answers") { model.codexCommand("answer", ["request_id": id, "answers": answers.mapValues { [$0] }]) }
      } else {
        Text(CodexTaskSettingsView.pretty(params)).font(.caption.monospaced()).textSelection(.enabled)
        HStack {
          Button("Allow") { model.codexCommand("approval", ["request_id": id, "decision": "accept"]) }
          Button("Deny") { model.codexCommand("approval", ["request_id": id, "decision": "decline"]) }
          Button("Open in Codex") { model.openInCodex() }
        }
      }
    }.padding(14).background(Color.orange.opacity(0.09), in: RoundedRectangle(cornerRadius: 10))
  }
}
