import XCTest
@testable import Companion

final class CodexStateTests: XCTestCase {
  func testVoiceUnavailableBlocksMicrophoneButKeepsTypedTaskInput() {
    var state = CodexPresentationState()
    state.apply(["type": "codex_state", "thread_id": "task", "task": "working", "voice": "off",
                 "submission_uncertain": false, "voice_unavailable_reason": "API-key authentication required"])
    XCTAssertTrue(state.canSubmit)
    XCTAssertFalse(state.canStartVoice)
    XCTAssertEqual(state.voiceUnavailableReason, "API-key authentication required")
    state.apply(["type": "codex_state", "thread_id": "task", "voice": "off", "submission_uncertain": false])
    XCTAssertTrue(state.canStartVoice)
  }

  @MainActor
  func testActivityClassificationPreservesConversationAndChronology() {
    let events: [[String: Any]] = [
      ["id": "user", "role": "user", "text": "List files"],
      ["id": "progress", "role": "assistant", "presentation": "activity", "text": "Checking"],
      ["id": "command", "role": "tool", "text": "ls"],
      ["id": "answer", "role": "assistant", "presentation": "conversation", "text": "Found 39 entries"],
      ["id": "legacy", "role": "assistant", "text": "An older answer"],
    ]
    let rows = AppModel.mergeCodexHistory([], event: ["messages": events])
    XCTAssertEqual(rows.filter { !$0.isTaskActivity }.map(\.id), ["user", "answer", "legacy"])
    XCTAssertEqual(rows.filter(\.isTaskActivity).map(\.id), ["progress", "command"])
    XCTAssertEqual(rows.map(\.id), events.compactMap { $0["id"] as? String })
  }

  @MainActor
  func testCompletionCanReclassifyStreamedMessageWithoutDuplicatingIt() {
    var rows = AppModel.mergeCodexHistory([], event: ["messages": [
      ["id": "a", "role": "assistant", "text": "", "presentation": "activity"]
    ]])
    rows = AppModel.mergeCodexHistory(rows, event: ["append": true, "messages": [
      ["id": "a", "role": "assistant", "text": "Done", "presentation": "conversation"]
    ]])
    XCTAssertEqual(rows.count, 1)
    XCTAssertFalse(rows[0].isTaskActivity)
    XCTAssertEqual(rows[0].text, "Done")
  }

  @MainActor
  func testHistoryBatchesReplaceThenAppendWithoutDuplicates() {
    let first: [String: Any] = ["messages": [["id": "a", "role": "user", "text": "hello"]], "append": false]
    let second: [String: Any] = ["messages": [["id": "b", "role": "assistant", "text": "안녕"]], "append": true]
    var rows = AppModel.mergeCodexHistory([], event: first)
    rows = AppModel.mergeCodexHistory(rows, event: second)
    rows = AppModel.mergeCodexHistory(rows, event: second)
    XCTAssertEqual(rows.map(\.id), ["a", "b"])
    XCTAssertEqual(rows.last?.text, "안녕")
    rows = AppModel.mergeCodexHistory(rows, event: first)
    XCTAssertEqual(rows.map(\.id), ["a"])
    XCTAssertTrue(AppModel.mergeCodexHistory(rows, event: ["messages": [], "append": false]).isEmpty)
  }
  func testStopDiscardsQueuedWorkButMutePreservesTypedTaskInput() {
    let text = URLSessionWebSocketTask.Message.string(#"{"type":"text","text":"do work"}"#)
    XCTAssertTrue(SpeechSender.discardOnCancellation(text, stopTask: true))
    XCTAssertFalse(SpeechSender.discardOnCancellation(text, stopTask: false))
    let approval = URLSessionWebSocketTask.Message.string(#"{"type":"codex","codex":{"action":"approval","request_id":"7"}}"#)
    XCTAssertTrue(SpeechSender.discardOnCancellation(approval, stopTask: true))
    let ack = URLSessionWebSocketTask.Message.string(#"{"type":"ack","samples":100}"#)
    XCTAssertFalse(SpeechSender.discardOnCancellation(ack, stopTask: true))
    XCTAssertTrue(SpeechSender.discardOnCancellation(.data(Data([0, 1])), stopTask: false))
  }
  func testMutedVoiceDoesNotMeanIdleTask() {
    var state = CodexPresentationState()
    state.apply(["type": "codex_state", "thread_id": "task", "task": "working", "voice": "muted", "submission_uncertain": false])
    XCTAssertEqual(state.task, "working")
    XCTAssertEqual(state.voice, "muted")
    XCTAssertTrue(state.canSubmit)
  }
  func testDisconnectAndOwnershipBlockInputUntilReconciled() {
    var state = CodexPresentationState()
    state.apply(["type": "codex_state", "thread_id": "task", "voice": "in_desktop", "submission_uncertain": false])
    XCTAssertFalse(state.canSubmit)
    state.apply(["type": "codex_state", "thread_id": "task", "voice": "off", "submission_uncertain": false])
    XCTAssertTrue(state.canSubmit)
    state.apply(["type": "codex_disconnected"])
    XCTAssertFalse(state.canSubmit)
    state.apply(["type": "codex_state", "thread_id": "task", "voice": "off", "submission_uncertain": false, "read_only": true])
    XCTAssertFalse(state.canSubmit)
  }
  func testApprovalResolvedByOtherClientDisappears() {
    var state = CodexPresentationState()
    state.apply(["type": "codex_request", "request_id": "7", "params": ["threadId": "task"]])
    XCTAssertEqual(state.requests.count, 1)
    state.apply(["type": "codex_request_resolved", "request_id": "7"])
    state.apply(["type": "codex_request_resolved", "request_id": "7"])
    XCTAssertTrue(state.requests.isEmpty)
  }
  func testTaskCompletionClearsStaleApprovalCards() {
    var state = CodexPresentationState()
    state.apply(["type": "codex_request", "request_id": "7"])
    state.apply(["type": "codex_requests_clear"])
    XCTAssertTrue(state.requests.isEmpty)
  }
  func testLiveVoiceAndTaskStateAreIndependent() {
    var state = CodexPresentationState()
    state.apply(["type": "codex_state", "thread_id": "task", "task": "working", "voice": "off", "submission_uncertain": false])
    XCTAssertEqual(state.task, "working")
    XCTAssertTrue(state.canSubmit)
    state.apply(["type": "codex_state", "thread_id": "task", "task": "idle", "voice": "starting", "submission_uncertain": false])
    XCTAssertFalse(state.canSubmit)
  }
}
