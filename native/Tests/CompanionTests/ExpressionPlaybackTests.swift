import XCTest
@testable import Companion

final class ExpressionPlaybackTests: XCTestCase {
  func testSpokenGesturesWaitForPlaybackAndNeverSurviveACancelledTurn() async {
    await MainActor.run {
      let audio = AudioController()
      defer { audio.close() }
      var applied: [String] = []
      audio.onGesture = { applied.append($0) }
      audio.beginTurn("hello")
      audio.applyGesture(turn: "hello", gesture: "wave", waitForAudio: true)
      XCTAssertEqual(applied, ["cancel"])
      audio.playbackStarted(turn: "stale")
      XCTAssertEqual(applied, ["cancel"])
      audio.playbackStarted(turn: "hello")
      audio.playbackStarted(turn: "hello")
      XCTAssertEqual(applied, ["cancel", "wave"])
      audio.beginTurn("cancelled")
      audio.applyGesture(turn: "cancelled", gesture: "explain", waitForAudio: true)
      audio.beginTurn("text-only")
      audio.playbackStarted(turn: "cancelled")
      audio.playbackStarted(turn: "text-only")
      audio.applyGesture(turn: "text-only", gesture: "nod")
      XCTAssertEqual(applied, ["cancel", "wave", "cancel", "cancel", "nod"])
    }
  }
  func testGesturesAreTurnScopedAndCancelledOnInterruption() async {
    await MainActor.run {
      let audio = AudioController()
      defer { audio.close() }
      var applied: [String] = []
      audio.onGesture = { applied.append($0) }
      audio.beginTurn("hello")
      audio.applyGesture(turn: "hello", gesture: "wave")
      audio.beginTurn("explanation")
      audio.applyGesture(turn: "hello", gesture: "wave")
      audio.applyGesture(turn: "explanation", gesture: "explain")
      audio.applyGesture(turn: "explanation", gesture: "invalid")
      audio.applyGesture(turn: "explanation", gesture: "dance")
      audio.stopPlayback()
      audio.applyGesture(turn: "explanation", gesture: "explain")
      XCTAssertEqual(applied, ["cancel", "wave", "cancel", "explain", "cancel"])
    }
  }
  func testExplicitAvatarActionsWaitForSpeechAndCancelLikeConversationalActions() async {
    await MainActor.run {
      let audio = AudioController()
      defer { audio.close() }
      var applied: [String] = []
      audio.onGesture = { applied.append($0) }
      for gesture in ["twirl", "raise-arms", "point"] {
        applied = []
        audio.beginTurn(gesture)
        audio.applyGesture(turn: gesture, gesture: gesture, waitForAudio: true)
        XCTAssertEqual(applied, ["cancel"])
        audio.playbackStarted(turn: gesture)
        audio.playbackStarted(turn: gesture)
        XCTAssertEqual(applied, ["cancel", gesture])
        audio.beginTurn("next")
        audio.applyGesture(turn: gesture, gesture: gesture)
        XCTAssertEqual(applied, ["cancel", gesture, "cancel"])
      }
      audio.applyGesture(turn: "next", gesture: "laugh")
      audio.applyGesture(turn: "next", gesture: "dance")
      XCTAssertEqual(applied, ["cancel", "point", "cancel"])
    }
  }
  func testNewTurnClearsPreviousMoodAndRejectsStaleExpressionEvents() async {
    await MainActor.run {
      let audio = AudioController()
      defer { audio.close() }
      var applied: [String] = []
      audio.onExpression = { applied.append($0) }
      audio.beginTurn("celebration")
      audio.queueExpression(turn: "celebration", offset: 0, expression: "happy")
      audio.beginTurn("sad-story")
      audio.queueExpression(turn: "celebration", offset: 0, expression: "happy")
      audio.queueExpression(turn: "sad-story", offset: 0, expression: "concerned")
      audio.stopPlayback()
      audio.queueExpression(turn: "sad-story", offset: 0, expression: "happy")
      XCTAssertEqual(applied, ["neutral", "happy", "neutral", "concerned"])
    }
  }
}
