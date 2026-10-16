import XCTest
@testable import Companion

final class SpeechPlaybackTimelineTests: XCTestCase {
  private func schedule(
    _ timeline: inout SpeechPlaybackTimeline, _ samples: [Float],
    offset: UInt64 = 0, playerSample: Int64? = nil
  ) -> Int64 {
    samples.withUnsafeBufferPointer {
      timeline.schedule(samples: $0, offset: offset, playerSample: playerSample)
    }
  }

  func testMouthTracksChangingAmplitudeAndClosesExactlyAtBufferEnd() {
    var timeline = SpeechPlaybackTimeline()
    let samples = [Float](repeating: 0.1, count: 800)
      + [Float](repeating: 0, count: 800) + [Float](repeating: 0.04, count: 123)
    XCTAssertEqual(schedule(&timeline, samples), 0)
    XCTAssertNil(timeline.position(at: -1))
    XCTAssertEqual(timeline.position(at: 400)!.level, 0.5, accuracy: 0.001)
    XCTAssertEqual(timeline.position(at: 800)!.level, 0)
    XCTAssertEqual(timeline.position(at: 1722)!.level, 0.2, accuracy: 0.001)
    XCTAssertNil(timeline.position(at: 1723))
  }

  func testBufferedPacketsStayContiguousAndRetainSpeechOffsets() {
    var timeline = SpeechPlaybackTimeline()
    let samples = [Float](repeating: 0.1, count: 7680)
    XCTAssertEqual(schedule(&timeline, samples), 0)
    XCTAssertEqual(schedule(&timeline, samples, offset: 7680, playerSample: 1000), 7680)
    XCTAssertEqual(timeline.position(at: 7679)!.offset, 7679)
    XCTAssertEqual(timeline.position(at: 7680)!.offset, 7680)
    XCTAssertEqual(timeline.position(at: 15359)!.offset, 15359)
    XCTAssertNil(timeline.position(at: 15360))
  }

  func testRepeatedSynthesisGapsDoNotSkipMouthFramesOrAdvanceExpressions() {
    var timeline = SpeechPlaybackTimeline()
    let samples = [Float](repeating: 0.1, count: 4800)
    _ = schedule(&timeline, samples)
    XCTAssertNil(timeline.position(at: 24000))
    for phrase in 1...4 {
      let now = Int64(phrase * 48000)
      let offset = UInt64(phrase * 4800)
      let start = schedule(&timeline, samples, offset: offset, playerSample: now)
      XCTAssertGreaterThan(start, now)
      XCTAssertNil(timeline.position(at: now))
      XCTAssertNil(timeline.position(at: start - 1))
      XCTAssertEqual(timeline.position(at: start)!.offset, offset)
      XCTAssertEqual(timeline.position(at: start + 2400)!.level, 0.5, accuracy: 0.001)
      XCTAssertEqual(timeline.position(at: start + 4799)!.offset, offset + 4799)
      XCTAssertNil(timeline.position(at: start + 4800))
    }
  }

  func testLatePacketHasSchedulingHeadroomAndNoPrematureMouthMovement() {
    var timeline = SpeechPlaybackTimeline()
    let samples = [Float](repeating: 0.1, count: 800)
    _ = schedule(&timeline, samples)
    let start = schedule(&timeline, samples, offset: 800, playerSample: 799)
    XCTAssertGreaterThan(start, 800)
    XCTAssertEqual(timeline.position(at: 799)!.offset, 799)
    XCTAssertNil(timeline.position(at: 800))
    XCTAssertNil(timeline.position(at: start - 1))
    XCTAssertEqual(timeline.position(at: start)!.offset, 800)
  }
}
