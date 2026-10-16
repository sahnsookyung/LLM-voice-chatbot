import XCTest
import AVFoundation
@testable import Companion

final class CapturePipeTests: XCTestCase {
  func testFramingPreservesCaptureSequenceAndSampleOffsets() {
    let id = UUID()
    var framing = MicrophoneFraming(capture: id)
    let first = framing.packet(Data(repeating: 7, count: 2048))
    let second = framing.packet(Data(repeating: 9, count: 128))
    XCTAssertEqual(String(data: first.prefix(4), encoding: .utf8), "LVM2")
    XCTAssertEqual(first.suffix(2048), Data(repeating: 7, count: 2048))
    XCTAssertEqual(second.withUnsafeBytes { $0.loadUnaligned(fromByteOffset: 20, as: UInt32.self) }, 1)
    XCTAssertEqual(second.withUnsafeBytes { $0.loadUnaligned(fromByteOffset: 24, as: UInt64.self) }, 1024)
    XCTAssertEqual(framing.offset, 1088)
    var next = MicrophoneFraming()
    XCTAssertNotEqual(next.packet(Data(count: 2))[4..<20], first[4..<20])
  }

  func testConverterDrainPreservesInitialAndFinalAudioAcrossPacketSizes() async throws {
    try await MainActor.run {
      let source = AVAudioFormat(standardFormatWithSampleRate: 48000, channels: 2)!
      @MainActor func record(chunk: Int) throws -> Data {
        let converter = try AudioController.inputConverter(from: source)
        let pipe = CapturePipe(converter)
        var result = Data()
        for start in stride(from: 0, to: 48003, by: chunk) {
          let count = min(chunk, 48003 - start)
          let buffer = AVAudioPCMBuffer(pcmFormat: source, frameCapacity: AVAudioFrameCount(count))!
          buffer.frameLength = AVAudioFrameCount(count)
          for i in 0..<count {
            buffer.floatChannelData![0][i] = Float(sin(Double(start+i) * 0.03)) * 0.25
            buffer.floatChannelData![1][i] = 0 // Preserve the microphone-channel correction.
          }
          pipe.accept(buffer)
          let (data, failed) = pipe.drain()
          XCTAssertFalse(failed); result.append(data)
        }
        let (tail, failed) = pipe.drain(finish: true)
        XCTAssertFalse(failed); result.append(tail)
        XCTAssertTrue(pipe.drain(finish: true).0.isEmpty)
        let late = AVAudioPCMBuffer(pcmFormat: source, frameCapacity: 1024)!
        late.frameLength = 1024; pipe.accept(late)
        XCTAssertTrue(pipe.drain().0.isEmpty)
        return result
      }
      let one = try record(chunk: 48003)
      let fragmented = try record(chunk: 1024)
      XCTAssertEqual(one, fragmented)
      XCTAssertEqual(one.count / 2, 16001)
      XCTAssertNotEqual(one, Data(count: one.count))
    }
  }

  func testSettingsRoundTripIncludesIndependentLanguages() throws {
    var preferences = SpeechSettings()
    preferences.input_language = "auto"
    preferences.reply_language = "korean"
    preferences.vocabulary = ["Ari", "Codex", "네모"]
    preferences.context_ms = 560
    let data = try JSONEncoder().encode(preferences)
    XCTAssertEqual(try JSONDecoder().decode(SpeechSettings.self, from: data), preferences)
  }
}
