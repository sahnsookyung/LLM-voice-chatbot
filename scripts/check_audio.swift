// Compile with native/Sources/Companion/Audio.swift; no microphone is opened.
@preconcurrency import AVFoundation

@main @MainActor struct AudioConversionCheck {
  static func main() throws {
    for sampleRate in [44100.0, 48000.0, 96000.0] {
      let mono = try convert(sampleRate: sampleRate, channels: 1)
      precondition(mono.count > 3000 && mono.contains { abs(Int($0)) > 1000 })
      for channels: AVAudioChannelCount in [2, 3, 7, 9] {
        let multichannel = try convert(sampleRate: sampleRate, channels: channels)
        precondition(
          multichannel == mono, "Microphone channel was lost or mixed with reference audio")
      }
      print("PASS: \(Int(sampleRate)) Hz, 1/2/3/7/9 channels → audible, identical 16 kHz mono PCM")
    }
  }

  static func convert(sampleRate: Double, channels: AVAudioChannelCount) throws -> [Int16] {
    let layout = AVAudioChannelLayout(layoutTag: kAudioChannelLayoutTag_DiscreteInOrder | channels)!
    let source = AVAudioFormat(standardFormatWithSampleRate: sampleRate, channelLayout: layout)
    let converter = try AudioController.inputConverter(from: source)
    precondition(converter.outputFormat.sampleRate == 16000)
    precondition(converter.outputFormat.channelCount == 1)
    precondition(converter.outputFormat.commonFormat == .pcmFormatInt16)
    var result: [Int16] = []
    for chunk in 0..<20 {
      let input = AVAudioPCMBuffer(pcmFormat: source, frameCapacity: 1024)!
      input.frameLength = 1024
      for channel in 0..<Int(channels) {
        for frame in 0..<1024 {
          // Reference channels intentionally differ from the microphone signal.
          input.floatChannelData![channel][frame] =
            channel == 0
            ? Float(sin(Double(chunk * 1024 + frame) * 0.04) * 0.2) : -0.6
        }
      }
      let capacity = AVAudioFrameCount(1024 * 16000 / sampleRate + 16)
      let output = AVAudioPCMBuffer(pcmFormat: converter.outputFormat, frameCapacity: capacity)!
      var supplied = false
      var error: NSError?
      let status = converter.convert(to: output, error: &error) { _, status in
        if supplied {
          status.pointee = .noDataNow
          return nil
        }
        supplied = true
        status.pointee = .haveData
        return input
      }
      if let error { throw error }
      precondition(status != .error)
      result.append(
        contentsOf: UnsafeBufferPointer(
          start: output.int16ChannelData![0], count: Int(output.frameLength)))
    }
    return result
  }
}
