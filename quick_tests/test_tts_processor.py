import unittest
import queue
import time
from unittest.mock import patch, MagicMock

# Assuming TTSProcessor is in a file named tts_processor.py
# and the other necessary libraries (like torch, sounddevice) are installed.
# We'll mock the parts that require real-world hardware or external dependencies.
from tts_processor import TTSProcessor


class TestTTSProcessor(unittest.TestCase):
    """
    A test suite for the TTSProcessor class.

    Note: Due to the nature of the TTSProcessor class, which
    involves multithreading and hardware interaction (audio playback),
    these tests will focus on mocking dependencies and testing
    the logical flow, particularly the `_clean_text_for_tts` and
    `speak_stream` functions.
    """

    def setUp(self):
        """
        Set up a TTSProcessor instance for each test.
        We'll disable TTS to prevent the background thread from running
        and interfering with tests.
        """
        # Patch `sounddevice.play` and `sounddevice.wait` via the alias `sd` in `tts_processor`
        # to prevent actual audio playback during testing.
        self.sd_patch = patch('tts_processor.sd.play')
        self.sd_wait_patch = patch('tts_processor.sd.wait')
        self.mock_play = self.sd_patch.start()
        self.mock_wait = self.sd_wait_patch.start()

        # Disable TTS to avoid dependency on Kokoro TTS being installed and loaded
        with patch('tts_processor.TTS_AVAILABLE', False):
            self.tts_processor = TTSProcessor(enable_tts=False)

        # Also need a TTSProcessor with TTS enabled to test streaming logic
        with patch('tts_processor.TTS_AVAILABLE', True):
            # Mock the `_initialize_tts` method so we don't need real models
            with patch.object(TTSProcessor, '_initialize_tts', return_value=None):
                # Patch threading.Thread to prevent the worker thread from starting
                # and consuming items from the queue during tests
                with patch('tts_processor.threading.Thread'):
                    self.tts_processor_enabled = TTSProcessor(enable_tts=True)
                    # Mock the worker thread's audio generation to prevent actual audio processing
                    self.tts_processor_enabled._generate_and_play_speech = MagicMock()

    def tearDown(self):
        """Clean up patches and resources after each test."""
        self.sd_patch.stop()
        self.sd_wait_patch.stop()
        if self.tts_processor_enabled.tts_thread and self.tts_processor_enabled.tts_thread.is_alive():
            self.tts_processor_enabled.cleanup()

    def test_clean_text_for_tts(self):
        """
        Test the text cleaning method to ensure it handles various inputs
        correctly for TTS.
        """
        test_cases = [
            ("Hello, world!", "Hello, world!"),
            ("A sentence with *markdown* and _underscores_.", "A sentence with markdown and underscores."),
            ("Text with... ellipses and—dashes.", "Text with... ellipses and dashes."),
            ("  Extra   spaces   here.  ", "Extra spaces here."),
            ("[This should be gone] A normal sentence.", " A normal sentence."),
            ("A sentence with (parentheses) in it.", "A sentence with  in it.")
        ]

        for text, expected in test_cases:
            with self.subTest(text=text):
                cleaned_text = self.tts_processor._clean_text_for_tts(text)
                self.assertEqual(cleaned_text, expected)

    def test_speak_stream_disabled(self):
        """
        Test that when TTS is disabled, speak_stream just joins the text
        and prints it, without putting anything into the queue.
        """
        generator = (chunk for chunk in ["This is", " a test", " of the streaming", " function."])

        # We'll capture the stdout to verify the output
        with patch('sys.stdout', new_callable=MagicMock) as mock_stdout:
            self.tts_processor.speak_stream(generator)
            # Verify that the full message was printed
            self.assertIn("This is a test of the streaming function.", mock_stdout.write.call_args_list[0].args[0])

        # Verify that the audio queue was never used
        self.assertTrue(self.tts_processor.audio_queue.empty())

    def test_speak_stream_enabled(self):
        """
        Test the streaming logic of speak_stream.
        It should collect the full response and queue it as one item.
        """
        test_text = "Hello, this is the first sentence. And this is the rest of the text, all together now!"
        generator = (chunk for chunk in [
            "Hello, this ",
            "is the first sentence. ",
            "And this is the rest ",
            "of the text, all together now!"
        ])

        # Mock join to prevent it from blocking since no worker thread will call task_done()
        with patch.object(self.tts_processor_enabled.audio_queue, 'join', return_value=None):
            self.tts_processor_enabled.speak_stream(generator)

        # Give the worker thread a moment to process the queue, if it were real
        # Though the test mocks _generate_and_play_speech, so we just check the queue.
        time.sleep(0.1)

        # Check that the audio queue received the correct concatenated text as a single block
        expected_text = "Hello, this is the first sentence. And this is the rest of the text, all together now!"
        try:
            queued_item = self.tts_processor_enabled.audio_queue.get_nowait()
            self.assertEqual(queued_item, expected_text)
        except queue.Empty:
            self.fail("Queue is empty, expected one block of text")

        # The queue should now be empty
        self.assertTrue(self.tts_processor_enabled.audio_queue.empty())


if __name__ == '__main__':
    unittest.main()
