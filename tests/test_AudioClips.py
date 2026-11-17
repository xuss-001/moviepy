"""Image sequencing clip tests meant to be run with pytest."""

import os
import subprocess as sp
import wave

import numpy as np

import pytest

from moviepy.audio.AudioClip import (
    AudioArrayClip,
    AudioClip,
    CompositeAudioClip,
    concatenate_audioclips,
)
from moviepy.audio.io.AudioFileClip import AudioFileClip
from moviepy.audio.io.readers import FFMPEG_AudioReader
from moviepy.config import FFMPEG_BINARY


def write_stereo_wav(filename, samples, fps):
    """Write stereo samples as 16-bit PCM and return their decoded values."""
    integer_samples = np.round(np.clip(samples, -1, 1) * 32767).astype("<i2")
    with wave.open(filename, "wb") as wav_file:
        wav_file.setnchannels(2)
        wav_file.setsampwidth(2)
        wav_file.setframerate(fps)
        wav_file.writeframes(integer_samples.tobytes())
    return integer_samples.astype(float) / 32768


def expected_samples(samples, times, fps, duration, n_frames):
    """Return the samples expected for arbitrary timestamps."""
    frames = np.round(fps * times).astype(int)
    in_time = (times >= 0) & (times < duration)
    in_time &= (frames >= 0) & (frames < n_frames)

    result = np.zeros((len(times), samples.shape[1]))
    result[in_time] = samples[frames[in_time]]
    return result


def test_audioclip(util, mono_wave):
    filename = os.path.join(util.TMP_DIR, "audioclip.mp3")
    audio = AudioClip(mono_wave(440), duration=2, fps=22050)
    audio.write_audiofile(filename, bitrate="16", logger=None)

    assert os.path.exists(filename)

    AudioFileClip(filename)

    # TODO Write better tests; find out why the following fail
    # assert clip.duration == 2
    # assert clip.fps == 22050
    # assert clip.reader.bitrate == 16


def test_audioclip_io(util):
    filename = os.path.join(util.TMP_DIR, "random.wav")

    # Generate a random audio clip of 4.989 seconds at 44100 Hz,
    # and save it to a file.
    input_array = np.random.random((220000, 2)) * 1.98 - 0.99
    clip = AudioArrayClip(input_array, fps=44100)
    clip.write_audiofile(filename, logger=None)
    # Load the clip.
    # The loaded clip will be slightly longer because the duration is rounded
    # up to 4.99 seconds.
    # Verify that the extra frames are all zero, and the remainder is identical
    # to the original signal.
    clip = AudioFileClip(filename)
    output_array = clip.to_soundarray()
    np.testing.assert_array_almost_equal(
        output_array[: len(input_array)], input_array, decimal=4
    )
    assert (output_array[len(input_array) :] == 0).all()


def test_audiofileclip_arbitrary_times(util):
    fps = 44100
    n_frames = 2 * fps
    rng = np.random.default_rng(1234)
    samples = rng.uniform(-0.9, 0.9, size=(n_frames, 2))
    filename = os.path.join(util.TMP_DIR, "audiofileclip_arbitrary_times.wav")
    samples = write_stereo_wav(filename, samples, fps)

    clip = AudioFileClip(filename, buffersize=32, fps=fps, nbytes=2)
    try:
        frames = np.array(
            [
                n_frames - 2,
                0,
                32,
                32,
                33,
                1000,
                1001,
                30000,
                30001,
                60000,
                60000,
                120000,
                7,
            ]
        )
        times = np.concatenate(
            [
                frames / fps,
                np.array([500.6, 500.6, 1200.4]) / fps,
                np.array([-1 / fps, clip.duration, clip.duration + 1]),
            ]
        )
        expected = expected_samples(
            samples, times, fps, clip.duration, clip.reader.n_frames
        )

        np.testing.assert_array_equal(clip.get_frame(times), expected)

        # Reading elsewhere first must not alter the result.
        clip.get_frame(np.array([0.9, 0.1]))
        np.testing.assert_array_equal(clip.get_frame(times), expected)

        point_samples = np.array([clip.get_frame(float(t)) for t in times])
        np.testing.assert_array_equal(point_samples, expected)

        empty = clip.get_frame(np.array([], dtype=float))
        assert empty.shape == (0, clip.nchannels)

        out_of_time = np.array([-1.0, clip.duration, clip.duration + 10])
        np.testing.assert_array_equal(
            clip.get_frame(out_of_time), np.zeros((3, clip.nchannels))
        )
        np.testing.assert_array_equal(clip.get_frame(-1.0), np.zeros(clip.nchannels))
        np.testing.assert_array_equal(
            clip.get_frame(clip.duration), np.zeros(clip.nchannels)
        )
    finally:
        clip.close()


def test_audiofileclip_repeated_times_reuse_buffer(util, monkeypatch):
    fps = 44100
    n_frames = 3 * fps
    rng = np.random.default_rng(5678)
    samples = rng.uniform(-0.9, 0.9, size=(n_frames, 2))
    filename = os.path.join(util.TMP_DIR, "audiofileclip_repeated_times.wav")
    write_stereo_wav(filename, samples, fps)

    initialize_calls = []
    original_initialize = FFMPEG_AudioReader.initialize

    def counting_initialize(reader, *args, **kwargs):
        initialize_calls.append((args, kwargs))
        return original_initialize(reader, *args, **kwargs)

    monkeypatch.setattr(FFMPEG_AudioReader, "initialize", counting_initialize)

    clip = AudioFileClip(filename, buffersize=128, fps=fps, nbytes=2)
    try:
        initialize_calls.clear()
        frames = np.concatenate(
            [
                np.arange(1000, 2001),
                np.repeat(np.array([12345, 12345, 12346, 12347]), 250),
            ]
        )
        clip.get_frame(frames / fps)

        # Adjacent and duplicate requests should be served from sliding buffers.
        assert len(initialize_calls) <= 2
        assert len(clip.reader.buffer) <= clip.reader.buffersize
    finally:
        clip.close()


def test_audiofileclip_audio_stream_index_with_small_buffers(util):
    fps = 44100
    n_frames = 2 * fps
    rng = np.random.default_rng(9012)
    track_samples = []
    source_filenames = []
    for index in range(2):
        samples = rng.uniform(-0.9, 0.9, size=(n_frames, 2))
        filename = os.path.join(util.TMP_DIR, f"dual_audio_source_{index}.wav")
        track_samples.append(write_stereo_wav(filename, samples, fps))
        source_filenames.append(filename)

    filename = os.path.join(util.TMP_DIR, "dual_audio_streams.mkv")
    sp.run(
        [
            FFMPEG_BINARY,
            "-y",
            "-loglevel",
            "error",
            "-i",
            source_filenames[0],
            "-i",
            source_filenames[1],
            "-map",
            "0:a",
            "-map",
            "1:a",
            "-c:a",
            "pcm_s16le",
            filename,
        ],
        check=True,
        capture_output=True,
    )

    frames = np.array(
        [
            n_frames - 2,
            0,
            64,
            64,
            65,
            5000,
            20000,
            20000,
            40000,
            7,
            60000,
        ]
    )
    times = np.concatenate(
        [
            frames / fps,
            np.array([-0.1, 2.0, 2.5]),
        ]
    )

    selected_arrays = []
    for audio_stream_index in range(2):
        clip = AudioFileClip(
            filename,
            buffersize=64,
            fps=fps,
            nbytes=2,
            audio_stream_index=audio_stream_index,
        )
        try:
            expected = expected_samples(
                track_samples[audio_stream_index],
                times,
                fps,
                clip.duration,
                clip.reader.n_frames,
            )
            np.testing.assert_array_equal(clip.get_frame(times), expected)

            # Jump around, then go back to the beginning.
            clip.get_frame(np.array([1.8, 0.2, 1.7]))
            np.testing.assert_array_equal(clip.get_frame(times), expected)

            clip.reader.initialize(0)
            np.testing.assert_array_equal(clip.get_frame(times), expected)

            reopened = AudioFileClip(
                filename,
                buffersize=64,
                fps=fps,
                nbytes=2,
                audio_stream_index=audio_stream_index,
            )
            try:
                np.testing.assert_array_equal(reopened.get_frame(times), expected)
            finally:
                reopened.close()

            selected_arrays.append(clip.get_frame(times))
            assert len(clip.reader.buffer) <= clip.reader.buffersize
        finally:
            clip.close()

    assert not np.array_equal(selected_arrays[0], selected_arrays[1])

    default_clip = AudioFileClip(filename, buffersize=64, fps=fps, nbytes=2)
    explicit_default = AudioFileClip(
        filename, buffersize=64, fps=fps, nbytes=2, audio_stream_index=0
    )
    try:
        np.testing.assert_array_equal(
            default_clip.get_frame(times), explicit_default.get_frame(times)
        )
    finally:
        default_clip.close()
        explicit_default.close()


def test_concatenate_audioclips_render(util, mono_wave):
    """Concatenated AudioClips through ``concatenate_audioclips`` should return
    a clip that can be rendered to a file.
    """
    filename = os.path.join(util.TMP_DIR, "concatenate_audioclips.mp3")

    clip_440 = AudioClip(mono_wave(440), duration=0.01, fps=44100)
    clip_880 = AudioClip(mono_wave(880), duration=0.000001, fps=22050)

    concat_clip = concatenate_audioclips((clip_440, clip_880))
    concat_clip.write_audiofile(filename, logger=None)

    assert concat_clip.duration == clip_440.duration + clip_880.duration


def test_concatenate_audioclips_CompositeAudioClip():
    """Concatenated AudioClips through ``concatenate_audioclips`` should return
    a CompositeAudioClip whose attributes should be consistent:

    - Returns CompositeAudioClip.
    - Their fps is taken from the maximum of their audios.
    - Audios are placed one after other:
      - Duration is the sum of their durations.
      - Ends are the accumulated sum of their durations.
      - Starts are the accumulated sum of their durations, but first start is 0
      and latest is ignored.
    - Channels are the max channels of their clips.
    """
    frequencies = [440, 880, 1760]
    durations = [2, 5, 1]
    fpss = [44100, 22050, 11025]

    clips = [
        AudioClip(
            lambda t: [np.sin(frequency * 2 * np.pi * t)], duration=duration, fps=fps
        )
        for frequency, duration, fps in zip(frequencies, durations, fpss)
    ]

    concat_clip = concatenate_audioclips(clips)

    # should return a CompositeAudioClip
    assert isinstance(concat_clip, CompositeAudioClip)

    # fps of the greatest fps passed into it
    assert concat_clip.fps == 44100

    # audios placed on after other
    assert concat_clip.duration == sum(durations)
    assert list(concat_clip.ends) == list(np.cumsum(durations))
    assert list(concat_clip.starts), list(np.cumsum([0, *durations[:-1]]))

    # channels are maximum number of channels of the clips
    assert concat_clip.nchannels == max(clip.nchannels for clip in clips)


def test_CompositeAudioClip_by__init__():
    """The difference between the CompositeAudioClip returned by
    ``concatenate_audioclips`` and a CompositeAudioClip created using the class
    directly, is that audios in ``concatenate_audioclips`` are played one after
    other and AudioClips passed to CompositeAudioClip can be played at different
    times, it depends on their ``start`` attributes.
    """
    frequencies = [440, 880, 1760]
    durations = [2, 5, 1]
    fpss = [44100, 22050, 11025]
    starts = [0, 1, 2]

    clips = [
        AudioClip(
            lambda t: [np.sin(frequency * 2 * np.pi * t)], duration=duration, fps=fps
        ).with_start(start)
        for frequency, duration, fps, start in zip(frequencies, durations, fpss, starts)
    ]

    compound_clip = CompositeAudioClip(clips)

    # should return a CompositeAudioClip
    assert isinstance(compound_clip, CompositeAudioClip)

    # fps of the greatest fps passed into it
    assert compound_clip.fps == 44100

    # duration depends on clips starts and durations
    ends = [start + duration for start, duration in zip(starts, durations)]
    assert compound_clip.duration == max(ends)
    assert list(compound_clip.ends) == ends
    assert list(compound_clip.starts) == starts

    # channels are maximum number of channels of the clips
    assert compound_clip.nchannels == max(clip.nchannels for clip in clips)


def test_concatenate_audioclip_with_audiofileclip(util, stereo_wave):
    clip1 = AudioClip(
        stereo_wave(left_freq=440, right_freq=880),
        duration=1,
        fps=44100,
    )
    clip2 = AudioFileClip("media/crunching.mp3")

    concat_clip = concatenate_audioclips((clip1, clip2))
    concat_clip.write_audiofile(
        os.path.join(util.TMP_DIR, "concat_clip_with_file_audio.mp3"),
        logger=None,
    )

    assert concat_clip.duration == clip1.duration + clip2.duration


def test_concatenate_audiofileclips(util):
    clip1 = AudioFileClip("media/crunching.mp3").subclipped(1, 4)

    # Checks it works with videos as well
    clip2 = AudioFileClip("media/big_buck_bunny_432_433.webm")
    concat_clip = concatenate_audioclips((clip1, clip2))

    concat_clip.write_audiofile(
        os.path.join(util.TMP_DIR, "concat_audio_file.mp3"),
        logger=None,
    )

    assert concat_clip.duration == clip1.duration + clip2.duration


def test_audioclip_mono_max_volume(mono_wave):
    clip = AudioClip(mono_wave(440), duration=1, fps=44100)
    max_volume = clip.max_volume()
    assert isinstance(max_volume, float)
    assert max_volume > 0


@pytest.mark.parametrize(("nchannels"), (2, 4, 8, 16))
@pytest.mark.parametrize(("channel_muted"), ("left", "right"))
def test_audioclip_stereo_max_volume(nchannels, channel_muted):
    def frame_function(t):
        frame = []
        # build channels (one of each pair muted)
        for i in range(int(nchannels / 2)):
            if channel_muted == "left":
                # if muted channel is left, [0, sound, 0, sound...]
                frame.append(np.sin(t * 0))
                frame.append(np.sin(440 * 2 * np.pi * t))
            else:
                # if muted channel is right, [sound, 0, sound, 0...]
                frame.append(np.sin(440 * 2 * np.pi * t))
                frame.append(np.sin(t * 0))
        return np.array(frame).T

    clip = AudioClip(frame_function, fps=44100, duration=1)
    max_volume = clip.max_volume(stereo=True)
    # if `stereo == True`, `AudioClip.max_volume` returns a Numpy array`
    assert isinstance(max_volume, np.ndarray)
    assert len(max_volume) == nchannels

    # check channels muted and with sound
    for i, channel_max_volume in enumerate(max_volume):
        if i % 2 == 0:
            if channel_muted == "left":
                assert channel_max_volume == 0
            else:
                assert channel_max_volume > 0
        else:
            if channel_muted == "right":
                assert channel_max_volume == 0
            else:
                assert channel_max_volume > 0


if __name__ == "__main__":
    pytest.main()
