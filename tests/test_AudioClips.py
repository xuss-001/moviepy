"""Image sequencing clip tests meant to be run with pytest."""

import os
import subprocess

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
from moviepy.tools import ffmpeg_escape_filename


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


def _make_tone_wav(util, frequency, fps=44100, duration=1.0):
    """Create a mono sine wave wav file and return its path."""
    filename = os.path.join(util.TMP_DIR, "tone_%dhz.wav" % frequency)

    def frame_function(t, frequency=frequency):
        return np.sin(2 * np.pi * frequency * np.asarray(t, dtype=float))

    AudioClip(frame_function, duration=duration, fps=fps).write_audiofile(
        filename, logger=None, codec="pcm_s16le"
    )
    return filename


def _dominant_frequency(signal, fps):
    """Return the dominant non-zero frequency of a 1D signal."""
    spectrum = np.abs(np.fft.rfft(signal))
    frequencies = np.fft.rfftfreq(len(signal), 1.0 / fps)
    return frequencies[np.argmax(spectrum[1:]) + 1]


def test_audio_reader_empty_time_array(util):
    """An empty time array returns zero rows but keeps the channel count."""
    reader = FFMPEG_AudioReader("media/crunching.mp3", buffersize=1000)
    try:
        frames = reader.get_frame(np.array([]))
        assert frames.shape == (0, reader.nchannels)
    finally:
        reader.close()


def test_audio_reader_out_of_time_returns_silence(util):
    """Negative and beyond-duration requests return silence, never an error."""
    filename = _make_tone_wav(util, 440)
    reader = FFMPEG_AudioReader(filename, buffersize=1000)
    try:
        duration = reader.duration

        # all requests out of range: the result is fully silent
        frames = reader.get_frame(np.array([-1.0, -0.001, duration, duration + 2]))
        assert frames.shape == (4, reader.nchannels)
        assert (frames == 0).all()

        # mixed in-range and out-of-range requests keep their request order
        times = np.array([-0.5, 0.25, duration, 0.75, duration + 1.0])
        mixed = reader.get_frame(times)
        assert mixed.shape == (5, reader.nchannels)
        assert (mixed[[0, 2, 4]] == 0).all()
        # in-range rows match point-by-point reads (0.25s and 0.75s happen
        # to be exact zero crossings of the 440 Hz sine)
        np.testing.assert_array_equal(
            mixed[[1, 3]],
            np.stack([reader.get_frame(0.25), reader.get_frame(0.75)]),
        )

        # scalar requests are silent too, as a single 1D frame
        assert reader.get_frame(-1.0).shape == (reader.nchannels,)
        assert (reader.get_frame(-1.0) == 0).all()
        assert (reader.get_frame(duration) == 0).all()
    finally:
        reader.close()


def test_audio_reader_shuffled_duplicated_times_match_single_reads(util):
    """Batch reads follow the request order, keep duplicates and match the
    point-by-point reads regardless of the previous reading position.
    """
    filename = _make_tone_wav(util, 440)
    small_buffer_reader = FFMPEG_AudioReader(filename, buffersize=120)
    reference_reader = FFMPEG_AudioReader(filename, buffersize=120)
    try:
        duration = small_buffer_reader.duration
        times = np.array(
            [0.8, 0.1, 0.55, 0.1, 0.9, 0.8, 0.0, 0.35, 0.55, duration]
        )

        batch_frames = small_buffer_reader.get_frame(times)
        single_frames = np.stack(
            [reference_reader.get_frame(float(t)) for t in times]
        )
        np.testing.assert_array_equal(batch_frames, single_frames)

        # duplicates are preserved row for row
        repeated_times = np.repeat(np.linspace(0.2, 0.3, 10), 7)
        repeated_frames = small_buffer_reader.get_frame(repeated_times)
        reference_frames = np.stack(
            [reference_reader.get_frame(float(t)) for t in repeated_times]
        )
        np.testing.assert_array_equal(repeated_frames, reference_frames)

        # jumping back and forth does not depend on the previous position
        for _ in range(3):
            jumping_times = np.array([0.95, 0.02, 0.7, 0.02, 0.95, 0.4])
            jumping_frames = small_buffer_reader.get_frame(jumping_times)
            reference_jumping = np.stack(
                [reference_reader.get_frame(float(t)) for t in jumping_times]
            )
            np.testing.assert_array_equal(jumping_frames, reference_jumping)
    finally:
        small_buffer_reader.close()
        reference_reader.close()


def test_audio_reader_scalar_and_array_share_rounding_rule(util):
    """Single-point and batch sampling use the same sample-point rule."""
    filename = _make_tone_wav(util, 440)
    reader = FFMPEG_AudioReader(filename, buffersize=200)
    try:
        # half a sample at 44100 fps rounds to the next sample in both paths
        half_sample_time = 0.5 / reader.fps
        scalar_frame = reader.get_frame(half_sample_time)
        array_frame = reader.get_frame(np.array([half_sample_time]))[0]
        np.testing.assert_array_equal(scalar_frame, array_frame)

        # the last valid sample is one frame before the file duration
        last_time = (reader.n_frames - 1) / reader.fps
        assert (reader.get_frame(np.array([last_time])) != 0).any()
    finally:
        reader.close()


def test_audio_reader_batches_do_not_restart_ffmpeg_per_time(util):
    """Dense or repeated requests reuse the streaming buffer."""
    filename = _make_tone_wav(util, 440)
    reader = FFMPEG_AudioReader(filename, buffersize=200)
    try:
        restarts = 0
        original_initialize = reader.initialize

        def counting_initialize(*args, **kwargs):
            nonlocal restarts
            restarts += 1
            return original_initialize(*args, **kwargs)

        reader.initialize = counting_initialize

        # 500 requests with many duplicates, all within one buffer window
        times = np.repeat(np.linspace(0.2, 0.202, 100), 5)
        frames = reader.get_frame(times)
        assert frames.shape == (500, reader.nchannels)
        assert restarts == 0

        # adjacent forward reads stream new chunks without reopening ffmpeg
        reader.get_frame(np.linspace(0.3, 0.35, 200))
        assert restarts == 0
    finally:
        reader.close()


def test_audio_reader_audio_stream_index_is_stable(util):
    """The selected audio track survives jumps, zero seeks and process restarts."""
    fps = 44100
    track_440 = _make_tone_wav(util, 440, fps=fps)
    track_880 = _make_tone_wav(util, 880, fps=fps)
    dual_track_filepath = os.path.join(util.TMP_DIR, "dual_audio_stream.mkv")

    cmd = [
        FFMPEG_BINARY,
        "-y",
        "-i",
        ffmpeg_escape_filename(track_440),
        "-i",
        ffmpeg_escape_filename(track_880),
        "-map",
        "0:a:0",
        "-map",
        "1:a:0",
        "-c",
        "copy",
        ffmpeg_escape_filename(dual_track_filepath),
    ]
    with open(os.devnull, "w") as stderr:
        subprocess.check_call(cmd, stderr=stderr)

    # a small buffer forces frequent buffer recenters while jumping around
    reader = FFMPEG_AudioReader(
        dual_track_filepath, buffersize=80, fps=fps, audio_stream_index=1
    )
    probe_times = np.linspace(0.3, 0.4, 4410, endpoint=False)
    try:
        assert reader.audio_stream_index == 1

        shuffled_times = np.array([0.9, 0.05, 0.6, 0.05, 0.9, 0.3])
        reader.get_frame(shuffled_times)
        assert _dominant_frequency(reader.get_frame(probe_times)[:, 0], fps) == 880

        # going back to zero triggers a fresh ffmpeg process
        reader.seek(0)
        reader.buffer_around(1)
        assert _dominant_frequency(reader.get_frame(probe_times)[:, 0], fps) == 880

        # explicitly reopening the decoder must still map the second track
        reader.initialize(0.0)
        reader.buffer_around(1)
        assert _dominant_frequency(reader.get_frame(probe_times)[:, 0], fps) == 880

        reader.initialize(0.5)
        reader.buffer_around(int(0.35 * fps))
        assert _dominant_frequency(reader.get_frame(probe_times)[:, 0], fps) == 880
    finally:
        reader.close()

    # without an explicit index, the default keeps selecting track 0
    default_reader = FFMPEG_AudioReader(
        dual_track_filepath, buffersize=80, fps=fps
    )
    try:
        assert default_reader.audio_stream_index == 0
        default_reader.seek(0)
        default_reader.buffer_around(1)
        assert (
            _dominant_frequency(default_reader.get_frame(probe_times)[:, 0], fps)
            == 440
        )
    finally:
        default_reader.close()


if __name__ == "__main__":
    pytest.main()
