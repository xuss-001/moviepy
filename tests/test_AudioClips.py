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
from moviepy.config import FFMPEG_BINARY


def _write_two_track_audio_file(path, fps=22050, duration=1.0):
    """Create a container with two distinguishable mono audio streams."""
    n_frames = int(fps * duration)
    times = np.arange(n_frames) / fps

    # Distinct frequencies make the two streams easy to tell apart.
    stream_0 = (0.6 * np.sin(2 * np.pi * 440 * times)).reshape(-1, 1)
    stream_1 = (0.6 * np.sin(2 * np.pi * 880 * times)).reshape(-1, 1)

    stream_0_path = os.path.join(path, "stream_0.wav")
    stream_1_path = os.path.join(path, "stream_1.wav")
    AudioArrayClip(stream_0, fps=fps).write_audiofile(
        stream_0_path, logger=None, codec="pcm_s16le"
    )
    AudioArrayClip(stream_1, fps=fps).write_audiofile(
        stream_1_path, logger=None, codec="pcm_s16le"
    )

    two_track_path = os.path.join(path, "two_tracks.mka")
    subprocess.check_call(
        [
            FFMPEG_BINARY,
            "-y",
            "-i",
            stream_0_path,
            "-i",
            stream_1_path,
            "-map",
            "0:a:0",
            "-map",
            "1:a:0",
            "-c",
            "copy",
            two_track_path,
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    return two_track_path


def test_audiofileclip_get_frame_empty_times_keeps_channels(util):
    """An empty array of times returns 0 rows with the current channels."""
    clip = AudioFileClip("media/crunching.mp3")
    empty = clip.reader.get_frame(np.array([]))
    assert empty.shape == (0, clip.nchannels)
    clip.close()


def test_audiofileclip_get_frame_out_of_range_is_silent(util):
    """Times outside ``[0, duration)`` read as silence, even when all are out."""
    clip = AudioFileClip("media/crunching.mp3")
    reader = clip.reader

    all_out = reader.get_frame(np.array([-1.0, reader.duration, reader.duration + 5]))
    assert all_out.shape == (3, reader.nchannels)
    assert (all_out == 0).all()

    mixed = reader.get_frame(np.array([-0.5, 0.25, reader.duration]))
    assert (mixed[0] == 0).all()
    assert (mixed[-1] == 0).all()
    assert (mixed[1] != 0).any()

    assert (reader.get_frame(-0.1) == 0).all()
    assert (reader.get_frame(reader.duration) == 0).all()

    # Non-finite times are treated as out of range without raising.
    non_finite = reader.get_frame(np.array([np.nan, 0.25, np.inf]))
    assert (non_finite[0] == 0).all()
    assert (non_finite[-1] == 0).all()
    assert (non_finite[1] != 0).any()
    clip.close()


def test_audiofileclip_get_frame_scalar_and_array_use_same_rule(util):
    """Single-point and batch reads follow the same rounded sampling rule."""
    clip = AudioFileClip("media/crunching.mp3")
    reader = clip.reader

    for time in (0.0, 0.123, 0.5, 0.987):
        scalar_frame = reader.get_frame(time)
        array_frame = reader.get_frame(np.array([time]))[0]
        np.testing.assert_allclose(scalar_frame, array_frame)
    clip.close()


def test_audiofileclip_get_frame_unordered_duplicated_and_spanning(util):
    """Batch reads return frames in request order, keeping duplicates, and
    match the pointwise reads even when times are unordered and span several
    read buffers. Repeated/neighbouring requests must not restart ffmpeg per
    requested time.
    """
    filename = os.path.join(util.TMP_DIR, "audio_reader_random.wav")
    fps = 44100
    n_frames = fps * 2
    rng = np.random.default_rng(42)
    audio = rng.uniform(-0.5, 0.5, size=(n_frames, 2))
    AudioArrayClip(audio, fps=fps).write_audiofile(
        filename, logger=None, codec="pcm_s16le"
    )

    # A deliberately tiny buffer (around 45 ms) forces many recenterings.
    clip = AudioFileClip(filename, buffersize=2000)
    reader = clip.reader

    times = np.array(
        [1.5, 0.1, 0.1, 1.9, 0.75, 1.5, 0.0, 1.99, 0.333, -1.0, 2.0, 0.1]
    )
    batch = reader.get_frame(times)

    assert batch.shape == (len(times), reader.nchannels)

    pointwise = np.vstack([reader.get_frame(float(time)) for time in times])
    np.testing.assert_allclose(batch, pointwise)

    # Duplicated times must produce duplicated frames, in request order.
    np.testing.assert_allclose(batch[1], batch[2])
    np.testing.assert_allclose(batch[1], batch[-1])
    np.testing.assert_allclose(batch[0], batch[5])

    # Out-of-range positions are silent, in-range ones are actual audio.
    assert (batch[times < 0] == 0).all()
    assert (batch[times >= reader.duration] == 0).all()
    assert (batch[(times >= 0) & (times < reader.duration)] != 0).any()

    # The same query is independent of the previous read position.
    reference = batch
    for warmup_time in (0.01, 1.99, 0.5, 1.25, 0.0):
        reader.get_frame(warmup_time)
        np.testing.assert_allclose(reader.get_frame(times), reference)

    # Many adjacent/duplicated times go through one forward buffer pass:
    # ffmpeg must only be restarted a small number of times, never per time.
    restarts = [0]
    original_initialize = reader.initialize

    def counting_initialize(start_time=0):
        restarts[0] += 1
        return original_initialize(start_time)

    reader.initialize = counting_initialize
    reader.buffer_around(1)
    restarts[0] = 0

    dense_times = np.repeat(np.arange(0.0, 1.5, 0.001), 3)
    rng.shuffle(dense_times)
    reader.get_frame(dense_times)
    assert restarts[0] <= 2

    clip.close()


def test_audiofileclip_audio_stream_index_is_kept_on_seek_and_reinit(util):
    """The selected audio stream survives jumps, zero seeks and process restarts."""
    two_track_path = _write_two_track_audio_file(util.TMP_DIR)

    clips = [
        AudioFileClip(two_track_path, buffersize=2000, audio_stream_index=index)
        for index in (0, 1)
    ]
    try:
        frames_by_track = []
        for stream_index, clip in enumerate(clips):
            reader = clip.reader
            assert reader.audio_stream_index == stream_index

            early = reader.get_frame(0.1).copy()

            # Unordered jump spanning several buffers forces buffer recentering
            # and ffmpeg process restarts.
            reader.get_frame(np.array([0.8, 0.2, 0.95, 0.05]))
            after_jump = reader.get_frame(0.1).copy()
            np.testing.assert_allclose(after_jump, early)

            # Going back to zero also restarts the decoding process.
            reader.get_frame(0.0)
            after_zero = reader.get_frame(0.1).copy()
            np.testing.assert_allclose(after_zero, early)

            # An explicit decoding process restart keeps the mapped stream.
            reader.initialize(0.5)
            reader.buffer_around(1)
            after_reinitialize = reader.get_frame(0.1).copy()
            np.testing.assert_allclose(after_reinitialize, early)

            frames_by_track.append(early)

        # The two selected streams must really read different audio content.
        assert not np.allclose(frames_by_track[0], frames_by_track[1], atol=1e-3)
    finally:
        for clip in clips:
            clip.close()

    # Without an explicit index the default stream 0 is still selected.
    default_clip = AudioFileClip(two_track_path)
    try:
        assert default_clip.reader.audio_stream_index == 0
        default_frame = default_clip.reader.get_frame(0.1)
        np.testing.assert_allclose(default_frame, frames_by_track[0], atol=1e-6)
    finally:
        default_clip.close()


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


if __name__ == "__main__":
    pytest.main()
