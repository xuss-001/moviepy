"""Regression tests for resource cleanup in the audio/video export chain.

Whatever happens during an export (success, failure while initializing,
generating frames or writing them), the ffmpeg processes, pipes and logfile
handles created for this export must be released deterministically, and
temporary audio files must be removed or kept according to ``remove_temp``.
"""

import os

import numpy as np

import pytest

from moviepy import AudioClip, ColorClip, VideoFileClip
from moviepy.audio.io import ffmpeg_audiowriter
from moviepy.audio.io.ffmpeg_audiowriter import FFMPEG_AudioWriter
from moviepy.video.io import ffmpeg_writer
from moviepy.video.io.ffmpeg_writer import FFMPEG_VideoWriter


def make_mono_clip(duration=2.0, fps=44100):
    return AudioClip(
        lambda t: np.sin(440 * 2 * np.pi * t), duration=duration, fps=fps
    )


def make_video_clip(duration=0.5, fps=10, with_audio=True):
    clip = ColorClip((128, 128), color=(255, 0, 0)).with_duration(duration)
    if with_audio:
        clip = clip.with_audio(make_mono_clip(duration=duration))
    return clip.with_fps(fps)


def count_open_fds():
    """Number of open file descriptors of this process, None if unsupported."""
    if not os.path.isdir("/proc/self/fd"):
        return None
    return len(os.listdir("/proc/self/fd"))


@pytest.fixture
def audio_writer_instances(monkeypatch):
    """Patches FFMPEG_AudioWriter so created instances can be inspected."""
    instances = []
    real_class = ffmpeg_audiowriter.FFMPEG_AudioWriter

    class SpyAudioWriter(real_class):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            instances.append(self)

    monkeypatch.setattr(ffmpeg_audiowriter, "FFMPEG_AudioWriter", SpyAudioWriter)
    return instances


@pytest.fixture
def video_writer_instances(monkeypatch):
    """Patches FFMPEG_VideoWriter so created instances can be inspected."""
    instances = []
    real_class = ffmpeg_writer.FFMPEG_VideoWriter

    class SpyVideoWriter(real_class):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            instances.append(self)

    monkeypatch.setattr(ffmpeg_writer, "FFMPEG_VideoWriter", SpyVideoWriter)
    return instances


def test_audio_writer_close_is_idempotent(util):
    filename = os.path.join(util.TMP_DIR, "close_idempotent.wav")
    writer = FFMPEG_AudioWriter(filename, 44100, 2, 2, codec="pcm_s16le")
    writer.close()
    assert writer.proc is None
    # Closing again must be a safe no-op.
    writer.close()
    assert writer.proc is None


def test_video_writer_close_is_idempotent(util):
    filename = os.path.join(util.TMP_DIR, "close_idempotent.mp4")
    writer = FFMPEG_VideoWriter(filename, (16, 16), 10)
    writer.close()
    assert writer.proc is None
    writer.close()
    assert writer.proc is None


def test_audio_writer_close_with_already_exited_process(util):
    filename = os.path.join(util.TMP_DIR, "close_dead_proc.wav")
    writer = FFMPEG_AudioWriter(filename, 44100, 2, 2, codec="pcm_s16le")
    writer.proc.kill()
    writer.proc.wait()
    # The process is already gone: close() must still release the pipes.
    writer.close()
    assert writer.proc is None


def test_video_writer_close_with_already_exited_process(util):
    filename = os.path.join(util.TMP_DIR, "close_dead_proc.mp4")
    writer = FFMPEG_VideoWriter(filename, (16, 16), 10)
    writer.proc.kill()
    writer.proc.wait()
    writer.close()
    assert writer.proc is None


def test_writers_do_not_close_external_logfile(util):
    logfile = open(os.path.join(util.TMP_DIR, "external.log"), "w+")
    try:
        audio_writer = FFMPEG_AudioWriter(
            os.path.join(util.TMP_DIR, "external_log.wav"),
            44100,
            2,
            2,
            codec="pcm_s16le",
            logfile=logfile,
        )
        audio_writer.close()
        assert not logfile.closed

        video_writer = FFMPEG_VideoWriter(
            os.path.join(util.TMP_DIR, "external_log.mp4"),
            (16, 16),
            10,
            logfile=logfile,
        )
        video_writer.close()
        assert not logfile.closed
    finally:
        logfile.close()


@pytest.mark.parametrize(
    "write_logfile", (False, True), ids=("write_logfile=False", "write_logfile=True")
)
def test_write_audiofile_failure_releases_resources(
    util, audio_writer_instances, write_logfile
):
    filename = os.path.join(util.TMP_DIR, "failing_audio.wav")
    fds_before = count_open_fds()

    with pytest.raises(IOError) as excinfo:
        make_mono_clip().write_audiofile(
            filename,
            codec="codec_that_does_not_exist",
            write_logfile=write_logfile,
            logger=None,
        )

    # The original ffmpeg error must be propagated.
    assert "MoviePy error: FFMPEG encountered the following error" in str(
        excinfo.value
    )

    # The ffmpeg process and its pipes are released.
    assert len(audio_writer_instances) == 1
    assert audio_writer_instances[0].proc is None

    # The logfile handle is closed, its content preserved on disk.
    if write_logfile:
        logfile_name = filename + ".log"
        assert os.path.isfile(logfile_name)
        with open(logfile_name) as logfile:
            assert logfile.read()

    # No leaked file descriptors.
    if fds_before is not None:
        assert count_open_fds() == fds_before


@pytest.mark.parametrize(
    "remove_temp", (True, False), ids=("remove_temp=True", "remove_temp=False")
)
def test_write_videofile_video_failure_releases_resources(
    util, audio_writer_instances, video_writer_instances, remove_temp
):
    filename = os.path.join(util.TMP_DIR, "failing_video.mp4")
    temp_audiofile = os.path.join(util.TMP_DIR, "temp_audio_video_failure.mp3")
    fds_before = count_open_fds()

    with pytest.raises(IOError) as excinfo:
        make_video_clip().write_videofile(
            filename,
            codec="codec_that_does_not_exist",
            temp_audiofile=temp_audiofile,
            remove_temp=remove_temp,
            write_logfile=True,
            logger=None,
        )

    assert "MoviePy error: FFMPEG encountered the following error" in str(
        excinfo.value
    )

    # Both ffmpeg processes (audio then video) are released.
    assert len(audio_writer_instances) == 1
    assert audio_writer_instances[0].proc is None
    assert len(video_writer_instances) == 1
    assert video_writer_instances[0].proc is None

    # The temporary audio file follows remove_temp, even on failure.
    assert os.path.exists(temp_audiofile) != remove_temp
    if not remove_temp:
        os.remove(temp_audiofile)

    # Logfiles are closed and their content preserved.
    for logfile_name in (temp_audiofile + ".log", filename + ".log"):
        assert os.path.isfile(logfile_name)
        with open(logfile_name) as logfile:
            assert logfile.read()

    if fds_before is not None:
        assert count_open_fds() == fds_before


@pytest.mark.parametrize(
    "remove_temp", (True, False), ids=("remove_temp=True", "remove_temp=False")
)
def test_write_videofile_audio_failure_handles_partial_temp(
    util, audio_writer_instances, remove_temp
):
    """If the audio generation fails halfway, the partially generated
    temporary audio file must still follow remove_temp."""

    def failing_make_frame(t):
        if np.max(np.atleast_1d(np.asarray(t, dtype=float))) >= 0.2:
            raise RuntimeError("boom while generating audio")
        return np.sin(440 * 2 * np.pi * t)

    filename = os.path.join(util.TMP_DIR, "failing_audio_gen.mp4")
    temp_audiofile = os.path.join(util.TMP_DIR, "temp_audio_partial.mp3")
    clip = ColorClip((16, 16), color=(255, 0, 0)).with_duration(1)
    clip = clip.with_audio(
        AudioClip(failing_make_frame, duration=1, fps=44100)
    ).with_fps(10)
    fds_before = count_open_fds()

    # The original error is the one raised while generating the audio.
    with pytest.raises(RuntimeError, match="boom while generating audio"):
        clip.write_videofile(
            filename,
            temp_audiofile=temp_audiofile,
            remove_temp=remove_temp,
            logger=None,
        )

    assert len(audio_writer_instances) == 1
    assert audio_writer_instances[0].proc is None

    assert os.path.exists(temp_audiofile) != remove_temp
    if not remove_temp:
        os.remove(temp_audiofile)

    if fds_before is not None:
        assert count_open_fds() == fds_before


def test_write_videofile_failure_keeps_user_audiofile(util):
    """An audio file passed by the user is not a temp file: it must not
    be deleted, even with remove_temp=True and a failing export."""
    audiofile = os.path.join(util.TMP_DIR, "user_provided_audio.mp3")
    make_mono_clip(duration=0.5).write_audiofile(audiofile, logger=None)
    assert os.path.isfile(audiofile)

    clip = ColorClip((128, 128), color=(255, 0, 0)).with_duration(0.5).with_fps(10)
    with pytest.raises(IOError):
        clip.write_videofile(
            os.path.join(util.TMP_DIR, "video_with_user_audio.mp4"),
            codec="codec_that_does_not_exist",
            audio=audiofile,
            remove_temp=True,
            logger=None,
        )

    assert os.path.isfile(audiofile)


def test_write_videofile_success_removes_temp_audiofile(util):
    filename = os.path.join(util.TMP_DIR, "success_remove_temp.mp4")
    temp_audiofile = os.path.join(util.TMP_DIR, "temp_audio_success.mp3")

    make_video_clip().write_videofile(
        filename, temp_audiofile=temp_audiofile, remove_temp=True, logger=None
    )

    assert os.path.isfile(filename)
    assert not os.path.exists(temp_audiofile)


def test_ffmpeg_write_video_failure_releases_resources(util, video_writer_instances):
    """A failure while generating frames must release the ffmpeg process
    and close the logfile opened by ffmpeg_write_video."""
    filename = os.path.join(util.TMP_DIR, "failing_frame_gen.mp4")
    clip = ColorClip((16, 16), color=(255, 0, 0)).with_duration(1).with_fps(10)

    original_get_frame = clip.get_frame

    def failing_get_frame(t):
        if t >= 0.2:
            raise RuntimeError("boom while generating frames")
        return original_get_frame(t)

    clip.get_frame = failing_get_frame
    fds_before = count_open_fds()

    with pytest.raises(RuntimeError, match="boom while generating frames"):
        ffmpeg_writer.ffmpeg_write_video(
            clip, filename, 10, write_logfile=True, logger=None
        )

    assert len(video_writer_instances) == 1
    assert video_writer_instances[0].proc is None

    logfile_name = filename + ".log"
    assert os.path.isfile(logfile_name)
    with open(logfile_name) as logfile:
        assert logfile.read()

    if fds_before is not None:
        assert count_open_fds() == fds_before


def test_process_can_export_again_after_failures(util):
    """After failed exports, the same Python process must still be able
    to export audio and video normally."""
    audio_filename = os.path.join(util.TMP_DIR, "audio_after_failure.wav")
    video_filename = os.path.join(util.TMP_DIR, "video_after_failure.mp4")

    with pytest.raises(IOError):
        make_mono_clip().write_audiofile(
            os.path.join(util.TMP_DIR, "will_fail.wav"),
            codec="codec_that_does_not_exist",
            logger=None,
        )
    with pytest.raises(IOError):
        make_video_clip().write_videofile(
            os.path.join(util.TMP_DIR, "will_fail.mp4"),
            codec="codec_that_does_not_exist",
            logger=None,
        )

    make_mono_clip(duration=0.5).write_audiofile(audio_filename, logger=None)
    assert os.path.isfile(audio_filename)

    make_video_clip().write_videofile(video_filename, logger=None)
    clip = VideoFileClip(video_filename)
    try:
        assert clip.duration > 0
        assert clip.audio is not None
    finally:
        clip.close()
