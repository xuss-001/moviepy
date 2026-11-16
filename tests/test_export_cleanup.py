"""Regression tests for resource cleanup in the audio/video export chain.

Whatever happens during an export (success, or failure at initialization,
frame generation or writing), the ffmpeg processes, pipes and logfile
handles created for this export must be released immediately, and temporary
audio files must be removed or kept according to ``remove_temp``.
"""

import os

import numpy as np

import pytest

from moviepy.audio.AudioClip import AudioClip
from moviepy.audio.io import ffmpeg_audiowriter
from moviepy.audio.io.ffmpeg_audiowriter import FFMPEG_AudioWriter
from moviepy.audio.io.AudioFileClip import AudioFileClip
from moviepy.Clip import Clip
from moviepy.video.io import ffmpeg_writer
from moviepy.video.io.ffmpeg_writer import FFMPEG_VideoWriter
from moviepy.video.io.VideoFileClip import VideoFileClip
from moviepy.video.VideoClip import ColorClip, VideoClip


# Long enough that the data written to ffmpeg exceeds the OS pipe capacity,
# so a failing ffmpeg process is always detected with a broken pipe error.
FAILURE_DETECTION_DURATION = 5


class WriterTracker:
    """Monkeypatches a writer class in its module to record every instance
    and subprocess created, so tests can assert everything was released.
    """

    def __init__(self, monkeypatch, module, class_name):
        self.writers = []
        self.procs = []
        original_class = getattr(module, class_name)
        tracker = self

        class TrackedWriter(original_class):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                tracker.writers.append(self)
                tracker.procs.append(self.proc)

        monkeypatch.setattr(module, class_name, TrackedWriter)

    def assert_all_released(self):
        assert self.writers, "no writer was created"
        for writer in self.writers:
            assert writer.proc is None
        for proc in self.procs:
            assert proc.poll() is not None


@pytest.fixture
def audio_writer_tracker(monkeypatch):
    return WriterTracker(monkeypatch, ffmpeg_audiowriter, "FFMPEG_AudioWriter")


@pytest.fixture
def video_writer_tracker(monkeypatch):
    return WriterTracker(monkeypatch, ffmpeg_writer, "FFMPEG_VideoWriter")


@pytest.fixture
def tracked_logfiles(monkeypatch):
    """Tracks the logfiles opened by ``ffmpeg_audiowrite`` and
    ``ffmpeg_write_video`` so tests can assert they are closed.
    """
    opened = []
    real_open = open

    def tracking_open(*args, **kwargs):
        file = real_open(*args, **kwargs)
        opened.append(file)
        return file

    monkeypatch.setattr(ffmpeg_audiowriter, "open", tracking_open, raising=False)
    monkeypatch.setattr(ffmpeg_writer, "open", tracking_open, raising=False)
    return opened


def make_videoclip_with_audio(mono_wave, duration=FAILURE_DETECTION_DURATION):
    clip = ColorClip((32, 32), color=(255, 0, 0), duration=duration).with_fps(10)
    clip.audio = AudioClip(mono_wave(440), duration=duration, fps=44100)
    return clip


def make_exploding_wave(fail_after=0.5):
    """A mono wave whose generation fails once ``t`` goes past ``fail_after``."""

    def make_frame(t):
        if np.max(t) > fail_after:
            raise ValueError("boom")
        return np.sin(440 * 2 * np.pi * t)

    return make_frame


def test_write_audiofile_error_releases_resources(
    mono_wave, audio_writer_tracker, tracked_logfiles, tmp_path
):
    """A failing ``write_audiofile`` must release the ffmpeg process and
    close the logfile, while preserving the original ffmpeg error message
    and the logfile content.
    """
    filename = str(tmp_path / "audio.mp3")
    clip = AudioClip(mono_wave(440), duration=FAILURE_DETECTION_DURATION, fps=44100)

    with pytest.raises(IOError) as excinfo:
        clip.write_audiofile(
            filename, codec="libcrazyfoobar", write_logfile=True, logger=None
        )

    assert "MoviePy error: FFMPEG encountered the following error" in str(
        excinfo.value
    )
    audio_writer_tracker.assert_all_released()
    assert tracked_logfiles
    assert all(logfile.closed for logfile in tracked_logfiles)
    with open(filename + ".log") as logfile:
        assert "libcrazyfoobar" in logfile.read()


def test_write_audiofile_generation_error_releases_resources(
    audio_writer_tracker, tracked_logfiles, tmp_path
):
    """When the audio frames generation fails, the ffmpeg process and the
    logfile must be released and the original error must propagate.
    """
    filename = str(tmp_path / "audio.mp3")
    clip = AudioClip(make_exploding_wave(), duration=2, fps=44100)

    with pytest.raises(ValueError, match="boom"):
        clip.write_audiofile(filename, write_logfile=True, logger=None)

    audio_writer_tracker.assert_all_released()
    assert tracked_logfiles
    assert all(logfile.closed for logfile in tracked_logfiles)


def test_write_audiofile_success_releases_resources(
    mono_wave, audio_writer_tracker, tracked_logfiles, tmp_path
):
    """A successful ``write_audiofile`` must also release everything."""
    filename = str(tmp_path / "audio.mp3")
    clip = AudioClip(mono_wave(440), duration=0.5, fps=44100)

    clip.write_audiofile(filename, write_logfile=True, logger=None)

    assert os.path.isfile(filename)
    AudioFileClip(filename)
    audio_writer_tracker.assert_all_released()
    assert tracked_logfiles
    assert all(logfile.closed for logfile in tracked_logfiles)
    assert os.path.getsize(filename + ".log") > 0


@pytest.mark.parametrize(
    "remove_temp", (True, False), ids=("remove_temp", "keep_temp")
)
@pytest.mark.parametrize(
    "failing_kwargs",
    (
        pytest.param({"audio_codec": "libcrazyfoobar"}, id="fail_in_audio_writing"),
        pytest.param({"codec": "libcrazyfoobar"}, id="fail_in_video_writing"),
    ),
)
def test_write_videofile_error_releases_resources(
    mono_wave,
    audio_writer_tracker,
    video_writer_tracker,
    tracked_logfiles,
    tmp_path,
    remove_temp,
    failing_kwargs,
):
    """A failing ``write_videofile`` must release every process and logfile
    handle, and the temporary audiofile must be removed when
    ``remove_temp=True`` (even if only partially generated) or kept when
    ``remove_temp=False``.
    """
    filename = str(tmp_path / "video.mp4")
    temp_audiofile = str(tmp_path / "temp_audio.mp3")
    clip = make_videoclip_with_audio(mono_wave)

    with pytest.raises(IOError) as excinfo:
        clip.write_videofile(
            filename,
            temp_audiofile=temp_audiofile,
            remove_temp=remove_temp,
            write_logfile=True,
            logger=None,
            **failing_kwargs,
        )

    assert "MoviePy error: FFMPEG encountered the following error" in str(
        excinfo.value
    )
    audio_writer_tracker.assert_all_released()
    if "codec" in failing_kwargs:
        video_writer_tracker.assert_all_released()
    assert tracked_logfiles
    assert all(logfile.closed for logfile in tracked_logfiles)
    if remove_temp:
        assert not os.path.exists(temp_audiofile)
    elif "codec" in failing_kwargs:
        # The audio export succeeded, so the temporary audiofile was fully
        # generated and must be kept. When the audio export itself fails,
        # ffmpeg may not have created the file at all; the partial-file
        # behavior is covered by
        # ``test_write_videofile_audio_generation_error_handles_partial_temp``.
        assert os.path.isfile(temp_audiofile)


def test_write_videofile_error_removes_default_temp_audiofile(
    mono_wave, audio_writer_tracker, tmp_path
):
    """The auto-named temporary audiofile must be removed on failure too."""
    filename = str(tmp_path / "video.mp4")
    clip = make_videoclip_with_audio(mono_wave)

    with pytest.raises(IOError):
        clip.write_videofile(
            filename,
            codec="libcrazyfoobar",
            temp_audiofile_path=str(tmp_path),
            logger=None,
        )

    temp_files = [
        name for name in os.listdir(tmp_path) if Clip._TEMP_FILES_PREFIX in name
    ]
    assert temp_files == []
    audio_writer_tracker.assert_all_released()


@pytest.mark.parametrize(
    "remove_temp", (True, False), ids=("remove_temp", "keep_temp")
)
def test_write_videofile_audio_generation_error_handles_partial_temp(
    audio_writer_tracker, tmp_path, remove_temp
):
    """When the audio generation fails midway, the partially generated
    temporary audiofile must be removed if ``remove_temp=True`` and kept
    otherwise, and the original error must propagate.
    """
    filename = str(tmp_path / "video.mp4")
    temp_audiofile = str(tmp_path / "temp_audio.mp3")
    clip = ColorClip((32, 32), color=(255, 0, 0), duration=2).with_fps(10)
    clip.audio = AudioClip(make_exploding_wave(), duration=2, fps=44100)

    with pytest.raises(ValueError, match="boom"):
        clip.write_videofile(
            filename,
            temp_audiofile=temp_audiofile,
            remove_temp=remove_temp,
            logger=None,
        )

    audio_writer_tracker.assert_all_released()
    if remove_temp:
        assert not os.path.exists(temp_audiofile)
    else:
        assert os.path.isfile(temp_audiofile)


def test_write_videofile_error_keeps_external_audio(
    mono_wave, video_writer_tracker, tmp_path
):
    """An audio file provided by the user must never be deleted, even when
    the export fails.
    """
    external_audio = str(tmp_path / "external.mp3")
    AudioClip(mono_wave(440), duration=0.5, fps=44100).write_audiofile(
        external_audio, logger=None
    )
    filename = str(tmp_path / "video.mp4")
    clip = ColorClip(
        (32, 32), color=(255, 0, 0), duration=FAILURE_DETECTION_DURATION
    ).with_fps(10)

    with pytest.raises(IOError):
        clip.write_videofile(
            filename, codec="libcrazyfoobar", audio=external_audio, logger=None
        )

    assert os.path.isfile(external_audio)
    video_writer_tracker.assert_all_released()


def test_write_videofile_success_removes_temp_audiofile(
    mono_wave, audio_writer_tracker, video_writer_tracker, tmp_path
):
    """On success the temporary audiofile is removed when ``remove_temp``
    and everything is released."""
    filename = str(tmp_path / "video.mp4")
    temp_audiofile = str(tmp_path / "temp_audio.mp3")
    clip = make_videoclip_with_audio(mono_wave)

    clip.write_videofile(filename, temp_audiofile=temp_audiofile, logger=None)

    assert os.path.isfile(filename)
    result = VideoFileClip(filename)
    assert result.audio is not None
    result.close()
    assert not os.path.exists(temp_audiofile)
    audio_writer_tracker.assert_all_released()
    video_writer_tracker.assert_all_released()


def test_export_still_works_after_failure(mono_wave, tmp_path):
    """After a failed export, the same Python process must be able to
    export normally.
    """
    clip = make_videoclip_with_audio(mono_wave)
    with pytest.raises(IOError):
        clip.write_videofile(
            str(tmp_path / "broken.mp4"), codec="libcrazyfoobar", logger=None
        )
    with pytest.raises(IOError):
        clip.audio.write_audiofile(
            str(tmp_path / "broken.mp3"), codec="libcrazyfoobar", logger=None
        )

    filename = str(tmp_path / "video.mp4")
    clip.write_videofile(filename, logger=None)
    result = VideoFileClip(filename)
    assert result.duration > 0
    assert result.audio is not None
    result.close()


def test_writers_close_is_idempotent(tmp_path):
    """Closing a writer twice, or after its process already exited, must
    be safe.
    """
    audio_writer = FFMPEG_AudioWriter(str(tmp_path / "audio.mp3"), 44100)
    audio_writer.close()
    audio_writer.close()

    video_writer = FFMPEG_VideoWriter(str(tmp_path / "video.avi"), (32, 32), 10)
    video_writer.close()
    video_writer.close()

    exited_writer = FFMPEG_AudioWriter(str(tmp_path / "audio2.mp3"), 44100)
    process = exited_writer.proc
    process.stdin.close()
    process.wait()
    assert process.poll() is not None
    exited_writer.close()
    assert exited_writer.proc is None
