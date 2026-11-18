from dataclasses import dataclass, field

import numpy as np

from moviepy.Clip import Clip
from moviepy.Effect import Effect
from moviepy.tools import convert_to_seconds


@dataclass
class DuckVolume(Effect):
    """Automatically lowers the volume of a clip whenever another audio
    clip (the *control* clip, e.g. a voice-over) contains sound, and
    restores it when the control clip is silent. This is commonly known
    as "sidechain ducking", typically used to lower background music
    while someone is speaking.

    The control clip is analyzed at its own frame rate: each sample
    ``i`` (at time ``control.start + i / control.fps``) is considered to
    represent the period up to the next sample, the last sample being
    truncated to the end of the control clip. A sample is considered
    *active* (someone is speaking) when the absolute amplitude of any of
    its channels is strictly greater than ``threshold``. Silence gaps
    shorter than ``hold_time`` between active samples are still
    considered continuous speech.

    While speech is detected, the volume of the clip is multiplied by
    ``factor``. Before each speech segment, the volume linearly
    decreases from ``1`` to ``factor`` over ``attack_time`` seconds, and
    after each segment it linearly increases back to ``1`` over
    ``release_time`` seconds. When several transitions overlap, the
    lowest volume wins. A zero ``attack_time`` or ``release_time`` means
    the volume switches immediately.

    The control clip only drives the volume, it is *not* mixed into the
    result. Both clips are aligned on a common timeline using their
    respective ``start`` attributes (for video clips, the ``start`` of
    their audio track is used). Outside of the control clip there is no
    activity, but attack and release transitions may extend beyond its
    limits.

    The effect works on audio clips and on video clips with an audio
    track. It preserves the duration, frame rate and relative channel
    volumes of the target clip, does not modify the input clips, and
    analyzes the control clip by chunks without loading all its samples
    in memory.

    Parameters
    ----------

    control : AudioClip
      The audio clip whose activity drives the volume (e.g. a
      voice-over). It must have a ``duration`` and a positive ``fps``.

    factor : float
      Volume multiplication factor applied while speech is detected.
      Must be between 0 and 1.

    threshold : float, optional
      Absolute amplitude strictly above which a control sample is
      considered active. Must be a finite non-negative number.

    attack_time : float, optional
      Time (in seconds) taken to linearly lower the volume from ``1``
      to ``factor`` before each speech segment. Zero means immediate
      switch. Must be a finite non-negative number.

    release_time : float, optional
      Time (in seconds) taken to linearly restore the volume from
      ``factor`` to ``1`` after each speech segment. Zero means
      immediate switch. Must be a finite non-negative number.

    hold_time : float, optional
      Silence gaps strictly shorter than this duration (in seconds) do
      not interrupt a speech segment. Must be a finite non-negative
      number.

    Examples
    --------

    .. code:: python

        from moviepy import AudioFileClip
        from moviepy.audio.fx import DuckVolume

        music = AudioFileClip("music.ogg")
        voice = AudioFileClip("voiceover.ogg")
        # lower the music to 20% of its volume while someone speaks
        ducked = music.with_effects([afx.DuckVolume(voice, factor=0.2)])
    """

    control: "Clip"
    factor: float
    threshold: float = 0.01
    attack_time: float = 0.05
    release_time: float = 0.3
    hold_time: float = 0.2
    _segments: list = field(default=None, init=False, repr=False)

    def __post_init__(self):
        self.attack_time = convert_to_seconds(self.attack_time)
        self.release_time = convert_to_seconds(self.release_time)
        self.hold_time = convert_to_seconds(self.hold_time)

        for name in ("threshold", "attack_time", "release_time", "hold_time"):
            value = getattr(self, name)
            if not np.isfinite(value) or value < 0:
                raise ValueError(
                    f"Attribute '{name}' must be a finite non-negative number"
                )

        if not np.isfinite(self.factor) or not 0 <= self.factor <= 1:
            raise ValueError("Attribute 'factor' must be a number between 0 and 1")

    def _iter_active_intervals(self, control, fps):
        """Streams the control clip by chunks and yields ``(start, end)``
        intervals (on the common timeline) of every active sample.
        """
        start = control.start or 0
        total_size = int(fps * control.duration)
        chunksize = max(1, min(50000, int(round(fps))))
        index = 0
        for chunk in control.iter_chunks(chunksize=chunksize, fps=fps, logger=None):
            amplitudes = np.abs(np.asarray(chunk, dtype=float))
            if amplitudes.ndim > 1:
                amplitudes = amplitudes.max(axis=tuple(range(1, amplitudes.ndim)))
            for is_active in amplitudes > self.threshold:
                if is_active:
                    interval_start = start + index / fps
                    if index == total_size - 1:
                        interval_end = start + control.duration
                    else:
                        interval_end = start + (index + 1) / fps
                    yield (interval_start, interval_end)
                index += 1

    def _find_speech_segments(self, control, fps):
        """Returns the list of ``[start, end]`` speech segments of the
        control clip, merging silence gaps shorter than ``hold_time``.
        """
        segments = []
        for interval_start, interval_end in self._iter_active_intervals(control, fps):
            if segments:
                gap = interval_start - segments[-1][1]
                if gap <= 0 or gap < self.hold_time:
                    segments[-1][1] = interval_end
                    continue
            segments.append([interval_start, interval_end])
        return segments

    def _volume_envelope(self):
        """Returns a stateless function giving the volume factor at any
        time (or array of times) on the common timeline.
        """
        factor = self.factor
        attack = self.attack_time
        release = self.release_time
        segments = self._segments

        def envelope(t):
            is_scalar = np.ndim(t) == 0
            t = np.atleast_1d(np.asarray(t, dtype=float))
            volume = np.ones(t.shape)
            for segment_start, segment_end in segments:
                segment_volume = np.ones(t.shape)
                if attack > 0:
                    mask = (t >= segment_start - attack) & (t < segment_start)
                    segment_volume[mask] = 1.0 - (1.0 - factor) * (
                        t[mask] - (segment_start - attack)
                    ) / attack
                mask = (t >= segment_start) & (t < segment_end)
                segment_volume[mask] = factor
                if release > 0:
                    mask = (t >= segment_end) & (t < segment_end + release)
                    segment_volume[mask] = factor + (1.0 - factor) * (
                        t[mask] - segment_end
                    ) / release
                volume = np.minimum(volume, segment_volume)
            return volume[0] if is_scalar else volume

        return envelope

    def _apply_to_audio(self, audio_clip):
        control = self.control

        fps = getattr(control, "fps", None)
        if fps is None or not np.isfinite(fps) or fps <= 0:
            raise ValueError("Control clip must have a positive finite 'fps'")
        if control.duration is None:
            raise ValueError("Control clip must have a 'duration'")

        self._segments = self._find_speech_segments(control, fps)
        envelope = self._volume_envelope()

        start = audio_clip.start or 0
        nchannels = audio_clip.nchannels

        def multiply(get_frame, t):
            frame = get_frame(t)
            factors = envelope(np.asarray(t, dtype=float) + start)
            if nchannels > 1 and np.ndim(factors) > 0:
                factors = factors[:, np.newaxis]
            return frame * factors

        return audio_clip.transform(multiply, keep_duration=True)

    def apply(self, clip: Clip) -> Clip:
        """Apply the effect to the clip."""
        if hasattr(clip, "audio"):
            if clip.audio is None:
                raise ValueError(
                    "Cannot apply DuckVolume to a video clip without audio track"
                )
            clip.audio = self._apply_to_audio(clip.audio)
            return clip
        return self._apply_to_audio(clip)
