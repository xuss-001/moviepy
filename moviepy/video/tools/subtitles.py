"""Experimental module for subtitles support."""

import re

import numpy as np

from moviepy.decorators import convert_path_to_string
from moviepy.tools import convert_to_seconds
from moviepy.video.VideoClip import TextClip, VideoClip


class SubtitlesClip(VideoClip):
    """A Clip that serves as "subtitle track" in videos.

    One particularity of this class is that the images of the
    subtitle texts are not generated beforehand, but only if
    needed.

    Parameters
    ----------

    subtitles
        Either the name of a file as a string or path-like object, or a list

    font
        Path to a font file to be used. Optional if make_textclip is provided.

    make_textclip
        A custom function to use for text clip generation. If None, a TextClip
        will be generated.

        The function must take a text as argument and return a VideoClip
        to be used as caption

    encoding
        Optional, specifies srt file encoding.
        Any standard Python encoding is allowed (listed at
        https://docs.python.org/3.8/library/codecs.html#standard-encodings)

    stacked
        Optional, default ``False``. When ``False``, only the first subtitle
        in input order that is active at the requested time is shown. When
        ``True``, all active subtitles are displayed as a vertically stacked
        block, preserving their input order. Shorter lines are centered and
        duplicate subtitle records are displayed separately.

        Stacked subtitles are generated on demand. Each text clip's animation
        uses its own subtitle start time, while the returned video and mask
        always have matching dimensions, positions and animation progress.
        The stacked clip has a transparent mask, including the empty space
        around shorter lines and times when no subtitle is active.

    overlap
        Optional alias for ``stacked``. It accepts a boolean when provided.

    Examples
    --------

    .. code:: python

        from moviepy.video.tools.subtitles import SubtitlesClip
        from moviepy.video.io.VideoFileClip import VideoFileClip
        generator = lambda text: TextClip(text, font='./path/to/font.ttf',
                                        font_size=24, color='white')
        sub = SubtitlesClip("subtitles.srt", make_textclip=generator, encoding='utf-8')
        myvideo = VideoFileClip("myvideo.avi")
        final = CompositeVideoClip([clip, subtitles])
        final.write_videofile("final.mp4", fps=myvideo.fps)

    To display overlapping subtitles instead of only the first active one:

    .. code:: python

        subtitles = SubtitlesClip(
            "subtitles.srt",
            make_textclip=generator,
            encoding="utf-8",
            stacked=True,
        )
        final = CompositeVideoClip(
            [myvideo, subtitles.with_position(("center", "bottom"))]
        )

    """

    def __init__(
        self,
        subtitles,
        font=None,
        make_textclip=None,
        encoding=None,
        stacked=False,
        overlap=None,
    ):
        if not isinstance(stacked, (bool, np.bool_)):
            raise TypeError("Argument stacked must be a boolean.")
        if overlap is not None:
            if not isinstance(overlap, (bool, np.bool_)):
                raise TypeError("Argument overlap must be a boolean.")
            if stacked and not overlap:
                raise ValueError(
                    "Arguments stacked and overlap cannot have conflicting values."
                )
            stacked = bool(overlap)

        stacked = bool(stacked)

        VideoClip.__init__(self, has_constant_size=False)

        if not isinstance(subtitles, list):
            # `subtitles` is a string or path-like object
            subtitles = file_to_subtitles(subtitles, encoding=encoding)

        # subtitles = [(map(convert_to_seconds, times), text)
        #              for times, text in subtitles]
        self.subtitles = subtitles
        self.textclips = dict()
        self.stacked = stacked
        self.overlap = stacked

        self.font = font

        if make_textclip is None:
            if self.font is None:
                raise ValueError("Argument font is required if make_textclip is None.")

            def make_textclip(txt):
                return TextClip(
                    font=self.font,
                    text=txt,
                    font_size=24,
                    color="#ffffff",
                    stroke_color="#000000",
                    stroke_width=1,
                )

        self.make_textclip = make_textclip
        self.start = 0
        self.duration = max([tb for ((ta, tb), txt) in self.subtitles])
        self.end = self.duration
        self.size = (1, 1)

        if self.stacked:
            self._stacked_cache_time = None
            self._stacked_cache = None

            def frame_function(t):
                return self._stacked_frame_and_mask(t)[0]

            def make_mask_frame(t):
                return self._stacked_frame_and_mask(t)[1]

            self.frame_function = frame_function
            self.mask = VideoClip(is_mask=True, has_constant_size=False)
            self.mask.frame_function = make_mask_frame
            self.mask.size = (1, 1)
            return

        def add_textclip_if_none(t):
            """Return the first subtitle active at ``t`` in input order.

            The text clip is generated and cached on first use. Looking only
            at ``self.subtitles`` makes the result independent from which
            times were requested previously.
            """
            for index, sub in enumerate(self.subtitles):
                (text_start, text_end), text = sub
                if text_start <= t < text_end:
                    break
            else:
                return None

            cache_key = self._subtitle_cache_key(index, sub)
            if cache_key not in self.textclips:
                self.textclips[cache_key] = self.make_textclip(sub[1])

            return cache_key

        def frame_function(t):
            sub = add_textclip_if_none(t)
            return (
                self.textclips[sub].get_frame(t)
                if sub is not None
                else np.array([[[0, 0, 0]]])
            )

        def make_mask_frame(t):
            sub = add_textclip_if_none(t)
            return (
                self.textclips[sub].mask.get_frame(t)
                if sub is not None
                else np.array([[0]])
            )

        self.frame_function = frame_function
        hasmask = bool(self.make_textclip("T").mask)
        self.mask = VideoClip(make_mask_frame, is_mask=True) if hasmask else None

    def _active_subtitles(self, t):
        """Return active subtitles in input order, with their record index."""
        return [
            (index, subtitle)
            for index, subtitle in enumerate(self.subtitles)
            if subtitle[0][0] <= t < subtitle[0][1]
        ]

    def _textclip_for(self, index, subtitle):
        """Return the cached text clip for a subtitle record."""
        cache_key = self._subtitle_cache_key(index, subtitle)
        if cache_key not in self.textclips:
            self.textclips[cache_key] = self.make_textclip(subtitle[1])

        return self.textclips[cache_key]

    def _subtitle_cache_key(self, index, subtitle):
        """Keep legacy record keys when possible and use the index otherwise."""
        if self.stacked:
            return index
        try:
            hash(subtitle)
        except TypeError:
            return index
        return subtitle

    @staticmethod
    def _resize_mask(mask, shape):
        """Crop or zero-pad a mask to match a frame's height and width."""
        height, width = shape
        mask = np.asarray(mask)
        if mask.ndim == 3:
            mask = mask[..., 0]

        resized = np.zeros((height, width), dtype=np.float64)
        copy_height = min(height, mask.shape[0])
        copy_width = min(width, mask.shape[1])
        resized[:copy_height, :copy_width] = mask[:copy_height, :copy_width]
        return resized

    def _stacked_frame_and_mask(self, t):
        """Build the stacked frame and its matching transparency mask."""
        if t == self._stacked_cache_time and self._stacked_cache is not None:
            return self._stacked_cache

        active_subtitles = self._active_subtitles(t)
        if not active_subtitles:
            frame = np.zeros((1, 1, 3), dtype=np.uint8)
            mask = np.zeros((1, 1), dtype=np.float64)
            self.size = (1, 1)
            self.mask.size = (1, 1)
            self._stacked_cache_time = t
            self._stacked_cache = (frame, mask)
            return frame, mask

        rows = []
        for index, subtitle in active_subtitles:
            textclip = self._textclip_for(index, subtitle)
            local_t = t - subtitle[0][0]
            frame = np.asarray(textclip.get_frame(local_t))

            if frame.ndim == 2:
                frame = np.dstack(3 * [frame])

            embedded_alpha = None
            if frame.ndim == 3 and frame.shape[2] == 4:
                embedded_alpha = frame[:, :, 3].astype(np.float64) / 255.0
                frame = frame[:, :, :3]

            if frame.ndim != 3 or frame.shape[2] != 3:
                raise ValueError(
                    "make_textclip must return clips with RGB or RGBA frames."
                )

            if textclip.mask is None:
                mask = np.ones(frame.shape[:2], dtype=np.float64)
            else:
                mask = self._resize_mask(
                    textclip.mask.get_frame(local_t), frame.shape[:2]
                )

            if embedded_alpha is not None:
                mask = mask * self._resize_mask(embedded_alpha, frame.shape[:2])

            rows.append((frame, mask))

        width = max(frame.shape[1] for frame, mask in rows)
        height = sum(frame.shape[0] for frame, mask in rows)
        frame_dtype = np.result_type(*(frame.dtype for frame, mask in rows))

        frame = np.zeros((height, width, 3), dtype=frame_dtype)
        mask = np.zeros((height, width), dtype=np.float64)
        y_start = 0
        for row_frame, row_mask in rows:
            row_height, row_width = row_frame.shape[:2]
            x_start = (width - row_width) // 2
            frame[y_start : y_start + row_height, x_start : x_start + row_width] = (
                row_frame
            )
            mask[y_start : y_start + row_height, x_start : x_start + row_width] = (
                row_mask
            )
            y_start += row_height

        self.size = (width, height)
        self.mask.size = (width, height)
        self._stacked_cache_time = t
        self._stacked_cache = (frame, mask)
        return frame, mask

    def in_subclip(self, start_time=None, end_time=None):
        """Returns a sequence of [(t1,t2), text] covering all the given subclip
        from start_time to end_time. The first and last times will be cropped so as
        to be exactly start_time and end_time if possible.
        """

        def is_in_subclip(t1, t2):
            try:
                return (start_time <= t1 < end_time) or (start_time < t2 <= end_time)
            except Exception:
                return False

        def try_cropping(t1, t2):
            try:
                return max(t1, start_time), min(t2, end_time)
            except Exception:
                return t1, t2

        return [
            (try_cropping(t1, t2), txt)
            for ((t1, t2), txt) in self.subtitles
            if is_in_subclip(t1, t2)
        ]

    def __iter__(self):
        return iter(self.subtitles)

    def __getitem__(self, k):
        return self.subtitles[k]

    def __str__(self):
        def to_srt(sub_element):
            (start_time, end_time), text = sub_element
            formatted_start_time = convert_to_seconds(start_time)
            formatted_end_time = convert_to_seconds(end_time)
            return "%s - %s\n%s" % (formatted_start_time, formatted_end_time, text)

        return "\n\n".join(to_srt(sub) for sub in self.subtitles)

    def match_expr(self, expr):
        """Matches a regular expression against the subtitles of the clip."""
        return SubtitlesClip(
            [sub for sub in self.subtitles if re.findall(expr, sub[1]) != []],
            font=self.font,
            make_textclip=self.make_textclip,
            stacked=self.stacked,
        )

    def write_srt(self, filename):
        """Writes an ``.srt`` file with the content of the clip."""
        with open(filename, "w+") as file:
            file.write(str(self))


@convert_path_to_string("filename")
def file_to_subtitles(filename, encoding=None):
    """Converts a srt file into subtitles.

    The returned list is of the form ``[((start_time,end_time),'some text'),...]``
    and can be fed to SubtitlesClip.

    Only works for '.srt' format for the moment.
    """
    times_texts = []
    current_times = None
    current_text = ""
    with open(filename, "r", encoding=encoding) as file:
        for line in file:
            times = re.findall("([0-9]*:[0-9]*:[0-9]*,[0-9]*)", line)
            if times:
                current_times = [convert_to_seconds(t) for t in times]
            elif line.strip() == "":
                times_texts.append((current_times, current_text.strip("\n")))
                current_times, current_text = None, ""
            elif current_times:
                current_text += line
    return times_texts
