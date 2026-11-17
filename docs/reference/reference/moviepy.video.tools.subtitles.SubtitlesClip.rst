.. custom class to enable complete documentation of every function
   see https://stackoverflow.com/a/62613202

moviepy.video.tools.subtitles.SubtitlesClip
===========================================

.. currentmodule:: moviepy.video.tools.subtitles

.. autoclass:: SubtitlesClip
   :members:

Overlapping subtitles
---------------------

By default, :class:`SubtitlesClip` shows only the first subtitle that is active
at the requested time, in the same order as the input list or SRT file. This
keeps the historical behavior where each subtitle appears at its start time and
disappears at its end time.

Pass ``stacked=True`` to show every active subtitle at once. The active lines
are placed from top to bottom in input order, without extra vertical spacing.
Narrower lines are centered, and identical records are kept as separate rows.

.. code:: python

    from moviepy.video.compositing.CompositeVideoClip import CompositeVideoClip
    from moviepy.video.tools.subtitles import SubtitlesClip
    from moviepy.video.VideoClip import TextClip


    def make_textclip(text):
        return TextClip(
            text=text,
            font="media/doc_medias/example.ttf",
            font_size=24,
            color="white",
            stroke_color="black",
            stroke_width=1,
        )


    subtitles = SubtitlesClip(
        "subtitles.srt",
        make_textclip=make_textclip,
        encoding="utf-8",
        stacked=True,
    )
    final = CompositeVideoClip(
        [video, subtitles.with_position(("center", "bottom"))]
    )

Stacked subtitles are generated lazily and cached per subtitle record. Each
line's animation starts at that line's own start time. Its image and mask use
the same dimensions, centering offsets, and animation time. The returned clip
always has a transparency mask: the mask uses each generated text clip's
opacity (fully opaque when that clip has no mask), while side padding and
periods without active subtitles remain transparent.

``overlap=True`` is an alias for ``stacked=True``. The option must be a boolean.
