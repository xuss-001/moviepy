"""SubtitlesClip tests."""

import os

import numpy as np
import pytest

from moviepy.video.compositing.CompositeVideoClip import (
    CompositeVideoClip,
    concatenate_videoclips,
)
from moviepy.video.io.VideoFileClip import VideoFileClip
from moviepy.video.tools.subtitles import SubtitlesClip, file_to_subtitles
from moviepy.video.VideoClip import ColorClip, TextClip, VideoClip


MEDIA_SUBTITLES_DATA = [
    ([0.0, 1.0], "Red!"),
    ([2.0, 3.5], "More Red!"),
    ([5.0, 6.0], "Green!"),
    ([7.0, 8.0], "Blue"),
    ([9.0, 10.0], "More Blue!"),
]

MEDIA_SUBTITLES_UNICODE_DATA = [
    ([0, 5.0], "ÁÉíöÙ"),
]


def test_subtitles(util):
    red = ColorClip((800, 600), color=(255, 0, 0)).with_duration(10)
    green = ColorClip((800, 600), color=(0, 255, 0)).with_duration(10)
    blue = ColorClip((800, 600), color=(0, 0, 255)).with_duration(10)
    myvideo = concatenate_videoclips([red, green, blue])
    assert myvideo.duration == 30

    generator = lambda txt: TextClip(
        text=txt,
        font=util.FONT,
        size=(800, 600),
        font_size=24,
        method="caption",
        vertical_align="bottom",
        color="white",
    )

    subtitles = SubtitlesClip("media/subtitles.srt", make_textclip=generator)
    final = CompositeVideoClip([myvideo, subtitles])
    final.subclipped(0, 0.5).write_videofile(
        os.path.join(util.TMP_DIR, "subtitles.mp4"),
        fps=5,
        logger=None,
    )

    assert subtitles.subtitles == MEDIA_SUBTITLES_DATA

    subtitles = SubtitlesClip(MEDIA_SUBTITLES_DATA, make_textclip=generator)
    assert subtitles.subtitles == MEDIA_SUBTITLES_DATA


def test_file_to_subtitles():
    assert MEDIA_SUBTITLES_DATA == file_to_subtitles("media/subtitles.srt")


def test_file_to_subtitles_unicode():
    assert MEDIA_SUBTITLES_UNICODE_DATA == file_to_subtitles(
        "media/subtitles-unicode.srt", encoding="utf-8"
    )


def test_default_mode_uses_first_active_subtitle_in_input_order():
    colors = {
        "first": (255, 0, 0),
        "second": (0, 255, 0),
        "T": (0, 0, 0),
    }

    def make_textclip(text):
        return ColorClip((1, 1), color=colors[text])

    subtitles = [([0.0, 3.0], "first"), ([1.0, 4.0], "second")]
    clip = SubtitlesClip(subtitles, make_textclip=make_textclip)

    assert clip.subtitles is subtitles
    assert clip.get_frame(3.5)[0, 0].tolist() == [0, 255, 0]
    assert clip.get_frame(2.5)[0, 0].tolist() == [255, 0, 0]
    assert clip.get_frame(1.5)[0, 0].tolist() == [255, 0, 0]
    assert clip.get_frame(2.5)[0, 0].tolist() == [255, 0, 0]
    assert clip.get_frame(4.0).shape == (1, 1, 3)


def test_default_mode_keeps_global_animation_time():
    times = []

    def make_textclip(text):
        if text == "T":
            return ColorClip((1, 1), color=(0, 0, 0))

        def frame(t):
            times.append(t)
            return np.full((1, 1, 3), [int(t * 10), 0, 0], dtype=np.uint8)

        return VideoClip(frame)

    clip = SubtitlesClip([([1.0, 2.0], "animated")], make_textclip=make_textclip)
    times.clear()
    clip.get_frame(1.5)

    assert times[-1] == 1.5
    assert 1.5 in times
    assert clip.textclips[0].get_frame(1.5)[0, 0, 0] == 15


def test_stacked_mode_layout_duplicates_opacity_and_cache():
    generated = []

    def make_textclip(text):
        generated.append(text)
        if text == "wide":
            frame = VideoClip(
                lambda t: np.full((2, 6, 3), [255, 0, 0], dtype=np.uint8)
            )
            frame.mask = VideoClip(
                lambda t: np.full((2, 6), 0.5), is_mask=True
            )
            return frame
        return VideoClip(
            lambda t: np.full((1, 2, 3), [0, 255, 0], dtype=np.uint8)
        )

    subtitles = [
        ([0.0, 3.0], "wide"),
        ([1.0, 4.0], "narrow"),
        ([1.0, 4.0], "narrow"),
    ]
    clip = SubtitlesClip(subtitles, make_textclip=make_textclip, stacked=True)

    assert generated == []

    frame = clip.get_frame(1.5)
    mask = clip.mask.get_frame(1.5)

    assert generated == ["wide", "narrow", "narrow"]
    assert frame.shape == (4, 6, 3)
    assert mask.shape == frame.shape[:2]
    assert clip.size == (6, 4)

    np.testing.assert_array_equal(frame[0:2, :], np.full((2, 6, 3), [255, 0, 0]))
    np.testing.assert_array_equal(frame[2, 2:4], np.full((2, 3), [0, 255, 0]))
    np.testing.assert_array_equal(frame[3, 2:4], np.full((2, 3), [0, 255, 0]))
    assert np.all(mask[0:2] == 0.5)
    np.testing.assert_array_equal(mask[2, 2:4], 1.0)
    np.testing.assert_array_equal(mask[3, 2:4], 1.0)
    np.testing.assert_array_equal(mask[2:4, 0:2], 0.0)
    np.testing.assert_array_equal(frame[2:4, 0:2], 0)

    clip.get_frame(1.5)
    clip.get_frame(2.0)
    clip.mask.get_frame(2.0)
    assert generated == ["wide", "narrow", "narrow"]

    gap_frame = clip.get_frame(5.0)
    gap_mask = clip.mask.get_frame(5.0)
    assert gap_frame.shape == (1, 1, 3)
    assert gap_mask.shape == (1, 1)
    assert gap_mask[0, 0] == 0.0
    assert generated == ["wide", "narrow", "narrow"]
    assert clip.subtitles is subtitles


def test_stacked_mode_uses_local_animation_time_and_consistent_frame_mask():
    def make_textclip(text):
        def frame(t):
            value = 10 + int(round(t * 10))
            return np.full((2, 3, 3), [value, value, value], dtype=np.uint8)

        def mask(t):
            return np.full((2, 3), min(max(t / 2.0, 0.0), 1.0))

        clip = VideoClip(frame)
        clip.mask = VideoClip(mask, is_mask=True)
        return clip

    subtitles = [([1.0, 5.0], "animated")]
    frame_first = SubtitlesClip(subtitles, make_textclip=make_textclip, stacked=True)
    mask_first = SubtitlesClip(subtitles, make_textclip=make_textclip, stacked=True)

    frame = frame_first.get_frame(2.5)
    mask = frame_first.mask.get_frame(2.5)
    reverse_mask = mask_first.mask.get_frame(2.5)
    reverse_frame = mask_first.get_frame(2.5)

    np.testing.assert_array_equal(frame, reverse_frame)
    np.testing.assert_array_equal(mask, reverse_mask)
    np.testing.assert_array_equal(frame, 25)
    np.testing.assert_array_equal(mask, 0.75)
    assert frame.shape[:2] == mask.shape


def test_stacked_mode_accepts_srt_and_overlap_alias(util):
    def make_textclip(text):
        return ColorClip((2, 1), color=(255, 255, 255))

    clip = SubtitlesClip(
        "media/subtitles.srt",
        make_textclip=make_textclip,
        overlap=True,
    )

    assert clip.stacked is True
    assert clip.subtitles == MEDIA_SUBTITLES_DATA
    assert clip.get_frame(0.5).shape == (1, 2, 3)
    assert clip.mask.get_frame(0.5).shape == (1, 2)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"stacked": "yes"},
        {"stacked": 1},
        {"overlap": "yes"},
        {"overlap": 1},
    ],
)
def test_invalid_stacked_options_raise(kwargs):
    with pytest.raises(TypeError):
        SubtitlesClip(
            [([0.0, 1.0], "text")],
            make_textclip=lambda text: ColorClip((1, 1), color=(0, 0, 0)),
            **kwargs,
        )


def test_conflicting_stacked_options_raise():
    with pytest.raises(ValueError):
        SubtitlesClip(
            [([0.0, 1.0], "text")],
            make_textclip=lambda text: ColorClip((1, 1), color=(0, 0, 0)),
            stacked=True,
            overlap=False,
        )


def test_stacked_generator_errors_propagate():
    def make_textclip(text):
        raise RuntimeError("generator failed")

    clip = SubtitlesClip(
        [([0.0, 1.0], "text")],
        make_textclip=make_textclip,
        stacked=True,
    )

    with pytest.raises(RuntimeError, match="generator failed"):
        clip.get_frame(0.5)
    with pytest.raises(RuntimeError, match="generator failed"):
        clip.mask.get_frame(0.5)


def test_stacked_subtitles_export_and_readback(util):
    def make_textclip(text):
        specs = {
            "top": ((255, 0, 0), 40, True),
            "bottom": ((0, 255, 0), 20, False),
            "solo": ((255, 255, 0), 30, False),
        }
        color, width, animated_opacity = specs[text]

        def frame(t):
            return np.full((8, width, 3), color, dtype=np.uint8)

        clip = VideoClip(frame)
        if animated_opacity:
            clip.mask = VideoClip(
                lambda t: np.full((8, width), min(0.25 + 0.25 * t, 0.75)),
                is_mask=True,
            )
        return clip

    background = ColorClip((100, 80), color=(0, 0, 255)).with_duration(4.0)
    subtitles = SubtitlesClip(
        [
            ([0.0, 2.0], "top"),
            ([0.5, 2.5], "bottom"),
            ([3.0, 4.0], "solo"),
        ],
        make_textclip=make_textclip,
        stacked=True,
    )
    final = CompositeVideoClip(
        [background, subtitles.with_position((30, 20))],
        size=(100, 80),
        use_bgclip=True,
    )
    output = os.path.join(util.TMP_DIR, "stacked_subtitles_readback.mp4")
    final.write_videofile(output, fps=10, audio=False, logger=None)

    with VideoFileClip(output) as exported:
        early = exported.get_frame(0.25)
        overlap = exported.get_frame(0.75)
        late = exported.get_frame(1.75)
        gap = exported.get_frame(2.75)
        solo = exported.get_frame(3.5)

    top_early = early[22, 40].astype(float)
    top_overlap = overlap[22, 40].astype(float)
    top_late = late[22, 40].astype(float)
    bottom = overlap[32, 50].astype(float)
    bottom_side = overlap[32, 35].astype(float)

    assert top_late[0] > top_early[0] + 40
    assert top_late[2] < top_early[2] - 40
    assert top_overlap[1] < 40
    assert bottom[1] > 80
    np.testing.assert_allclose(bottom_side, [0, 0, 255], atol=25)
    np.testing.assert_allclose(gap[40, 50], [0, 0, 255], atol=10)
    np.testing.assert_allclose(solo[22, 45], [255, 255, 0], atol=35)


if __name__ == "__main__":
    pytest.main()
