"""Compositing tests for use with pytest."""

import os

import numpy as np

import pytest

from moviepy import *


class ClipPixelTest:
    ALLOWABLE_COLOR_VARIATION = 3  # from 0-767: how much mismatch do we allow

    def __init__(self, clip):
        self.clip = clip

    def expect_color_at(self, ts, expected, xy=[0, 0]):
        frame = self.clip.frame_function(ts)
        r, g, b = expected
        actual = frame[xy[1]][xy[0]]
        diff = abs(actual[0] - r) + abs(actual[1] - g) + abs(actual[2] - b)

        mismatch = diff > ClipPixelTest.ALLOWABLE_COLOR_VARIATION
        assert (
            not mismatch
        ), "Expected (%02x,%02x,%02x) but got (%02x,%02x,%02x) at timestamp %s" % (
            *expected,
            *actual,
            ts,
        )


def _tone_audio(value, duration, fps=8000):
    def frame_function(t):
        if isinstance(t, np.ndarray):
            return np.full((len(t), 1), value, dtype=float)
        return np.array([value])

    return AudioClip(frame_function, duration=duration, fps=fps)


def _gap_test_clips():
    first = (
        ColorClip((8, 6), color=(255, 0, 0), duration=1)
        .with_fps(4)
        .with_audio(_tone_audio(0.1, 1.5))
    )
    second = (
        ColorClip((4, 2), color=(0, 255, 0, 128), duration=2)
        .with_fps(4)
        .with_audio(_tone_audio(0.2, 2))
    )
    third = (
        ColorClip((6, 4), color=(0, 0, 255, 128), duration=1)
        .with_fps(4)
        .with_audio(_tone_audio(0.3, 1))
    )
    return [first, second, third]


def _clip_state(clip):
    mask = clip.mask
    audio = clip.audio
    return (
        clip.start,
        clip.end,
        clip.duration,
        clip.pos(0),
        clip.layer_index,
        clip.get_frame(0).tobytes(),
        None
        if mask is None
        else (
            mask.start,
            mask.end,
            mask.duration,
            mask.pos(0),
            mask.layer_index,
        ),
        None
        if audio is None
        else (audio.start, audio.end, audio.duration),
    )


def test_clips_array(util):
    red = ColorClip((1024, 800), color=(255, 0, 0))
    green = ColorClip((1024, 800), color=(0, 255, 0))
    blue = ColorClip((1024, 800), color=(0, 0, 255))

    video = clips_array([[red, green, blue]])

    with pytest.raises(ValueError):  # duration not set
        video.with_effects([vfx.Resize(width=480)]).write_videofile(
            os.path.join(util.TMP_DIR, "test_clips_array.mp4")
        )


def test_clips_array_duration(util):
    filename = os.path.join(util.TMP_DIR, "test_clips_array.mp4")

    # NOTE: anyone knows what behaviour this sets ? If yes please replace
    # this comment.
    red = ColorClip((256, 200), color=(255, 0, 0))
    green = ColorClip((256, 200), color=(0, 255, 0))
    blue = ColorClip((256, 200), color=(0, 0, 255))

    video = clips_array([[red, green, blue]]).with_duration(5)
    with pytest.raises(AttributeError):  # fps not set
        video.write_videofile(filename)

    # this one should work correctly
    red.fps = green.fps = blue.fps = 30
    video = clips_array([[red, green, blue]]).with_duration(5)
    video.write_videofile(filename)


def test_concatenate_self(util):
    clip = BitmapClip([["AAA", "BBB"], ["CCC", "DDD"]], fps=1)
    target = BitmapClip([["AAA", "BBB"], ["CCC", "DDD"]], fps=1)

    concatenated = concatenate_videoclips([clip])

    concatenated.write_videofile(
        os.path.join(util.TMP_DIR, "test_concatenate_self.mp4")
    )
    assert concatenated == target


def test_concatenate_floating_point(util):
    """
    >>> print("{0:.20f}".format(1.12))
    1.12000000000000010658

    This test uses duration=1.12 to check that it still works when the clip
    duration is represented as being bigger than it actually is. Fixed in #1195.
    """
    clip = ColorClip([100, 50], color=[255, 128, 64], duration=1.12).with_fps(25.0)
    concat = concatenate_videoclips([clip])
    concat.write_videofile(os.path.join(util.TMP_DIR, "concat.mp4"), preset="ultrafast")


def test_blit_with_opacity():
    # has one second R, one second G, one second B
    size = (2, 2)
    clip1 = (
        ColorClip(size, color=(255, 0, 0), duration=1)
        + ColorClip(size, color=(0, 255, 0), duration=1)
        + ColorClip(size, color=(0, 0, 255), duration=1)
    )

    # overlay green at half opacity during first 2 sec
    clip2 = ColorClip(size, color=(0, 255, 0), duration=2).with_opacity(0.5)
    composite = CompositeVideoClip([clip1, clip2])
    bt = ClipPixelTest(composite)

    # red + 50% green
    bt.expect_color_at(0.5, (0x7F, 0x7F, 0x00))
    # green + 50% green
    bt.expect_color_at(1.5, (0x00, 0xFF, 0x00))
    # blue is after 2s, so keep untouched
    bt.expect_color_at(2.5, (0x00, 0x00, 0xFF))


def test_compositing_masks(util):
    # Has one R 30%, one G 30%, one B 30%
    clip1 = ColorClip((100, 100), (255, 0, 0, 76.5)).with_duration(2)
    clip2 = ColorClip((50, 50), (0, 255, 0, 76.5)).with_duration(2)
    clip3 = ColorClip((25, 25), (0, 0, 255, 76.5)).with_duration(2)

    compostite_clip1 = CompositeVideoClip(
        [clip1, clip2.with_position(("center", "center"))],
        bg_color=None,
    )
    compostite_clip2 = CompositeVideoClip(
        [compostite_clip1, clip3.with_position(("center", "center"))],
        bg_color=None,
    )

    # Load output file and check transparency
    frame = compostite_clip2.mask.get_frame(1)

    # We check opacity with one, two and three layers
    # Allow for a bit of tolerance (about 1%) to account
    # for rounding errors
    opacity1 = frame[50, 10]
    opacity2 = frame[50, 30]
    opacity3 = frame[50, 50]
    assert abs(opacity1 - 0.3) < 0.01
    assert abs(opacity2 - 0.51) < 0.01
    assert abs(opacity3 - 0.657) < 0.01


def test_compositing_with_transparency_colors(util):
    # Has one R 30%, one G 30%, one B 30%
    clip1 = ColorClip((100, 100), (255, 0, 0, 76.5)).with_duration(2)
    clip2 = ColorClip((50, 50), (0, 255, 0, 76.5)).with_duration(2)
    clip3 = ColorClip((25, 25), (0, 0, 255, 76.5)).with_duration(2)

    compostite_clip1 = CompositeVideoClip(
        [clip1, clip2.with_position(("center", "center"))],
        bg_color=None,  # No background color, so it should be transparent
    )
    compostite_clip2 = CompositeVideoClip(
        [compostite_clip1, clip3.with_position(("center", "center"))],
        bg_color=None,
    )

    # Load output file and check transparency
    frame = compostite_clip2.get_frame(1)
    mask = compostite_clip2.mask.get_frame(1)

    # We check color with 1 layer
    # We add a bit of tolerance (about 1%) to account
    # For possible rounding errors
    color1 = frame[50, 10]
    opacity1 = mask[50, 10]
    assert np.allclose(color1, [255, 0, 0], atol=1)
    assert abs(opacity1 - 0.3) < 0.01

    # With 2 layers
    color2 = frame[50, 30]
    opacity2 = mask[50, 30]
    assert np.allclose(color2, [105, 150, 0], atol=1)
    assert abs(opacity2 - 0.51) < 0.01

    # With 3 layers
    color3 = frame[50, 50]
    opacity3 = mask[50, 50]
    assert np.allclose(color3, [56, 81, 116], atol=1)
    assert abs(opacity3 - 0.657) < 0.01


def test_slide_in():
    duration = 0.1
    size = (10, 1)
    fps = 10
    color = (255, 0, 0)

    # left and right sides
    clip = ColorClip(
        color=color,
        duration=duration,
        size=size,
    ).with_fps(fps)

    for side in ["left", "right"]:
        new_clip = CompositeVideoClip(
            [clip.with_effects([vfx.SlideIn(duration, side)])]
        )

        for t in np.arange(0, duration, duration / fps):
            n_reds, n_reds_expected = (0, int(t * 100))

            if t:
                assert n_reds_expected

            if n_reds_expected == 7:  # skip 7 due to inaccurate frame
                continue

            for r, g, b in new_clip.get_frame(t)[0]:
                if r == color[0] and g == color[1] and g == color[2]:
                    n_reds += 1

            assert n_reds == n_reds_expected

    # top and bottom sides
    clip = ColorClip(
        color=color,
        duration=duration,
        size=(size[1], size[0]),
    ).with_fps(fps)

    for side in ["top", "bottom"]:
        new_clip = CompositeVideoClip(
            [clip.with_effects([vfx.SlideIn(duration, side)])]
        )
        for t in np.arange(0, duration, duration / fps):
            n_reds, n_reds_expected = (0, int(t * 100))

            if t:
                assert n_reds_expected

            if n_reds_expected == 7:  # skip 7 due to inaccurate frame
                continue

            for row in new_clip.get_frame(t):
                r, g, b = row[0]

                if r == color[0] and g == color[1] and g == color[2]:
                    n_reds += 1

            assert n_reds == n_reds_expected


def test_slide_out():
    duration = 0.1
    size = (11, 1)
    fps = 10
    color = (255, 0, 0)

    # left and right sides
    clip = ColorClip(
        color=color,
        duration=duration,
        size=size,
    ).with_fps(fps)

    for side in ["left", "right"]:
        new_clip = CompositeVideoClip(
            [clip.with_effects([vfx.SlideOut(duration, side)])]
        )

        for t in np.arange(0, duration, duration / fps):
            n_reds, n_reds_expected = (0, round(11 - t * 100, 6))

            if t:
                assert n_reds_expected

            for r, g, b in new_clip.get_frame(t)[0]:
                if r == color[0] and g == color[1] and g == color[2]:
                    n_reds += 1

            assert n_reds == n_reds_expected

    # top and bottom sides
    clip = ColorClip(
        color=color,
        duration=duration,
        size=(size[1], size[0]),
    ).with_fps(fps)

    for side in ["top", "bottom"]:
        new_clip = CompositeVideoClip(
            [clip.with_effects([vfx.SlideOut(duration, side)])]
        )
        for t in np.arange(0, duration, duration / fps):
            n_reds, n_reds_expected = (0, round(11 - t * 100, 6))

            if t:
                assert n_reds_expected

            for row in new_clip.get_frame(t):
                r, g, b = row[0]

                if r == color[0] and g == color[1] and g == color[2]:
                    n_reds += 1

            assert n_reds == n_reds_expected


def test_concatenate_with_masks(util):
    video_without_mask = (
        ColorClip(size=(10, 10), color=(255, 0, 0)).with_duration(1).with_fps(1)
    )
    video_with_mask = (
        ColorClip(size=(5, 5), color=(0, 255, 0))
        .with_duration(1)
        .with_fps(1)
        .with_mask()
    )

    output = os.path.join(util.TMP_DIR, "test_concatenate_with_masks.mp4")
    concatenate_videoclips([video_without_mask, video_with_mask]).write_videofile(
        output
    )


def test_concatenate_with_gaps_timeline_video_mask_and_audio():
    clips = _gap_test_clips()
    original_states = [_clip_state(clip) for clip in clips]

    result = concatenate_videoclips(clips, gaps=[0.5, -0.4], bg_color=None)

    assert result.size == (8, 6)
    assert result.duration == pytest.approx(4.1)
    assert result.end == pytest.approx(4.1)
    assert result.timings == pytest.approx([0, 1.5, 3.1, 4.1])
    assert result.start_times == pytest.approx([0, 1.5, 3.1])
    assert result.mask is not None
    assert result.mask.duration == pytest.approx(4.1)
    assert result.mask.end == pytest.approx(4.1)

    # The first clip ends at 1.0 and the second starts at 1.5. The interval is
    # transparent when no background color is requested and contains no audio.
    frame = result.get_frame(1.2)
    mask = result.mask.get_frame(1.2)
    assert np.all(mask == 0)
    np.testing.assert_array_equal(frame, np.zeros_like(frame))
    assert result.audio.get_frame(1.2)[0] == pytest.approx(0)

    colored_gap = concatenate_videoclips(
        _gap_test_clips(), gaps=[0.5, -0.4], bg_color=(1, 2, 3)
    )
    assert colored_gap.mask is None
    np.testing.assert_array_equal(colored_gap.get_frame(1.2)[0, 0], [1, 2, 3])

    # A smaller, semi-transparent clip is centered on the fixed canvas.
    frame = result.get_frame(1.75)
    mask = result.mask.get_frame(1.75)
    np.testing.assert_allclose(frame[3, 4], [0, 255, 0], atol=1)
    assert mask[3, 4] == pytest.approx(128 / 255, abs=0.01)
    assert result.audio.get_frame(1.75)[0] == pytest.approx(0.2)

    # The third clip starts at 3.1, while the second still plays until 3.5.
    # The later clip is on top and both alpha values are composited correctly.
    frame = result.get_frame(3.2)
    mask = result.mask.get_frame(3.2)
    np.testing.assert_allclose(frame[3, 4], [0, 85, 170], atol=2)
    assert mask[3, 4] == pytest.approx(0.752, abs=0.01)
    assert result.audio.get_frame(3.2)[0] == pytest.approx(0.5)

    # The total duration ends with the last clip, without trailing blank time.
    frame = result.get_frame(4.05)
    mask = result.mask.get_frame(4.05)
    np.testing.assert_allclose(frame[3, 4], [0, 0, 255], atol=1)
    assert mask[3, 4] == pytest.approx(128 / 255, abs=0.01)
    assert result.audio.get_frame(4.05)[0] == pytest.approx(0.3)

    assert [_clip_state(clip) for clip in clips] == original_states


@pytest.mark.parametrize(
    "case",
    [
        "transition",
        "padding",
        "wrong_count",
        "not_a_sequence",
        "nan",
        "infinity",
        "overlap_previous",
        "overlap_next",
    ],
)
def test_concatenate_with_gaps_rejects_invalid_combinations(case):
    clips = [
        ColorClip((4, 4), color=(255, 0, 0), duration=1),
        ColorClip((4, 4), color=(0, 255, 0), duration=1),
        ColorClip((4, 4), color=(0, 0, 255), duration=1),
    ]

    if case == "transition":
        kwargs = {
            "gaps": [0, 0],
            "transition": ColorClip((4, 4), color=(0, 0, 0), duration=0.1),
        }
    elif case == "padding":
        kwargs = {"gaps": [0, 0], "padding": 0.1}
    elif case == "wrong_count":
        kwargs = {"gaps": [0]}
    elif case == "not_a_sequence":
        kwargs = {"gaps": 0}
    elif case == "nan":
        kwargs = {"gaps": [np.nan, 0]}
    elif case == "infinity":
        kwargs = {"gaps": [0, np.inf]}
    elif case == "overlap_previous":
        kwargs = {"gaps": [-1, 0]}
    else:
        kwargs = {"gaps": [0, -1]}

    with pytest.raises(ValueError):
        concatenate_videoclips(clips, **kwargs)


def test_concatenate_padding_and_transition_compatibility():
    first = ColorClip((8, 6), color=(255, 0, 0), duration=1).with_fps(2)
    second = ColorClip((4, 2), color=(0, 255, 0), duration=1).with_fps(2)

    chained = concatenate_videoclips([first, second])
    assert not isinstance(chained, CompositeVideoClip)
    assert chained.duration == pytest.approx(2)
    assert chained.mask is None

    padded = concatenate_videoclips([first, second], padding=0.25)
    assert isinstance(padded, CompositeVideoClip)
    assert padded.size == (8, 6)
    assert padded.duration == pytest.approx(2.25)
    assert padded.start_times == pytest.approx([0, 1.25])

    overlapped = concatenate_videoclips([first, second], padding=-0.25)
    assert isinstance(overlapped, CompositeVideoClip)
    assert overlapped.duration == pytest.approx(1.75)

    transition = ColorClip((4, 2), color=(0, 0, 0), duration=0.5)
    with_transition = concatenate_videoclips([first, second], transition=transition)
    assert with_transition.duration == pytest.approx(2.5)


def test_concatenate_with_gaps_exports_video_with_audio(util):
    clips = _gap_test_clips()
    result = concatenate_videoclips(clips, gaps=[0.5, -0.4], bg_color=None)
    output = os.path.join(util.TMP_DIR, "test_concatenate_with_gaps.webm")

    result.write_videofile(
        output,
        fps=4,
        codec="libvpx",
        audio_codec="libvorbis",
        bitrate="50k",
        audio_bitrate="32k",
        logger=None,
    )

    exported = VideoFileClip(output)
    try:
        assert exported.size == [8, 6]
        assert exported.duration == pytest.approx(result.duration, abs=0.1)
        assert exported.audio is not None
        assert exported.audio.duration == pytest.approx(result.duration, abs=0.1)
    finally:
        exported.close()


if __name__ == "__main__":
    pytest.main()
