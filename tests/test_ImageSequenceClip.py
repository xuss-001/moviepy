"""Image sequencing clip tests meant to be run with pytest."""

import os
from unittest import mock

import numpy as np
import pytest
from imageio.v2 import imread
from PIL import Image

from moviepy import VideoFileClip
from moviepy.video.io.ImageSequenceClip import ImageSequenceClip


def rgb_image(red, green, blue, size=(8, 8)):
    image = np.zeros((size[0], size[1], 3), dtype=np.uint8)
    image[:, :] = (red, green, blue)
    return image


def rgba_image(alpha, size=(8, 8)):
    image = np.zeros((size[0], size[1], 4), dtype=np.uint8)
    image[:, :, :3] = (255, 0, 0)
    image[:, :, 3] = alpha
    return image


@pytest.fixture
def mixed_images(tmp_path):
    """Writes a mixed RGB/RGBA sequence of equally sized PNG files."""
    images = [
        rgb_image(255, 0, 0),  # opaque red (RGB)
        rgba_image(64),  # semi-transparent red (RGBA)
        rgb_image(0, 255, 0),  # opaque green (RGB)
        rgba_image(192),  # mostly opaque red (RGBA)
    ]
    files = []
    for index, image in enumerate(images):
        path = tmp_path / "image_{:02d}.png".format(index)
        Image.fromarray(image).save(path)
        files.append(str(path))
    return files


@pytest.fixture
def mixed_durations():
    return [0.5, 1.0, 1.5, 0.25]


def test_1(util):
    images = []
    durations = []

    for i in range(5):
        durations.append(i)
        images.append("media/python_logo.png")
        durations.append(i)
        images.append("media/python_logo_upside_down.png")

    with ImageSequenceClip(images, durations=durations) as clip:
        assert clip.duration == sum(durations)
        clip.write_videofile(
            os.path.join(util.TMP_DIR, "ImageSequenceClip1.mp4"), fps=30, logger=None
        )


def test_2():
    images = []
    durations = []

    durations.append(1)
    images.append("media/python_logo.png")
    durations.append(2)
    images.append("media/matplotlib_demo1.png")

    # images are not the same size..
    with pytest.raises(Exception):
        ImageSequenceClip(images, durations=durations).close()


def test_mixed_rgba_sequence_switching_and_alpha(mixed_images, mixed_durations):
    with ImageSequenceClip(mixed_images, durations=mixed_durations) as clip:
        assert clip.duration == pytest.approx(3.25)
        assert clip.size == (8, 8)
        assert clip.mask is not None

        # Each image appears at its own start time, never earlier.
        assert np.array_equal(clip.get_frame(0.0)[0, 0], [255, 0, 0])
        assert np.array_equal(clip.mask.get_frame(0.0), np.ones((8, 8)))

        assert np.array_equal(clip.get_frame(0.5)[0, 0], [255, 0, 0])
        assert np.array_equal(
            clip.mask.get_frame(0.5), np.full((8, 8), 64 / 255.0)
        )

        # The switch happens exactly at the next image start.
        assert np.array_equal(clip.get_frame(0.9999)[0, 0], [255, 0, 0])
        assert np.array_equal(clip.get_frame(1.5)[0, 0], [0, 255, 0])
        assert np.array_equal(clip.mask.get_frame(1.5), np.ones((8, 8)))

        assert np.array_equal(clip.get_frame(3.0)[0, 0], [255, 0, 0])
        assert np.array_equal(
            clip.mask.get_frame(3.0), np.full((8, 8), 192 / 255.0)
        )
        assert np.array_equal(clip.get_frame(3.2499)[0, 0], [255, 0, 0])


def test_rgb_frame_and_mask_share_the_same_image(mixed_images, mixed_durations):
    """The mask must be derived from the same picture as the RGB frame."""
    with ImageSequenceClip(mixed_images, durations=mixed_durations) as clip:
        probe_times = [0.0, 0.49, 0.5, 1.49, 1.5, 2.99, 3.0, 3.24]
        for t in probe_times:
            index = clip._find_image_index(t)
            frame = clip.get_frame(t)
            mask = clip.mask.get_frame(t)
            expected_mask = clip._image_as_mask(clip._load_image(index))
            assert np.array_equal(mask, expected_mask)
            assert frame.shape[:2] == mask.shape


def test_backward_and_forward_seeking_never_shifts_frames(
    mixed_images, mixed_durations
):
    with ImageSequenceClip(mixed_images, durations=mixed_durations) as clip:
        expected = {
            0.0: [255, 0, 0],
            0.5: [255, 0, 0],
            1.5: [0, 255, 0],
            3.0: [255, 0, 0],
        }
        seek_order = [3.0, 0.0, 1.5, 0.5, 3.0, 0.5, 0.0, 1.5]
        for t in seek_order:
            assert np.array_equal(clip.get_frame(t)[0, 0], expected[t])

        frame_at_start = clip.get_frame(0.0).copy()
        for _ in range(3):
            clip.get_frame(3.24)
        assert np.array_equal(clip.get_frame(0.0), frame_at_start)


def test_zero_duration_images_are_skipped(mixed_images):
    with ImageSequenceClip(
        mixed_images, durations=[0.5, 0.0, 1.5, 0.0]
    ) as clip:
        assert clip.duration == pytest.approx(2.0)
        assert len(clip.sequence) == 2
        assert clip.images_starts.tolist() == pytest.approx([0.0, 0.5, 2.0])
        assert np.array_equal(clip.get_frame(0.0)[0, 0], [255, 0, 0])
        assert np.array_equal(clip.get_frame(0.5)[0, 0], [0, 255, 0])
        assert np.array_equal(clip.get_frame(1.99)[0, 0], [0, 255, 0])
        assert clip.mask is not None
        assert np.array_equal(clip.mask.get_frame(0.5), np.ones((8, 8)))


def test_all_zero_durations_raise(mixed_images):
    with pytest.raises(ValueError):
        ImageSequenceClip(mixed_images, durations=[0, 0, 0, 0])


def test_disk_sequence_decodes_each_frame_at_most_once(
    mixed_images, mixed_durations
):
    with ImageSequenceClip(mixed_images, durations=mixed_durations) as clip:
        # Construction already decoded and cached the first image to compute
        # the clip size.
        with mock.patch(
            "moviepy.video.io.ImageSequenceClip.imread", wraps=imread
        ) as mocked_imread:
            # Consecutive or repeated reads of the same frame, including the
            # mask at the same time, must not decode the same file twice.
            clip.get_frame(0.2)
            clip.get_frame(0.2)
            clip.mask.get_frame(0.2)
            clip.mask.get_frame(0.2)
            clip.get_frame(0.6)
            clip.get_frame(0.6)
            clip.mask.get_frame(0.6)
            decoded = [call.args[0] for call in mocked_imread.call_args_list]
            assert decoded == [mixed_images[1]]


def test_images_are_not_kept_in_memory_by_default(mixed_images, mixed_durations):
    with ImageSequenceClip(mixed_images, durations=mixed_durations) as clip:
        assert clip.fromfiles is True
        assert clip.sequence == mixed_images
        # Only one decoded picture is cached, never the whole sequence.
        assert clip.last_index == 0
        assert all(isinstance(image, str) for image in clip.sequence)


def test_load_images_keeps_the_whole_sequence_in_memory(
    mixed_images, mixed_durations
):
    with ImageSequenceClip(
        mixed_images, durations=mixed_durations, load_images=True
    ) as clip:
        assert clip.fromfiles is False
        assert all(isinstance(image, np.ndarray) for image in clip.sequence)
        with mock.patch(
            "moviepy.video.io.ImageSequenceClip.imread"
        ) as mocked_imread:
            clip.get_frame(0.0)
            clip.mask.get_frame(0.0)
            assert mocked_imread.call_count == 0


def test_folder_sequence(tmp_path, mixed_images, mixed_durations):
    with ImageSequenceClip(str(tmp_path), durations=mixed_durations) as clip:
        assert clip.duration == pytest.approx(3.25)
        assert clip.sequence == sorted(mixed_images)
        assert np.array_equal(clip.get_frame(1.5)[0, 0], [0, 255, 0])


def test_uint8_array_sequence():
    images = [rgb_image(255, 0, 0), rgba_image(128), rgb_image(0, 0, 255)]
    with ImageSequenceClip(images, durations=[1, 2, 1]) as clip:
        assert clip.fromfiles is False
        assert clip.duration == 4.0
        assert clip.mask is not None
        assert np.array_equal(clip.get_frame(0.0)[0, 0], [255, 0, 0])
        assert np.array_equal(
            clip.mask.get_frame(1.0), np.full((8, 8), 128 / 255.0)
        )
        assert np.array_equal(clip.get_frame(3.0)[0, 0], [0, 0, 255])
        assert np.array_equal(clip.mask.get_frame(3.0), np.ones((8, 8)))


def test_with_mask_false_ignores_alpha():
    images = [rgba_image(0), rgba_image(128)]
    with ImageSequenceClip(images, durations=[1, 1], with_mask=False) as clip:
        assert clip.mask is None
        assert clip.get_frame(0.0).shape == (8, 8, 3)


def test_is_mask_clip():
    images = [rgba_image(64), rgb_image(1, 2, 3)]
    with ImageSequenceClip(
        images, durations=[1, 1], is_mask=True
    ) as mask_clip:
        assert mask_clip.is_mask is True
        assert mask_clip.mask is None
        assert np.array_equal(
            mask_clip.get_frame(0.0), np.full((8, 8), 64 / 255.0)
        )
        # RGB images are fully opaque when used as a mask.
        assert np.array_equal(mask_clip.get_frame(1.0), np.ones((8, 8)))


def test_fps_takes_precedence_over_durations(mixed_images):
    with ImageSequenceClip(
        mixed_images, fps=2, durations=[10, 10, 10, 10]
    ) as clip:
        assert clip.fps == 2
        assert clip.duration == pytest.approx(2.0)
        assert clip.images_starts.tolist() == pytest.approx(
            [0.0, 0.5, 1.0, 1.5, 2.0]
        )


def test_fps_based_sequence_switches_at_frame_boundaries():
    images = [
        rgb_image(255, 0, 0),
        rgb_image(0, 255, 0),
        rgb_image(0, 0, 255),
    ]
    with ImageSequenceClip(images, fps=1) as clip:
        assert clip.duration == 3.0
        assert np.array_equal(clip.get_frame(0.99)[0, 0], [255, 0, 0])
        assert np.array_equal(clip.get_frame(1.0)[0, 0], [0, 255, 0])
        assert np.array_equal(clip.get_frame(2.0)[0, 0], [0, 0, 255])


def test_export_mixed_alpha_animation(util, mixed_images, mixed_durations):
    output_path = os.path.join(util.TMP_DIR, "ImageSequenceClip_mixed.mp4")
    with ImageSequenceClip(mixed_images, durations=mixed_durations) as clip:
        clip.write_videofile(output_path, fps=10, logger=None)
        rendered_times = list(np.arange(0, clip.duration - 1e-9, 1.0 / 10))
        rendered_frames = [clip.get_frame(t) for t in rendered_times]

    with VideoFileClip(output_path) as rendered:
        assert rendered.duration == pytest.approx(3.25, abs=0.05)
        assert list(rendered.size) == [8, 8]
        for index, t in enumerate(rendered_times):
            frame = rendered.get_frame(min(t, rendered.duration - 1e-6))
            assert np.allclose(frame, rendered_frames[index], atol=15)


def test_neither_fps_nor_durations_raises(mixed_images):
    with pytest.raises(ValueError):
        ImageSequenceClip(mixed_images)


@pytest.mark.parametrize("bad_fps", [0, -1, float("nan"), float("inf")])
def test_invalid_fps_raises(mixed_images, bad_fps):
    with pytest.raises(ValueError):
        ImageSequenceClip(mixed_images, fps=bad_fps)


def test_empty_list_raises():
    with pytest.raises(ValueError):
        ImageSequenceClip([], fps=1)


def test_empty_folder_raises(tmp_path):
    with pytest.raises(ValueError):
        ImageSequenceClip(str(tmp_path), fps=1)


def test_size_mismatch_between_arrays_raises():
    images = [
        rgb_image(1, 2, 3, size=(8, 8)),
        rgb_image(4, 5, 6, size=(6, 6)),
    ]
    with pytest.raises(ValueError):
        ImageSequenceClip(images, durations=[1, 1])


def test_durations_count_mismatch_raises(mixed_images):
    with pytest.raises(ValueError):
        ImageSequenceClip(mixed_images, durations=[1, 1, 1])


@pytest.mark.parametrize(
    "bad_durations",
    [
        [1, -1, 1, 1],
        [1, float("nan"), 1, 1],
        [1, float("inf"), 1, 1],
        [1, "1", 1, 1],
    ],
)
def test_invalid_durations_raises(mixed_images, bad_durations):
    with pytest.raises(ValueError):
        ImageSequenceClip(mixed_images, durations=bad_durations)


def test_durations_must_be_iterable_raises(mixed_images):
    with pytest.raises(ValueError):
        ImageSequenceClip(mixed_images, durations=4)


if __name__ == "__main__":
    pytest.main()
