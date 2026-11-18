"""Image sequencing clip tests meant to be run with pytest."""

import os

import numpy as np

import pytest
from imageio.v2 import imwrite

import moviepy.video.io.ImageSequenceClip as isc_module
from moviepy.video.io.ImageSequenceClip import ImageSequenceClip


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


def _rgb_image(color, size=(24, 16)):
    """Solid RGB image (HxWx3 uint8) of the given ``(height, width)`` size."""
    image = np.zeros((size[0], size[1], 3), dtype=np.uint8)
    image[:, :] = color
    return image


def _rgba_image(color, alpha, size=(24, 16)):
    """Solid RGBA image (HxWx4 uint8) of the given ``(height, width)`` size."""
    image = np.zeros((size[0], size[1], 4), dtype=np.uint8)
    image[:, :, :3] = color
    image[:, :, 3] = alpha
    return image


def _write_mixed_sequence(folder):
    """Write a mixed RGB/RGBA sequence of PNG files, return the file names."""
    images = [
        _rgb_image((255, 0, 0)),
        _rgba_image((0, 255, 0), alpha=128),
        _rgb_image((0, 0, 255)),
        _rgba_image((255, 255, 0), alpha=64),
    ]
    filenames = []
    for i, image in enumerate(images):
        filename = os.path.join(folder, f"image_{i}.png")
        imwrite(filename, image)
        filenames.append(filename)
    return filenames


def test_mixed_rgb_rgba_arrays():
    """Mixed RGB/RGBA arrays: RGB is opaque, RGBA keeps its transparency."""
    images = [
        _rgb_image((255, 0, 0)),
        _rgba_image((0, 255, 0), alpha=128),
        _rgb_image((0, 0, 255)),
    ]
    with ImageSequenceClip(images, durations=[0.5, 1.0, 0.5]) as clip:
        assert clip.duration == 2.0
        assert clip.mask is not None

        # frames are always RGB, whatever the input channel count
        assert clip.get_frame(0.0).shape == (24, 16, 3)
        assert np.array_equal(clip.get_frame(0.0)[0, 0], (255, 0, 0))
        assert np.array_equal(clip.get_frame(0.6)[0, 0], (0, 255, 0))
        assert np.array_equal(clip.get_frame(1.6)[0, 0], (0, 0, 255))

        # mask: RGB images are opaque, RGBA images keep their alpha
        assert clip.mask.get_frame(0.0).shape == (24, 16)
        assert np.allclose(clip.mask.get_frame(0.0), 1.0)
        assert np.allclose(clip.mask.get_frame(0.6), 128 / 255.0)
        assert np.allclose(clip.mask.get_frame(1.6), 1.0)


def test_switching_times_with_durations():
    """Each image appears at its start time, never before."""
    images = [
        _rgb_image((255, 0, 0)),
        _rgb_image((0, 255, 0)),
        _rgb_image((0, 0, 255)),
    ]
    durations = [0.5, 0.25, 0.75]
    with ImageSequenceClip(images, durations=durations) as clip:
        assert clip.duration == 1.5
        assert list(clip.images_starts) == [0, 0.5, 0.75]

        # just before a switch, the previous image is still displayed
        assert np.array_equal(clip.get_frame(0.49)[0, 0], (255, 0, 0))
        assert np.array_equal(clip.get_frame(0.74)[0, 0], (0, 255, 0))
        # exactly at a start time, the new image is displayed
        assert np.array_equal(clip.get_frame(0.5)[0, 0], (0, 255, 0))
        assert np.array_equal(clip.get_frame(0.75)[0, 0], (0, 0, 255))
        # last image stays until the end of the clip
        assert np.array_equal(clip.get_frame(1.49)[0, 0], (0, 0, 255))


def test_zero_duration_images_are_skipped():
    images = [
        _rgb_image((255, 0, 0)),
        _rgb_image((0, 255, 0)),
        _rgb_image((0, 0, 255)),
    ]
    with ImageSequenceClip(images, durations=[0.5, 0.0, 0.5]) as clip:
        assert clip.duration == 1.0
        # the zero-duration image never shows up, even exactly at its start
        assert np.array_equal(clip.get_frame(0.5)[0, 0], (0, 0, 255))
        for t in np.linspace(0, 0.99, 20):
            assert not np.array_equal(clip.get_frame(t)[0, 0], (0, 255, 0))


def test_fps_takes_precedence_over_durations():
    images = [_rgb_image((i, 0, 0)) for i in (10, 20, 30, 40)]
    with ImageSequenceClip(images, fps=2, durations=[10, 10, 10, 10]) as clip:
        assert clip.duration == 2.0
        assert list(clip.images_starts) == [0, 0.5, 1.0, 1.5]
        assert np.array_equal(clip.get_frame(0.6)[0, 0], (20, 0, 0))


def test_without_mask_ignores_alpha():
    images = [_rgba_image((0, 255, 0), alpha=128)]
    with ImageSequenceClip(images, fps=1, with_mask=False) as clip:
        assert clip.mask is None
        frame = clip.get_frame(0)
        assert frame.shape == (24, 16, 3)
        assert np.array_equal(frame[0, 0], (0, 255, 0))


def test_frame_and_mask_stay_in_sync_when_seeking(tmp_path):
    """Frame and mask come from the same image, even when seeking back
    and forth in time."""
    filenames = _write_mixed_sequence(str(tmp_path))
    durations = [0.5, 0.25, 0.75, 0.5]
    expected = [
        ((255, 0, 0), 1.0),
        ((0, 255, 0), 128 / 255.0),
        ((0, 0, 255), 1.0),
        ((255, 255, 0), 64 / 255.0),
    ]
    with ImageSequenceClip(filenames, durations=durations) as clip:
        assert clip.mask is not None
        # visit the images out of order, then back and forth again
        times = [1.6, 0.1, 0.6, 1.0, 0.2, 1.7, 0.55, 0.0]
        for t in times:
            index = max(i for i, start in enumerate(clip.images_starts) if start <= t)
            color, alpha = expected[index]
            assert np.array_equal(clip.get_frame(t)[0, 0], color), f"t={t}"
            assert np.allclose(clip.mask.get_frame(t), alpha), f"t={t}"


def test_disk_sequence_does_not_redecode_same_image(tmp_path, monkeypatch):
    """Repeated or consecutive reads of a same picture decode it only once."""
    filenames = _write_mixed_sequence(str(tmp_path))

    reads = []
    real_imread = isc_module.imread

    def counting_imread(file, *args, **kwargs):
        reads.append(file)
        return real_imread(file, *args, **kwargs)

    monkeypatch.setattr(isc_module, "imread", counting_imread)
    with ImageSequenceClip(filenames, fps=2) as clip:
        reads.clear()  # ignore the size checks/decodes done at init time
        clip.get_frame(0.6)
        clip.get_frame(0.7)
        clip.get_frame(0.9)  # still the second picture (fps=2)
        assert len(reads) == 1
        clip.get_frame(1.1)  # third picture
        assert len(reads) == 2
        clip.get_frame(0.6)  # seeking back decodes again, but only once
        assert len(reads) == 3
        clip.get_frame(0.7)
        assert len(reads) == 3
        # only the last decoded image is kept in memory
        assert clip.last_image is not None


def test_load_images_still_supported(tmp_path):
    filenames = _write_mixed_sequence(str(tmp_path))
    with ImageSequenceClip(filenames, fps=2, load_images=True) as clip:
        assert not clip.fromfiles
        assert clip.mask is not None
        assert np.array_equal(clip.get_frame(0.6)[0, 0], (0, 255, 0))
        assert np.allclose(clip.mask.get_frame(0.6), 128 / 255.0)


def test_invalid_arguments_raise_errors(tmp_path):
    images = [_rgb_image((255, 0, 0)), _rgb_image((0, 0, 255))]

    # empty sequences
    with pytest.raises(ValueError):
        ImageSequenceClip([], fps=1)
    with pytest.raises(ValueError):
        ImageSequenceClip(str(tmp_path), fps=1)

    # neither fps nor durations
    with pytest.raises(ValueError):
        ImageSequenceClip(images)

    # invalid fps
    for fps in (0, -2, float("inf"), float("nan")):
        with pytest.raises(ValueError):
            ImageSequenceClip(images, fps=fps)

    # durations must match the number of images
    with pytest.raises(ValueError):
        ImageSequenceClip(images, durations=[1.0])
    with pytest.raises(ValueError):
        ImageSequenceClip(images, durations=[1.0, 1.0, 1.0])

    # durations must be finite and non-negative
    for durations in ([1.0, -1.0], [1.0, float("inf")], [1.0, float("nan")]):
        with pytest.raises(ValueError):
            ImageSequenceClip(images, durations=durations)

    # total duration must be greater than zero
    with pytest.raises(ValueError):
        ImageSequenceClip(images, durations=[0.0, 0.0])

    # images of different sizes
    with pytest.raises(Exception):
        ImageSequenceClip(
            [_rgb_image((255, 0, 0)), _rgb_image((0, 0, 255), size=(8, 8))],
            durations=[1.0, 1.0],
        )


def test_mixed_sequence_export(util, tmp_path):
    """Export a short animation mixing RGB/RGBA images and uneven durations,
    and check switching times, transparency and back-and-forth reads."""
    filenames = _write_mixed_sequence(str(tmp_path))
    durations = [0.5, 0.25, 0.5, 0.25]
    with ImageSequenceClip(filenames, durations=durations) as clip:
        assert clip.duration == 1.5
        assert clip.mask is not None

        # switching times: no image shows up before its start time
        checks = [
            (0.0, (255, 0, 0), 1.0),
            (0.49, (255, 0, 0), 1.0),
            (0.5, (0, 255, 0), 128 / 255.0),
            (0.74, (0, 255, 0), 128 / 255.0),
            (0.75, (0, 0, 255), 1.0),
            (1.24, (0, 0, 255), 1.0),
            (1.25, (255, 255, 0), 64 / 255.0),
        ]
        # read forward, then backward, then forward again
        for t, color, alpha in checks + checks[::-1] + checks:
            assert np.array_equal(clip.get_frame(t)[0, 0], color), f"t={t}"
            assert np.allclose(clip.mask.get_frame(t), alpha), f"t={t}"

        output = os.path.join(util.TMP_DIR, "ImageSequenceClip_mixed.mp4")
        clip.write_videofile(output, fps=10, logger=None)
        assert os.path.exists(output)


if __name__ == "__main__":
    pytest.main()
