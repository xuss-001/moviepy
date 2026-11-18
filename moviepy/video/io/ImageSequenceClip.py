"""Implements ImageSequenceClip, a class to create a video clip from a set
of image files.
"""

import os
from bisect import bisect_right

import numpy as np
from imageio.v2 import imread

from moviepy.video.VideoClip import VideoClip


class ImageSequenceClip(VideoClip):
    """A VideoClip made from a series of images.

    Parameters
    ----------

    sequence
      Can be one of these:

      - The name of a folder (containing only pictures). The pictures
        will be considered in alphanumerical order.
      - A list of names of image files. In this case you can choose to
        load the pictures in memory pictures
      - A list of Numpy arrays representing images.

      The images may mix RGB and RGBA pictures, as long as they all share
      the same width and height. RGB pictures are treated as fully opaque,
      while the transparency of RGBA pictures is preserved in the clip's
      mask (unless ``with_mask=False``).

    fps
      Number of picture frames to read per second. Instead, you can provide
      the duration of each image with durations (see below). If both ``fps``
      and ``durations`` are provided, ``fps`` takes precedence.

    durations
      List of the duration of each picture. Must contain exactly one finite,
      non-negative duration per picture, and the total duration must be
      greater than zero. Each picture appears at its start time (the sum of
      the previous durations) and is replaced by the next picture exactly
      when the next one starts. Pictures with a duration of 0 are skipped.

    with_mask
      Should the alpha layer of PNG images be considered as a mask ?

    is_mask
      Will this sequence of pictures be used as an animated mask.

    load_images
      Specify that all images should be loaded into the RAM. This is only
      interesting if you have a small number of images that will be used
      more than once.

    Notes
    -----

    When the sequence is read from files, the pictures are decoded lazily,
    one at a time, and only the last decoded picture is kept in memory, so
    reading the same frame repeatedly (or reading consecutive frames of a
    same picture) does not decode the file again. The clip's picture and its
    mask always come from the same image at any given time, including when
    seeking back and forth.
    """

    def __init__(
        self,
        sequence,
        fps=None,
        durations=None,
        with_mask=True,
        is_mask=False,
        load_images=False,
    ):
        if (fps is None) and (durations is None):
            raise ValueError("Please provide either 'fps' or 'durations'.")

        if fps is not None and (not np.isfinite(fps) or fps <= 0):
            raise ValueError(
                f"'fps' must be a finite number greater than 0, got {fps!r}."
            )

        VideoClip.__init__(self, is_mask=is_mask)

        # Parse the data

        self.fromfiles = True

        if isinstance(sequence, list):
            if len(sequence) == 0:
                raise ValueError(
                    "ImageSequenceClip requires at least one image, "
                    "got an empty sequence."
                )
            if isinstance(sequence[0], str):
                if load_images:
                    sequence = [imread(file) for file in sequence]
                    self.fromfiles = False
            else:
                # sequence is already a list of numpy arrays
                self.fromfiles = False
        else:
            # sequence is a folder name, make it a list of files:
            folder = sequence
            sequence = sorted(
                [os.path.join(folder, file) for file in os.listdir(folder)]
            )
            if len(sequence) == 0:
                raise ValueError(
                    f"ImageSequenceClip found no image in folder {folder!r}."
                )

        # Check that all the images have the same width and height, and
        # record which images carry an alpha (transparency) channel, so
        # that sequences mixing RGB and RGBA pictures are supported.
        size = None
        self._has_alpha = []
        for image in sequence:
            array = imread(image) if isinstance(image, str) else image
            image_size = array.shape[:2]
            if size is None:
                size = image_size
            elif image_size != size:
                raise ValueError(
                    "MoviePy: ImageSequenceClip requires all images to be "
                    "the same size"
                )
            self._has_alpha.append(array.ndim == 3 and array.shape[2] == 4)

        if fps is not None:
            # 'fps' takes precedence over 'durations' when both are given.
            durations = [1.0 / fps for _ in sequence]
        else:
            durations = list(durations)
            if len(durations) != len(sequence):
                raise ValueError(
                    "ImageSequenceClip requires exactly one duration per "
                    f"image, got {len(durations)} durations for "
                    f"{len(sequence)} images."
                )
            for duration in durations:
                if not np.isfinite(duration) or duration < 0:
                    raise ValueError(
                        "ImageSequenceClip durations must be finite, "
                        f"non-negative numbers, got {duration!r}."
                    )

        self.fps = fps
        self.durations = durations
        self.duration = sum(durations)
        if self.duration <= 0:
            raise ValueError(
                "ImageSequenceClip requires a total duration greater than 0."
            )
        self.end = self.duration
        self.sequence = sequence

        # Start time of each picture: picture i appears exactly at
        # images_starts[i] and is replaced by picture i+1 at
        # images_starts[i+1]. Pictures with a zero duration share their
        # start time with the next picture and are therefore skipped.
        if fps is not None:
            self.images_starts = [1.0 * i / fps for i in range(len(sequence))]
        else:
            self.images_starts = [0] + list(np.cumsum(durations))[:-1]

        if fps is None:
            self.fps = len(sequence) / self.duration

        if self.fromfiles:
            self.last_index = None
            self.last_image = None

        if with_mask and any(self._has_alpha):
            # The mask shares the sequence and the timing of the clip, so
            # the picture and its transparency always come from the same
            # image at any given time.
            self.mask = ImageSequenceClip(
                sequence=sequence,
                fps=fps,
                durations=durations,
                with_mask=False,
                is_mask=True,
                load_images=load_images,
            )

        self.size = self.frame_function(0).shape[:2][::-1]

    def _find_image_index(self, t):
        # bisect_right inserts after any start time equal to ``t``, so a
        # picture starting exactly at ``t`` is selected, while pictures
        # with a zero duration (same start as the next picture) are
        # skipped.
        return max(0, bisect_right(self.images_starts, t) - 1)

    def frame_function(self, t):
        """Retrieves the frame corresponding to the given time `t`.

        Depending on whether the frames are loaded from files or provided as
        an in-memory sequence, this function either reads the frame from disk
        or accesses it directly from the sequence.

        Parameters
        ----------

        t (float): The time (in seconds) for which the corresponding frame
            is to be retrieved.

        Returns
        -------

        numpy.ndarray: The image frame at the specified time. If `is_mask`
            is True, the alpha channel of the image is returned as a float
            array normalized to the range [0, 1] (images without an alpha
            channel are fully opaque). Otherwise, the RGB channels of the
            image are returned.

        """
        index = self._find_image_index(t)

        if self.fromfiles:
            # Only decode a file when the frame falls on a new picture:
            # consecutive (or repeated) reads of a same picture reuse the
            # last decoded image instead of decoding the file again.
            if index != self.last_index:
                self.last_index = index
                self.last_image = imread(self.sequence[index])
            image = self.last_image
        else:
            image = self.sequence[index]

        if self.is_mask:
            if image.ndim == 3 and image.shape[2] == 4:
                return image[:, :, 3].astype(float) / 255.0
            # Images without an alpha channel are fully opaque.
            return np.ones(image.shape[:2], dtype=float)
        return image[:, :, :3]
