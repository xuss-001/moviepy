"""Implements ImageSequenceClip, a class to create a video clip from a set
of image files.
"""

import os
from numbers import Real

import numpy as np
from imageio.v2 import imread

from moviepy.video.VideoClip import VideoClip


class ImageSequenceClip(VideoClip):
    """A VideoClip made from a series of images.

    The sequence can mix RGB and RGBA pictures as long as they all share the
    same width and height. RGB pictures are treated as fully opaque while
    RGBA pictures keep their alpha channel: when ``with_mask`` is ``True`` and
    at least one picture has an alpha channel, the clip gets a mask whose
    frame at any time is built from the very same picture as the matching RGB
    frame, so seeking backwards and forwards can never desynchronize them.

    Parameters
    ----------

    sequence
      Can be one of these:

      - The name of a folder (containing only pictures). The pictures
        will be considered in alphanumerical order.
      - A list of names of image files. In this case you can choose to
        load the pictures in memory pictures
      - A list of Numpy arrays representing images.

      An empty folder or list raises a ``ValueError``.

    fps
      Number of picture frames to read per second. Instead, you can provide
      the duration of each image with ``durations`` (see below). If both are
      provided, ``fps`` always takes precedence and ``durations`` is ignored.
      It must be a finite positive number.

    durations
      List of the duration of each picture, in seconds. It must contain as
      many finite non-negative numbers as there are pictures. Pictures with
      a duration of 0 are skipped and never displayed, so the total duration
      must remain strictly positive.

    with_mask
      Should the alpha layer of the images be considered as a mask? When
      ``False`` the alpha channel is ignored and every image is opaque.

    is_mask
      Will this sequence of pictures be used as an animated mask.

    load_images
      Specify that all images should be loaded into the RAM. This is only
      interesting if you have a small number of images that will be used more
      than once. By default the images stay on disk and only the most recently
      decoded picture is kept in a cache, so reading the same frame several
      times in a row (as happens when rendering or scrubbing) does not decode
      the same file repeatedly.

    Raises
    ------

    ValueError
      If neither ``fps`` nor ``durations`` is provided, if ``fps`` is not a
      finite positive number, if ``durations`` does not match the number of
      images or contains invalid values, if the sequence is empty, if the
      images have different sizes, or if the resulting total duration is not
      strictly positive.
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
        if fps is not None and not (
            isinstance(fps, Real) and np.isfinite(fps) and fps > 0
        ):
            raise ValueError(
                "'fps' must be a finite positive number, got %r." % (fps,)
            )

        VideoClip.__init__(self, is_mask=is_mask)

        # Parse the sequence into a list of file names or numpy arrays.
        if isinstance(sequence, str):
            sequence = sorted(
                os.path.join(sequence, filename)
                for filename in os.listdir(sequence)
            )
        elif not isinstance(sequence, list):
            raise ValueError(
                "'sequence' must be a folder name or a list of image file "
                "names or numpy arrays, got %r." % (type(sequence),)
            )

        if len(sequence) == 0:
            raise ValueError(
                "ImageSequenceClip requires at least one image but the "
                "sequence is empty."
            )

        self.fromfiles = any(isinstance(image, str) for image in sequence)
        if load_images:
            sequence = [
                imread(image) if isinstance(image, str) else image
                for image in sequence
            ]
            self.fromfiles = False

        # Validate the durations against the number of images. When fps is
        # given it takes precedence and the provided durations are ignored.
        if fps is not None:
            durations = [1.0 / fps for _ in sequence]
        else:
            try:
                durations = list(durations)
            except TypeError:
                raise ValueError(
                    "'durations' must be a list of finite non-negative "
                    "numbers, one per image."
                )
            if len(durations) != len(sequence):
                raise ValueError(
                    "ImageSequenceClip got %d images but %d durations: "
                    "'durations' must contain one value per image."
                    % (len(sequence), len(durations))
                )
            for index, image_duration in enumerate(durations):
                if not (
                    isinstance(image_duration, Real)
                    and np.isfinite(image_duration)
                    and image_duration >= 0
                ):
                    raise ValueError(
                        "All durations must be finite non-negative numbers "
                        "but durations[%d] = %r." % (index, image_duration)
                    )

        # Check that every image shares the same width and height while
        # remembering whether the sequence contains any alpha channel.
        reference_shape = self._shape_of(sequence[0])
        has_alpha = self._shape_has_alpha(reference_shape)
        for image in sequence[1:]:
            shape = self._shape_of(image)
            if shape[:2] != reference_shape[:2]:
                raise ValueError(
                    "MoviePy: ImageSequenceClip requires all images to be "
                    "the same size"
                )
            has_alpha = has_alpha or self._shape_has_alpha(shape)
        self.has_alpha = has_alpha

        # Images with a duration of 0 are never displayed: drop them before
        # computing the start times and the total duration.
        kept = [
            (index, image_duration)
            for index, image_duration in enumerate(durations)
            if image_duration > 0
        ]
        if not kept:
            raise ValueError(
                "ImageSequenceClip requires a strictly positive total "
                "duration but every image duration is 0."
            )

        self.sequence = [sequence[index] for index, _ in kept]
        self.durations = [image_duration for _, image_duration in kept]
        images_starts = np.zeros(len(self.durations) + 1)
        images_starts[1:] = np.cumsum(self.durations)
        self.images_starts = images_starts

        self.duration = float(sum(self.durations))
        self.end = self.duration
        self.fps = fps if fps is not None else len(self.sequence) / self.duration

        # Cache of the last decoded file. Reading the same frame again (a
        # common case when rendering or scrubbing) never re-decodes it.
        self.last_index = None
        self.last_image = None

        self.size = self.frame_function(0).shape[:2][::-1]

        if with_mask and not is_mask and has_alpha:
            # The mask shares this clip's image index and decode cache, so
            # the RGB frame and the mask frame always come from the exact
            # same image at any given time, including after seeking.
            def mask_frame_function(t):
                index = self._find_image_index(t)
                return self._image_as_mask(self._load_image(index))

            mask = VideoClip(
                is_mask=True,
                frame_function=mask_frame_function,
                duration=self.duration,
            )
            mask.fps = self.fps
            self.mask = mask

    @staticmethod
    def _shape_of(image):
        return imread(image).shape if isinstance(image, str) else image.shape

    @staticmethod
    def _shape_has_alpha(shape):
        return len(shape) == 3 and shape[2] == 4

    @staticmethod
    def _image_as_mask(image):
        """Returns the alpha channel of an image as a float mask in [0, 1].

        RGB images without an alpha channel are treated as fully opaque.
        """
        if image.ndim == 3 and image.shape[2] == 4:
            return image[:, :, 3].astype(float) / 255.0
        return np.ones(image.shape[:2], dtype=float)

    def _find_image_index(self, t):
        """Returns the index of the image displayed at time ``t``.

        Image ``i`` is displayed on the half-open interval
        ``[images_starts[i], images_starts[i + 1])``: each image appears at
        its own start time and is replaced by the next image exactly when the
        next one starts, never earlier.
        """
        index = np.searchsorted(self.images_starts, t, side="right") - 1
        return int(min(max(index, 0), len(self.sequence) - 1))

    def _load_image(self, index):
        """Returns the raw image at ``index``, decoding files only once."""
        if not self.fromfiles:
            return self.sequence[index]
        if index != self.last_index:
            self.last_index = index
            self.last_image = imread(self.sequence[index])
        return self.last_image

    def frame_function(self, t):
        """Retrieves the frame corresponding to the given time `t`.

        Depending on whether the frames are loaded from files or provided as
        an in-memory sequence, this function either reads the frame from disk
        or accesses it directly from the sequence. Files are decoded lazily
        and the most recently decoded image is cached, so consecutive reads
        of the same frame never decode the same picture twice.

        Parameters
        ----------

        t (float): The time (in seconds) for which the corresponding frame
            is to be retrieved.

        Returns
        -------

        numpy.ndarray: The image frame at the specified time. If `is_mask`
            is True, the alpha channel of the image is returned
            as a float array normalized to the range [0, 1] (RGB images
            without an alpha channel produce a fully opaque mask of ones).
            Otherwise, the RGB channels of the image are returned.

        """
        index = self._find_image_index(t)
        image = self._load_image(index)

        if self.is_mask:
            return self._image_as_mask(image)
        return image[:, :, :3]
