import numbers
from dataclasses import dataclass
from typing import Union

import cv2
import numpy as np

from moviepy.Effect import Effect


@dataclass
class Resize(Effect):
    """Effect returning a video clip that is a resized version of the clip.

    Parameters
    ----------

    new_size : tuple or float or function, optional
        Can be either
        - ``(width, height)`` in pixels or a float representing
        - A scaling factor, like ``0.5``.
        - A function of time returning one of these.

    height : int, optional
        Height of the new clip in pixels. The width is then computed so
        that the width/height ratio is conserved.

    width : int, optional
        Width of the new clip in pixels. The height is then computed so
        that the width/height ratio is conserved.

    apply_to_mask : bool, optional
        Whether the mask of the clip should be rescaled along with the
        frames. Default is ``True``.

    preserve_transparency : bool, optional
        If ``True`` and the clip has a mask, the RGB frames and the mask
        are rescaled together using premultiplied alpha: the colors of
        fully transparent pixels (which may contain arbitrary hidden
        values) are prevented from bleeding into the visible edges, and
        both the colors and the coverage of semi-transparent edge pixels
        are preserved. Fully transparent regions are guaranteed to stay
        free of invalid values. This mode requires ``apply_to_mask=True``
        and a mask with the same width/height as the clip's frames, and
        raises a ``ValueError`` otherwise. If the clip has no mask, the
        frames are processed as usual. Default is ``False``, which keeps
        the historical behavior of rescaling frames and mask separately.

    Examples
    --------

    .. code:: python

        clip.with_effects([vfx.Resize((460,720))]) # New resolution: (460,720)
        clip.with_effects([vfx.Resize(0.6)]) # width and height multiplied by 0.6
        clip.with_effects([vfx.Resize(width=800)]) # height computed automatically.
        clip.with_effects([vfx.Resize(lambda t : 1+0.02*t)]) # slow clip swelling
        # icon with transparent background, protected against color bleeding
        clip.with_effects([vfx.Resize(0.5, preserve_transparency=True)])
    """

    new_size: Union[tuple, float, callable] = None
    height: int = None
    width: int = None
    apply_to_mask: bool = True
    preserve_transparency: bool = False

    def resizer(self, pic, new_size):
        """Resize the image using openCV."""
        lx, ly = int(new_size[0]), int(new_size[1])
        if lx > pic.shape[1] or ly > pic.shape[0]:
            # For upsizing use linear for good quality & decent speed
            interpolation = cv2.INTER_LINEAR
        else:
            # For dowsizing use area to prevent aliasing
            interpolation = cv2.INTER_AREA
        return cv2.resize(+pic.astype("uint8"), (lx, ly), interpolation=interpolation)

    def _interpolation(self, pic, new_size):
        """Pick the openCV interpolation for a resize of ``pic`` to ``new_size``."""
        lx, ly = int(new_size[0]), int(new_size[1])
        if lx > pic.shape[1] or ly > pic.shape[0]:
            # For upsizing use linear for good quality & decent speed
            return cv2.INTER_LINEAR
        # For dowsizing use area to prevent aliasing
        return cv2.INTER_AREA

    def alpha_resizer(self, mask_pic, new_size):
        """Resize a float (0-1) mask picture, keeping full precision."""
        lx, ly = int(new_size[0]), int(new_size[1])
        interpolation = self._interpolation(mask_pic, new_size)
        resized = cv2.resize(
            mask_pic.astype("float32"), (lx, ly), interpolation=interpolation
        )
        return np.clip(resized, 0.0, 1.0)

    def resizer_preserving_transparency(self, pic, mask_pic, new_size):
        """Resize an RGB picture along with its mask using premultiplied alpha.

        The RGB channels are multiplied by the mask before resizing and
        divided back afterwards, so hidden colors of fully transparent
        pixels cannot bleed into visible pixels. Fully transparent pixels
        (resized mask equal to 0) are left at 0 instead of producing
        invalid values.
        """
        lx, ly = int(new_size[0]), int(new_size[1])
        interpolation = self._interpolation(pic, new_size)

        alpha = mask_pic.astype("float32")
        rgb = pic.astype("float32")
        if rgb.ndim == 2:
            rgb = rgb[:, :, np.newaxis]

        premultiplied = rgb * alpha[:, :, np.newaxis]
        resized_alpha = cv2.resize(alpha, (lx, ly), interpolation=interpolation)
        resized_premultiplied = cv2.resize(
            premultiplied, (lx, ly), interpolation=interpolation
        )

        result = np.zeros_like(resized_premultiplied)
        np.divide(
            resized_premultiplied,
            resized_alpha[:, :, np.newaxis],
            out=result,
            where=resized_alpha[:, :, np.newaxis] > 0,
        )
        result = np.clip(result, 0, 255).astype("uint8")
        if pic.ndim == 2:
            result = result[:, :, 0]
        return result

    def _apply_preserving_transparency(self, clip, get_new_size):
        """Resize ``clip`` frames and mask jointly (premultiplied alpha).

        ``get_new_size`` is a function of time returning the target
        ``(width, height)``. Both the frames and the mask of the returned
        clip are pure functions of time, so reading them in any order
        yields consistent sizes and contents.
        """
        mask = clip.mask

        def filter(get_frame, t):
            pic = get_frame(t)
            mask_pic = mask.get_frame(t)
            if mask_pic.shape[:2] != pic.shape[:2]:
                raise ValueError(
                    "`preserve_transparency` requires the clip and its mask "
                    "to have the same width/height, got "
                    f"{pic.shape[1]}x{pic.shape[0]} for the frames and "
                    f"{mask_pic.shape[1]}x{mask_pic.shape[0]} for the mask."
                )
            return self.resizer_preserving_transparency(pic, mask_pic, get_new_size(t))

        new_clip = clip.transform(filter, keep_duration=True)

        def mask_filter(get_frame, t):
            return self.alpha_resizer(get_frame(t), get_new_size(t))

        new_clip.mask = mask.transform(mask_filter, keep_duration=True)
        return new_clip

    def _use_transparency_mode(self, clip):
        """Whether the transparency-preserving path applies to ``clip``."""
        return (
            self.preserve_transparency
            and not clip.is_mask
            and clip.mask is not None
        )

    def apply(self, clip):
        """Apply the effect to the clip."""
        if self.preserve_transparency and not self.apply_to_mask:
            raise ValueError(
                "`preserve_transparency=True` requires `apply_to_mask=True`, "
                "as the frames and the mask must be rescaled together."
            )

        w, h = clip.size

        if self.new_size is not None:

            def translate_new_size(new_size_):
                """Returns a [w, h] pair from `new_size_`. If `new_size_` is a
                scalar, then work out the correct pair using the clip's size.
                Otherwise just return `new_size_`
                """
                if isinstance(new_size_, numbers.Number):
                    return [new_size_ * w, new_size_ * h]
                else:
                    return new_size_

            if hasattr(self.new_size, "__call__"):
                # The resizing is a function of time

                def get_new_size(t):
                    return translate_new_size(self.new_size(t))

                if self._use_transparency_mode(clip):
                    return self._apply_preserving_transparency(clip, get_new_size)

                if clip.is_mask:

                    def filter(get_frame, t):
                        return (
                            self.resizer(
                                (255 * get_frame(t)).astype("uint8"), get_new_size(t)
                            )
                            / 255.0
                        )

                else:

                    def filter(get_frame, t):
                        return self.resizer(
                            get_frame(t).astype("uint8"), get_new_size(t)
                        )

                newclip = clip.transform(
                    filter,
                    keep_duration=True,
                    apply_to=(["mask"] if self.apply_to_mask else []),
                )
                if self.apply_to_mask and clip.mask is not None:
                    newclip.mask = clip.mask.with_effects(
                        [Resize(self.new_size, apply_to_mask=False)]
                    )

                return newclip

            else:
                self.new_size = translate_new_size(self.new_size)

        elif self.height is not None:
            if hasattr(self.height, "__call__"):

                def func(t):
                    return 1.0 * int(self.height(t)) / h

                return clip.with_effects(
                    [Resize(func, preserve_transparency=self.preserve_transparency)]
                )

            else:
                self.new_size = [w * self.height / h, self.height]

        elif self.width is not None:
            if hasattr(self.width, "__call__"):

                def func(t):
                    return 1.0 * self.width(t) / w

                return clip.with_effects(
                    [Resize(func, preserve_transparency=self.preserve_transparency)]
                )

            else:
                self.new_size = [self.width, h * self.width / w]
        else:
            raise ValueError(
                "You must provide either 'new_size' or 'height' or 'width'"
            )

        # From here, the resizing is constant (not a function of time), size=newsize

        if self._use_transparency_mode(clip):
            return self._apply_preserving_transparency(clip, lambda t: self.new_size)

        if clip.is_mask:

            def image_filter(pic):
                return (
                    1.0
                    * self.resizer((255 * pic).astype("uint8"), self.new_size)
                    / 255.0
                )

        else:

            def image_filter(pic):
                return self.resizer(pic.astype("uint8"), self.new_size)

        new_clip = clip.image_transform(image_filter)

        if self.apply_to_mask and clip.mask is not None:
            new_clip.mask = clip.mask.with_effects(
                [Resize(self.new_size, apply_to_mask=False)]
            )

        return new_clip
