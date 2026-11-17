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
        Resize the clip mask together with the image. Defaults to ``True``.

    transparent_edge_protection : bool, optional
        Resize the image and its mask together using premultiplied alpha.
        This prevents colors hidden in fully transparent pixels from bleeding
        into visible or semi-transparent edges. Requires a mask and
        ``apply_to_mask=True``. If the clip has no mask, the image is resized
        normally. Defaults to ``False``, which preserves the previous behavior.

    Examples
    --------

    .. code:: python

        clip.with_effects([vfx.Resize((460,720))]) # New resolution: (460,720)
        clip.with_effects([vfx.Resize(0.6)]) # width and height multiplied by 0.6
        clip.with_effects([vfx.Resize(width=800)]) # height computed automatically.
        clip.with_effects([vfx.Resize(lambda t : 1+0.02*t)]) # slow clip swelling
        clip.with_effects(
            [vfx.Resize(0.6, transparent_edge_protection=True)]
        ) # resize alpha without color bleeding
    """

    new_size: Union[tuple, float, callable] = None
    height: int = None
    width: int = None
    apply_to_mask: bool = True
    transparent_edge_protection: bool = False

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

    def apply(self, clip):
        """Apply the effect to the clip."""
        if self.transparent_edge_protection:
            if not self.apply_to_mask:
                raise ValueError(
                    "'transparent_edge_protection=True' requires "
                    "'apply_to_mask=True'"
                )
            if clip.mask is not None and not clip.is_mask:
                return self._apply_with_transparent_edge_protection(clip)

        return self._apply_standard(clip)

    @staticmethod
    def _interpolation_for(source_shape, new_size):
        """Return the interpolation used by the regular resizer."""
        width, height = int(new_size[0]), int(new_size[1])
        if width > source_shape[1] or height > source_shape[0]:
            return cv2.INTER_LINEAR
        return cv2.INTER_AREA

    @staticmethod
    def _resize_rgb_and_mask(image, mask, new_size):
        """Resize RGB and alpha together without mixing hidden RGB colors."""
        if image.shape[:2] != mask.shape[:2]:
            raise ValueError(
                "Image and mask dimensions must match for transparent edge "
                f"protection, got image {image.shape[:2]} and mask "
                f"{mask.shape[:2]}"
            )

        width, height = int(new_size[0]), int(new_size[1])
        image_float = image.astype(np.float32, copy=False)
        mask_float = np.clip(mask.astype(np.float32, copy=False), 0.0, 1.0)
        premultiplied = image_float * mask_float[..., None]

        if (height, width) == image.shape[:2]:
            resized_image = premultiplied
            resized_mask = mask_float
        else:
            interpolation = Resize._interpolation_for(image.shape, new_size)
            resized_image = cv2.resize(
                premultiplied, (width, height), interpolation=interpolation
            )
            resized_mask = cv2.resize(
                mask_float, (width, height), interpolation=interpolation
            )

        resized_mask = np.clip(
            np.nan_to_num(resized_mask, nan=0.0, posinf=1.0, neginf=0.0),
            0.0,
            1.0,
        )
        visible = resized_mask > 0
        safe_mask = np.where(visible, resized_mask, 1.0)
        resized_image = resized_image / safe_mask[..., None]
        resized_image = np.where(visible[..., None], resized_image, 0.0)
        resized_image = np.clip(
            np.nan_to_num(resized_image, nan=0.0, posinf=255.0, neginf=0.0),
            0.0,
            255.0,
        )

        if np.issubdtype(image.dtype, np.integer):
            image_info = np.iinfo(image.dtype)
            resized_image = np.clip(
                np.rint(resized_image), image_info.min, image_info.max
            ).astype(image.dtype)
        else:
            resized_image = resized_image.astype(image.dtype, copy=False)

        if np.issubdtype(mask.dtype, np.floating):
            resized_mask = resized_mask.astype(mask.dtype, copy=False)
        else:
            resized_mask = resized_mask.astype(float)

        return resized_image, resized_mask

    def _get_size_function(self, clip):
        """Return a function giving the requested size at a given time."""
        width, height = clip.size

        def translate_new_size(new_size):
            if isinstance(new_size, numbers.Number):
                return [new_size * width, new_size * height]
            return new_size

        if self.new_size is not None:
            if callable(self.new_size):
                return lambda t: translate_new_size(self.new_size(t))
            return lambda t: translate_new_size(self.new_size)

        if self.height is not None:
            if callable(self.height):

                def size_from_height(t):
                    new_height = int(self.height(t))
                    return [width * new_height / height, new_height]

            else:

                def size_from_height(t):
                    return [width * self.height / height, self.height]

            return size_from_height

        if self.width is not None:
            if callable(self.width):

                def size_from_width(t):
                    new_width = self.width(t)
                    return [new_width, height * new_width / width]

            else:

                def size_from_width(t):
                    return [self.width, height * self.width / width]

            return size_from_width

        raise ValueError("You must provide either 'new_size' or 'height' or 'width'")

    def _apply_with_transparent_edge_protection(self, clip):
        """Resize an RGB clip and its mask as one premultiplied-alpha pair."""
        mask = clip.mask
        if tuple(clip.size) != tuple(mask.size):
            raise ValueError(
                "Image and mask dimensions must match for transparent edge "
                f"protection, got image {tuple(clip.size)} and mask "
                f"{tuple(mask.size)}"
            )

        get_new_size = self._get_size_function(clip)
        get_image = clip.get_frame
        get_mask = mask.get_frame

        def resize_at(t):
            image, resized_mask = self._resize_rgb_and_mask(
                get_image(t), get_mask(t), get_new_size(t)
            )
            return image, resized_mask

        new_clip = clip.transform(
            lambda get_frame, t: resize_at(t)[0],
            keep_duration=True,
        )
        new_mask = mask.transform(
            lambda get_frame, t: resize_at(t)[1],
            keep_duration=True,
        )
        new_clip.mask = new_mask
        return new_clip

    def _apply_standard(self, clip):
        """Apply the effect using the historical resize implementation."""
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

                return clip.with_effects([Resize(func)])

            else:
                self.new_size = [w * self.height / h, self.height]

        elif self.width is not None:
            if hasattr(self.width, "__call__"):

                def func(t):
                    return 1.0 * self.width(t) / w

                return clip.with_effects([Resize(func)])

            else:
                self.new_size = [self.width, h * self.width / w]
        else:
            raise ValueError(
                "You must provide either 'new_size' or 'height' or 'width'"
            )

        # From here, the resizing is constant (not a function of time), size=newsize

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
