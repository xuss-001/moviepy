from moviepy import ImageSequenceClip

# A clip with a list of images showed for 1 second each
myclip = ImageSequenceClip(
    [
        "example_img_dir/image_0001.jpg",
        "example_img_dir/image_0002.jpg",
        "example_img_dir/image_0003.jpg",
    ],
    durations=[1, 1, 1],
)
# 3 images, 1 seconds each, duration = 3
print("Clip duration: {}".format(myclip.duration))
# 3 seconds, 3 images, fps is 3/3 = 1
print("Clip fps: {}".format(myclip.fps))

# This time we will load all images in the dir, and instead of showing theme
# for X seconds, we will define FPS
myclip2 = ImageSequenceClip("./example_img_dir", fps=30)
# fps = 30, so duration = nb images in dir / 30
print("Clip duration: {}".format(myclip2.duration))
print("Clip fps: {}".format(myclip2.fps))  # fps = 30

# the gif will be 30 fps, its duration will depend on the number of
# images in dir
myclip.write_gif("result.gif")  # the gif will be 3 sec and 1 fps
myclip2.write_gif("result2.gif")

# The sequence can mix RGB and RGBA images as long as they share the same
# size. RGB images are treated as fully opaque while RGBA images keep their
# alpha channel as the clip's mask, and each image can be displayed for a
# different duration (in seconds). A duration of 0 simply skips the image.
import numpy as np
from PIL import Image


def solid_image(color):
    return np.array(Image.new("RGB", (8, 8), color=color))


def transparent_image(alpha):
    rgba = np.zeros((8, 8, 4), dtype=np.uint8)
    rgba[:, :, :3] = 255
    rgba[:, :, 3] = alpha
    return rgba


rgba_clip = ImageSequenceClip(
    [
        solid_image((255, 0, 0)),  # opaque RGB image
        transparent_image(64),  # semi-transparent RGBA image
        solid_image((0, 255, 0)),  # opaque again
    ],
    durations=[0.5, 1.0, 1.5],
)
print("Mixed clip duration: {}".format(rgba_clip.duration))  # 3.0 seconds
print("Mixed clip has a mask: {}".format(rgba_clip.mask is not None))
