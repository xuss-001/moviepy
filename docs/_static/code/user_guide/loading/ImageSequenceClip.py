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

# Images may mix RGB and RGBA pictures (same width/height): RGB ones are
# treated as opaque, RGBA ones keep their transparency in the clip's mask.
# You can also give each image its own duration (images with a duration
# of 0 are skipped, and fps takes precedence if both are provided).
import numpy as np
from imageio.v2 import imwrite

rgb_image = np.full((100, 100, 3), (255, 0, 0), dtype=np.uint8)
rgba_image = np.full((100, 100, 4), (0, 255, 0, 128), dtype=np.uint8)
imwrite("rgb_image.png", rgb_image)
imwrite("rgba_image.png", rgba_image)  # semi-transparent green

myclip3 = ImageSequenceClip(
    ["rgb_image.png", "rgba_image.png"],
    durations=[1, 2],  # RGB image for 1s, then RGBA image for 2s
)
# the RGB image is opaque, the RGBA one keeps its alpha in the mask
print(myclip3.mask.get_frame(0.5)[0, 0])  # 1.0
print(myclip3.mask.get_frame(1.5)[0, 0])  # 128/255 ~= 0.5
