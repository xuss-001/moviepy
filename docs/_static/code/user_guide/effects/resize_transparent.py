from moviepy import ImageClip, vfx


# This example image has an alpha layer and automatically receives a mask.
icon = ImageClip("example2.png")

# Keep premultiplied color and coverage together while resizing.
small_icon = icon.with_effects(
    [vfx.Resize(width=128, transparent_edge_protection=True)]
)

# Size arguments may also change over time.
animated_icon = icon.with_effects(
    [vfx.Resize(lambda t: 0.5 + 0.1 * t, transparent_edge_protection=True)]
)
