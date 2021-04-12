"""
2D rendering framework
"""
import os
import sys

if "Apple" in sys.version:
    if "DYLD_FALLBACK_LIBRARY_PATH" in os.environ:
        os.environ["DYLD_FALLBACK_LIBRARY_PATH"] += ":/usr/lib"
        # (JDS 2016/04/15): avoid bug on Anaconda 2.3.0 / Yosemite

from gym import error


import pyglet

from pyglet import gl


RAD2DEG = 57.29577951308232


def get_display(spec):
    """Convert a display specification (such as :0) into an actual Display
    object.

    Pyglet only supports multiple Displays on Linux.
    """
    if spec is None:
        return pyglet.canvas.get_display()
        # returns already available pyglet_display,
        # if there is no pyglet display available then it creates one
    elif isinstance(spec, str):
        return pyglet.canvas.Display(spec)
    else:
        raise error.Error(
            f"Invalid display specification: {spec}. (Must be a string like :0 or None.)"
        )


def get_window(width, height, display, **kwargs):
    """
    Will create a pyglet window from the display specification provided.
    """
    screen = display.get_screens()  # available screens
    config = screen[0].get_best_config()  # selecting the first screen
    context = config.create_context(None)  # create GL context

    return pyglet.window.Window(
        width=width,
        height=height,
        display=display,
        config=config,
        context=context,
        **kwargs,
    )


class ImageViewer:
    def __init__(self, width, height, display=None):
        display = get_display(display)

        self.width = width
        self.height = height
        self.window = get_window(width=width, height=height, display=display)
        self.window.on_close = self.window_closed_by_user
        self.isopen = True

        gl.glEnable(gl.GL_BLEND)
        gl.glBlendFunc(gl.GL_SRC_ALPHA, gl.GL_ONE_MINUS_SRC_ALPHA)

    def imshow(self, arr):
        assert len(arr.shape) == 3, "You passed in an image with the wrong number shape"
        self.window.clear()
        self.window.switch_to()
        self.window.dispatch_events()
        image = pyglet.image.ImageData(
            arr.shape[1], arr.shape[0], "RGB", arr.tobytes(), pitch=arr.shape[1] * -3
        )
        image.blit(0, 0)  # draw
        self.window.flip()

    def close(self):
        if self.isopen and sys.meta_path:
            self.window.close()
            self.isopen = False

    def window_closed_by_user(self):
        self.isopen = False

    def __del__(self):
        self.close()
