import numpy as np

from OpenGL.GL import *
from typing import Callable, List, Optional, Tuple

from smg.opengl import OpenGLFrameBuffer, OpenGLMatrixContext, OpenGLUtil
from smg.rigging.helpers import CameraPoseConverter

from .skeleton3d import Skeleton3D


class PeopleMaskRenderer:
    """TODO"""

    # CONSTRUCTOR

    def __init__(self):
        self.__framebuffer = None  # type: Optional[OpenGLFrameBuffer]

    # PUBLIC METHODS

    def render_people_mask(self, render_person_mask: Callable[[Skeleton3D], None], skeletons: List[Skeleton3D],
                           world_from_camera: np.ndarray, intrinsics: Optional[Tuple[float, float, float, float]],
                           width: int, height: int) -> np.ndarray:
        # If the camera intrinsics aren't available, early out.
        if intrinsics is None:
            return np.zeros((height, width), dtype=np.uint8)

        # If the OpenGL framebuffer hasn't been constructed yet, construct it now.
        # FIXME: Support image size changes.
        if self.__framebuffer is None:
            self.__framebuffer = OpenGLFrameBuffer(width, height)

        # Render a people mask based on the skeletons into the framebuffer.
        with self.__framebuffer:
            # Set the viewport to encompass the whole framebuffer.
            OpenGLUtil.set_viewport((0.0, 0.0), (1.0, 1.0), (width, height))

            # Clear the background to black.
            glClearColor(0.0, 0.0, 0.0, 1.0)
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

            # Set the projection matrix.
            with OpenGLMatrixContext(GL_PROJECTION, lambda: OpenGLUtil.set_projection_matrix(
                intrinsics, width, height
            )):
                # Set the model-view matrix.
                with OpenGLMatrixContext(GL_MODELVIEW, lambda: OpenGLUtil.load_matrix(
                    CameraPoseConverter.pose_to_modelview(np.linalg.inv(world_from_camera))
                )):
                    # Render the people mask into the framebuffer.
                    for skeleton in skeletons:
                        render_person_mask(skeleton)

                    # Make a binary mask from the contents of the framebuffer, and return it.
                    return OpenGLUtil.read_bgr_image(width, height)[:, :, 0]
