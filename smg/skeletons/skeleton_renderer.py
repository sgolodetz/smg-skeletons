import numpy as np

from OpenGL.GL import *
from typing import Dict, List, Optional, Tuple

from smg.opengl import OpenGLUtil, ShapeRenderer
from smg.utility import ShapeUtil

from .skeleton import Skeleton


class SkeletonRenderer:
    """Utility functions to render 3D skeletons."""

    # PUBLIC STATIC METHODS

    @staticmethod
    def render_bounding_shapes(skeleton: Skeleton) -> None:
        """
        Render the specified 3D skeleton's bounding shapes using OpenGL.

        :param skeleton:    The 3D skeleton.
        """
        shape_renderer = ShapeRenderer()  # type: ShapeRenderer
        for shape in skeleton.bounding_shapes:
            shape.accept(shape_renderer)

    @staticmethod
    def render_bounding_voxels(skeleton: Skeleton, *, voxel_size: float) -> None:
        """
        Render the voxels touched by the bounding shapes of the specified 3D skeleton using OpenGL.

        :param skeleton:    The 3D skeleton.
        :param voxel_size:  The voxel size.
        """
        voxel_centres = ShapeUtil.rasterise_shapes(skeleton.bounding_shapes, voxel_size)  # type: List[np.ndarray]
        offset = np.full(3, voxel_size / 2)  # type: np.ndarray
        for voxel_centre in voxel_centres:
            OpenGLUtil.render_aabb(voxel_centre - offset, voxel_centre + offset)

    @staticmethod
    def render_skeleton(skeleton: Skeleton) -> None:
        """
        Render the specified 3D skeleton using OpenGL.

        :param skeleton:    The 3D skeleton.
        """
        bone_colours = {
            ('MidHip', 'Neck'): np.array([153., 0., 0.]),
            ('Neck', 'RShoulder'): np.array([153., 51., 0.]),
            ('LShoulder', 'Neck'): np.array([102., 153., 0.]),
            ('RElbow', 'RShoulder'): np.array([153., 102., 0.]),
            ('RElbow', 'RWrist'): np.array([153., 153., 0.]),
            ('LElbow', 'LShoulder'): np.array([51., 153., 0.]),
            ('LElbow', 'LWrist'): np.array([0., 153., 0.]),
            ('MidHip', 'RHip'): np.array([0., 153., 51.]),
            ('RHip', 'RKnee'): np.array([0., 153., 102.]),
            ('RAnkle', 'RKnee'): np.array([0., 153., 153.]),
            ('LHip', 'MidHip'): np.array([0., 102., 153.]),
            ('LHip', 'LKnee'): np.array([0., 51., 153.]),
            ('LAnkle', 'LKnee'): np.array([0., 0., 153.]),
            ('Neck', 'Nose'): np.array([153., 0., 51.]),
            ('Nose', 'REye'): np.array([153., 0., 102.]),
            ('REar', 'REye'): np.array([153., 0., 153.]),
            ('LEye', 'Nose'): np.array([102., 0., 153.]),
            ('LEar', 'LEye'): np.array([51., 0., 153.]),
            ('REar', 'RShoulder'): np.array([0., 0., 255.]),
            ('LEar', 'LShoulder'): np.array([0., 0., 255.]),
            ('LAnkle', 'LBigToe'): np.array([0., 0., 153.]),
            ('LBigToe', 'LSmallToe'): np.array([0., 0., 153.]),
            ('LAnkle', 'LHeel'): np.array([0., 0., 153.]),
            ('RAnkle', 'RBigToe'): np.array([0., 153., 153.]),
            ('RBigToe', 'RSmallToe'): np.array([0., 153., 153.]),
            ('RAnkle', 'RHeel'): np.array([0., 153., 153.])
        }  # type: Dict[Tuple[str, str], np.ndarray]

        # Enable lighting.
        glEnable(GL_LIGHTING)

        # Set up the first light to cast light in the +z direction.
        glEnable(GL_LIGHT0)
        pos = np.array([0.0, 0.0, -1.0, 0.0])  # type: np.ndarray
        glLightfv(GL_LIGHT0, GL_POSITION, pos)

        # Set up the second light to cast light in the -z direction.
        glEnable(GL_LIGHT1)
        glLightfv(GL_LIGHT1, GL_DIFFUSE, np.array([1, 1, 1, 1]))
        glLightfv(GL_LIGHT1, GL_SPECULAR, np.array([1, 1, 1, 1]))
        glLightfv(GL_LIGHT1, GL_POSITION, -pos)

        # Enable colour-based materials (i.e. let material properties be defined by glColor).
        glEnable(GL_COLOR_MATERIAL)

        # Render the keypoints themselves, colouring them on a scale according to their score (0 = red, 1 = green).
        for keypoint_name, keypoint in skeleton.keypoints.items():
            glColor3f(1 - keypoint.score, keypoint.score, 0.0)
            OpenGLUtil.render_sphere(keypoint.position, 0.03, slices=10, stacks=10)

        # Render the bones between the keypoints.
        for keypoint1, keypoint2 in skeleton.bones:
            bone_key = Skeleton.make_bone_key(keypoint1, keypoint2)  # type: Tuple[str, str]
            bone_colour = bone_colours.get(bone_key)  # type: Optional[np.ndarray]
            if bone_colour is not None:
                # Note: We divide by 153 because that's the maximum value of a component in the colours table,
                #       and we want the colours to be nice and vibrant.
                bone_colour = bone_colour / 153
                glColor3f(*bone_colour)
                OpenGLUtil.render_cylinder(keypoint1.position, keypoint2.position, 0.025, 0.025, slices=10)

        # Disable colour-based materials and lighting again.
        glDisable(GL_COLOR_MATERIAL)
        glDisable(GL_LIGHTING)
