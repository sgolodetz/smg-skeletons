import numpy as np

from OpenGL.GL import *
from typing import Dict, List, Optional, Tuple

from smg.opengl import CameraRenderer, OpenGLLightingContext, OpenGLUtil, ShapeRenderer
from smg.rigging.helpers import CameraPoseConverter
from smg.utility import ShapeUtil

from .skeleton3d import Skeleton3D


class SkeletonRenderer:
    """Utility functions to render 3D skeletons."""

    # PUBLIC STATIC METHODS

    @staticmethod
    def default_lighting_context() -> OpenGLLightingContext:
        """
        Get the default OpenGL lighting context to use when rendering skeletons.

        :return:    The default OpenGL lighting context to use when rendering skeletons.
        """
        direction = np.array([0.0, 0.0, 1.0, 0.0])  # type: np.ndarray
        return OpenGLLightingContext({
            0: OpenGLLightingContext.DirectionalLight(direction),
            1: OpenGLLightingContext.DirectionalLight(-direction),
        })

    @staticmethod
    def render_bounding_shapes(skeleton: Skeleton3D) -> None:
        """
        Render the specified 3D skeleton's bounding shapes using OpenGL.

        :param skeleton:    The 3D skeleton.
        """
        shape_renderer = ShapeRenderer()  # type: ShapeRenderer
        for shape in skeleton.bounding_shapes:
            shape.accept(shape_renderer)

    @staticmethod
    def render_bounding_voxels(skeleton: Skeleton3D, *, voxel_size: float) -> None:
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
    def render_keypoint_orienters(skeleton: Skeleton3D) -> None:
        """
        Render the orienters of the specified skeleton's keypoints (for debugging purposes).

        :param skeleton:    The skeleton.
        """
        # Try to get the current pose of the mid-hip keypoint. If this isn't possible, early out.
        world_from_midhip = skeleton.global_keypoint_poses.get("MidHip")  # type: Optional[np.ndarray]
        if world_from_midhip is None:
            return

        # For each keypoint orienter the skeleton has:
        for keypoint_name, orienter in skeleton.keypoint_orienters.items():
            # Render the associated triangle.
            v0, v1, v2 = orienter.triangle_vertices

            glPushAttrib(GL_CURRENT_BIT | GL_POLYGON_BIT)
            glPolygonMode(GL_FRONT_AND_BACK, GL_LINE)
            glBegin(GL_TRIANGLES)

            glColor3f(1.0, 0.0, 0.0)
            glVertex3f(*v0)
            glColor3f(0.0, 1.0, 0.0)
            glVertex3f(*v1)
            glColor3f(0.0, 0.0, 1.0)
            glVertex3f(*v2)

            glEnd()
            glPopAttrib()

            # Render the rest orientation of the associated keypoint at its current position.
            world_from_rest = skeleton.global_keypoint_poses[keypoint_name].copy()  # type: np.ndarray
            world_from_rest[0:3, 0:3] = world_from_midhip[0:3, 0:3] @ skeleton.midhip_from_rests[keypoint_name]

            glPushAttrib(GL_ENABLE_BIT | GL_LINE_BIT)
            glLineStipple(1, 0xCCCC)
            glEnable(GL_LINE_STIPPLE)
            glLineWidth(2)

            CameraRenderer.render_camera(
                CameraPoseConverter.pose_to_camera(np.linalg.inv(world_from_rest)), axis_scale=0.1
            )

            glPopAttrib()

    @staticmethod
    def render_keypoint_poses(skeleton: Skeleton3D) -> None:
        """
        Render the current global poses of the specified skeleton's keypoints (for debugging purposes).

        :param skeleton:    The skeleton.
        """
        for keypoint_name in skeleton.global_keypoint_poses:
            world_from_current = skeleton.global_keypoint_poses[keypoint_name]  # type: np.ndarray

            glPushAttrib(GL_LINE_BIT)
            glLineWidth(2)

            CameraRenderer.render_camera(
                CameraPoseConverter.pose_to_camera(np.linalg.inv(world_from_current)), axis_scale=0.1
            )

            glPopAttrib()

    @staticmethod
    def render_skeleton(skeleton: Skeleton3D) -> None:
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

        # Render the keypoints themselves, colouring them on a scale according to their score (0 = red, 1 = green).
        for keypoint_name, keypoint in skeleton.keypoints.items():
            glColor3f(1 - keypoint.score, keypoint.score, 0.0)
            OpenGLUtil.render_sphere(keypoint.position, 0.03, slices=10, stacks=10)

        # Render the bones between the keypoints.
        for keypoint1, keypoint2 in skeleton.bones:
            bone_key = Skeleton3D.make_bone_key(keypoint1, keypoint2)  # type: Tuple[str, str]
            bone_colour = bone_colours.get(bone_key)                 # type: Optional[np.ndarray]
            if bone_colour is not None:
                # Note: We divide by 153 because that's the maximum value of a component in the colours table,
                #       and we want the colours to be nice and vibrant.
                bone_colour = bone_colour / 153
                glColor3f(*bone_colour)
            else:
                glColor3f(0.0, 0.0, 0.0)

            OpenGLUtil.render_cylinder(keypoint1.position, keypoint2.position, 0.025, 0.025, slices=10)
