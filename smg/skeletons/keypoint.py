import numpy as np


class Keypoint:
    """A keypoint in a 2D or 3D skeleton."""

    # CONSTRUCTOR

    def __init__(self, name: str, position: np.ndarray, score: float = 1.0):
        """
        Construct a keypoint.

        :param name:        The name of the keypoint.
        :param position:    The position of the keypoint.
        :param score:       The score assigned to the keypoint (a float in [0,1]).
        """
        self.__name = name          # type: str
        self.__position = position  # type: np.ndarray
        self.__score = score        # type: float

    # SPECIAL METHODS

    def __repr__(self) -> str:
        """
        Get a string representation of the keypoint.

        :return:    A string representation of the keypoint.
        """
        return "Keypoint({}, {}, {})".format(repr(self.__name), repr(self.__position), repr(self.__score))

    # PROPERTIES

    @property
    def name(self) -> str:
        """
        Get the name of the keypoint.

        :return:    The name of the keypoint.
        """
        return self.__name

    @property
    def position(self) -> np.ndarray:
        """
        Get the position of the keypoint.

        :return:    The position of the keypoint.
        """
        return self.__position

    @property
    def score(self) -> float:
        """
        Get the score assigned to the keypoint.

        :return:    The score assigned to the keypoint (a float in [0,1]).
        """
        return self.__score
