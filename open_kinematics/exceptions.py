"""
Custom exception classes for OpenKinematics.

These exceptions provide clear, domain-specific error handling for robotics and kinematics operations.
"""

class InvalidDHParameters(ValueError):
    """
    Raised when robot kinematic parameters or related input data are malformed or inconsistent.
    """
    pass

class JointLimitViolation(ValueError):
    """
    Raised when a joint angle exceeds the defined min/max range for that joint.
    """
    pass

class IKNoSolution(RuntimeError):
    """
    Raised when inverse kinematics cannot produce a valid joint configuration for the requested target pose.

    This may occur because the target is unreachable, the iterative solver fails to converge,
    numerical issues arise during optimization, or joint constraints prevent a valid solution.
    """
    pass

class SingularityWarning(UserWarning):
    """
    Issued when the robot is near a singular configuration — manipulability index is close to zero.
    """
    pass