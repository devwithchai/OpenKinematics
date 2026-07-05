"""
Custom exception classes for OpenKinematics.

These exceptions provide clear, domain-specific error handling
for robotics and kinematics operations.
"""

class InvalidDHParameters(ValueError):
    """
    Raised when DH parameter table is malformed - wrong number of parameters per row,
    wrong types, or empty table.
    """
    pass

class JointLimitViolation(ValueError):
    """
    Raised when a joint angle exceeds the defined min/max range for that joint.
    """
    pass

class IKNoSolution(RuntimeError):
    """
    Raised when inverse kinematics has no real solution — typically when the target pose
    is outside the robot's workspace.
    """
    pass

class SingularityWarning(UserWarning):
    """
    Issued when the robot is near a singular configuration — manipulability index is
    close to zero.
    """
    pass