"""
common.py - Shared utilities for robot control system
"""


def common_print(caller):
    print(
        "This module is common to both driver and slave controllers (called from "
        + caller
        + ")."
    )


def calculate_distance(p1, p2):
    """Calculate Euclidean distance between two points"""
    return ((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2) ** 0.5


def normalize_angle(angle):
    """Normalize angle to range [-pi, pi]"""
    import math

    while angle > math.pi:
        angle -= 2 * math.pi
    while angle < -math.pi:
        angle += 2 * math.pi
    return angle
