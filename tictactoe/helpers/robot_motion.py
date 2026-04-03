# High-level motion helper for the Dobot tic-tac-toe project.

import time
import math
from typing import Sequence, Tuple, Optional


XYZ = Tuple[float, float, float]
XYZR = Tuple[float, float, float, float]


class RobotMotion:
    def __init__(self, robot, approach_offset: float, retract_distance: float, pose_tol_mm: float, pose_poll_s: float):
        """
        Parameters
        ----------
        robot
            Your Dobot instance (from dobot_python.dobot.Dobot).
        approach_offset
            How many mm above the target Z to approach from before descending.
        retract_distance
            How many mm to retract upwards after picking/placing.
        pose_tol_mm
            Distance tolerance (in mm) for wait_pose.
        pose_poll_s
            Polling interval (in seconds) for wait_pose.
        """
        self.robot = robot
        self.approach_offset = approach_offset
        self.retract_distance = retract_distance
        self.pose_tol_mm = pose_tol_mm
        self.pose_poll_s = pose_poll_s

    # -----------------
    # Low-level helpers
    # -----------------

    @staticmethod
    def _distance(a: XYZ, b: XYZ) -> float:
        dx = a[0] - b[0]
        dy = a[1] - b[1]
        dz = a[2] - b[2]
        return math.sqrt(dx * dx + dy * dy + dz * dz)

    def wait_pose(self, target_xyz: Sequence[float], timeout: Optional[float] = None):
        """
        Block until the TCP is within pose_tol_mm of target_xyz.
        Don't allow the program to proceed until the robot has performed the action.

        Parameters
        ----------
        target_xyz
            (x, y, z) you expect the robot to reach.
        timeout
            Optional max seconds to wait; if exceeded, raises TimeoutError.
        """
        target = (float(target_xyz[0]), float(target_xyz[1]), float(target_xyz[2]))
        start = time.time()

        while True:
            pose = self.robot.get_pose()
            current = pose[0:3]  # x, y, z
            dist = self._distance(current, target)

            if dist <= self.pose_tol_mm:
                return

            if timeout is not None and (time.time() - start) > timeout:
                raise TimeoutError(f"wait_pose timeout: dist={dist:.2f} mm, "f"target={target}, current={current}")

            time.sleep(self.pose_poll_s)

    def move_joint_and_wait(self, x: float, y: float, z: float, r: float, timeout: Optional[float] = None):
        """
        Joint move + wait until pose is reached.
        """
        self.robot.move_joint(x, y, z, r)
        self.wait_pose((x, y, z), timeout=timeout)

    def move_linear_and_wait(self, x: float, y: float, z: float, r: float, timeout: Optional[float] = None):
        """
        Linear move + wait until pose is reached.
        """
        self.robot.move_linear(x, y, z, r)
        self.wait_pose((x, y, z), timeout=timeout)

    # -------------------
    # High-level routines
    # -------------------

    def pick_object(self, pos: XYZR, mode: str = "pickup"):
        """
        Pick a piece from the tic-tac-toe board and return to safe height.
        """
        x, y, z, r = pos

        # 1) Approach above the piece
        self.move_joint_and_wait(x, y, z + self.approach_offset, r)

        # 2) Descend onto the piece
        self.move_linear_and_wait(x, y, z, r)

        # 3) Turn suction on
        self.robot.set_suction(True); time.sleep(0.2)

        if mode == "pickup":
            # 4) Retract a bit upwards
            self.move_joint_and_wait(x, y, z + self.retract_distance, r)

        elif mode == "cleanup":
            # 4) Picking up pieces for cleanup: retract fully upwards
            self.move_linear_and_wait(x, y, z + self.approach_offset, r)
        else:
            raise ValueError("mode must be 'pickup' or 'cleanup'")

    def place_object(self, pos: XYZR):
        """
        Dropping a pice on to the tic-tac-toe board..
        """
        x, y, z, r = pos
        self.move_joint_and_wait(x, y, z + self.approach_offset, r)
        self.move_linear_and_wait(x, y, z, r)
        self.robot.set_suction(False);  time.sleep(0.2)
        self.move_linear_and_wait(x, y, z + self.retract_distance, r)

    def special_pick(self, pos: XYZR):
        """
        Special pick sequence for picking items from the slide.
        """
        x, y, z, r = pos
        self.move_joint_and_wait(x, y, z + self.approach_offset, r)
        self.move_linear_and_wait(x, y, z, r)
        self.robot.set_suction(True);  time.sleep(0.2)
        self.move_linear_and_wait(x, y, z + 6, r)
        self.move_linear_and_wait(x, y - 10, z + 10, r)
        self.move_linear_and_wait(x, y - 10, z + 30, r)