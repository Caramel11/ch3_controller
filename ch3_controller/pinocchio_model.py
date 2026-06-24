import os
import tempfile
import subprocess
from typing import Optional, Tuple

import numpy as np
import pinocchio as pin


class PinocchioModelHelper:
    """
    7DoF arm-only 版本的 Pinocchio helper。

    设计原则：
    - 直接从 /robot_state_publisher 读取 robot_description
    - 假定当前 URDF 已经是 arm-only（即 nq=nv=7）
    - reference_frame / tip_frame 都直接在同一个 7DoF 模型里找
    - 不再做 full->controlled 的裁剪
    """

    def __init__(
        self,
        node,
        rsp_node: str,
        controlled_joints,
        reference_frame: str,
        tip_frame: str,
    ):
        self.node = node
        self.rsp_node = rsp_node
        self.controlled_joints = list(controlled_joints)
        self.reference_frame_name = reference_frame
        self.tip_frame_name = tip_frame

        self.model: Optional[pin.Model] = None
        self.data: Optional[pin.Data] = None

        self.reference_frame_id: Optional[int] = None
        self.tip_frame_id: Optional[int] = None

    # ============================================================
    # Public API
    # ============================================================

    def load(self) -> bool:
        urdf_xml = self._load_urdf_from_ros2_param()
        if urdf_xml is None:
            return False
        return self._build_model_from_xml(urdf_xml)

    def build_full_state(
        self, q_meas: np.ndarray, qd_meas: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        对于 arm-only 7DoF 模型：
            full state == measured state
        """
        return q_meas.copy(), qd_meas.copy()

    def get_frame_state(
        self,
        q: np.ndarray,
        qd: np.ndarray,
        frame_id: int,
    ):
        """
        返回：
          p         : (3,)
          R         : (3,3)
          J6        : (6,7)  LOCAL_WORLD_ALIGNED
          Jv        : (3,7)
          Jw        : (3,7)
          v_linear  : (3,)
        """
        assert self.model is not None and self.data is not None

        pin.forwardKinematics(self.model, self.data, q, qd)
        pin.updateFramePlacements(self.model, self.data)

        oMf = self.data.oMf[frame_id]
        p = oMf.translation.copy()
        R = oMf.rotation.copy()

        J6 = pin.computeFrameJacobian(
            self.model,
            self.data,
            q,
            frame_id,
            pin.ReferenceFrame.LOCAL_WORLD_ALIGNED,
        )
        Jv = J6[:3, :].copy()
        Jw = J6[3:, :].copy()
        v_linear = Jv @ qd

        return p, R, J6, Jv, Jw, v_linear

    def extract_controlled_columns(self, J: np.ndarray) -> np.ndarray:
        """
        7DoF arm-only 模型下，Jacobian 已经就是控制空间 Jacobian。
        """
        return J.copy()

    def extract_controlled_effort(self, tau: np.ndarray) -> np.ndarray:
        """
        7DoF arm-only 模型下，力矩已经就是控制空间力矩。
        """
        return tau.copy()

    # ============================================================
    # Internal
    # ============================================================

    def _load_urdf_from_ros2_param(self) -> Optional[str]:
        try:
            out = subprocess.check_output(
                ["ros2", "param", "get", self.rsp_node, "robot_description"],
                text=True,
                stderr=subprocess.STDOUT,
            )
            prefix = "String value is:"
            if prefix not in out:
                self.node.get_logger().error(
                    f"Unexpected ros2 param output: {out[:200]}"
                )
                return None

            urdf_xml = out.split(prefix, 1)[1].strip()
            return urdf_xml

        except subprocess.CalledProcessError as e:
            self.node.get_logger().error(
                f"ros2 param get failed: {e.output[:300]}"
            )
            return None
        except Exception as e:
            self.node.get_logger().error(
                f"Exception while reading URDF: {e}"
            )
            return None

    def _build_model_from_xml(self, urdf_xml: str) -> bool:
        try:
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".urdf", delete=False
            ) as f:
                f.write(urdf_xml)
                urdf_path = f.name

            self.model = pin.buildModelFromUrdf(urdf_path)
            self.data = self.model.createData()

            os.unlink(urdf_path)

            self.reference_frame_id = self.model.getFrameId(self.reference_frame_name)
            self.tip_frame_id = self.model.getFrameId(self.tip_frame_name)

            if self.reference_frame_id >= len(self.model.frames):
                self.node.get_logger().error(
                    f"Reference frame '{self.reference_frame_name}' not found."
                )
                self.model = None
                self.data = None
                return False

            if self.tip_frame_id >= len(self.model.frames):
                self.node.get_logger().error(
                    f"Tip frame '{self.tip_frame_name}' not found."
                )
                self.model = None
                self.data = None
                return False

            self.node.get_logger().info("Pinocchio model ready (URDF from ros2 param).")
            self.node.get_logger().info(
                f"model.nq={self.model.nq}, model.nv={self.model.nv}"
            )
            self.node.get_logger().info(
                f"reference_frame={self.reference_frame_name}, tip_frame={self.tip_frame_name}"
            )
            self.node.get_logger().info(
                f"model joints: {list(self.model.names)}"
            )
            return True

        except Exception as e:
            self.node.get_logger().error(f"Failed to build Pinocchio model: {e}")
            self.model = None
            self.data = None
            return False