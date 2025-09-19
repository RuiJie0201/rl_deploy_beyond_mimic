"""
 * @file mujoco_simulation.py
 * @brief simulation in mujoco with torque data saved to CSV
 * @author mayuxuan
 * @version 1.3
 * @date 2025-09-18
 *
 * @copyright Copyright (c) 2024  DeepRobotics
"""

import os
import time
import socket
import struct
import threading
from pathlib import Path
from scipy.spatial.transform import Rotation
import numpy as np
import mujoco
import mujoco.viewer
import csv

MODEL_NAME = "CR01B-pro"
XML_PATH = "urdf_model/CR01B-pro/B-20250522.xml"
LOCAL_PORT = 20001
CTRL_IP = "127.0.0.1"
CTRL_PORT = 30010
USE_VIEWER = True
DT = 0.001
RENDER_INTERVAL = 40
SAVE_INTERVAL = 100  # Steps between CSV writes

URDF_INIT = {
    "CR01B-pro": np.array([0, ] * 29, dtype=np.float32)
}

virtual_joint_num = {
    "k1w": 0,
    "m20": 0,
    "CR1LEG": 0,
    "CR1PRO": 2,
    "CR1STANDARD": 0,
}

class MuJoCoSimulation:
    def __init__(self,
                 model_key: str = MODEL_NAME,
                 xml_relpath: str = XML_PATH,
                 local_port: int = LOCAL_PORT,
                 ctrl_ip: str = CTRL_IP,
                 ctrl_port: int = CTRL_PORT):

        self.robot_name = "CR1PRO"

        # UDP 通信
        self.recive_bool = True
        self.local_port = local_port
        self.ctrl_addr = (ctrl_ip, ctrl_port)
        self.recv_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.recv_sock.bind(("127.0.0.1", local_port))
        self.send_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

        # 加载 MJCF
        xml_full = str(Path(__file__).resolve().parent / xml_relpath)
        if not os.path.isfile(xml_full):
            raise FileNotFoundError(f"Cannot find MJCF: {xml_full}")

        self.model = mujoco.MjModel.from_xml_path(xml_full)
        self.model.opt.timestep = DT
        self.data = mujoco.MjData(self.model)

        # 机器人自由度列表
        self.actuator_ids = [a for a in range(self.model.nu)]
        self.dof_num = len(self.actuator_ids)
        print("model_key", model_key)

        # Initialize torque history for CSV
        self.torque_history = []
        self.time_history = []
        self.csv_file = 'joint_torques.csv'
        # Write CSV header
        with open(self.csv_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['time'] + [f'joint_{i}' for i in range(self.dof_num)])

        # 初始化站立姿态
        self._set_initial_pose(model_key)
        self.pyload_old = None
        # 缓存
        self.kp_cmd = np.zeros((self.dof_num, 1), np.float32)
        self.kd_cmd = np.zeros_like(self.kp_cmd)
        self.pos_cmd = np.zeros_like(self.kp_cmd)
        self.vel_cmd = np.zeros_like(self.kp_cmd)
        self.tau_ff = np.zeros_like(self.kp_cmd)
        self.input_tq = np.zeros_like(self.kp_cmd)

        # IMU
        self.last_base_linvel = np.zeros((3, 1), np.float64)
        self.timestamp = 0.0
        self.timestamp_last = 0

        self.virtual_joint_num = virtual_joint_num[self.robot_name]
        if virtual_joint_num[self.robot_name] > 0:
            self.virtual_joint = np.zeros(virtual_joint_num[self.robot_name])
        else:
            self.virtual_joint = None

        print(f"[INFO] MuJoCo model loaded, dof = {self.dof_num}")

        # 可视化
        self.viewer = None
        if USE_VIEWER:
            try:
                self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
                self.viewer.cam.distance = 3.0
                self.viewer.cam.lookat = [0.0, 0.0, 0.8]
                self.viewer.cam.azimuth = 45.0
                self.viewer.cam.elevation = -30.0
            except Exception as e:
                print(f"[WARN] Failed to launch viewer: {e}")

        self.arr_pos = np.array([0., 0., 0.9], dtype=np.float64)
        self.arr_quat = np.array([0.957521915435791,
        -0.0633149966597557,
        -0.27774500846862793,
        -0.04473799839615822], dtype=np.float64)
        dof_list = [-0.011264000087976456,
    0.07964000105857849,
    0.7619820237159729,
    -0.5531160235404968,
    0.3225969970226288,
    -1.2476969957351685,
    1.310541033744812,
    0.0,
    0.0,
    0.0,
    -0.5200409889221191,
    -0.42575201392173767,
    1.0707900524139404,
    1.2147979736328125,
    0.0,
    0.0,
    0.0,
    0.5245800018310547,
    0.22095300257205963,
    0.7726830244064331,
    0.4251680076122284,
    0.0,
    0.0,
    0.5106850266456604,
    0.0630439966917038,
    -0.23514999449253082,
    0.29128000140190125,
    0.0,
    0.0# right_ankle_y/x
                    ]
        self.arr_init_dof = np.array(dof_list, dtype=np.float64)

    def _set_initial_pose(self, key: str):
        """关节位置设置为与 PyBullet 脚本一致的初始角度"""
        qpos0 = self.data.qpos.copy()
        print("qpos0", qpos0.shape)
        print(" self.dof_num", self.dof_num, qpos0[7:7 + self.dof_num].shape)
        qpos0[7:7 + self.dof_num] = URDF_INIT[key]
        qpos0[:3] = np.array([0, 0, 0.5])
        qpos0[3:7] = np.array([0.707, 0, -0.707, 0])
        self.data.qpos[:] = qpos0
        mujoco.mj_forward(self.model, self.data)

    def save_torques_to_csv(self):
        """Save torque data to CSV file"""
        if not self.torque_history or not self.time_history:
            print("[INFO] No torque data to save yet")
            return

        if len(self.torque_history[0]) != self.dof_num:
            print(f"[WARN] Torque array size ({len(self.torque_history[0])}) does not match DoF ({self.dof_num})")
            return

        try:
            with open(self.csv_file, 'a', newline='') as f:
                writer = csv.writer(f)
                for t, torques in zip(self.time_history[-SAVE_INTERVAL:], self.torque_history[-SAVE_INTERVAL:]):
                    writer.writerow([t] + list(torques))
            print(f"[INFO] Torque data appended to {self.csv_file}")
            # Clear history to save memory
            self.torque_history = self.torque_history[-SAVE_INTERVAL:]
            self.time_history = self.time_history[-SAVE_INTERVAL:]
        except Exception as e:
            print(f"[ERROR] Failed to write to CSV: {e}")

    def start(self):
        threading.Thread(target=self._udp_receiver, daemon=True).start()
        print(f"[INFO] UDP receiver on 127.0.0.1:{self.local_port}")

        step = 0
        last_time = time.time()
        init_time = time.time()
        while True:
            if time.time() - last_time >= DT:
                last_time = time.time()
                step += 1

                # Apply control and collect torque data
                if self.tau_ff[0] > 88:
                    self._apply_joint_torque()
                else:
                    self._apply_joint_torque()
                    self._set_mimic_init_state()
                
                # Debug torque calculation
                if self.input_tq.size == 0 or self.input_tq.flatten().shape[0] != self.dof_num:
                    print(f"[WARN] Invalid torque array: size={self.input_tq.flatten().shape[0]}, expected={self.dof_num}")
                    print(f"[DEBUG] kp_cmd shape: {self.kp_cmd.shape}, pos_cmd shape: {self.pos_cmd.shape}")
                    print(f"[DEBUG] kd_cmd shape: {self.kd_cmd.shape}, vel_cmd shape: {self.vel_cmd.shape}")
                else:
                    self.torque_history.append(self.input_tq.flatten().copy())
                    self.time_history.append(self.timestamp)

                # Save to CSV every SAVE_INTERVAL steps
                if step % SAVE_INTERVAL == 0 and self.torque_history:
                    self.save_torques_to_csv()

                # Simulate one step
                mujoco.mj_step(self.model, self.data)
                self.timestamp = time.time() - init_time
                self._send_robot_state()

                # Visualize
                if self.viewer and step % RENDER_INTERVAL == 0:
                    try:
                        self.viewer.sync()
                    except Exception as e:
                        print(f"[WARN] Viewer sync failed: {e}")

    def _udp_receiver(self):
        if self.virtual_joint is not None:
            fmt = f'{self.dof_num+self.virtual_joint_num}f' * 5
            joint_all_num = self.dof_num + self.virtual_joint_num
        else:
            fmt = f'{self.dof_num}f' * 5
            joint_all_num = self.dof_num
        expected = struct.calcsize(fmt)
        print(f"[DEBUG] Expected UDP packet size: {expected} bytes, format: {fmt}")
        while True:
            try:
                data, addr = self.recv_sock.recvfrom(expected)
                if len(data) != expected:
                    print(f"[WARN] UDP packet size {len(data)} != {expected}")
                    continue
                unpacked = struct.unpack(fmt, data)
                print(f"[DEBUG] Received UDP data: {len(unpacked)} values from {addr}")

                if self.virtual_joint is not None:
                    self.kp_cmd = np.asarray(unpacked[0:joint_all_num], dtype=np.float32).reshape(joint_all_num, 1)[:-2]
                    self.pos_cmd = np.asarray(unpacked[joint_all_num:joint_all_num * 2], dtype=np.float32).reshape(joint_all_num, 1)[:-2]
                    self.kd_cmd = np.asarray(unpacked[joint_all_num * 2:joint_all_num * 3], dtype=np.float32).reshape(joint_all_num, 1)[:-2]
                    self.vel_cmd = np.asarray(unpacked[joint_all_num * 3:joint_all_num * 4], dtype=np.float32).reshape(joint_all_num, 1)[:-2]
                    self.tau_ff = np.asarray(unpacked[joint_all_num * 4:], dtype=np.float32).reshape(joint_all_num, 1)[:-2]
                else:
                    self.kp_cmd = np.asarray(unpacked[0:joint_all_num], dtype=np.float32).reshape(joint_all_num, 1)
                    self.pos_cmd = np.asarray(unpacked[joint_all_num:joint_all_num * 2], dtype=np.float32).reshape(joint_all_num, 1)
                    self.kd_cmd = np.asarray(unpacked[joint_all_num * 2:joint_all_num * 3], dtype=np.float32).reshape(joint_all_num, 1)
                    self.vel_cmd = np.asarray(unpacked[joint_all_num * 3:joint_all_num * 4], dtype=np.float32).reshape(joint_all_num, 1)
                    self.tau_ff = np.asarray(unpacked[joint_all_num * 4:], dtype=np.float32).reshape(joint_all_num, 1)
                
                print(f"[DEBUG] Updated kp_cmd shape: {self.kp_cmd.shape}, first few values: {self.kp_cmd[:3].flatten()}")
            except socket.error as e:
                print(f"[ERROR] UDP receive error: {e}")
                continue

    def _apply_joint_torque(self):
        q = self.data.qpos[7:7+self.dof_num].reshape(-1, 1)
        dq = self.data.qvel[6:6+self.dof_num].reshape(-1, 1)
        self.input_tq = (
            self.kp_cmd * (self.pos_cmd - q) +
            self.kd_cmd * (self.vel_cmd - dq)
        )
        print(f"[DEBUG] Applying torque: input_tq shape={self.input_tq.shape}, first few values={self.input_tq[:3].flatten()}")
        self.data.ctrl[:] = self.input_tq.flatten()

    def _set_mimic_init_state(self):
        self.data.qpos[-self.dof_num:] = self.arr_init_dof.astype(np.float64)
        self.data.qpos[0:3] = self.arr_pos.astype(np.float64)
        self.data.qpos[3:7] = self.arr_quat.astype(np.float64)
        self.data.qvel[:] = np.zeros((35, ), dtype=np.float64)

    def quaternion_to_euler(self, q):
        w, x, y, z = q
        t0 = 2.0 * (w * x + y * z)
        t1 = 1.0 - 2.0 * (x * x + y * y)
        roll = np.arctan2(t0, t1)
        t2 = 2.0 * (w * y - z * x)
        t2 = np.clip(t2, -1.0, 1.0)
        pitch = np.arcsin(t2)
        t3 = 2.0 * (w * z + x * y)
        t4 = 1.0 - 2.0 * (y * y + z * z)
        yaw = np.arctan2(t3, t4)
        return np.array([roll, pitch, yaw], dtype=np.float32)

    def _send_robot_state(self):
        q_world = self.data.sensordata[:4]
        rpy = self.quaternion_to_euler(q_world)
        body_acc = self.data.sensordata[4:7]
        angvel_b = self.data.sensordata[7:10]
        q = self.data.qpos[7:7+self.dof_num]
        dq = self.data.qvel[6:6+self.dof_num]
        tau = self.input_tq.flatten()

        if self.virtual_joint is None:
            payload = np.concatenate([
                [self.timestamp],
                rpy.flatten(),
                body_acc.flatten(),
                angvel_b.flatten(),
                q.flatten(),
                dq.flatten(),
                tau.flatten()
            ])
        else:
            payload = np.concatenate([
                [self.timestamp],
                rpy.flatten(),
                body_acc.flatten(),
                angvel_b.flatten(),
                np.concatenate([q.flatten(), self.virtual_joint]),
                np.concatenate([dq.flatten(), self.virtual_joint]),
                np.concatenate([tau.flatten(), self.virtual_joint]),
            ])

        self.pyload_old = payload
        try:
            self.send_sock.sendto(
                struct.pack(f'{len(payload)}f', *payload),
                self.ctrl_addr
            )
        except socket.error as ex:
            print(f"[UDP send] {ex}")

if __name__ == "__main__":
    np.set_printoptions(precision=4, suppress=True)
    sim = MuJoCoSimulation()
    sim.start()