import onnxruntime as ort
import numpy as np

# 加载ONNX模型
session = ort.InferenceSession("/home/ubuntu/rl_deploy_for_Cr1-beyondmimic/policy/policy_getup.onnx")

# 准备输入数据
obs_input = np.zeros((1, 118), dtype=np.float32)  # in_features需要替换为实际值
time_step_input = np.zeros((1, 1), dtype=np.float32)

# 推理
inputs = {
    "obs": obs_input,
    "time_step": time_step_input
}
outputs = session.run(None, inputs)

# 输出结果
actions, joint_pos, joint_vel, body_pos_w, body_quat_w, body_lin_vel_w, body_ang_vel_w = outputs
import ipdb;ipdb.set_trace()