import torch
from rsl_rl.modules import ActorCritic
from rsl_rl.algorithms import PPO
# from actor_critic import ActorCritic, get_activation
# from ppo import PPO

export_policy_name = './policy/temp_policy.onnx'


state = torch.rand((1, 76*6), dtype=torch.float32)
# command = torch.rand((1, 2), dtype=torch.float32)
# uncertain = torch.rand((1, 1), dtype=torch.float32)
# reflect = torch.rand((1, 1), dtype=torch.float32)

model = torch.jit.load('./policy/model_10000.pt')

print(model.forward(state))
model.training = False
torch.onnx.export(
    model,  # model to export
    (state),  # inputs of the model,
    export_policy_name,  # filename of the ONNX model
    input_names=["state"],  # Rename inputs for the ONNX model
    output_names=["output"],  # Rename inputs for the ONNX model
    # dynamo=True  # True or False to select the exporter to use
)
print("export success")
#
import onnx

# 加载模型
model = onnx.load(export_policy_name)
# 检查模型格式是否完整及正确
onnx.checker.check_model(model)
# 获取输出层，包含层名称、维度信息
output = model.graph.output
print(output)

import onnxruntime
#
# 创建一个InferenceSession的实例，并将模型的地址传递给该实例
sess = onnxruntime.InferenceSession(export_policy_name)
# 调用实例sess的润方法进行推理
outputs = sess.run(["output"], {"state": state.numpy()})
print(outputs)
