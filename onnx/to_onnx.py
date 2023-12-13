import numpy as np
import torch


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def convert():
    # set the model to inference mode
    torch_model.eval()
    x = torch.randn(batch_size,*input_shape, requires_grad=True)
    x = x.to(device)

    torch.onnx.export(
        torch_model,
        x,
        export_onnx_file,
        opset_version=10,
        export_params=True,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input":{0:"batch_size"}, "output":{0:"batch_size"}}
    )


SET_MODELS = ["MLP", "LSTM"] # 需要转换的模型们

if "MLP" in SET_MODELS:
    torch_model = MLP()
    torch_model.load_state_dict(torch.load("MLP/MLP.pth")) #加载.pth文件
    torch_model.to(device)
    export_onnx_file = "MLP/MLP.onnx"
    batch_size = 1
    input_shape = (3,)  # 模型的输入，根据训练时数据集的输入
    convert()
if "LSTM" in SET_MODELS:
    torch_model = LstmRNN()
    torch_model.load_state_dict(torch.load("LSTM/LSTM.pth")) #加载.pth文件
    torch_model.to(device)
    export_onnx_file = "LSTM/LSTM.onnx"
    batch_size = 1
    input_shape = (1, 1, 3,)  # 模型的输入，根据训练时数据集的输入
    convert()
