import numpy as np
import torch
from Img2Latex.Model import myYOLO

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


SET_MODELS = ["MyYolo"] # 需要转换的模型们


if "MyYolo" in SET_MODELS:
    torch_model = myYOLO(device, input_size=416, num_classes=1, trainable=False)
    torch_model.load_state_dict(torch.load("../Img2Latex/out/model_80.pth")) #加载.pth文件
    torch_model.to(device)
    export_onnx_file = "../Img2Latex/out/myYolo.onnx"
    batch_size = 1
    input_shape = (1, 3, 416, 416)  # 模型的输入，根据训练时数据集的输入
    convert()
