# import torch
# import torchvision

# # 1. Load the model
# model = torchvision.models.resnet18(weights="DEFAULT")
# model.eval()

# # 2. Create dummy input
# dummy_input = torch.randn(1, 3, 224, 224)

# # 3. Export using the STABLE API
# print("Exporting model to ONNX...")
# torch.onnx.export(
#     model, 
#     dummy_input, 
#     "resnet18.onnx",
#     input_names=["input"], 
#     output_names=["output"],
#     # Optional: Allow the batch size (dimension 0) to change
#     dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}}
# )
# print("Model exported successfully as 'resnet18.onnx'")

import onnx
model = onnx.load("resnet18.onnx")
onnx.checker.check_model(model)
print(onnx.helper.printable_graph(model.graph))