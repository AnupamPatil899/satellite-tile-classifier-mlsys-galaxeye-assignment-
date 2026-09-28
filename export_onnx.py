import os
import torch
from train import SatelliteCNN
from app.config import settings

def export():
    model_path = os.path.join(settings.BASE_DIR, "models", "classifier.pt")
    onnx_path = os.path.join(settings.BASE_DIR, "models", "classifier.onnx")
    
    model = SatelliteCNN(num_classes=len(settings.CLASSES))
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()

    dummy_input = torch.randn(1, 3, 64, 64, requires_grad=False)

    try:
        # Legacy exporter
        torch.onnx.export(
            model,
            dummy_input,
            onnx_path,
            export_params=True,
            opset_version=14,
            do_constant_folding=True,
            input_names=["input"],
            output_names=["output"],
            dynamic_axes={
                "input": {0: "batch_size"},
                "output": {0: "batch_size"}
            },
            dynamo=False
        )
        print(f"Successfully exported ONNX model to {onnx_path}")
    except TypeError:
        torch.onnx.export(
            model,
            dummy_input,
            onnx_path,
            export_params=True,
            opset_version=14,
            do_constant_folding=True,
            input_names=["input"],
            output_names=["output"],
            dynamic_axes={
                "input": {0: "batch_size"},
                "output": {0: "batch_size"}
            }
        )
        print(f"Successfully exported ONNX model to {onnx_path}")

if __name__ == "__main__":
    export()
