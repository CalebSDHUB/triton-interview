# argparse
import argparse
import warnings

import onnxruntime
import torch



if __name__ == "__main__":
    # TODO: Run the script with the correct arguments
    # e.g. python scripts/onnx_export.py -c checkpoint.pth -m type -p result.onnx
    parser = argparse.ArgumentParser()
    # model checkpoint
    parser.add_argument(
        "-c", "--checkpoint", type=str, default="checkpoint.pth"
    )
    # model type
    parser.add_argument("-m", "--model_type", type=str, default="type")
    # onnx model path
    parser.add_argument("-p", "--onnx_model_path", type=str, default="result.onnx")

    model_type = parser.parse_args().model_type
    checkpoint = parser.parse_args().checkpoint
    onnx_model_path = parser.parse_args().onnx_model_path

    optimization = onnxruntime.__version__.startswith("1.15")
    quantization = onnxruntime.__version__.startswith(
        "1.15"
    ) or onnxruntime.__version__.startswith("1.16")


    # TODO: Implement model init here
    onnx_model = None

    # TODO: Implement model forward here
    dynamic_axes = {
        "point_coords": {1: "num_points"},
        "point_labels": {1: "num_points"},
    }
    # TODO: Implement dummy inputs here
    dummy_inputs = {
        # "image": torch.randn(3, 720, 1280, dtype=torch.float),
    }
    # TODO: Implement results here
    output_names = ["result"]

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=torch.jit.TracerWarning)
        warnings.filterwarnings("ignore", category=UserWarning)
        with open(onnx_model_path, "wb") as f:
            torch.onnx.export(
                onnx_model,
                tuple(dummy_inputs.values()),
                f,
                export_params=True,
                verbose=False,
                opset_version=17, # TODO: set to the correct opset version
                do_constant_folding=True,
                input_names=list(dummy_inputs.keys()),
                output_names=output_names,
                dynamic_axes=dynamic_axes,
            )

    if quantization:
        from onnxruntime.quantization import QuantType
        from onnxruntime.quantization.quantize import quantize_dynamic

        quantization_params = {
            "model_input": onnx_model_path,
            "model_output": f"quantized_{onnx_model_path}",
            "per_channel": False,
            "reduce_range": False,
            "weight_type": QuantType.QUInt8,
        }
        if optimization:
            quantization_params["optimize_model"] = True
        quantize_dynamic(
            **quantization_params,
        )
