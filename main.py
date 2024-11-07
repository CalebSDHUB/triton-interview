import gc
import os

import cv2
import numpy as np
import torch
from PIL import Image
from helpers import helpers

from pytriton.model_config import ModelConfig, Tensor
from pytriton.triton import Triton, TritonConfig


class PyTritonServer:
    """triton server for segment anything"""

    def __init__(self):
        self.model_name = os.getenv("MODEL_NAME", "example")
        self.logger = helpers.get_logger()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.mode = os.getenv("MODE", "development")
        self.model_path = os.getenv("LOCAL_MODEL_PATH", "models/checkpoint.pth")

        self.logger.info("Loading model...")
        ...

        if os.getenv("WARMUP_MODEL", False):
            self.logger.info("Warming up model...")
            self._warmup()

    def _warmup(self):
        """warmup model"""
        self.logger.info("Warming up model...")
        ...
        self.logger.info("Model warmed up!")

    def _infer_fn(self, requests):
        """infer function"""
        responses = []
        for req in requests:
            req_data = req.data
            req_parameters = helpers.uppercase_keys(req.parameters)
            self.logger.info(f"Received request: {req_parameters}")
            self.logger.info("Processing input...")

            # get input and output paths
            output_path = helpers.get_output_path(req_parameters)
            input_path = helpers.get_input_path(req_parameters)
            max_msg_size = helpers.get_max_msg_size(req_parameters)
            req_id = helpers.get_request_id(req_parameters)
            asset_ids = helpers.get_asset_id(req_parameters)


            image = None
            prompt = ''
            seed = 0
            
            try:
                image_asset_name = helpers.numpy_array_to_variable(req_data.get("image"))
                image = Image.open(os.path.join(input_path, image_asset_name))
            except Exception as e:
                self.logger.warning("No image provided")


            try:
                mask_rgb_asset_name = helpers.numpy_array_to_variable(req_data.get("mask_rgb"))
                mask_rgb = Image.open(os.path.join(input_path, mask_rgb_asset_name))
            except Exception:
                self.logger.warning("No mask data provided")
            

            try:
                prompt = helpers.numpy_array_to_variable(req_data.get("prompt", ""))
                negative_prompt = helpers.numpy_array_to_variable(
                    req_data.get("negative_prompt", None)
                )
            except Exception:
                self.logger.warning("No prompt provided")


            try:
                seed = helpers.numpy_array_to_variable(req_data.get("seed", 0))
                num_inference_steps = helpers.numpy_array_to_variable(
                    req_data.get("num_inference_steps", 50)
                )
                guidance_scale = helpers.numpy_array_to_variable(
                    req_data.get("guidance_scale", 7.0)
                )
                eta = helpers.numpy_array_to_variable(req_data.get("eta", 0.5))
                strength = helpers.numpy_array_to_variable(req_data.get("strength", 0.9))
            except Exception:
                self.logger.warning("No SD params provided")

            if seed is not None:
                gen = torch.Generator(torch.device("cuda"))
                gen.manual_seed(int(seed))
            else:
                gen = None

            # Preprocess image or input
            if image is not None:
                init_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            
                # Process image or input
                self.logger.info("Processing...")
                output = Image.fromarray(init_image)
                
            # Postprocess image or output
                image = Image.fromarray(output)
            else:
                result = "example"

            # Encode output to easily transfer response back
            # Note that there is a max message size limit
            # If the response is too large, only a reference to download the file will be returned
            if image is not None:
                self.logger.info("Encoding image...")
                img_encoded = helpers.encode_image_to_base64(image)
            elif result is not None:
                self.logger.info("Encoding result...")
                result_encoded = helpers.encode_text_to_base64(result)

            # Return response
            self.logger.info("Image generated successfully!")
            if img_encoded is not None:
                responses.append({"image_generated": np.array([img_encoded])})
            elif result_encoded is not None:
                responses.append({"result": np.array([result_encoded])})
            else:
                responses.append({"error": "No image or result generated"})

        self.logger.info("Returning responses...")
        gc.collect()
        torch.cuda.empty_cache()
        return responses

    def run(self):
        """run triton server"""
        with Triton(
            config=TritonConfig(
                allow_http=True,
                http_port=8000,
                grpc_port=8001,
                metrics_port=8002,
                allow_metrics=True,
                allow_gpu_metrics=True,
                allow_cpu_metrics=True,
                metrics_interval_ms=500,
                http_header_forward_pattern="(nvcf-.*|NVCF-.*)",
                strict_readiness=True,
            )
        ) as triton:
            triton.bind(
                model_name=os.getenv("MODEL_NAME", "example"),
                infer_func=self._infer_fn,
                inputs=[
                    # Tensor(name="image", dtype=np.bytes_, shape=(1,)),
                    Tensor(name="prompt", dtype=np.bytes_, shape=(1,)),
                    # Tensor(name="seed", dtype=np.uint64, shape=(1,)),
                    # Tensor(name="num_inference_steps", dtype=np.uint16, shape=(1,)),
                    # Tensor(name="strength", dtype=np.float32, shape=(1,)),
                ],
                outputs=[
                    # Tensor(name="image_generated", dtype=np.bytes_, shape=(1,)),
                    Tensor(name="result", dtype=np.bytes_, shape=(1,)),
                    ],
                config=ModelConfig(batching=False),
            )
            triton.serve()


if __name__ == "__main__":
    server = PyTritonServer()
    server.run()
