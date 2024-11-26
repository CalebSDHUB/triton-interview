import gc
import os
import io
import base64
import logging

import numpy as np
import torch

from PIL import Image
from pytriton.model_config import ModelConfig, Tensor
from pytriton.triton import Triton, TritonConfig
from diffusers import StableDiffusionPipeline


class PyTritonServer:
    """
    The inference Triton server (GPU).
    """

    def __init__(self):
        """
        Initialize the PyTritonServer class and load the model.
        """
        # Set up logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

        # NOTE: you can set your model name here
        self.model_name = os.getenv("MODEL_NAME", "Curat/StableDiffusion1.4")

        try:
            # NOTE: you can change the device to "cpu" if you don't have a GPU, otherwise you may perform any optimizations you may find necessary
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.logger.info(f"Using device: {self.device}")

            # Loading model
            self.logger.info(f"Loading model: {self.model_name}")
            self.pipe = StableDiffusionPipeline.from_pretrained(self.model_name)

            # Model computes on chosen device
            self.pipe.to(self.device)
            self.logger.info("Model loaded successfully.")

        except Exception as e:
            self.logger.error(f"Error while initializing the model: {e}", exc_info=True)
            raise

        if os.getenv("WARMUP_MODEL", False):
            self._warmup()

    def _warmup(self):
        """
        Model warm-up is needed to avoid a large latency for the first request
        """
        try:
            warmup_prompt = "A white cat walking in the forest"
            self.logger.info("Warming up the model")
            self.pipe(warmup_prompt)
            self.logger.info("Model warm-up completed.")
        except Exception as e:
            self.logger.error(f"Error during model warm-up: {e}", exc_info=True)

    def _encode_image_to_base64(self, image: Image.Image) -> bytes:
        """
        Encode the given image as a base64 string.

        Args:
            image (Image.Image): The image to encode.

        Returns:
            bytes: The base64 encoded image.
        """
        try:
            raw_bytes = io.BytesIO()
            image.save(raw_bytes, "PNG")
            raw_bytes.seek(0)
            return base64.b64encode(raw_bytes.read())
        except Exception as e:
            self.logger.error(f"Error encoding image to base64: {e}", exc_info=True)
            raise

    def _generate_noise_image(self) -> Image.Image:
        """
        Generate a random noisy image for testing purposes (client-server).

        Returns:
            Image.Image: The generated noise image.
        """
        return Image.fromarray(np.random.randint(0, 256, (512, 512, 3), dtype=np.uint8))

    def _inference(self, prompt: str) -> Image.Image:
        """
        Perform inference on the given prompt and return the generated image.

        Args:
            prompt (str): The prompt to generate the image.

        Returns:
            Image.Image: The generated image.
        """
        try:
            self.logger.info(f"Running model inference with prompt: '{prompt}'")
            return self.pipe(prompt).images[0]
        except Exception as e:
            self.logger.error(f"Error during model inference with prompt '{prompt}': {e}", exc_info=True)
            raise

    def _infer_fn(self, requests):
        """
        The inference function that will be called by Triton when a request is made.

        This processes a list of requests and returns a list of responses, but in this case, we only have one request at a time.

        Args:
            requests (list): A list of requests.

        Returns:
            list: A list of responses.
        """
        responses = []
        try:
            for req in requests:
                # Decode the prompt from the request into a string
                decoded_prompt = req.data["prompt"][0].decode("utf-8")
                self.logger.info(f"Received inference request with prompt: {decoded_prompt}")
                # Perform inference on the decoded prompt
                image = self._inference(decoded_prompt)
                # Converts to raw data and encodes it to base64
                raw_data = self._encode_image_to_base64(image)

                responses.append({"result": np.array([raw_data])})

        except Exception as e:
            self.logger.error(f"Error processing inference requests: {e}", exc_info=True)
            raise
        finally:
            # NOTE: this is important to free up memory
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        return responses

    def run(self):
        """
        The main function that runs the Triton server and sets up the bindings.
        """
        try:
            # Configuring of the Triton server
            self.logger.info("Starting the Triton server...")
            with Triton(
                    config=TritonConfig(
                        http_address="0.0.0.0",
                        http_port="8000",
                        log_verbose=0,
                        exit_on_error=True
                    )
            ) as triton:
                # Binding the model to the Triton server and defines the communication format.
                triton.bind(
                    model_name="stable-diffusion",
                    infer_func=self._infer_fn,
                    inputs=[
                        Tensor(name="prompt", dtype=np.bytes_, shape=(1,)),
                    ],
                    outputs=[
                        Tensor(name="result", dtype=np.bytes_, shape=(1,)),
                    ],
                    config=ModelConfig(batching=False),
                )
                self.logger.info("Model successfully bound to the Triton server")
                triton.serve()

        except Exception as e:
            self.logger.error(f"Error running the Triton server: {e}", exc_info=True)
            raise


if __name__ == "__main__":
    server = PyTritonServer()
    server.run()
