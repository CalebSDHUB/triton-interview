import gc
import os

import numpy as np
import torch

from PIL import Image

from pytriton.model_config import ModelConfig, Tensor
from pytriton.triton import Triton, TritonConfig


class PyTritonServer:

    def __init__(self):
        """
        Initialize the PyTritonServer class and load the model.
        """
        # NOTE: you can set your model name here
        self.model_name = os.getenv("MODEL_NAME", "example")
        # NOTE: you can change the device to "cpu" if you don't have a GPU, otherwise you may perform any optimizations you may find necessary
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # TODO: load model

        # TODO: set up model warmup
        # TODO: explain why this might be necessary
        if os.getenv("WARMUP_MODEL", False):
            self._warmup()

    def _warmup(self):
        """
        Why would you want to warm up your model?
        """
        # TODO: implement model warmup
        # TODO: explain why this might be necessary

    def _infer_fn(self, requests):
        """
        The inference function that will be called by Triton when a request is made.
        
        This processes a list of requests and returns a list of responses, but in this case, we only have one request at a time.
        """
        responses = []
        for req in requests:
            req_data = req.parameters
            
            # TODO: interpret the request data
            
            # TODO: implement model inference
            
            # TODO: transform the model output into a byte array            
            
            responses.append({"result": np.array([...])})

        # NOTE: this is important to free up memory
        gc.collect()
        torch.cuda.empty_cache()
        return responses

    def run(self):
        """
        The main function that runs the Triton server and sets up the bindings.
        """
        with Triton(
            config=TritonConfig(
                # TODO: set up the triton server configuration
            )
        ) as triton:
            triton.bind(
                model_name="",  # TODO: set up the model name
                infer_func=self._infer_fn,
                inputs=[
                    # TODO: set up model inputs
                ],
                outputs=[
                    # TODO: set up model outputs
                    ],
                config=ModelConfig(batching=False),
            )
            triton.serve()


if __name__ == "__main__":
    server = PyTritonServer()
    server.run()
