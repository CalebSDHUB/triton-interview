import os
import json
import io
import base64
import numpy as np
import logging

from typing import Dict
from PIL import Image
from pytriton.client import ModelClient


class PyTritonClient:
    """
    A Triton client to interact with the Triton server.
    """

    def __init__(
        self, server_url: str = "localhost", 
        model_name: str = "stable-diffusion",
        inference_timeout: int = 60
        ):
        """
        Initialize the PyTritonClient class.

        Args:
            server_url (str): The URL of the Triton server.
            model_name (str): The name matching the Triton server model name.
            inference_timeout (int): The timeout for the inference request.
        """
        # Set up logging
        logging.basicConfig(
            level=logging.INFO, 
            filemode='a',
            format='%(asctime)s - %(levelname)s - %(message)s'
            )
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"Initializing Triton client for model: {model_name}")

        self.client = ModelClient(
            server_url,
            model_name=model_name,
            inference_timeout_s=inference_timeout
            )

    def _infer_sample(self, prompt: str):
        """
        Send an inference request to the Triton server with the given prompt.

        Args:
            prompt (str): The prompt to generate the image.

        Returns:
            dict: The response from the Triton server.
        """
        try:
            # Encode the prompt as a byte array
            encoded_prompt = np.char.encode(np.array(prompt), "utf-8")
            self.logger.info(f"Sending inference request with prompt: {prompt}")
            
            # Send the inference request to the Triton server
            response = self.client.infer_sample(encoded_prompt)
            self.logger.info("Inference request completed successfully.")
            return response
        except Exception as e:
            self.logger.error(f"Error during inference with prompt '{prompt}': {e}", exc_info=True)
            raise

    def _load_file(self, file_path: str) -> Dict[str, str]:
        """
        Load the file from the given path.

        Args:
            file_path (str): The path to the file.

        Returns:
            dict: The loaded JSON data.
        """
        try:
            with open(file_path, "r") as f:
                request_data = json.load(f)
                self.logger.info(f"Loaded JSON file successfully from: {file_path}")
            return request_data
        except FileNotFoundError as e:
            self.logger.error(f"Request JSON file not found: {file_path}", exc_info=True)
            raise
        except json.JSONDecodeError as e:
            self.logger.error(f"Error decoding JSON file: {file_path}", exc_info=True)
            raise
        except Exception as e:
            self.logger.error(f"Unexpected error while loading file {file_path}: {e}", exc_info=True)
            raise

    def _request_image_with_prompt(self, file_path: str):
        """
        Send a prompt to the server, receive the generated image, and save it.

        Args:
            file_path (str): The path to the JSON file.
        """
        try:
            # Load the request JSON data
            self.logger.info(f"Loading request data from: {file_path}")
            request_data = self._load_file(file_path)

            # Extract the prompt from the request data
            prompt = request_data["inputs"][0]["data"]
            self.logger.info(f"Prompt extracted from request data: {prompt}")

            # Send the inference request to the Triton server
            response = self._infer_sample(prompt)

            # Extract the image data (payload) from the response
            response_payload = response["result"][0]

            # Convert the byte array into a PIL image
            image = Image.open(io.BytesIO(base64.b64decode(response_payload)))
            self.logger.info("Image successfully generated from the response payload.")

            # Ensure the results directory exists
            os.makedirs("results", exist_ok=True)
            self.logger.info("Created 'results' directory for saving images.")

            # Save the image on the disk
            image_path = os.path.join("results", "generated_image.png")
            image.save(image_path)
            self.logger.info(f"Image saved as {image_path}")
        except KeyError as e:
            self.logger.error(f"Missing key in response or request data: {e}", exc_info=True)
        except Exception as e:
            self.logger.error(f"Unexpected error while requesting image from server: {e}", exc_info=True)
            raise

    def run(self, file_path: str):
        """
        Run the client with the given request JSON file.

        Args:
            file_path (str): The path to the request JSON file.
        """
        try:
            self.logger.info("Starting image generation process.")
            # Generate the image using the request JSON file
            self._request_image_with_prompt(file_path)
        except Exception as e:
            self.logger.error(f"Error during the image generation process: {e}", exc_info=True)


if __name__ == "__main__":
    try:
        client = PyTritonClient()
        client.run("request.json")
    except Exception as e:
        logging.error(f"Client execution failed: {e}", exc_info=True)
