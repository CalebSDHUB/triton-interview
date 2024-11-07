# Instructions
**Please read the instructions carefully before starting the assignment.**

At Cuebric, we often utilize small python projects, which we refer to as "functions", to perform inference on our models. These functions are then deployed on Triton servers to serve as endpoints for our applications. This repository contains a template for such a function, which you can use to create your own functions.

We do not expect this assignment to take more than 2-3 hours. If you find yourself spending more time than that, please make a note of it in your submission. Please ensure that your code is well-documented and that you have provided appropriate logging for the server. Code quality is more important than the completion of the assignment.

For this assignment, you will be creating such a function to perform inference on an open-source diffusion model of your choice via a text-to-image pipeline. You may utilize models available on hugginface, github, or any other repository if you wish to test it locally. You do not need to provide the model weights or the code to download the model weights in your assignment.

The function should take in a bytestring prompt, decode it into a string, run it through the diffusion-based generation model and return a bytestring representing the encoded image as a response. You may also provide additional inputs or outputs alongside motivation or example use cases as you see appropriate. You may use PIL, numpy, or any other library to encode and decode the image. Additionally, the function should have appropriate logging set up to log the setup (initialization), model warmup, requests and responses of the server.

There are various `NOTE:` comments in the codebase that you should pay attention to. In addition, there are some `TODO:` comments that you should complete. Please ensure that you have completed all the `TODO:` comments before submitting your assignment. Furthermore, please make regular commits to your repository with appropriate commit messages. We will be looking at your commit history to understand how you have approached the assignment.

# Requirements
This assignment requires you to have the following installed on your local machine or implicit knowledge of the following:

0. A machine capable of running Linux-based operating systems OR Windows Subsystem for Linux (WSL).
1. Python 3.10 or higher.
2. Docker.
3. Basic knowledge of git.
4. Basic knowledge of Triton Inference Server.
5. Basic knowledge of PyTorch.
6. Basic knowledge of bash.
7. At least ~6GB of free space on your local machine for the virtual environment (without the model weights, which are not required for this assignment). You may use a smaller virtual environment (e.g. install pytorch[cpu]) if you do not have enough space by adjusting the `requirements.txt` file. You may utilize the provided `create_venv_and_install.sh` script to set up the virtual environment, which will also set up a bash alias to activate the venv via `source ~/.bashrc | technical_assessment`.

# Step-by-step
1. Clone this repository to your local machine and set it up as a new project in your favorite IDE.

2. Set up a new virtual environment and install the required packages (via `requirements.txt`, you may use any python package manager of your choice).

3. Set up `main.py` as a Triton server, providing appropriate inputs and output configurations.

    1. You may set up a `client.py` file as a Triton client to test the server.

    2. You may utilize `curl` via the `request.json` file to test the server.

    3. You may utilize any other method to test the server.

    4. **In any case**, ensure you provide appropriate instructions on how to test the server. You may do so in a separate markdown file, at the end of this file, or as a comment in the code.

4. Provide some code in your `main.py` to save some intermediate results during the denoising process of your model. You do not need to provide any images, but you may provide some example code to save the images.

5. Set up a `test.py` file and provide some example tests for the server. These may be unit tests, integration tests, or end-to-end tests via any testing framework of your choice.

6. Set up the `Docekrfile`. You may optionally set up a `docker-compose.yml` file, but it is not required.

7. Push the code to a new repository on your GitHub account. Alternatively, you may use `git bundle` to create a bundle of your repository and send it to us via email.

# Optional

These are optional steps that you may complete if you have time, all of which reflect our expectations of the ideal candidate. While **they are not required**, they will help us understand your experience and expertise better, and we will address similar concepts during the technical part of the interview process regardless. You may choose any one or more of the following steps to complete:

1. Set up a CI/CD pipeline for your repository using GitHub Actions, GitLab CI, or any other CI/CD service of your choice.

2. Apply adjustments to the latent space to improve the quality of the generated images.

3. Adjust `main.py` to perform inference via TensorRT or ONNX Runtime.

4. Utilize a quantized version of the model to reduce the model size and improve inference speed. You do not need to perform the quantization yourself.

5. Implement an open-source super-resolution model to improve the quality of the generated images.

6. Implement an open-source image-to-text pipeline to generate a text prompt from an image.

7. Implement an open-source image-to-image pipeline to improve the quality of the generated images.

8. Instead of doing individual inference, implement a streaming pipeline for faster communication between the client and the server.

9. Provide an architecture diagram of your server and client setup, and how you would integrate it into a larger application. You may use any tool of your choice to create the diagram, but please be prepared to discuss it during the interview process.

# Submission
**You may write any additional notes, explanations, instructions or comments pertaining to the assignment here.**