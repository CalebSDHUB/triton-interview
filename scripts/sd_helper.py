import PIL
import torch
import math

import numpy as np
import skimage



# this function is taken from https://github.com/parlance-zz/g-diffuser-bot
def get_matched_noise(_np_src_image, np_mask_rgb, noise_q=1, color_variation=0.05):
    # helper fft routines that keep ortho normalization and auto-shift before and after fft
    def _fft2(data):
        if data.ndim > 2:  # has channels
            out_fft = np.zeros((data.shape[0], data.shape[1], data.shape[2]), dtype=np.complex128)
            for c in range(data.shape[2]):
                c_data = data[:, :, c]
                out_fft[:, :, c] = np.fft.fft2(np.fft.fftshift(c_data), norm="ortho")
                out_fft[:, :, c] = np.fft.ifftshift(out_fft[:, :, c])
        else:  # one channel
            out_fft = np.zeros((data.shape[0], data.shape[1]), dtype=np.complex128)
            out_fft[:, :] = np.fft.fft2(np.fft.fftshift(data), norm="ortho")
            out_fft[:, :] = np.fft.ifftshift(out_fft[:, :])

        return out_fft

    def _ifft2(data):
        if data.ndim > 2:  # has channels
            out_ifft = np.zeros((data.shape[0], data.shape[1], data.shape[2]), dtype=np.complex128)
            for c in range(data.shape[2]):
                c_data = data[:, :, c]
                out_ifft[:, :, c] = np.fft.ifft2(np.fft.fftshift(c_data), norm="ortho")
                out_ifft[:, :, c] = np.fft.ifftshift(out_ifft[:, :, c])
        else:  # one channel
            out_ifft = np.zeros((data.shape[0], data.shape[1]), dtype=np.complex128)
            out_ifft[:, :] = np.fft.ifft2(np.fft.fftshift(data), norm="ortho")
            out_ifft[:, :] = np.fft.ifftshift(out_ifft[:, :])

        return out_ifft

    def _get_gaussian_window(width, height, std=3.14, mode=0):
        window_scale_x = float(width / min(width, height))
        window_scale_y = float(height / min(width, height))

        window = np.zeros((width, height))
        x = (np.arange(width) / width * 2. - 1.) * window_scale_x
        for y in range(height):
            fy = (y / height * 2. - 1.) * window_scale_y
            if mode == 0:
                window[:, y] = np.exp(-(x ** 2 + fy ** 2) * std)
            else:
                window[:, y] = (1 / ((x ** 2 + 1.) * (fy ** 2 + 1.))) ** (std / 3.14)  # hey wait a minute that's not gaussian

        return window

    def _get_masked_window_rgb(np_mask_grey, hardness=1.):
        np_mask_rgb = np.zeros((np_mask_grey.shape[0], np_mask_grey.shape[1], 3))
        if hardness != 1.:
            hardened = np_mask_grey[:] ** hardness
        else:
            hardened = np_mask_grey[:]
        for c in range(3):
            np_mask_rgb[:, :, c] = hardened[:]
        return np_mask_rgb

    width = _np_src_image.shape[0]
    height = _np_src_image.shape[1]
    num_channels = _np_src_image.shape[2]

    _np_src_image[:] * (1. - np_mask_rgb)
    np_mask_grey = (np.sum(np_mask_rgb, axis=2) / 3.)
    img_mask = np_mask_grey > 1e-6
    ref_mask = np_mask_grey < 1e-3

    windowed_image = _np_src_image * (1. - _get_masked_window_rgb(np_mask_grey))
    windowed_image /= np.max(windowed_image)
    windowed_image += np.average(_np_src_image) * np_mask_rgb  # / (1.-np.average(np_mask_rgb))  # rather than leave the masked area black, we get better results from fft by filling the average unmasked color

    src_fft = _fft2(windowed_image)  # get feature statistics from masked src img
    src_dist = np.absolute(src_fft)
    src_phase = src_fft / src_dist

    # create a generator with a static seed to make outpainting deterministic / only follow global seed
    rng = np.random.default_rng(0)

    noise_window = _get_gaussian_window(width, height, mode=1)  # start with simple gaussian noise
    noise_rgb = rng.random((width, height, num_channels))
    noise_grey = (np.sum(noise_rgb, axis=2) / 3.)
    noise_rgb *= color_variation  # the colorfulness of the starting noise is blended to greyscale with a parameter
    for c in range(num_channels):
        noise_rgb[:, :, c] += (1. - color_variation) * noise_grey

    noise_fft = _fft2(noise_rgb)
    for c in range(num_channels):
        noise_fft[:, :, c] *= noise_window
    noise_rgb = np.real(_ifft2(noise_fft))
    shaped_noise_fft = _fft2(noise_rgb)
    shaped_noise_fft[:, :, :] = np.absolute(shaped_noise_fft[:, :, :]) ** 2 * (src_dist ** noise_q) * src_phase  # perform the actual shaping

    brightness_variation = 0.  # color_variation # todo: temporarily tying brightness variation to color variation for now
    contrast_adjusted_np_src = _np_src_image[:] * (brightness_variation + 1.) - brightness_variation * 2.

    # scikit-image is used for histogram matching, very convenient!
    shaped_noise = np.real(_ifft2(shaped_noise_fft))
    shaped_noise -= np.min(shaped_noise)
    shaped_noise /= np.max(shaped_noise)
    shaped_noise[img_mask, :] = skimage.exposure.match_histograms(shaped_noise[img_mask, :] ** 1., contrast_adjusted_np_src[ref_mask, :], channel_axis=1)
    shaped_noise = _np_src_image[:] * (1. - np_mask_rgb) + shaped_noise * np_mask_rgb

    matched_noise = shaped_noise[:]

    return np.clip(matched_noise, 0., 1.)


class Processed:
    def __init__(
        self,
        image: PIL.Image,
        prompt="",
        negative_prompt="",
        seed=42,
        num_inference_steps=10,
        guidance_scale=7.0,
        eta=0.5,
        strength=0.9,
    ):
        self.image = image
        self.prompt = prompt
        self.negative_prompt = negative_prompt
        self.seed = seed
        self.num_inference_steps = num_inference_steps
        self.guidance_scale = guidance_scale
        self.eta = eta
        self.strength = strength
        
        self.mask_blur = None
        self.image_mask = None
        self.latent_mask = None
        self.width = image.width
        self.height = image.height
        self.init_images = [image]
        self.inpaint_full_res = False
        self.inpainting_fill = 1
        self.do_not_save_samples = True
        self.do_not_save_grid = True
        self.mask_blur_x = 0
        self.mask_blur_y = 0
        

class Pipe:
    def __init__(
        self,
        pipe,
        processed=None,
    ):
        self.pipe = pipe
        self.processed = processed
        self.device = self.pipe.device
        
    def __call__(self):
        
        self.seed = self.processed.seed
        self.gen = torch.Generator(self.device)
        self.gen.manual_seed(int(self.seed))
        
        with torch.no_grad() and torch.cuda.amp.autocast() and self.pipe.ema_scope():
            # reload model weights
            self.pipe.load_model()
            
        
        result = self.pipe(
            self.processed.image,
            self.processed.prompt,
            self.processed.negative_prompt,
            self.processed.seed,
            self.processed.num_inference_steps,
            self.processed.guidance_scale,
            self.processed.eta,
            self.processed.strength,
        ).images[0]
        
        return result
