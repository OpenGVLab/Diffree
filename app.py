from __future__ import annotations

import math
import random
import sys
from argparse import ArgumentParser

from tqdm.auto import trange
import einops
import gradio as gr
import k_diffusion as K
import numpy as np
import torch
import torch.nn as nn
from einops import rearrange 
from omegaconf import OmegaConf
from PIL import Image, ImageOps, ImageFilter
from torch import autocast
import cv2
import imageio

sys.path.append("./stable_diffusion")

from stable_diffusion.ldm.util import instantiate_from_config

class CFGDenoiser(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.inner_model = model

    def forward(self, z_0, z_1, sigma, cond, uncond, text_cfg_scale, image_cfg_scale):
        cfg_z_0 = einops.repeat(z_0, "1 ... -> n ...", n=3)
        cfg_z_1 = einops.repeat(z_1, "1 ... -> n ...", n=3)
        cfg_sigma = einops.repeat(sigma, "1 ... -> n ...", n=3)
        cfg_cond = {
            "c_crossattn": [torch.cat([cond["c_crossattn"][0], uncond["c_crossattn"][0], uncond["c_crossattn"][0]])],
            "c_concat": [torch.cat([cond["c_concat"][0], cond["c_concat"][0], uncond["c_concat"][0]])],
        }
        output_0, output_1 = self.inner_model(cfg_z_0, cfg_z_1, cfg_sigma, cond=cfg_cond)
        out_cond_0, out_img_cond_0, out_uncond_0 = output_0.chunk(3)
        out_cond_1, _, _ = output_1.chunk(3)
        return out_uncond_0 + text_cfg_scale * (out_cond_0 - out_img_cond_0) + image_cfg_scale * (out_img_cond_0 - out_uncond_0), \
            out_cond_1

def load_model_from_config(config, ckpt, vae_ckpt=None, verbose=False):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    if "global_step" in pl_sd:
        print(f"Global Step: {pl_sd['global_step']}")
    sd = pl_sd["state_dict"]
    if vae_ckpt is not None:
        print(f"Loading VAE from {vae_ckpt}")
        vae_sd = torch.load(vae_ckpt, map_location="cpu")["state_dict"]
        sd = {
            k: vae_sd[k[len("first_stage_model.") :]] if k.startswith("first_stage_model.") else v
            for k, v in sd.items()
        }
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=True)
    if len(m) > 0 and verbose:
        print("missing keys:")
        print(m)
    if len(u) > 0 and verbose:
        print("unexpected keys:")
        print(u)
    return model

def append_dims(x, target_dims):
    """Appends dimensions to the end of a tensor until it has target_dims dimensions."""
    dims_to_append = target_dims - x.ndim
    if dims_to_append < 0:
        raise ValueError(f'input has {x.ndim} dims but target_dims is {target_dims}, which is less')
    return x[(...,) + (None,) * dims_to_append]

class CompVisDenoiser(K.external.CompVisDenoiser):
    def __init__(self, model, quantize=False, device='cpu'):
        super().__init__(model, quantize, device)
    
    def get_eps(self, *args, **kwargs):
        return self.inner_model.apply_model(*args, **kwargs)
    
    def forward(self, input_0, input_1, sigma, **kwargs):
        c_out, c_in = [append_dims(x, input_0.ndim) for x in self.get_scalings(sigma)]
        # eps_0, eps_1 = self.get_eps(input_0 * c_in, input_1 * c_in, self.sigma_to_t(sigma), **kwargs)
        eps_0, eps_1 = self.get_eps(input_0 * c_in, self.sigma_to_t(sigma).cuda(), **kwargs)
        
        return input_0 + eps_0 * c_out, eps_1

def to_d(x, sigma, denoised):
    """Converts a denoiser output to a Karras ODE derivative."""
    return (x - denoised) / append_dims(sigma, x.ndim)

def default_noise_sampler(x):
    return lambda sigma, sigma_next: torch.randn_like(x)

def get_ancestral_step(sigma_from, sigma_to, eta=1.):
    """Calculates the noise level (sigma_down) to step down to and the amount
    of noise to add (sigma_up) when doing an ancestral sampling step."""
    if not eta:
        return sigma_to, 0.
    sigma_up = min(sigma_to, eta * (sigma_to ** 2 * (sigma_from ** 2 - sigma_to ** 2) / sigma_from ** 2) ** 0.5)
    sigma_down = (sigma_to ** 2 - sigma_up ** 2) ** 0.5
    return sigma_down, sigma_up

def decode_mask(mask, height = 256, width = 256):
    mask = nn.functional.interpolate(mask, size=(height, width), mode="bilinear", align_corners=False)
    mask = torch.where(mask > 0, 1, -1)  # Thresholding step
    mask = torch.clamp((mask + 1.0) / 2.0, min=0.0, max=1.0)
    mask = 255.0 * rearrange(mask, "1 c h w -> h w c")
    mask = torch.cat([mask, mask, mask], dim=-1)
    mask = mask.type(torch.uint8).cpu().numpy()
    return mask

def sample_euler_ancestral(model, x_0, x_1, sigmas, height, width, extra_args=None, disable=None, eta=1., s_noise=1., noise_sampler=None):
    """Ancestral sampling with Euler method steps."""
    extra_args = {} if extra_args is None else extra_args
    noise_sampler = default_noise_sampler(x_0) if noise_sampler is None else noise_sampler
    s_in = x_0.new_ones([x_0.shape[0]])

    mask_list = []
    image_list = []
    for i in trange(len(sigmas) - 1, disable=disable):
        denoised_0, denoised_1 = model(x_0, x_1, sigmas[i] * s_in, **extra_args)
        image_list.append(denoised_0)

        sigma_down, sigma_up = get_ancestral_step(sigmas[i], sigmas[i + 1], eta=eta)
        d_0 = to_d(x_0, sigmas[i], denoised_0)
        
        # Euler method
        dt = sigma_down - sigmas[i]
        x_0 = x_0 + d_0 * dt

        if sigmas[i + 1] > 0:
            x_0 = x_0 + noise_sampler(sigmas[i], sigmas[i + 1]) * s_noise * sigma_up

        x_1 = denoised_1
        mask_list.append(decode_mask(x_1, height, width))
        
    image_list = torch.cat(image_list, dim=0)

    return x_0, x_1, image_list, mask_list

parser = ArgumentParser()
parser.add_argument("--resolution", default=512, type=int)
parser.add_argument("--config", default="config/generate.yaml", type=str)
parser.add_argument("--ckpt", default="checkpoints/diffree-step=000010999.ckpt", type=str)
parser.add_argument("--vae-ckpt", default=None, type=str)
args = parser.parse_args()

config = OmegaConf.load(args.config)
model = load_model_from_config(config, args.ckpt, args.vae_ckpt)
model.eval().cuda()
model_wrap = CompVisDenoiser(model)
model_wrap_cfg = CFGDenoiser(model_wrap)
null_token = model.get_learned_conditioning([""])

def generate(
    input_image: Image.Image,
    instruction: str,
    steps: int,
    randomize_seed: bool,
    seed: int,
    randomize_cfg: bool,
    text_cfg_scale: float,
    image_cfg_scale: float,
    weather_close_video: bool,
    decode_image_batch: int
):
    seed = random.randint(0, 100000) if randomize_seed else seed
    text_cfg_scale = round(random.uniform(6.0, 9.0), ndigits=2) if randomize_cfg else text_cfg_scale
    image_cfg_scale = round(random.uniform(1.2, 1.8), ndigits=2) if randomize_cfg else image_cfg_scale

    width, height = input_image.size
    factor = args.resolution / max(width, height)
    factor = math.ceil(min(width, height) * factor / 64) * 64 / min(width, height)
    width = int((width * factor) // 64) * 64
    height = int((height * factor) // 64) * 64
    input_image = ImageOps.fit(input_image, (width, height), method=Image.Resampling.LANCZOS)
    input_image_copy = input_image.convert("RGB")

    if instruction == "":
        return [input_image, seed]
    
    model.cuda()
    with torch.no_grad(), autocast("cuda"), model.ema_scope():
        cond = {}
        cond["c_crossattn"] = [model.get_learned_conditioning([instruction]).to(model.device)]
        input_image = 2 * torch.tensor(np.array(input_image)).float() / 255 - 1
        input_image = rearrange(input_image, "h w c -> 1 c h w").to(model.device)
        cond["c_concat"] = [model.encode_first_stage(input_image).mode().to(model.device)]

        uncond = {}
        uncond["c_crossattn"] = [null_token.to(model.device)]
        uncond["c_concat"] = [torch.zeros_like(cond["c_concat"][0])]
        

        sigmas = model_wrap.get_sigmas(steps).to(model.device)

        extra_args = {
            "cond": cond,
            "uncond": uncond,
            "text_cfg_scale": text_cfg_scale,
            "image_cfg_scale": image_cfg_scale,
        }
        torch.manual_seed(seed)
        z_0 = torch.randn_like(cond["c_concat"][0]).to(model.device) * sigmas[0]
        z_1 = torch.randn_like(cond["c_concat"][0]).to(model.device) * sigmas[0]
        
        z_0, z_1, image_list, mask_list = sample_euler_ancestral(model_wrap_cfg, z_0, z_1, sigmas, height, width, extra_args=extra_args)
        
        x_0 = model.decode_first_stage(z_0)

        if model.first_stage_downsample:
            x_1 = nn.functional.interpolate(z_1, size=(height, width), mode="bilinear", align_corners=False)
            x_1 = torch.where(x_1 > 0, 1, -1)  # Thresholding step
        else:
            x_1 = model.decode_first_stage(z_1)
        
        x_0 = torch.clamp((x_0 + 1.0) / 2.0, min=0.0, max=1.0)
        x_1 = torch.clamp((x_1 + 1.0) / 2.0, min=0.0, max=1.0)
        x_0 = 255.0 * rearrange(x_0, "1 c h w -> h w c")
        x_1 = 255.0 * rearrange(x_1, "1 c h w -> h w c")
        x_1 = torch.cat([x_1, x_1, x_1], dim=-1)
        edited_image = Image.fromarray(x_0.type(torch.uint8).cpu().numpy())
        edited_mask = Image.fromarray(x_1.type(torch.uint8).cpu().numpy())

        image_video_path = None
        if not weather_close_video:
            image_video = []
            
            for i in range(0, len(image_list), decode_image_batch):
                if i + decode_image_batch < len(image_list):
                    tmp_image_list = image_list[i:i+decode_image_batch]
                else:
                    tmp_image_list = image_list[i:]
                tmp_image_list = model.decode_first_stage(tmp_image_list)
                tmp_image_list = torch.clamp((tmp_image_list + 1.0) / 2.0, min=0.0, max=1.0)
                tmp_image_list = 255.0 * rearrange(tmp_image_list, "b c h w -> b h w c")
                tmp_image_list = tmp_image_list.type(torch.uint8).cpu().numpy()
                # image list to image
                for image in tmp_image_list:
                    image_video.append(image)

            image_video_path = "image.mp4"
            fps = 30
            with imageio.get_writer(image_video_path, fps=fps) as video:
                for image in image_video:
                    video.append_data(image)

        # 对edited_mask做膨胀
        edited_mask_copy = edited_mask.copy()
        kernel = np.ones((3, 3), np.uint8)
        edited_mask = cv2.dilate(np.array(edited_mask), kernel, iterations=3)
        edited_mask = Image.fromarray(edited_mask)


        m_img = edited_mask.filter(ImageFilter.GaussianBlur(radius=3))
        m_img = np.asarray(m_img).astype('float') / 255.0
        img_np = np.asarray(input_image_copy).astype('float') / 255.0
        ours_np = np.asarray(edited_image).astype('float') / 255.0

        mix_image_np =  m_img * ours_np + (1 - m_img) * img_np
        mix_image = Image.fromarray((mix_image_np * 255).astype(np.uint8)).convert('RGB')


        red = np.array(mix_image).astype('float') * 1
        red[:, :, 0] = 180.0
        red[:, :, 2] = 0
        red[:, :, 1] = 0
        mix_result_with_red_mask = np.array(mix_image)
        mix_result_with_red_mask = Image.fromarray(
            (mix_result_with_red_mask.astype('float') * (1 - m_img.astype('float') / 2.0) +
            m_img.astype('float') / 2.0 * red).astype('uint8'))



        mask_video_path = "mask.mp4"
        fps = 30
        with imageio.get_writer(mask_video_path, fps=fps) as video:
            for image in mask_list:
                video.append_data(image)

        return [int(seed), text_cfg_scale, image_cfg_scale, edited_image, mix_image, edited_mask_copy, mask_video_path, image_video_path, input_image_copy, mix_result_with_red_mask]

def reset():
    return [100, "Randomize Seed", 1372, "Fix CFG", 7.5, 1.5, None, None, None, None, None, None, None, "Close Image Video", 10]

def get_example():
    return [
        ["example_images/dufu.png", "black and white suit", 100, "Fix Seed", 1372, "Fix CFG", 7.5, 1.5],
        ["example_images/girl.jpeg", "reflective sunglasses", 100, "Fix Seed", 1372, "Fix CFG", 7.5, 1.5],
        ["example_images/road_sign.png", "stop sign", 100, "Fix Seed", 1372, "Fix CFG", 7.5, 1.5],
        ["example_images/dufu.png", "blue medical mask", 100, "Fix Seed", 1372, "Fix CFG", 7.5, 1.5],
        ["example_images/people_standing.png", "dark green pleated skirt", 100, "Fix Seed", 1372, "Fix CFG", 7.5, 1.5],
        ["example_images/girl.jpeg", "shiny golden crown", 100, "Fix Seed", 1372, "Fix CFG", 7.5, 1.5],
        ["example_images/dufu.png", "sunglasses", 100, "Fix Seed", 1372, "Fix CFG", 7.5, 1.5],
        ["example_images/girl.jpeg", "diamond necklace", 100, "Fix Seed", 1372, "Fix CFG", 7.5, 1.5],
        ["example_images/iron_man.jpg", "sunglasses", 100, "Fix Seed", 1372, "Fix CFG", 7.5, 1.5],
        ["example_images/girl.jpeg", "the queen's crown", 100, "Fix Seed", 1372, "Fix CFG", 7.5, 1.5],
        ["example_images/girl.jpeg", "gorgeous yellow gown", 100, "Fix Seed", 1372, "Fix CFG", 7.5, 1.5],
    ]

with gr.Blocks(css="footer {visibility: hidden}") as demo:
    with gr.Row():
        gr.Markdown(
            "<div align='center'><font size='14'>Diffree: Text-Guided Shape Free Object Inpainting with Diffusion Model</font></div>"  # noqa
        )

    with gr.Row():
        with gr.Column(scale=1, min_width=100):
            with gr.Row():
                input_image = gr.Image(label="Input Image", type="pil", interactive=True)
            with gr.Row():
                instruction = gr.Textbox(lines=1, label="Object description", interactive=True)
            with gr.Row():
                steps = gr.Number(value=100, precision=0, label="Steps", interactive=True)
                randomize_seed = gr.Radio(
                    ["Fix Seed", "Randomize Seed"],
                    value="Randomize Seed",
                    type="index",
                    label="Seed Selection",
                    show_label=False,
                    interactive=True,
                )
                seed = gr.Number(value=1372, precision=0, label="Seed", interactive=True)
                randomize_cfg = gr.Radio(
                    ["Fix CFG", "Randomize CFG"],
                    value="Fix CFG",
                    type="index",
                    label="CFG Selection",
                    show_label=False,
                    interactive=True,
                )
                text_cfg_scale = gr.Number(value=7.5, label=f"Text CFG", interactive=True)
                image_cfg_scale = gr.Number(value=1.5, label=f"Image CFG", interactive=True)
            with gr.Row():
                reset_button = gr.Button("Reset")
                generate_button = gr.Button("Generate")
        with gr.Column(scale=1, min_width=100):
            with gr.Column():
                mix_image = gr.Image(label=f"Mix Image", type="pil", interactive=False)
            with gr.Column():
                edited_mask = gr.Image(label=f"Output Mask", type="pil", interactive=False)
    
    
    with gr.Accordion('More outputs', open=False):
        with gr.Row():
            weather_close_video = gr.Radio(
                ["Show Image Video", "Close Image Video"],
                value="Close Image Video",
                type="index",
                label="Image Generation Process Selection ()",
                interactive=True,
            )
            decode_image_batch = gr.Number(value=10, precision=0, label="Decode Image Batch (<steps)", interactive=True)
        with gr.Row():
            image_video = gr.Video(label="Image Video of Generation Process")
            mask_video = gr.Video(label="Mask Video of Generation Process")
        with gr.Row():
            original_image = gr.Image(label=f"Original Image", type="pil", interactive=False)
            edited_image = gr.Image(label=f"Output Image", type="pil", interactive=False)
            mix_result_with_red_mask = gr.Image(label=f"Mix Image With Red Mask", type="pil", interactive=False)
            
    
    with gr.Row():
        gr.Examples(
            examples=get_example(),
            fn=generate,
            inputs=[input_image, instruction, steps, randomize_seed, seed, randomize_cfg, text_cfg_scale, image_cfg_scale],
            outputs=[seed, text_cfg_scale, image_cfg_scale, edited_image, mix_image, edited_mask, mask_video, image_video, original_image, mix_result_with_red_mask],
            cache_examples=False,
        )
    
    generate_button.click(
        fn=generate,
        inputs=[
            input_image,
            instruction,
            steps,
            randomize_seed,
            seed,
            randomize_cfg,
            text_cfg_scale,
            image_cfg_scale,
            weather_close_video,
            decode_image_batch
        ],
        outputs=[seed, text_cfg_scale, image_cfg_scale, edited_image, mix_image, edited_mask, mask_video, image_video, original_image, mix_result_with_red_mask],
    )
    reset_button.click(
        fn=reset,
        inputs=[],
        outputs=[steps, randomize_seed, seed, randomize_cfg, text_cfg_scale, image_cfg_scale, edited_image, mix_image, edited_mask, mask_video, image_video, original_image, mix_result_with_red_mask, weather_close_video, decode_image_batch],
    )


# demo.queue(concurrency_count=1)
# demo.launch(share=True)


demo.queue().launch()
