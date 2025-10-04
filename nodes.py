import os
import re
from glob import glob
import torch
from torchvision.transforms import InterpolationMode
import torchvision.transforms.v2.functional as F
from llama_cpp import Llama
from scipy import ndimage
import numpy as np
import node_helpers
import folder_paths
import comfy
from nodes import LoraLoader
from comfy_api.latest import io
import math

class PadAndResize:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            'required': {
                'image': ('IMAGE',),
                'latent': ('LATENT',),
                'mode': (['resize', 'crop', 'fit'],),
                'left': ('FLOAT', {'min': 0, 'step': .01, 'default': 0}),
                'top': ('FLOAT', {'min': 0, 'step': .01, 'default': 0}),
                'right': ('FLOAT', {'min': 0, 'step': .01, 'default': 0}),
                'bottom': ('FLOAT', {'min': 0, 'step': .01, 'default': 0}),
                }
            }
    
    RETURN_TYPES = ('IMAGE',)
    
    FUNCTION = 'run'

    CATEGORY = 'davcha'

    def run(self, image, latent, mode, left, top, right, bottom):
        image = image.permute(0,3,1,2)
        image = self.pad(image, left, top, right, bottom)
        image = self.resize(image, latent, mode)
        image = image.permute(0,2,3,1)
        return (image,)
    
    def resize(self, image, latent, mode):
        b, c, sh, sw = image.shape
        *_, th, tw = latent['samples'].shape
        th, tw = th * 8, tw * 8
        if (th, tw) == (sh, sw):
            return image
        if mode == 'resize':
            result = F.resize(image, (th, tw), InterpolationMode.NEAREST_EXACT)
        else:
            fun = max if mode == 'crop' else min
            scale = fun(torch.tensor((th/sh,tw/sw)))
            shape = tuple(int(x * scale) for x in [sh,sw])
            mask = torch.ones((1, sh, sw))
            mask = F.resize(mask, shape, InterpolationMode.NEAREST_EXACT)
            result = F.resize(image, shape, InterpolationMode.NEAREST_EXACT)
            phw = (
                (tw - int(sw*scale))//2,
                (th - int(sh*scale))//2,
                (tw - int(sw*scale)+1)//2,
                (th - int(sh*scale)+1)//2,
            )
            mask = F.pad(mask, phw, fill=0, padding_mode='constant')
            if mask.sum() != np.prod(mask.shape[-2:]):
                dist = ndimage.distance_transform_edt(mask)
                soft_m = np.minimum(dist / 32, 1)
                soft_m = torch.from_numpy(soft_m).type(torch.float32)
                mask = soft_m
            result = F.pad(result, phw, fill=0.5, padding_mode='edge')
            if c != 4:
                result = torch.cat((result, mask.unsqueeze(1)), 1)
            else:
                result[:,3,:,:] *= mask
        return result

    def pad(self, image, left, top, right, bottom):
        b, c, h, w = image.shape
        left, right = [int(x * w) for x in [left, right]]
        top, bottom = [int(x * h) for x in [top, bottom]]
        mask = torch.ones(b, h, w, dtype=torch.float32)
        mask = F.pad(mask, (left, top, right, bottom), 0, padding_mode='constant')
        dist = ndimage.distance_transform_edt(mask)
        soft_m = np.minimum(dist / 32, 1)
        soft_m = torch.from_numpy(soft_m).type(torch.float32)
        mask = soft_m
        image = F.pad(image, (left, top, right, bottom), 0.5, padding_mode='edge')
        if c != 4:
            image = torch.cat((image, mask.unsqueeze(1)), 1)
        else:
            image[:,3,...] *= mask
        return image
    
class PercentPadding:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            'required': {
                'image': ('IMAGE',),
                'left': ('FLOAT', {'min': 0, 'step': .01, 'default': 0}),
                'top': ('FLOAT', {'min': 0, 'step': .01, 'default': 0}),
                'right': ('FLOAT', {'min': 0, 'step': .01, 'default': 0}),
                'bottom': ('FLOAT', {'min': 0, 'step': .01, 'default': 0}),
                }
            }
    
    RETURN_TYPES = ('IMAGE',)

    FUNCTION = 'run'

    CATEGORY = 'davcha'

    def run(self, image, left, top, right, bottom):
        image = image.permute(0,3,1,2)
        b, c, h, w = image.shape
        left, right = [int(x * w) for x in [left, right]]
        top, bottom = [int(x * h) for x in [top, bottom]]
        mask = torch.ones(b, h, w, dtype=torch.float32)
        mask = F.pad(mask, (left, top, right, bottom), 0, padding_mode='constant')
        dist = ndimage.distance_transform_edt(mask)
        soft_m = np.minimum(dist / 32, 1)
        soft_m = torch.from_numpy(soft_m).type(torch.float32)
        mask = soft_m
        image = F.pad(image, (left, top, right, bottom), 0.5, padding_mode='constant')
        if c != 4:
            image = torch.cat((image, mask.unsqueeze(1)), 1)
        else:
            image[:,3,...] *= mask
        image = image.permute(0,2,3,1)
        return (image, )
    
class StringScheduleHelper:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            'required': {
                'text': ('STRING', {'multiline': True, 'dynamicPrompts': True}),
                }
            }
    
    RETURN_NAMES = ('base_prompt', 'schedule', 'has_schedule')
    RETURN_TYPES = ('STRING', 'STRING', 'BOOLEAN')

    FUNCTION = 'run'

    CATEGORY = 'davcha'

    def run(self, text):
        base = []
        sched = []
        for a,b,c in re.findall(r"(\d+):(.+)|(.+)", text):
            if a == '':
                base.append(c.strip())
            else:
                sched.append(f'"{a}": "{b.strip()}"')
        base_prompt = '\n'.join(base)
        schedule = ',\n'.join(sched)
        has_schedule = len(sched) > 0
        if not has_schedule:
            schedule = '"0": ""'
        return (base_prompt, schedule, has_schedule)

class SmartMask:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            'required': {
                'mask': ('MASK',),
                }
            }
    
    RETURN_TYPES = ('MASK',)

    FUNCTION = 'mask'

    CATEGORY = 'davcha'

    def mask(self, mask):
        if torch.sum(mask) > 0:
            return (mask, )
        else:
            return (1 - mask, )
        
class ResizeCropFit:
    RETURN_TYPES = ('IMAGE',)
    
    FUNCTION = 'run'

    CATEGORY = 'davcha'

    @classmethod
    def INPUT_TYPES(cls):
        return {
            'required': {
                'image': ('IMAGE',),
                'latent': ('LATENT',),
                'mode': (['resize', 'crop', 'fit'],)
                }
            }
    
    def run(self, image, latent, mode):
        image = image.permute(0,3,1,2)
        b, c, sh, sw = image.shape
        _, _, th, tw = latent['samples'].shape
        th, tw = th * 8, tw * 8
        if (th, tw) == (sh, sw):
            return (image.permute(0,2,3,1), )
        if mode == 'resize':
            result = F.resize(image, (th, tw), InterpolationMode.NEAREST_EXACT)
        else:
            fun = max if mode == 'crop' else min
            scale = fun(torch.tensor((th/sh,tw/sw)))
            shape = tuple(int(x * scale) for x in [sh,sw])
            mask = torch.ones((1, sh, sw))
            mask = F.resize(mask, shape, InterpolationMode.NEAREST_EXACT)
            result = F.resize(image, shape, InterpolationMode.NEAREST_EXACT)
            phw = (
                (tw - int(sw*scale))//2,
                (th - int(sh*scale))//2,
                (tw - int(sw*scale)+1)//2,
                (th - int(sh*scale)+1)//2,
            )
            mask = F.pad(mask, phw, fill=0, padding_mode='constant')
            dist = ndimage.distance_transform_edt(mask)
            soft_m = np.minimum(dist / 32, 1)
            soft_m = torch.from_numpy(soft_m).type(torch.float32)
            mask = soft_m
            result = F.pad(result, phw, fill=0.5, padding_mode='constant')
            if c != 4:
                result = torch.cat((result, mask.unsqueeze(1)), 1)
            #     result[:,3,:,:] = 1
            else:
                result[:,3,:,:] *= mask
        result = result.permute(0,2,3,1)
        return (result, )

class SoftErosion:
    RETURN_TYPES = ('MASK',)
    FUNCTION = 'run'
    CATEGORY = 'davcha'

    @classmethod
    def INPUT_TYPES(cls):
        return {
            'required': {
                'mask': ('MASK',),
                'size': ('INT', {'min': 0, 'default': 5}),
            }
        }
    
    def run(self, mask, size):
        if size == 0:
            return (mask, )
        soft_mask = []
        for m in mask:
            dist = ndimage.distance_transform_edt(m)
            soft_m = np.minimum(dist / size, 1)
            soft_m = torch.from_numpy(soft_m)
            soft_mask.append(soft_m)
        soft_mask = torch.from_numpy(np.array(soft_mask)).type(torch.FloatTensor)
        return (soft_mask, )
    
class ApplyMask:
    RETURN_TYPES = ('IMAGE',)
    FUNCTION = 'run'
    CATEGORY = 'davcha'

    @classmethod
    def INPUT_TYPES(cls):
        return {
            'required': {
                'image': ('IMAGE',),
                'mask': ('MASK',),
            }
        }
    
    def run(self, image, mask):
        image = image.permute(0,3,1,2)
        _, _, h, w = image.shape
        mask = F.resize(mask, (h, w), InterpolationMode.BICUBIC)
        image = image.permute(0,2,3,1)[:,:,:,:3]
        image *= mask[...,None]
        return (image, )
    
class DStack:
    RETURN_TYPES = ('IMAGE',)
    FUNCTION = 'run'
    CATEGORY = 'davcha'

    @classmethod
    def INPUT_TYPES(cls):
        return {
            'required': {
                'image1': ('IMAGE',),
                'image2': ('IMAGE',),
            }
        }
    
    def run(self, image1, image2):
        stack = torch.dstack((image1, image2))
        return (stack, )

ratio_merge = ("FLOAT", {"default": 1.0, "min": -5.0, "max": 6.0, "step": 0.001})

class DavchaConditioningConcat:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "conditioning_to": ("CONDITIONING",),
            "conditioning_from": ("CONDITIONING",),
            }}
    RETURN_TYPES = ("CONDITIONING",)
    FUNCTION = "concat"

    CATEGORY = "conditioning"

    def concat(self, conditioning_to, conditioning_from):
        out = []
        
        if len(conditioning_from) < 1: return (conditioning_to,)
        if len(conditioning_to) < 1: return (conditioning_from,)

        cond_from = conditioning_from[0][0].to(conditioning_to[0][0].device)

        for i in range(len(conditioning_to)):
            t1 = conditioning_to[i][0]
            mb = max(t1.shape[0], cond_from.shape[0])
            tw = torch.cat((t1.repeat(mb // t1.shape[0],1,1), cond_from.repeat(mb // cond_from.shape[0],1,1)), 1)
            n = [tw, conditioning_to[i][1].copy()]
            out.append(n)

        return (out, )

class DavchaModelMergeSimple:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": { "model1": ("MODEL",),
                              "model2": ("MODEL",),
                              "ratio": ratio_merge,
                              }}
    RETURN_TYPES = ("MODEL",)
    FUNCTION = "merge"

    CATEGORY = "advanced/model_merging"

    def merge(self, model1, model2, ratio):
        m = model1.clone()
        kp = model2.get_key_patches("diffusion_model.")
        for k in kp:
            m.add_patches({k: kp[k]}, 1.0 - ratio, ratio)
        return (m, )
    
class DavchaCLIPMergeSimple:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": { "clip1": ("CLIP",),
                              "clip2": ("CLIP",),
                              "ratio": ratio_merge,
                              }}
    RETURN_TYPES = ("CLIP",)
    FUNCTION = "merge"

    CATEGORY = "advanced/model_merging"

    def merge(self, clip1, clip2, ratio):
        m = clip1.clone()
        kp = clip2.get_key_patches()
        for k in kp:
            if k.endswith(".position_ids") or k.endswith(".logit_scale"):
                continue
            m.add_patches({k: kp[k]}, 1.0 - ratio, ratio)
        return (m, )

import comfy_extras.nodes_model_merging

class DavchaModelMergeSD1(comfy_extras.nodes_model_merging.ModelMergeBlocks):
    CATEGORY = "advanced/model_merging/model_specific"
    @classmethod
    def INPUT_TYPES(s):
        arg_dict = { "model1": ("MODEL",),
                              "model2": ("MODEL",)}

        argument = ratio_merge

        arg_dict["time_embed."] = argument
        arg_dict["label_emb."] = argument

        for i in range(12):
            arg_dict["input_blocks.{}.".format(i)] = argument

        for i in range(3):
            arg_dict["middle_block.{}.".format(i)] = argument

        for i in range(12):
            arg_dict["output_blocks.{}.".format(i)] = argument

        arg_dict["out."] = argument

        return {"required": arg_dict}


class DavchaModelMergeSDXL(comfy_extras.nodes_model_merging.ModelMergeBlocks):
    CATEGORY = "advanced/model_merging/model_specific"

    @classmethod
    def INPUT_TYPES(s):
        arg_dict = { "model1": ("MODEL",),
                              "model2": ("MODEL",)}

        argument = ratio_merge

        arg_dict["time_embed."] = argument
        arg_dict["label_emb."] = argument

        for i in range(9):
            arg_dict["input_blocks.{}".format(i)] = argument

        for i in range(3):
            arg_dict["middle_block.{}".format(i)] = argument

        for i in range(9):
            arg_dict["output_blocks.{}".format(i)] = argument

        arg_dict["out."] = argument

        return {"required": arg_dict}
    
class ConditioningCompress:
    CATEGORY = "davcha"

    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "conditioning": ("CONDITIONING",),
            "max_tokens": ("INT", {"default": 75, "min": 1, "step": 1})
            }}
    RETURN_TYPES = ("CONDITIONING",)
    FUNCTION = "run"
    
    
    def run(self, conditioning, max_tokens):
        def reduc(o):
            u,s,v = torch.svd(o)
            return (u[0,:max_tokens,:max_tokens]@s[0,:max_tokens].diag()@v[0,:,:max_tokens].T)[None]
        
        out = [[c[0].clone(), c[1]] for c in conditioning]
        for o in out:
            o[0] = reduc(o[0])
        return (out, )
    
################ DavchaCLIPTextEncode

import regex
from itertools import product

def parse(text):
    def _inner(text, start=0, end=1):
        matches = regex.findall(r'(\[((?:(?:embedding:)?[^\[\]])*?):((?:(?:embedding:)?[^\[\]])*?):(\d*\.?\d+)\])', text, regex.DOTALL)
        if len(matches) == 0:
            return [(text, (start, end))]
        x = [m[1:3] for m in matches]
        prod = list(product(*x))
        results = {}
        for p in prod:
            if p not in results: results[p] = []
            for f, b, a, t in matches:
                t = float(t)
                if b in p:
                    results[p].append((f, b, (start, t)))
                if a in p:
                    results[p].append((f, a, (t, end)))
        final = []
        for items in results.values():
            txt = text
            st, en = start, end
            for f, r, (s, e) in items:
                txt = txt.replace(f, r)
                st = max(st, s)
                en = min(en, e)
            if st <= en:
                for prompt, (a, b) in _inner(txt, st, en):
                    final.append((prompt, (a, b)))
        return final
    candidates = _inner(text)
    dic = {}
    for prompt, (s, e) in candidates:
        if prompt not in dic:
            dic[prompt] = (s, e)
        else:
            dic[prompt] = (min(s, dic[prompt][0]), max(e, dic[prompt][1]))
    return [(prompt, (s,e)) for prompt, (s, e) in dic.items()]

class DavchaCLIPTextEncode:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "text": ("STRING", {"multiline": True, "dynamicPrompts": True, "tooltip": "The text to be encoded."}), 
                "clip": ("CLIP", {"tooltip": "The CLIP model used for encoding the text."})
            }
        }
    RETURN_TYPES = ("CONDITIONING",)
    OUTPUT_IS_LIST = (True,)
    FUNCTION = "run"

    CATEGORY = "davcha"
    
    def run(self, clip, text):
        text = text.split('|')
        result = []
        for txt in text:
            texts = re.split(r"\b(AREA\(\s*(?:\d*\.?\d+\s*){4,5}\))", txt)
            
            areas = []
            current_area = None

            for item in texts:
                if item.startswith('AREA'):
                    current_area = item
                else:
                    prompt = parse(item)
                    if current_area is None:
                        areas.append((prompt, None))
                    else:
                        values = re.findall(r"(\d*\.?\d+)", current_area)
                        x, y, w, h, s = [float(i) for i in values + [1.0] * (5 - len(values))]
                        areas.append((prompt, (x, y, w, h, s)))
                    current_area = None
            
            for schedule, area in areas:
                for prompt, (start, end) in schedule:
                    print(f'{area} {start}-{end}: {prompt}')

            cs = []
            for prompts, area in areas:
                for txt, (lb, ub) in prompts:
                    if len(txt) > 0:
                        tokens = clip.tokenize(txt)
                        output = clip.encode_from_tokens(tokens, return_pooled=True, return_dict=True)
                        cond = output.pop("cond")
                        c = [[cond, output]]
                        c = node_helpers.conditioning_set_values(c, {"start_percent": lb,
                                                                    "end_percent": ub})
                        if area is not None:
                            x, y, w, h, s = area
                            c = node_helpers.conditioning_set_values(c, {"area": ("percentage", h, w, y, x),
                                                                        "strength": s,
                                                                        "set_area_to_bounds": False})
                        cs += c
            result.append(cs)
        return (result, )

from pathlib import Path
from webp import WebPData, WebPAnimDecoderOptions, WebPAnimDecoder, mimread

def video_reader(file, start=0, count=0, target_fps=0):
    if Path(file).suffix.lower() == '.webp':
        return _webp_reader(file, start, count, target_fps)
    else:
        return _ffmpeg_reader(file, start, count, target_fps)
    
def _webp_reader(file, start, count, target_fps):
    with open(file, 'rb') as f:
        webp_data = WebPData.from_buffer(f.read())
        dec_opts = WebPAnimDecoderOptions.new(use_threads=True)
        dec = WebPAnimDecoder.new(webp_data, dec_opts)
        eps = 1e-7
        
        frames_data = list(dec.frames())

    frames = [arr[:,:,:3] for arr, _ in frames_data]
    fps = 1000 * len(frames_data) / frames_data[-1][1]
    if target_fps > 0:
        if fps > target_fps:
            frames = [frame[:,:,:3] for frame in mimread(file, fps=target_fps)]
    else:
        target_fps = fps
    
    if count > 0:
        frames = frames[int(start):int(start+count)]
    else:
        frames = frames[int(start):]
    
    frames = torch.from_numpy(np.array(frames)).float() / 255.0

    return frames, target_fps

import cv2
import numpy as np

def _ffmpeg_reader(file, start, count, target_fps):
    cap = cv2.VideoCapture(file)
    
    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    if target_fps == 0:
        target_fps = fps
    start_frame = int(min(max(0, start), frame_count - 1) * fps / target_fps)
    if count == 0:
        count = int(frame_count * target_fps / fps)
    count = min(int(frame_count * target_fps / fps - start), count)
    
    frames = np.empty((count, frame_height, frame_width, 3), dtype=np.uint8)
    for i in range(count):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(start_frame + i * fps / target_fps))
        ret, frame = cap.read()
        if not ret:
            break
    
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames[i] = frame_rgb
    
    cap.release()
    
    frames = torch.from_numpy(frames).float() / 255.0
    
    return frames, target_fps

class DavchaLoadVideo:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "path": ("STRING", {"multiline": False}),
                "start": ("INT", {"default": 0, "min": 0, "step": 1}),
                "count": ("INT", {"default": 0, "min": 0, "step": 1}),
                "target_fps": ("FLOAT", {"default": 0, "min": 0, "step": 0.01}),
            }
        }
    RETURN_TYPES = ("IMAGE", "INT", "FLOAT")
    RETURN_NAMES = ("frames", "frame_count", "fps")
    FUNCTION = "run"

    CATEGORY = "davcha"
    
    def run(self, path, start, count, target_fps):
        fullpath = os.path.join(folder_paths.get_input_directory(), path)
        frames, fps = video_reader(fullpath, start, count, target_fps)
        return (frames, len(frames), fps)
    
MAX_RESOLUTION = 16384

class AnyType(str):
    def __eq__(self, _) -> bool:
        return True
    def __ne__(self, __value: object) -> bool:
        return False
    
class DavchaEmptyLatentImage:
    def __init__(self):
        self.device = comfy.model_management.intermediate_device()

    @classmethod
    def INPUT_TYPES(s):        
        return {
            "required": { 
                "width": ("INT", {"default": 512, "min": 16, "max": MAX_RESOLUTION, "step": 8, "tooltip": "The width of the latent images in pixels."}),
                "height": ("INT", {"default": 512, "min": 16, "max": MAX_RESOLUTION, "step": 8, "tooltip": "The height of the latent images in pixels."}),
                "option": ([AnyType("*")], ),
                "batch_size": ("INT", {"default": 1, "min": 1, "max": 4096, "tooltip": "The number of latent images in the batch."})
            }
        }
    RETURN_TYPES = ("LATENT", "FLOAT", "INT")
    RETURN_NAMES = ("latent", "upscale_factor", "batch_size")
    OUTPUT_TOOLTIPS = ("The empty latent image batch.",)
    FUNCTION = "generate"

    CATEGORY = "davcha"
    DESCRIPTION = "Create a new batch of empty latent images to be denoised via sampling."

    def generate(self, width, height, option, batch_size=1):
        w, h, _ = re.findall(r"(\d+)x(\d+): (\d+\.?\d*)", option)[0]
        w, h = int(w), int(h)
        upscale_factor = width / w
        latent = torch.zeros([batch_size, 4, h // 8, w // 8], device=self.device)
        return ({"samples":latent}, upscale_factor, batch_size)

class DavchaMaskImage:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": { "image": ("IMAGE",),
                              "mask": ("MASK",),
                              }}
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "run"

    CATEGORY = "davcha"

    def run(self, image, mask):
        b, w, h, c = image.shape
        mask = F.resize(mask, (w, h), InterpolationMode.BICUBIC)
        image = image.permute(0, 3, 1, 2)
        if c != 4:
            image = torch.cat((image, torch.ones((b, 1, w, h), dtype=torch.float32)), 1)# mask.unsqueeze(1)), 1)
        image = image * mask.unsqueeze(1)
        image = image.permute(0, 2, 3, 1)
        return (image, )

class DavchaPop:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": { "items": (AnyType("*"),),
                              }}
    INPUT_IS_LIST = (True, )
    OUTPUT_IS_LIST = (False, True)
    RETURN_TYPES = (AnyType("*"), AnyType("*"))
    FUNCTION = "run"

    CATEGORY = "davcha"

    def run(self, items):
        return (items[0], items[1:])

gguf_folder = os.path.join(folder_paths.models_dir, "llm_gguf")
if os.path.isdir(gguf_folder):
    gguf_files = [file for file in os.listdir(gguf_folder) if file.endswith('.gguf')]
else:
    gguf_files = []
    
class DavchaLoadLLM:
    @classmethod
    def INPUT_TYPES(s):
        return {'required':{
            'modelname': (gguf_files,),
        }}
    RETURN_NAMES = ('model',)
    RETURN_TYPES = ('DavchaLLModel',)
    FUNCTION = "run"

    CATEGORY = "davcha"
    
    def run(self, modelname):
        model_path = os.path.join(gguf_folder, modelname)
        model = Llama(
            model_path=model_path,
            n_gpu_layers=-1,
            verbose=False,
            n_ctx=2048,
        )
        return (model,)        

class DavchaLLMAdvanced:
    @classmethod
    def INPUT_TYPES(s):
        return {'required':{
            'model': ('DavchaLLModel',),
            'seed': ('INT', {'default': 0, 'min': 0, 'max': 0xffffffffffffffff}),
            'system': ('STRING', {'multiline': True, 'dynamicPrompts': True}),
            'text': ('STRING', {'multiline': True, 'dynamicPrompts': True}),
            'max_tokens': ('INT', {'default': 512, 'min': 1, 'max': 8192}),
            'temperature': ('FLOAT', {'default': 1.0, 'min': 0, 'max': 1.0, 'step': 0.1}),
            'top_p': ('FLOAT', {'default': 0.9, 'min': 0, 'max': 1.0, 'step': 0.1}),
            'top_k': ('INT', {'default': 50, 'min': 0, 'max': 100}),
            'repeat_penalty': ('FLOAT', {'default': 1.2, 'min': 0, 'max': 5.0, 'step': 0.1}),
        }}
    
    RETURN_NAMES = ('text',)
    RETURN_TYPES = ('STRING',)
    FUNCTION = "run"

    CATEGORY = "davcha"
    
    def run(self, model, seed, system, text, max_tokens, temperature, top_p, top_k, repeat_penalty):
        generate_kwargs = {
            'max_tokens': max_tokens,
            'temperature': temperature,
            'top_p': top_p,
            'top_k': top_k,
            'repeat_penalty': repeat_penalty
        }
        msgs = []
        if len(system.strip()) > 0:
            msgs += [{'role': 'system', 'content': system}]
        msgs += [{"role": "user", "content": text}]
        model.set_seed(seed)
        llm_result = model.create_chat_completion(msgs, **generate_kwargs)
        return (llm_result['choices'][0]['message']['content'].strip(),)

class DavchaLLM:
    @classmethod
    def INPUT_TYPES(s):
        return {'required':{
            'model': ('DavchaLLModel',),
            'seed': ('INT', {'default': 0, 'min': 0, 'max': 0xffffffffffffffff}),
            'text': ('STRING', {'multiline': True, 'dynamicPrompts': True}),
        }}
    
    RETURN_NAMES = ('text',)
    RETURN_TYPES = ('STRING',)
    FUNCTION = "run"

    CATEGORY = "davcha"
    
    def run(self, model, seed, text):
        generate_kwargs = {
            'max_tokens': 512,
            'temperature': 1.0,
            'top_p': 0.9,
            'top_k': 50,
            'repeat_penalty': 1.2
        }
        msgs = []
        msgs += [{"role": "user", "content": text}]
        model.set_seed(seed)
        llm_result = model.create_chat_completion(msgs, **generate_kwargs)
        return (llm_result['choices'][0]['message']['content'].strip(),)

def viterbi_distance(a, b):
    """Compute the Viterbi distance between two sequences."""
    n = len(a)
    m = len(b)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if a[i - 1] == b[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1]) + 1
    return dp[n][m]

def viterbi_diff(a, b):
    """Compute the Viterbi diff between two sequences."""
    n = len(a)
    m = len(b)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if a[i - 1] == b[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1]) + 1
    i, j = n, m
    diff = []
    while i > 0 or j > 0:
        if i > 0 and j > 0 and a[i - 1] == b[j - 1]:
            diff.append((' ', a[i - 1], b[j - 1]))
            i -= 1
            j -= 1
        elif j > 0 and (i == 0 or dp[i][j] == dp[i][j - 1] + 1):
            diff.append(('+', '', b[j - 1]))
            j -= 1
        elif i > 0 and (j == 0 or dp[i][j] == dp[i - 1][j] + 1):
            diff.append(('-', a[i - 1], ''))
            i -= 1
        else:
            diff.append(('*', a[i - 1], b[j - 1]))
            i -= 1
            j -= 1
    diff.reverse()
    return diff

def get_highlow(lorapath, m):
    x = os.path.normpath(m[0][0])
    files = [os.path.relpath(f, lorapath) for f in glob(os.path.join(lorapath, os.path.dirname(x), '*.safetensors'))]
    files.sort(key=lambda f: viterbi_distance(x, f))
    other = files[1]
    diff = viterbi_diff(x,other)
    x_tag = ''.join(d[1] for d in diff if d[0]=='*').lower()
    high, low = (x,other) if 'h' in x_tag else (other,x)
    return (high, low)
    
class DavchaWan22LoraTagLoader:
    @classmethod
    def INPUT_TYPES(s):
        return {'required':{
            'high': ('MODEL',),
            'low': ('MODEL',),
            'txt': ('STRING', {'multiline': True, 'dynamicPrompts': True}),
        }}
    RETURN_NAMES = ('high','low', 'txt')
    RETURN_TYPES = ('MODEL','MODEL', 'STRING')
    FUNCTION = "run"
    CATEGORY = "davcha"
        
    def run(self, high, low, txt):
        loraspath = folder_paths.get_folder_paths('loras')[0]
        m = re.findall(r'<lora:([^:]+):([^>]+)>', txt)
        for model, weight in m:
            lora_high, lora_low = get_highlow(loraspath, [(model, weight)])
            high, _ = LoraLoader().load_lora(high, None, lora_high, float(weight), float(weight))
            low, _ = LoraLoader().load_lora(low, None, lora_low, float(weight), float(weight))
        txt = re.sub(r'<lora:[^:]+:[^>]+>', '', txt)
        return (high, low, txt)

class DavchaTextEncodeQwenImageEditPlus(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="TextEncodeQwenImageEditPlus",
            category="advanced/conditioning",
            inputs=[
                io.Clip.Input("clip"),
                io.String.Input("prompt", multiline=True, dynamic_prompts=True),
                io.Vae.Input("vae", optional=True),
                io.Image.Input("image1", optional=True),
                io.Image.Input("image2", optional=True),
                io.Image.Input("image3", optional=True),
            ],
            outputs=[
                io.Conditioning.Output(),
            ],
        )

    @classmethod
    def execute(cls, clip, prompt, vae=None, image1=None, image2=None, image3=None) -> io.NodeOutput:
        ref_latents = []
        images = [image1, image2, image3]
        images_vl = []
        llama_template = "<|im_start|>system\nDescribe the key features of the input image (color, shape, size, texture, objects, background), then explain how the user's text instruction should alter or modify the image. Generate a new image that meets the user's requirements while maintaining consistency with the original input where appropriate.<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n"
        image_prompt = ""

        for i, image in enumerate(images):
            if image is not None:
                samples = image.movedim(-1, 1)
                total = int(384 * 384)

                scale_by = math.sqrt(total / (samples.shape[3] * samples.shape[2]))
                width = round(samples.shape[3] * scale_by)
                height = round(samples.shape[2] * scale_by)

                s = comfy.utils.common_upscale(samples, width, height, "area", "disabled")
                images_vl.append(s.movedim(1, -1))
                if vae is not None:
                    ref_latents.append(vae.encode(samples.movedim(1, -1)[:, :, :, :3]))

                image_prompt += "Picture {}: <|vision_start|><|image_pad|><|vision_end|>".format(i + 1)

        tokens = clip.tokenize(image_prompt + prompt, images=images_vl, llama_template=llama_template)
        conditioning = clip.encode_from_tokens_scheduled(tokens)
        if len(ref_latents) > 0:
            conditioning = node_helpers.conditioning_set_values(conditioning, {"reference_latents": ref_latents}, append=True)
        return io.NodeOutput(conditioning)

NODE_CLASS_MAPPINGS = {
    'DavchaTextEncodeQwenImageEditPlus': DavchaTextEncodeQwenImageEditPlus,
    'DavchaWan22LoraTagLoader': DavchaWan22LoraTagLoader,
    'DavchaLoadLLM': DavchaLoadLLM,
    'DavchaLLM': DavchaLLM,
    'DavchaLLMAdvanced': DavchaLLMAdvanced,
    'PadAndResize': PadAndResize,
    'SmartMask': SmartMask,
    'ResizeCropFit': ResizeCropFit,
    'PercentPadding': PercentPadding,
    'SoftErosion': SoftErosion,
    'ApplyMask': ApplyMask,
    'StringScheduleHelper': StringScheduleHelper,
    'DStack': DStack,
    'DavchaConditioningConcat': DavchaConditioningConcat,
    'DavchaModelMergeSimple' : DavchaModelMergeSimple,
    'DavchaCLIPMergeSimple' : DavchaCLIPMergeSimple,
    'DavchaModelMergeSD1' : DavchaModelMergeSD1,
    'DavchaModelMergeSDXL' : DavchaModelMergeSDXL,
    'ConditioningCompress': ConditioningCompress,
    'DavchaCLIPTextEncode': DavchaCLIPTextEncode,
    'DavchaLoadVideo': DavchaLoadVideo,
    'DavchaEmptyLatentImage': DavchaEmptyLatentImage,
    'DavchaMaskImage': DavchaMaskImage,
    'DavchaPop': DavchaPop,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    'DavchaTextEncodeQwenImageEditPlus': 'Text Encode Qwen Image Edit Plus',
    'DavchaWan22LoraTagLoader': 'Wan22 Lora Tag Loader',
    'DavchaLoadLLM': 'DavchaLoadLLM',
    'DavchaLLM': 'DavchaLLM',
    'DavchaLLMAdvanced': 'DavchaLLMAdvanced',
    'PadAndResize': 'PadAndResize',
    'SmartMask': 'SmartMask',
    'ResizeCropFit': 'Resize, Crop or Fit',
    'PercentPadding': 'Percent Padding',
    'SoftErosion': 'SoftErosion',
    'ApplyMask': 'ApplyMask',
    'StringScheduleHelper': 'StringScheduleHelper',
    'DStack': 'DStack',
    'DavchaConditioningConcat': 'DavchaConditioningConcat',
    'DavchaModelMergeSimple': 'DavchaModelMergeSimple',
    'DavchaCLIPMergeSimple': 'DavchaCLIPMergeSimple',
    'DavchaModelMergeSD1': 'DavchaModelMergeSD1',
    'DavchaModelMergeSDXL': 'DavchaModelMergeSDXL',
    'ConditioningCompress': 'ConditioningCompress',
    'DavchaCLIPTextEncode': 'CLIP Text Encode (Davcha)',
    'DavchaLoadVideo': 'Load Video (Davcha)',
    'DavchaEmptyLatentImage': 'Empty Latent Image (Davcha)',
    'DavchaMaskImage': 'Mask Image (Davcha)',
    'DavchaPop': 'Pop',
}
