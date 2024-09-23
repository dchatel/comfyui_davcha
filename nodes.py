import os
import re
import torch
from torchvision.transforms import InterpolationMode
import torchvision.transforms.v2.functional as F
import torchvision as tv
from scipy import ndimage
import numpy as np
import node_helpers
import folder_paths
import comfy

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
        soft_m = torch.from_numpy(soft_m)
        mask = soft_m
        image = F.pad(image, (left, top, right, bottom), 0, padding_mode='constant')
        if c != 4:
            image = torch.cat((image, mask.unsqueeze(1)), 1)
        # image[:,3,...] *= mask
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
            soft_m = torch.from_numpy(soft_m)
            mask = soft_m
            result = F.pad(result, phw, fill=0, padding_mode='constant')
            if c != 4:
                result = torch.cat((result, mask.unsqueeze(1)), 1)
            #     result[:,3,:,:] = 1
            # result[:,3,:,:] *= mask
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
        soft_mask = torch.from_numpy(np.array(soft_mask))
        return (soft_mask, )
    
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

        print(conditioning_to[0][0].dtype, conditioning_from[0][0].dtype)
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

import lark

schedule_parser = lark.Lark(r"""
!start: (prompt | /[][():]/+)*
prompt: (emphasised | scheduled | plain | embedding | WHITESPACE)*
emphasised: "(" prompt ":" [WHITESPACE] NUMBER [WHITESPACE] ")"
scheduled: "[" [prompt ":"] prompt ":" [WHITESPACE] NUMBER [WHITESPACE] "]"
embedding: "embedding:" /[A-Za-z0-9-_.]+/
WHITESPACE: /\s+/
plain: /([^\\\[\]():|]|\\.)+/
%import common.SIGNED_NUMBER -> NUMBER
""")

def reduce(l):
    if isinstance(l, list):
        text = [('', (0, 1))]
        for i in l:
            text = [(f'{t}{x}', (max(xl, lb), min(xu, ub))) for t, (lb, ub) in text for x, (xl, xu) in i if max(xl, lb) < min(xu, ub)]
        return text
    else:
        return l

def visit(tree, lb=0.0, ub=1.0):
    if isinstance(tree, list):
        l = [visit(item, lb, ub) for item in tree]
        if len(l) == 1:
            l = l[0]
        else:
            l = [i for i in l if i is not None]
            l = reduce(l)
        return l
      
    elif isinstance(tree, lark.tree.Tree):
        if tree.data.type == 'RULE':
            if tree.data.value == 'start':
                return visit(tree.children[0], lb, ub)
            if tree.data.value == 'prompt':
                return visit(tree.children, lb, ub)
            if tree.data.value == 'plain':
                return visit(tree.children, lb, ub)
            if tree.data.value == 'scheduled':
                ts = float(tree.children[3].value)
                left = visit(tree.children[0], lb, ts)
                right = visit(tree.children[1], ts, ub)
                return left + right
            if tree.data.value == 'emphasised':
                prompts = visit(tree.children[0], lb, ub)
                value = float(tree.children[2])
                return [(f'({prompt}:{value})', (max(lb, pb), min(ub, pu))) for prompt, (pb, pu) in prompts if max(lb, pb) < min(ub, pu)]
            if tree.data.value == 'embedding':
                name = visit(tree.children[0])[0][0]
                return [(f'embedding:{name}', (lb, ub))]
    if isinstance(tree, lark.lexer.Token):
        if tree.type == 'WHITESPACE':
            return [(' ', (lb, ub))]
        return [(tree.value, (lb, ub))]

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
    FUNCTION = "run"

    CATEGORY = "davcha"
    
    def run(self, clip, text):
        texts = re.split(r"\b(AREA\(\s*(?:\d*\.?\d+\s*){4,5}\))", text)
        
        areas = []
        current_area = None

        for item in texts:
            if item.startswith('AREA'):
                current_area = item
            else:
                prompt = visit(schedule_parser.parse(item))
                if current_area is None:
                    areas.append((prompt, None))
                else:
                    values = re.findall(r"(\d*\.?\d+)", current_area)
                    x, y, w, h, s = [float(i) for i in values + [1.0] * (5 - len(values))]
                    areas.append((prompt, (x, y, w, h, s)))
                current_area = None
        
        print(areas)
        cs = []
        for prompts, area in areas:
            for text, (lb, ub) in prompts:
                if len(text) > 0:
                    tokens = clip.tokenize(text)
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
        return (cs, )

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
                "target_fps": ("FLOAT", {"default": 0, "min": 0, "max": 60.0, "step": 0.01}),
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
        width, height, upscale_factor = re.findall("(\d+)x(\d+): (\d+\.?\d*)", option)[0]
        width, height, upscale_factor = int(width), int(height), float(upscale_factor)
        latent = torch.zeros([batch_size, 4, height // 8, width // 8], device=self.device)
        return ({"samples":latent}, upscale_factor, batch_size)

NODE_CLASS_MAPPINGS = {
    'SmartMask': SmartMask,
    'ResizeCropFit': ResizeCropFit,
    'PercentPadding': PercentPadding,
    'SoftErosion': SoftErosion,
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
}

NODE_DISPLAY_NAME_MAPPINGS = {
    'SmartMask': 'SmartMask',
    'ResizeCropFit': 'Resize, Crop or Fit',
    'PercentPadding': 'Percent Padding',
    'SoftErosion': 'SoftErosion',
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
}
