import re
import torch
from torchvision.transforms import InterpolationMode
import torchvision.transforms.v2.functional as F
from scipy import ndimage
import numpy as np

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
        image = F.pad(image, (left, top, right, bottom), 0, padding_mode='constant')
        if c == 4:
            image[:,3,...][mask==0] = 0
        else:
            image = torch.cat((image, mask.unsqueeze(1)),1)
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
        if mode == 'resize':
            result = F.resize(image, (th, tw), InterpolationMode.NEAREST_EXACT)
        else:
            fun = max if mode == 'crop' else min
            scale = fun(torch.tensor((th/sh,tw/sw)))
            shape = tuple(int(x * scale) for x in [sh,sw])
            result = F.resize(image, shape, InterpolationMode.NEAREST_EXACT)
            phw = (
                (tw - int(sw*scale))//2,
                (th - int(sh*scale))//2,
                (tw - int(sw*scale)+1)//2,
                (th - int(sh*scale)+1)//2,
            )
            result = F.pad(result, phw, fill=0, padding_mode='constant')
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
}
