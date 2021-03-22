import argparse
import json
import os
import pickle

import numpy as np
import torch.nn as nn

from models import experimental, common, yolo


def save_model(module: nn.Module):
    if isinstance(module, common.Focus):
        return {
            "type": "Focus",
            "conv": save_model(module.conv)
        }
    elif isinstance(module, common.Conv):
        return {
            "type": "Conv",
            "conv": save_model(module.conv),
            "act": save_model(module.act)
        }
    elif isinstance(module, nn.Conv2d):
        return {
            "type": "nn.Conv2d",
            "args": str(module),
            "weight": module.weight.data.cpu().numpy(),
            "bias": module.bias.data.cpu().numpy()
        }
    elif isinstance(module, nn.SiLU):
        return {
            "type": "nn.SiLU",
            "args": str(module)
        }
    elif isinstance(module, common.C3):
        return {
            "type": "C3",
            "cv1": save_model(module.cv1),
            "cv2": save_model(module.cv2),
            "cv3": save_model(module.cv3),
            "m": [save_model(lm) for lm in module.m],
        }
    elif isinstance(module, common.Bottleneck):
        return {
            "type": "Bottleneck",
            "cv1": save_model(module.cv1),
            "cv2": save_model(module.cv2),
            "add": module.add
        }
    elif isinstance(module, common.SPP):
        return {
            "type": "SPP",
            "cv1": save_model(module.cv1),
            "cv2": save_model(module.cv2),
            "m": [save_model(lm) for lm in module.m]
        }
    elif isinstance(module, nn.MaxPool2d):
        return {
            "type": "nn.MaxPool2d",
            "args": str(module)
        }
    elif isinstance(module, nn.Upsample):

        return {
            "type": "nn.Upsample",
            "args": "Upsample(scale_factor={},mode=\'{}\')".format(module.scale_factor, module.mode)
        }
    elif isinstance(module, common.Concat):
        return {
            "type": "Concat",
            "d": module.d
        }
    elif isinstance(module, yolo.Detect):
        return {
            "type": "Detect",
            "m": [save_model(lm) for lm in module.m],
            "nc": module.nc,
            "stride": module.stride.data.cpu().numpy(),
            "anchors": list(module.named_buffers())[0][1].data.cpu().numpy().tolist(),
            "anchor_grid": list(module.named_buffers())[1][1].data.cpu().numpy(),
        }
    else:
        print(module)
        assert False, "unknown type"


def remove_ndarray(d: dict):
    for (k, v) in d.items():
        if isinstance(v, dict):
            remove_ndarray(v)
        elif isinstance(v, list):
            [remove_ndarray(var) for var in v if isinstance(var, dict)]
        elif type(v) is np.ndarray:
            d[k] = k


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", type=str, default="weights/yolov5s.pt")
    parser.add_argument("--dst", type=str)
    parser.add_argument("--json-dst", type=str)
    opt = parser.parse_args()
    if opt.dst is None:
        pre, ext = os.path.splitext(opt.src)
        opt.dst = pre + ".pkl"

    model = experimental.attempt_load(opt.src)
    results = []
    for m in model.model:
        res = save_model(m)
        res['attached_info'] = {
            "i": m.i,
            "f": m.f,
            "type": m.type,
            "np": m.np,
        }
        results.append(res)
    output = {
        "modules": results,
        "names": model.names,
        "save": model.save
    }
    with open(opt.dst, "wb") as f:
        pickle.dump(output, f)

    for result in results:
        remove_ndarray(result)

    if opt.json_dst is not None:
        with open(opt.json_dst, "w") as f:
            json.dump({"model": results}, f, indent=2)
