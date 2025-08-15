"""
StarGAN v2
Copyright (c) 2020-present NAVER Corp.

This work is licensed under the Creative Commons Attribution-NonCommercial
4.0 International License. To view a copy of this license, visit
http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to
Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.
"""

import os
import torch


class CheckpointIO(object):
    def __init__(self, fname_template, data_parallel=False, **kwargs):
        # Extract the directory from the template, handling format strings
        if "{" in fname_template and "}" in fname_template:
            # For format strings like '{:06d}_nets.ckpt', extract the base directory
            # by splitting on the first path separator and taking everything before it
            parts = fname_template.split(os.sep)
            if len(parts) > 1:
                base_dir = os.sep.join(parts[:-1])
            else:
                # If no directory separator, assume current directory
                base_dir = "."
        else:
            # Regular filename, use dirname normally
            base_dir = os.path.dirname(fname_template)

        # Only create directory if it's not empty and not current directory
        if base_dir and base_dir != ".":
            os.makedirs(base_dir, exist_ok=True)

        self.fname_template = fname_template
        self.module_dict = kwargs
        self.data_parallel = data_parallel

    def register(self, **kwargs):
        self.module_dict.update(kwargs)

    def save(self, step):
        fname = self.fname_template.format(step)
        print("Saving checkpoint into %s..." % fname)
        outdict = {}
        for name, module in self.module_dict.items():
            if self.data_parallel:
                outdict[name] = module.module.state_dict()
            else:
                outdict[name] = module.state_dict()

        torch.save(outdict, fname)

    def load(self, step):
        fname = self.fname_template.format(step)
        assert os.path.exists(fname), fname + " does not exist!"
        print("Loading checkpoint from %s..." % fname)
        if torch.cuda.is_available():
            module_dict = torch.load(fname)
        else:
            module_dict = torch.load(fname, map_location=torch.device("cpu"))

        for name, module in self.module_dict.items():
            if self.data_parallel:
                module.module.load_state_dict(module_dict[name])
            else:
                module.load_state_dict(module_dict[name])
