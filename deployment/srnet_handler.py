# https://towardsdatascience.com/deploy-models-and-create-custom-handlers-in-torchserve-fc2d048fbe91
# https://www.pento.ai/blog/custom-torchserve-handlers

import io
import os
import sys
import importlib
from PIL import Image
import torch
from torchvision import transforms
from ts.torch_handler.base_handler import BaseHandler
from ts.utils.util import list_classes_from_module, load_label_mapping
import pytorch_lightning as pl
import json
from collections import namedtuple

import logging

logger = logging.getLogger(__name__)


class SRNetHandler(BaseHandler):

    def __init__(self):
        super(SRNetHandler, self).__init__()
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
            ])


    def _load_pickled_model(self, model_dir, model_file, model_pt_path):
        # We are overriding the BaseHandler's protected method here as it
        # would try to instantiate a Torch model instead of a lightning module.
        # Also, we improve finding the model class so we don't have to
        # restructure our original model module layouts.

        logger.info(f"Model dir is: {model_dir}")
        logger.info(f"Model file is: {model_file}")
        

        model_def_path = os.path.join(model_dir, model_file)

        if not os.path.isfile(model_def_path):
            raise RuntimeError("Missing the model.py file")
    
        module = importlib.import_module(model_file.split(".")[0])

        model_class_definitions = list_classes_from_module(module, parent_class=pl.LightningModule)
        
        # this is part of the base handler code, but we have to delete it, because we have multiple classes 
        #if len(model_class_definitions) != 1:
        #    raise ValueError(
        #        "Expected exactly one class subclassing LightningModule as model definition. Found classes: {}".format(
        #            model_class_definitions
        #        )
        #    )

        # change the class used from 0 to 2, this number is our main class and known from debugging
        model_class = model_class_definitions[2] # model_class_definitions[0]
        # Until here, the code is the same as in the base handler; but now
        # we will use lightning's checkpointing mechanism to restore the model
        if model_pt_path:
            # hyperparamters -> 'tpu_cores' needs to be set to "None" ! 
            hyperparams = {
                'debug': False, 
                'data_dir': '/work/ka1176/frauke/super-resolution/data', 
                'output_path': '/work/ka1176/frauke/super-resolution/nni/nni-experiments/abj60ped/trials/aDYYA/preds.h5', 
                'save_model_path': '/work/ka1176/frauke/super-resolution/nni/nni-experiments/abj60ped/trials/aDYYA/saved_models', 
                'batch_size': 8, 
                'learning_rate': 0.001, 
                'scaling_factor': 4, 
                'n_channels': 16, 
                'large_kernel_size': 9, 
                'small_kernel_size': 3, 
                'n_blocks': 28, 
                'n_epochs': 1000, 
                'nni': True, 
                'logger': True, 
                'checkpoint_callback': True, 
                'default_root_dir': None, 
                'gradient_clip_val': 0, 
                'process_position': 0, 
                'num_nodes': 1, 'num_processes': 1, 
                'gpus': 1, 'auto_select_gpus': False, 
                'tpu_cores': None, #<function _gpus_arg_default at 0x7fd9a6fab0d0>, 
                'log_gpu_memory': None, 
                'progress_bar_refresh_rate': None, 
                'overfit_batches': 0.0, 
                'track_grad_norm': -1, 
                'check_val_every_n_epoch': 1, 
                'fast_dev_run': False, 
                'accumulate_grad_batches': 1, 
                'max_epochs': None, 
                'min_epochs': None, 
                'max_steps': None, 
                'min_steps': None, 
                'limit_train_batches': 1.0, 
                'limit_val_batches': 1.0, 
                'limit_test_batches': 1.0, 
                'limit_predict_batches': 1.0, 
                'val_check_interval': 1.0, 
                'flush_logs_every_n_steps': 100, 
                'log_every_n_steps': 50, 
                'accelerator': None, 
                'sync_batchnorm': False, 
                'precision': 32, 
                'weights_summary': 'top', 
                'weights_save_path': None, 
                'num_sanity_val_steps': 2, 
                'truncated_bptt_steps': None, 
                'resume_from_checkpoint': None, 
                'profiler': None, 
                'benchmark': False, 
                'deterministic': False, 
                'reload_dataloaders_every_epoch': False, 
                'auto_lr_find': False, 
                'replace_sampler_ddp': True, 
                'terminate_on_nan': False, 
                'auto_scale_batch_size': False, 
                'prepare_data_per_node': True, 
                'plugins': None, 
                'amp_backend': 'native', 
                'amp_level': 'O2', 
                'distributed_backend': None, 
                'automatic_optimization': None, 
                'move_metrics_to_cpu': False, 
                'enable_pl_optimizer': None, 
                'multiple_trainloader_mode': 'max_size_cycle', 
                'stochastic_weight_avg': False
                }
            # convert dictionary to object to access values via dot notation
            args = namedtuple("ObjectName", hyperparams.keys())(*hyperparams.values())
            model = model_class.load_from_checkpoint(model_pt_path, args=args).eval()
        else:
            raise Exception("Model checkpoint file not provided")

        return model

    def preprocess_one_image(self, req):
        """
        Process one single image.
        """
        # get image from the request
        image = req.get("data")
        if image is None:
            image = req.get("body")
         # create a stream from the encoded image
        image = Image.open(io.BytesIO(image))
        image = self.transform(image)
        # add batch dim
        image = image.unsqueeze(0)
        return image

    def preprocess(self, requests):
        """
        Process all the images from the requests and batch them in a Tensor.
        """
        images = [self.preprocess_one_image(req) for req in requests]
        images = torch.cat(images)
        return images

    def postprocess(self, data):
        # process inference output, e.g. extracting top k
        # package output for web delivery

        mean = torch.tensor([0.485, 0.456, 0.406])
        std = torch.tensor([0.229, 0.224, 0.225])
        mean_rev = -(mean/std)
        std_rev = 1/std
        un_transform = transforms.Compose([transforms.Normalize(mean_rev, std_rev)])

        img_hr = un_transform(data.squeeze()).unsqueeze(dim=0)
        img_hr = img_hr.permute(0, 2, 3, 1)
        logger.info(f"output size: {img_hr.size()}")

        return img_hr.tolist()
