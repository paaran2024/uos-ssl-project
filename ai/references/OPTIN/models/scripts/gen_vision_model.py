import timm
import timm.data
from utils.utility import GLUE_TASKS
from transformers import ViTForImageClassification, AutoModelForImageClassification
import torch

def gen_vision_model(args):
    
    if args.dataset == "ImageNet":
        # Classifiation Task
            
        if "deit" in args.model_name:
            model = ViTForImageClassification.from_pretrained("facebook/{}".format(args.model_name))#vit-base-patch16-224"
            config = model.config
        
            
            
        elif "vit" in args.model_name:
            model = ViTForImageClassification.from_pretrained("google/{}".format(args.model_name))#vit-base-patch16-224"
            config = model.config
            
            
        elif "swin" in args.model_name:
            model = AutoModelForImageClassification.from_pretrained("microsoft/swin-tiny-patch4-window7-224")
            config = model.config
            print(config)
            
    elif args.dataset == "Cifar10":
        
        if "vit-base" in args.model_name:
            print(args.model_name)
            model = ViTForImageClassification.from_pretrained('nateraw/{}'.format(args.model_name))
            config = model.config
            
        elif "deit" in args.model_name:
            model = ViTForImageClassification.from_pretrained("facebook/{}".format(args.model_name))
            config = model.config
            
    elif args.dataset == "Cifar100":
        
        if "vit-base" in args.model_name:
            print(args.model_name)
            model = ViTForImageClassification.from_pretrained('Ahmed9275/Vit-Cifar100')
            config = model.config
            print(config)
    
        
    return model, config