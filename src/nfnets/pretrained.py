from pathlib import Path

import torch

from src.nfnets.model import NFNet


def pretrained_nfnet(pretrained_path, config: dict) -> NFNet:
    if isinstance(pretrained_path, str):
        pretrained_path = Path(pretrained_path)

    # Load parameters
    state_dict = torch.load(pretrained_path, map_location=torch.device('cpu'))['model']

    model = NFNet(
        num_classes=config['num_classes'],
        variant=config['variant'],
        stochdepth_rate=config['stochdepth_rate'],
        alpha=config['alpha'],
        se_ratio=config['se_ratio'],
        activation=config['activation']
    )

    model.load_state_dict(state_dict, strict=True)
    return model
