import argparse
import math
from pathlib import Path

import PIL
import torch
import torch.nn as nn
import torchvision.transforms.functional as tF
import torchvision.transforms.functional_pil as tF_pil
import yaml
from PIL.Image import Image
from datasets import load_dataset
from torch.utils.data.dataloader import DataLoader
from torchvision.transforms.transforms import Compose, Normalize, Resize, ToTensor

from src.nfnets.model import NFNet
from src.nfnets.pretrained import pretrained_nfnet


# Evaluation method used in the paper
# This seems to perform slightly worse than a simple resize
class Pad32CenterCrop(nn.Module):
    def __init__(self, size: int):
        super().__init__()
        self.size = size
        self.scaled_size = (size + 32, size + 32)

    def forward(self, img: Image):
        img = tF_pil.resize(img=img, size=self.scaled_size, interpolation=PIL.Image.BICUBIC)
        return tF.center_crop(img, self.size)


def test_on_dataset(model: NFNet, dataset_name: str, batch_size=50, device='cuda:0'):
    transforms = Compose([
        # Pad32CenterCrop(model.test_imsize),
        ToTensor(),
        Resize((model.test_imsize, model.test_imsize), PIL.Image.BICUBIC),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    def transform_fn(examples):
        img = [transforms(img) for img in examples["img"]]
        label = [label for label in examples["label"]]
        return {"img": img, "label": label}

    dataset = load_dataset(dataset_name, split="train[:90%]")
    dataset.set_transform(transform_fn)

    dataloader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,  # F0: 120, F1: 100, F2: 80
        shuffle=False,
        pin_memory=False,
        num_workers=8,
    )

    print(f"Validation set contains {len(dataset)} images.")

    model.to(device)
    model.eval()

    processed_imgs = 0
    correct_labels = 0
    for step, data in enumerate(dataloader):
        with torch.no_grad():
            inputs = data[0].to(device)
            targets = data[1].to(device)

            output = model(inputs).type(torch.float32)

            processed_imgs += targets.size(0)
            _, predicted = torch.max(output, 1)
            correct_labels += (predicted == targets).sum().item()

    print(f"\nFinished eval. Accuracy: {100.0 * correct_labels / processed_imgs:6.4f}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate NFNets.')
    parser.add_argument('--config', type=Path, help='Path to config.yaml', default='default_config.yaml')
    parser.add_argument('--dataset-name', type=str, help='name of the dataset', required=True)
    parser.add_argument('--pretrained', type=Path, help='Path to pre-trained weights in haiku format', required=True)
    parser.add_argument('--batch-size', type=int, help='Validation batch size', default=50)
    parser.add_argument('--device', type=str, help='Validation device. Either \'cuda:0\' or \'cpu\'', default='cuda:0')
    args = parser.parse_args()

    if not args.pretrained.exists():
        raise FileNotFoundError(f"Could not find file {args.pretrained.absolute()}")

    with args.config.open() as file:
        config = yaml.safe_load(file)

    model = pretrained_nfnet(args.pretrained, config)

    test_on_dataset(model, dataset_name=args.dataset_name, batch_size=args.batch_size, device=args.device)
