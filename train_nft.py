import argparse
import time
from pathlib import Path

import torch
import torch.nn as nn
import wandb
import yaml
from datasets import load_dataset
from lightning.fabric import Fabric
from lightning.fabric import seed_everything
from torch.utils.data.dataloader import DataLoader
from torchvision.transforms.transforms import Compose, Normalize, ToTensor, RandomHorizontalFlip
from tqdm import tqdm

from src.nfnets.optim import SGD_AGC
from src.vit.model import ViT, ViTPostNorm, ViTWithoutNorm, ViTWS


def train(config: dict) -> None:
    wandb.init(project="normalizer-free-transformers",
               entity="wade3han",
               name=config['model_type'] + f'lr-{config["learning_rate"]}')
    torch.set_float32_matmul_precision('high')

    fabric = Fabric(accelerator="cuda", devices=1, precision="16-mixed")
    seed_everything(config['seed'] + fabric.global_rank)

    fabric.launch()

    model_type = config.get('model_type')
    if model_type == 'vit':
        model_class = ViT
    elif model_type == 'vit_postnorm':
        model_class = ViTPostNorm
    elif model_type == 'vit_nonorm':
        model_class = ViTWithoutNorm
    elif model_type == 'vit_ws':
        model_class = ViTWS
    else:
        raise NotImplementedError

    if config['dataset'] == 'cifar10':
        normalize = Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.247, 0.243, 0.261])
        size = 32
        num_classes = 10
    elif config['dataset'] == 'cifar100':
        normalize = Normalize(mean=[0.5071, 0.4867, 0.4408], std=[0.2675, 0.2565, 0.2761])
        size = 32
        num_classes = 100
    else:
        raise NotImplementedError

    transforms = Compose([
        RandomHorizontalFlip(),
        ToTensor(),
        normalize,
    ])

    def transform_fn(examples):
        img = [transforms(img) for img in examples["img"]]
        label = [label for label in examples["label"]]
        return {"img": img, "label": label}

    model = model_class(
        image_size=size,
        patch_size=4,
        num_classes=num_classes,
        dim=384,
        depth=12,
        heads=6,
        mlp_dim=1536,
        dropout=0.1,
        emb_dropout=0.1
    )

    sum_of_params = sum(p.numel() for p in model.parameters())
    print(f"Model has {sum_of_params} parameters.")

    train_dataset = load_dataset(config['dataset'], split="train[:90%]")
    train_dataset.set_transform(transform_fn)
    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=config['num_workers'],
        pin_memory=config['pin_memory'],
    )

    val_dataset = load_dataset(config['dataset'], split="train[90%:]")
    val_dataset.set_transform(transform_fn)
    val_dataloader = DataLoader(
        dataset=val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config['num_workers'],
        pin_memory=config['pin_memory'],
    )

    if config['scale_lr']:
        learning_rate = config['learning_rate'] * config['batch_size'] / 256
    else:
        learning_rate = config['learning_rate']

    if not config['do_clip']:
        config['clipping'] = None

    optimizer = SGD_AGC(
        # The optimizer needs all parameter names
        # to filter them by hand later
        named_params=model.named_parameters(),
        lr=learning_rate,
        momentum=config['momentum'],
        clipping=config['clipping'],
        weight_decay=config['weight_decay'],
        nesterov=config['nesterov']
    )

    criterion = nn.CrossEntropyLoss()

    model, optimizer = fabric.setup(model, optimizer)
    train_dataloader, val_dataloader = fabric.setup_dataloaders(train_dataloader, val_dataloader)

    global_step = 0
    first_epoch = 0
    max_train_steps = config['epochs'] * len(train_dataloader)

    # Only show the progress bar on the master process
    progress_bar = tqdm(range(global_step, max_train_steps))

    # logging
    runs_dir = Path('runs')
    run_index = 0
    while (runs_dir / ('run' + str(run_index))).exists():
        run_index += 1
    runs_dir = runs_dir / ('run' + str(run_index))
    runs_dir.mkdir(exist_ok=False, parents=True)
    checkpoints_dir = runs_dir / 'checkpoints'
    checkpoints_dir.mkdir()

    wandb.watch(model)

    for epoch in range(first_epoch, config['epochs']):
        model.train()
        running_loss = 0.0
        processed_imgs = 0
        correct_labels = 0
        epoch_time = time.time()

        for step, data in enumerate(train_dataloader, 1):
            progress_bar.update(1)

            inputs = data["img"]
            targets = data["label"]

            output = model(inputs)
            loss = criterion(output, targets)

            fabric.backward(loss)
            optimizer.step()
            optimizer.zero_grad()

            wandb.log({"train/loss_per_steps": loss})

            running_loss += loss.item()
            processed_imgs += targets.size(0)
            _, predicted = torch.max(output, 1)
            correct_labels += (predicted == targets).sum().item()

            progress_bar.set_description(f"# step: {global_step}, epoch: {epoch}, "
                                         f"loss: {running_loss / step:6.4f}, "
                                         f"acc: {100.0 * correct_labels / processed_imgs:5.3f}%")
            global_step += 1

        elapsed = time.time() - epoch_time
        progress_bar.set_description(f"({elapsed:.3f}s, {elapsed / len(train_dataloader):.3}s/step, "
                                     f"{elapsed / len(train_dataloader):.3}s/img)")

        wandb.log({"train/loss_per_epoch": running_loss / (len(train_dataloader))})
        wandb.log({"train/accuracy_per_epoch": 100.0 * correct_labels / processed_imgs})

        # do evaluation
        with torch.no_grad():
            model.eval()
            running_loss = 0.0
            processed_imgs = 0
            correct_labels = 0
            for step, data in enumerate(val_dataloader):
                inputs = data["img"]
                targets = data["label"]

                output = model(inputs)
                loss = criterion(output, targets)

                running_loss += loss.item()
                processed_imgs += targets.size(0)
                _, predicted = torch.max(output, 1)
                correct_labels += (predicted == targets).sum().item()

            progress_bar.set_description(f"# step: {global_step}, epoch: {epoch}, "
                                         f"validation loss: {running_loss / len(val_dataloader):6.4f}, "
                                         f"validation acc: {100.0 * correct_labels / processed_imgs:5.3f}%")

            wandb.log({"validation/loss_per_epoch": running_loss / (len(val_dataloader))})
            wandb.log({"validation/accuracy_per_epoch": 100.0 * correct_labels / processed_imgs})

        # save models
        if epoch % 10 == 9:
            cp_path = checkpoints_dir / ("checkpoint_epoch" + str(epoch + 1) + ".pth")

            torch.save({
                'epoch': epoch,
                'model': model.state_dict(),
                'optim': optimizer.state_dict(),
            }, str(cp_path))

            print(f"Saved checkpoint to {str(cp_path)}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train NFTs.')
    parser.add_argument('--config', type=Path, help='Path to config.yaml', default='default_config.yaml')
    parser.add_argument('--model-type', type=str, help='Type of model to train', default='vit')
    parser.add_argument('--seed', type=int, help='Random seed', default=None)
    parser.add_argument('--learning-rate', type=float, help='Learning rate', default=None)
    args = parser.parse_args()

    if not args.config.exists():
        print(f"Config file \"{args.config}\" does not exist!\n")
        exit()

    with args.config.open() as file:
        config = yaml.safe_load(file)

    # Override config.yaml settings with command line settings
    for arg in vars(args):
        if getattr(args, arg) is not None:
            config[arg] = getattr(args, arg)

    train(config=config)
