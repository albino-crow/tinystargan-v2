import argparse
from core.model import build_model
import timm
import torch
import torch.nn as nn
import torch.optim as optim
from trainer import Trainer, create_data_loader
from core.checkpoint import CheckpointIO
from os.path import join as ospj

from pipeline import (
    create_fixed_domain_style_codes,
    Pipeline,
    FlexibleClassifier,
    StarGanV2Generator,
)


def main(args):
    backbone = timm.create_model(
        "hf-hub:1aurent/vit_small_patch8_224.lunit_dino",
        pretrained=True,
    )
    generator = None
    style_codes = None
    print(
        "----------------------------------->",
        "cuda" if torch.cuda.is_available() else "cpu",
        "<-----------------------------------",
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.mode == "star":
        _, nets_ema = build_model(args)
        ckptios = CheckpointIO(
            ospj(args.generator_checkpoint_dir, "{:06d}_nets_ema.ckpt"), **nets_ema
        )
        ckptios.load(args.generator_iter)

        style_codes = create_fixed_domain_style_codes(
            nets_ema.mapping_network, args.num_domains, args.latent_dim, seed=args.seed
        )
        generator = StarGanV2Generator(
            nets_ema.generator, output_dim=args.backbone_img_size
        )
    elif args.mode == "tiny":
        _, nets_ema = build_model(args)
        ckptios = CheckpointIO(
            ospj(args.generator_checkpoint_dir, "{:06d}_nets_ema.ckpt"), **nets_ema
        )
        ckptios.load(args.generator_iter)

        style_codes = create_fixed_domain_style_codes(
            nets_ema.mapping_network, args.num_domains, args.latent_dim, seed=args.seed
        )
        generator = StarGanV2Generator(
            nets_ema.generator, output_dim=args.backbone_img_size
        )
    pipeline = Pipeline(
        generator=generator,
        style_codes=style_codes if style_codes is not None else {},
        number_domain=args.num_domains,
        feature_extractor=FlexibleClassifier(backbone, num_classes=args.num_labels),
        fast_forward=(args.mode == "forward"),
        mix_up=args.mix_up,
        mix_up_start=args.mix_up_start,
        mix_up_end=args.mix_up_end,
        mix_up_growth=args.mix_up_growth,
    )

    # Move pipeline to device after creation
    pipeline.to(device)

    train_loader, valid_loader, test_loader = create_data_loader(
        args.data_dir,
        image_size=args.img_size,
        num_workers=args.num_workers,
        train_batch_size=args.batch_size,
        val_batch_size=args.val_batch_size,
        test_batch_size=args.test_batch_size,
    )

    # Create Adam optimizer using pipeline parameters
    optimizer = optim.Adam(
        pipeline.parameters(),
        lr=args.lr,
        betas=(args.beta1, args.beta2),
        weight_decay=args.weight_decay,
    )

    # Create loss function based on number of labels
    criterion = nn.CrossEntropyLoss()
    trainer = Trainer(
        model=pipeline,
        train_loader=train_loader,
        val_loader=valid_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        save_every_n_epochs=args.save_every,
        checkpoint_dir=args.checkpoint_dir,
    )
    trainer.train(num_epochs=args.total_epoch, resume_epoch=args.resume_iter)
    trainer.test(test_loader)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # model arguments
    parser.add_argument("--img_size", type=int, default=256, help="Image resolution")
    parser.add_argument(
        "--backbone_img_size", type=int, default=256, help="Image resolution"
    )
    parser.add_argument("--num_domains", type=int, default=2, help="Number of domains")
    parser.add_argument("--num_labels", type=int, default=2, help="Number of labels")

    parser.add_argument(
        "--latent_dim", type=int, default=16, help="Latent vector dimension"
    )
    parser.add_argument(
        "--hidden_dim",
        type=int,
        default=512,
        help="Hidden dimension of mapping network",
    )
    parser.add_argument(
        "--style_dim", type=int, default=64, help="Style code dimension"
    )
    parser.add_argument(
        "--alpha",
        type=int,
        default=512,
        help="Maximum channels for convolution and maximum hidden nodes for linear layers",
    )
    parser.add_argument(
        "--efficient",
        type=int,
        default=0,
        help="Efficient network with separable convolution and reduced MACs/FLOPs",
    )

    # weight for objective functions

    parser.add_argument(
        "--w_hpf", type=float, default=1, help="weight for high-pass filtering"
    )

    # training arguments
    parser.add_argument(
        "--data_dir",
        type=str,
        default="data",
        help="Directory containing training data",
    )
    parser.add_argument(
        "--total_epoch", type=int, default=100, help="Number of total iterations"
    )
    parser.add_argument(
        "--generator_iter",
        type=int,
        default=0,
        help="Iterations to resume training/testing",
    )
    parser.add_argument(
        "--resume_iter",
        type=int,
        default=0,
        help="Iterations to resume training/testing",
    )
    parser.add_argument(
        "--batch_size", type=int, default=8, help="Batch size for training"
    )
    parser.add_argument(
        "--val_batch_size", type=int, default=32, help="Batch size for validation"
    )
    parser.add_argument(
        "--test_batch_size", type=int, default=32, help="Batch size for testing"
    )
    parser.add_argument(
        "--lr", type=float, default=1e-4, help="Learning rate for D, E and G"
    )
    parser.add_argument("--f_lr", type=float, default=1e-6, help="Learning rate for F")
    parser.add_argument(
        "--beta1", type=float, default=0.0, help="Decay rate for 1st moment of Adam"
    )
    parser.add_argument(
        "--beta2", type=float, default=0.99, help="Decay rate for 2nd moment of Adam"
    )
    parser.add_argument(
        "--weight_decay", type=float, default=1e-4, help="Weight decay for optimizer"
    )

    # mix-up arguments
    parser.add_argument(
        "--mix_up", type=bool, default=False, help="Enable mix-up augmentation"
    )
    parser.add_argument(
        "--mix_up_start", type=float, default=0.0, help="Mix-up start value"
    )
    parser.add_argument(
        "--mix_up_end", type=float, default=1.0, help="Mix-up end value"
    )
    parser.add_argument(
        "--mix_up_growth", type=float, default=0.0001, help="Mix-up growth rate"
    )

    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="Number of workers used in DataLoader",
    )
    parser.add_argument(
        "--seed", type=int, default=777, help="Seed for random number generator"
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        default="celeba_hq_test_sample.jpg",
        help="Filename of generated big sample image",
    )

    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default="expr/checkpoints",
        help="Directory for saving network checkpoints",
    )
    parser.add_argument(
        "--generator_checkpoint_dir",
        type=str,
        default="expr/checkpoints",
        help="Directory for saving generator checkpoints",
    )
    # directory for calculating metrics
    parser.add_argument(
        "--eval_dir",
        type=str,
        default="expr/eval",
        help="Directory for saving metrics, i.e., loss and accuracy",
    )

    parser.add_argument("--save_every", type=int, default=10)

    parser.add_argument(
        "--mode",
        type=str,
        required=True,
        choices=["star", "tiny", "forward"],
        help="This argument is used in solver",
    )
    args = parser.parse_args()
    main(args)
