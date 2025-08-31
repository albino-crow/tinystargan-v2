import argparse
from core.model import build_model, build_teacher_model
import timm
import torch
import torch.nn as nn
import torch.optim as optim
from trainer import Trainer, create_data_loader
from core.checkpoint import CheckpointIO
from os.path import join as ospj

from pipeline import (
    MultiplePipeline,
    create_fixed_domain_style_codes,
    Pipeline,
    FlexibleClassifier,
    StarGanV2Generator,
)
# from torchao.quantization import quantize_, float8_weight_only


def str2bool(v):
    """Convert string to boolean value with flexible parsing."""
    if isinstance(v, bool):
        return v
    if v.lower() in ("true", "t", "1", "yes", "y"):
        return True
    elif v.lower() in ("false", "f", "0", "no", "n"):
        return False
    else:
        raise argparse.ArgumentTypeError(f"Boolean value expected, got: {v}")


def main(args):
    print("Creating backbone model...")
    backbone = timm.create_model(
        "hf-hub:1aurent/vit_small_patch8_224.lunit_dino",
        pretrained=True,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    backbone = backbone.to(device)
    generator = None
    style_codes = None
    print(
        "----------------------------------->",
        "cuda" if torch.cuda.is_available() else "cpu",
        "<-----------------------------------",
    )
    if args.quantized:
        print("\nOriginal model's first linear layer:")
        print(f"Type: {type(backbone.blocks[0].attn.qkv)}")
        print(f"Weight dtype: {backbone.blocks[0].attn.qkv.weight.dtype}")
        print(f"Weight shape: {backbone.blocks[0].attn.qkv.weight.shape}")
        print(
            f"Has quantization metadata: {hasattr(backbone.blocks[0].attn.qkv.weight, '_quantized_dtype')}"
        )

        backbone.half()

        print("\nQuantized model's first linear layer:")
        print(f"Type: {type(backbone.blocks[0].attn.qkv)}")
        print(f"Weight dtype: {backbone.blocks[0].attn.qkv.weight.dtype}")
        print(f"Weight shape: {backbone.blocks[0].attn.qkv.weight.shape}")
        print(
            f"Has quantization metadata: {hasattr(backbone.blocks[0].attn.qkv.weight, '_quantized_dtype')}"
        )

        # Check if weight is a quantized tensor
        if hasattr(backbone.blocks[0].attn.qkv.weight, "__tensor_flatten__"):
            print("Weight is a quantized tensor")

        # Show actual tensor implementation
        print(f"Weight tensor type: {type(backbone.blocks[0].attn.qkv.weight)}")
    print("setup:", print(" ".join(f"{k}={v}" for k, v in vars(args).items())))
    if args.mode == "forward" and args.loss_method != "normal":
        raise NotImplementedError("there is no such action available for this task")
    if args.mode == "star":
        nets_ema = build_teacher_model(args)
        ckptios = CheckpointIO(
            ospj(args.generator_checkpoint_dir, "{:06d}_nets_ema.ckpt"), **nets_ema
        )
        ckptios.load(args.generator_iter)

        style_codes = create_fixed_domain_style_codes(
            nets_ema.mapping_network, args.num_domains, args.latent_dim, seed=args.seed
        )
        generator = StarGanV2Generator(nets_ema.generator)
    elif args.mode == "tiny":
        _, nets_ema = build_model(args)
        ckptios = CheckpointIO(
            ospj(args.generator_checkpoint_dir, "{:06d}_nets_ema.ckpt"), **nets_ema
        )
        ckptios.load(args.generator_iter)

        style_codes = create_fixed_domain_style_codes(
            nets_ema.mapping_network, args.num_domains, args.latent_dim, seed=args.seed
        )
        generator = StarGanV2Generator(nets_ema.generator)

    pipeline = None
    if args.loss_method == "normal":
        pipeline = Pipeline(
            generator=generator,
            style_codes=style_codes if style_codes is not None else {},
            number_domain=args.num_domains,
            feature_extractor=FlexibleClassifier(backbone, num_classes=args.num_labels),
            fast_forward=(args.mode == "forward"),
            backbone_input_size=args.backbone_img_size,
            mix_up=args.mix_up,
            mix_up_start=args.mix_up_start,
            mix_up_end=args.mix_up_end,
            mix_up_growth=args.mix_up_growth,
            fake_guide=args.fake_guide,
            fake_guide_epsilon=args.fake_guide_epsilon,
        )
    else:
        pipeline = MultiplePipeline(
            generator=generator,
            style_codes=style_codes if style_codes is not None else {},
            number_domain=args.num_domains,
            feature_extractor=backbone,
            feature_extractor_embedding=384,
            number_label=args.num_labels,
            backbone_input_size=args.backbone_img_size,
            conv_type=args.convex_type,
            all_domain=args.all_domain,
            number_convex=args.number_convex,
            include_image=args.include_image,
            use_residual=args.use_residual,
            mode=args.loss_method,
            quantized=args.quantized,
        )
    pipeline.to(device)
    loss_method = (
        args.loss_method
        if args.loss_method
        in [
            "average",
            "majority",
            "leastrisk",
        ]
        else "normal"
    )
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
        test_loader=test_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        checkpoint_dir=args.checkpoint_dir,
        save_every_n_epochs=args.save_every,
        name=args.checkpoint_dir,
        mode=loss_method,
        description=f"Training pipeline with {args.loss_method} loss, weight generator {args.generator_checkpoint_dir}, location of weight {args.checkpoint_dir}",
    )
    trainer.train(num_epochs=args.total_epoch, resume_epoch=args.resume_iter)


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
    parser.add_argument(
        "--beta1", type=float, default=0.9, help="Decay rate for 1st moment of Adam"
    )
    parser.add_argument(
        "--beta2", type=float, default=0.999, help="Decay rate for 2nd moment of Adam"
    )
    parser.add_argument(
        "--weight_decay", type=float, default=1e-4, help="Weight decay for optimizer"
    )

    # mix-up arguments
    parser.add_argument(
        "--mix_up", type=str2bool, default=False, help="Enable mix-up augmentation"
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
        "--fake_guide", type=str2bool, default=False, help="Enable fake guide"
    )

    parser.add_argument(
        "--fake_guide_epsilon", type=float, default=0.2, help="Epsilon for fake guide"
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

    parser.add_argument("--save_every", type=int, default=2)

    parser.add_argument(
        "--mode",
        type=str,
        required=True,
        choices=["star", "tiny", "forward"],
        help="This argument is used in solver",
    )

    parser.add_argument(
        "--loss_method",
        type=str,
        default="normal",
        choices=[
            "normal",
            "average",
            "majority",
            "leastrisk",
            "ensemble",
            "vector_ensemble",
            "matrix_ensemble",
            "affine_vector_ensemble",
            "affine_matrix_ensemble",
            "attention_ff",
            "attention_fr",
            "attention_bb",
            "fake_guide",
            "attention_br",
            "attention_m",
        ],
        help="This argument is used in trainer to choose the loss method",
    )

    parser.add_argument(
        "--convex_type",
        type=str,
        default="blind",
        choices=["blind", "linear", "conv"],
        help="This argument is used in pipeline to find best combination of domains",
    )

    parser.add_argument(
        "--all_domain", type=str2bool, default=False, help="Enable all domain"
    )
    parser.add_argument(
        "--include_image", type=str2bool, default=False, help="Enable fake guide"
    )

    parser.add_argument(
        "--quantized", type=str2bool, default=False, help="Enable quantized model"
    )
    parser.add_argument(
        "--use_residual",
        type=str2bool,
        default=True,
        help="Enable residual connections",
    )
    parser.add_argument("--number_convex", type=int, default=5)

    args = parser.parse_args()
    main(args)
