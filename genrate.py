import glob
import os
import argparse
from os.path import join as ospj
from core.utils import save_image
from PIL import Image
import torchvision.transforms as T
import torch
from core.checkpoint import CheckpointIO
from core.model import build_model


def load_image(image_path, size, device):
    transform = T.Compose(
        [
            T.Resize((size, size)),
            T.ToTensor(),  # converts to [0,1]
            T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),  # -> [-1,1]
        ]
    )
    img = Image.open(image_path).convert("RGB")
    x = transform(img).unsqueeze(0).to(device)  # add batch dimension
    return x


def main(args):
    _, nets_ema = build_model(args)
    ckptios = CheckpointIO(
        ospj(args.checkpoint_dir, "{:06d}_nets_ema.ckpt"), **nets_ema
    )
    ckptios.load(args.step)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(42)

    fixed_latents = [
        torch.randn(1, args.latent_dim).to(device) for _ in range(args.num_domains)
    ]

    for folder in ["train", "test", "valid"]:
        for subfolder in ["0", "1"]:
            input_path = ospj(args.input_dir, folder, subfolder)
            output_path = ospj(args.output_dir, folder, subfolder)
            png_files = glob.glob(ospj(input_path, "*.png"))

            for file_path in png_files:
                generate_and_save_image(
                    fixed_latents,
                    nets_ema,
                    file_path,
                    output_path,
                    args.img_size,
                    args.num_domains,
                    args.latent_dim,
                    device,
                )

@torch.no_grad()
def generate_and_save_image(
    fixed_latents, nets, image_path, output_dir, size, domain, device
):
    file_name = os.path.basename(image_path)
    output_folder = ospj(output_dir, file_name)
    os.makedirs(output_folder, exist_ok=True)
    real_image = load_image(image_path, size, device)
    for i in range(domain):
        latent = fixed_latents[i]
        z = nets.mapping_network(latent, i)
        fake_image = nets.generator(real_image, z)
        save_image(fake_image, ospj(output_folder, f"{file_name}_{i}.png"))
        del z
        del fake_image
        torch.cuda.empty_cache()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # model arguments
    parser.add_argument("--img_size", type=int, default=96, help="Image resolution")
    parser.add_argument("--num_domains", type=int, default=5, help="Number of domains")
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

    # misc
    parser.add_argument(
        "--mode",
        type=str,
        required=True,
        choices=["latent", "ref"],
        help="This argument is used in solver",
    )

    # directory for training
    parser.add_argument(
        "--input_dir",
        type=str,
        default="data",
        help="Directory containing training images",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/celeba_hq/val",
        help="Directory containing validation images",
    )
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default="expr/checkpoints",
        help="Directory for saving network checkpoints",
    )

    parser.add_argument(
        "--step",
        type=str,
        default=100000,
        help="checkpoint saved",
    )
    args = parser.parse_args()
    main(args)
