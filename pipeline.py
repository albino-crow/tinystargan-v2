"""
StarGAN v2 Generator
Simple class for generating fake images using StarGAN v2.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from core.model import build_model

from core.checkpoint import CheckpointIO


class FlexibleClassifier(nn.Module):
    def __init__(self, backbone, embedding_dim=384, num_classes=2):
        super().__init__()
        self.backbone = backbone
        self.classifier = nn.Linear(
            embedding_dim, num_classes
        )  # flexible number of classes

    def forward(self, x):
        with torch.no_grad():  # freeze backbone
            features = self.backbone(x)
        out = self.classifier(features)
        return out


class ColorStyleExtractor(nn.Module):
    def __init__(self, number_domain=5):  # Match StarGAN v2's style_dim
        super().__init__()

        # Multi-scale convolutions in first layer
        self.conv_3x3 = nn.Conv2d(
            3, 12, kernel_size=3, padding=1
        )  # 12 channels with 3x3 kernel
        self.conv_1x1 = nn.Conv2d(
            3, 12, kernel_size=1, padding=0
        )  # 12 channels with 1x1 kernel
        self.conv_5x5 = nn.Conv2d(
            3, 12, kernel_size=5, padding=2
        )  # 12 channels with 5x5 kernel
        # Total: 12 + 12 + 12 = 36 channels

        self.conv1 = nn.Conv2d(36, 36, kernel_size=1)  # 36 -> 36 with 1x1 kernel
        self.conv2 = nn.Conv2d(36, number_domain, kernel_size=1)

        # Batch normalization layers
        self.bn_3x3 = nn.BatchNorm2d(12)
        self.bn_1x1 = nn.BatchNorm2d(12)
        self.bn_5x5 = nn.BatchNorm2d(12)
        self.bn1 = nn.BatchNorm2d(36)
        self.bn2 = nn.BatchNorm2d(36)
        self.bn3 = nn.BatchNorm2d(number_domain)

        # Add attention for better feature selection
        self.attention = nn.Conv2d(number_domain, 1, kernel_size=1)

    def forward(self, x):
        # Multi-scale feature extraction in parallel
        feat_3x3 = F.relu(self.bn_3x3(self.conv_3x3(x)))  # [N, 12, H, W]
        feat_1x1 = F.relu(self.bn_1x1(self.conv_1x1(x)))  # [N, 12, H, W]
        feat_5x5 = F.relu(self.bn_5x5(self.conv_5x5(x)))  # [N, 12, H, W]

        # Concatenate all features: 12 + 12 + 12 = 36 channels
        x = torch.cat([feat_3x3, feat_1x1, feat_5x5], dim=1)  # [N, 36, H, W]

        x = F.relu(self.bn1(self.conv1(x)))  # [N, 36, H, W] -> [N, 36, H, W]
        x = self.bn3(self.conv2(x))  # [N, 36, H, W] -> [N, number_domain, H, W]

        # Apply attention weighting
        attention = torch.sigmoid(self.attention(x))
        x = x * attention

        x = F.adaptive_avg_pool2d(x, (1, 1)).squeeze(-1).squeeze(-1)
        return F.normalize(x, p=2, dim=1)


class StarGanV2Generator(nn.Module):
    def __init__(
        self,
        generator,
        mapping_network=None,
        fan=None,
        generator_only_gpu=False,
        output_dim=None,
        w_hpf=0,
    ):
        """
        Initialize the StarGAN v2 generator.

        Args:
            generator: StarGAN v2 generator network
            mapping_network: StarGAN v2 mapping network (style mapper)
            args: Optional arguments object containing model configuration
            fan: Optional FAN network for high-pass filtering
            generator_only_gpu: If True, only generator goes to GPU, others stay on CPU
            output_dim: Optional output dimension for resizing generated images
        """
        super().__init__()
        self.generator_only_gpu = generator_only_gpu
        self.w_hpf = w_hpf
        self.output_dim = output_dim

        # Initialize resize transform if output_dim is specified
        self.resize_transform = (
            None
            if output_dim is None
            else transforms.Resize(
                (output_dim, output_dim),
                interpolation=transforms.InterpolationMode.BILINEAR,
                antialias=True,
            )
        )

        if generator_only_gpu:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.cpu_device = torch.device("cpu")
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.cpu_device = self.device

        # Store the provided networks
        self.generator = generator
        self.mapping_network = mapping_network
        self.fan = fan

        # Move networks to appropriate devices and set to eval mode
        self.generator.to(self.device).eval()

        if mapping_network is not None:
            if generator_only_gpu:
                self.mapping_network.to(self.cpu_device).eval()
            else:
                self.mapping_network.to(self.device).eval()

        if self.fan is not None:
            if generator_only_gpu:
                self.fan.to(self.cpu_device).eval()
            else:
                self.fan.to(self.device).eval()

    @torch.no_grad()
    def generate_with_style_code(self, input_images, style_code):
        """
        Generate fake images using pre-computed style code.

        Args:
            input_images: Input images tensor [batch_size, 3, H, W] (normalized to [-1, 1])
            style_code: Pre-computed style code tensor [batch_size, style_dim]

        Returns:
            Generated fake images tensor [batch_size, 3, H, W] (normalized to [-1, 1])
        """
        input_images = input_images.to(self.device)
        style_code = style_code.to(self.device)

        # Get masks if using high-pass filtering
        masks = None
        if self.w_hpf > 0 and self.fan is not None:
            if self.generator_only_gpu:
                # Move input to CPU for FAN processing, then back to GPU
                input_cpu = input_images.to(self.cpu_device)
                masks = self.fan.get_heatmap(input_cpu)
                if masks is not None:
                    masks = masks.to(self.device)
            else:
                masks = self.fan.get_heatmap(input_images)

        # Generate fake images using the provided style code
        fake_images = self.generator(input_images, style_code, masks=masks)

        fake_images = (fake_images + 1.0) / 2.0

        if self.resize_transform is not None:
            if self.output_dim != fake_images.size(2):
                fake_images = self.resize_transform(fake_images)

            # Keep in [0, 1] range for pretrained models (DINO, ResNet, etc.)
            # Most pretrained models expect inputs in [0, 1] range, not [-1, 1]

        return fake_images


class Pipeline(nn.Module):
    def __init__(
        self,
        generator,
        style_codes,
        number_domain,
        feature_extractor,
        fast_forward=False,
    ):
        super().__init__()
        self.generator = generator
        self.number_domain = number_domain
        self.color_extractor = (
            ColorStyleExtractor(self.number_domain) if not fast_forward else None
        )
        self.feature_extractor = feature_extractor
        self.fast_forward = fast_forward

        # Convert style_codes dict from create_fixed_domain_style_codes to tensor once
        # and register as buffer so it moves with the model to different devices
        if not fast_forward and style_codes is not None:
            style_codes_tensor = torch.stack(
                [style_codes[i] for i in range(number_domain)]
            )
            self.register_buffer("style_codes_tensor", style_codes_tensor)
        else:
            self.register_buffer("style_codes_tensor", None)

    def forward(self, x):
        # Extract domain weights from input image
        if not self.fast_forward:
            domain_weights = self.color_extractor(x)  # [batch_size, number_domain]

            # Weight the style codes: [batch_size, number_domain] @ [number_domain, style_dim] = [batch_size, style_dim]
            weighted_style_code = torch.matmul(domain_weights, self.style_codes_tensor)

            # Generate fake image using the generator with weighted style code
            x = self.generator.generate_with_style_code(x, weighted_style_code)

        # Extract features and classify
        logits = self.feature_extractor(x)

        # Let PyTorch handle memory management automatically
        # Don't manually delete fake_image as it may be needed for backprop

        return logits


def create_fixed_domain_style_codes(
    mapping_network, num_domains, latent_dim, device=None, seed=777, force_cpu=False
):
    """
    Create fixed domain style codes using a mapping network with reproducible random latents.
    This function is separate from the class and uses a fixed seed for consistent results.

    Args:
        mapping_network: StarGAN v2 mapping network
        num_domains: Number of domains to create style codes for
        latent_dim: Latent vector dimension
        device: Device to run on (if None, uses cuda if available)
        seed: Random seed for reproducible latent vectors (default: 777)
        force_cpu: If True, forces computation on CPU regardless of device availability

    Returns:
        Dict {domain: style_code} containing fixed style codes for all domains
    """
    if device is None:
        if force_cpu:
            device = torch.device("cpu")
        else:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Set fixed seed for reproducible results
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    domain_style_codes = {}
    mapping_network = mapping_network.to(device)

    with torch.no_grad():
        # Generate one fixed latent vector and copy it for all domains
        latent_vector = torch.randn(1, latent_dim).to(device)
        latent_vectors = latent_vector.repeat(
            num_domains, 1
        )  # [num_domains, latent_dim]

        # Create domain tensor for all domains at once
        domain_tensor = (
            torch.arange(num_domains).to(device).long()
        )  # [0, 1, 2, ..., num_domains-1]

        # Generate style codes for all domains in one forward pass
        style_codes = mapping_network(
            latent_vectors, domain_tensor
        )  # [num_domains, style_dim]

        # Convert to dictionary format and detach to ensure they act as constants
        for domain in range(num_domains):
            domain_style_codes[domain] = style_codes[domain].detach()

    return domain_style_codes


def create_generator(generator, mapping_network=None, fan=None, output_dim=None):
    """
    Convenience function to create StarGAN v2 generator from networks.

    Args:
        generator: StarGAN v2 generator network
        mapping_network: StarGAN v2 mapping network
        args: Optional model configuration arguments
        fan: Optional FAN network for high-pass filtering
        output_dim: Optional output dimension for resizing generated images

    Returns:
        StarGanV2Generator instance
    """
    return StarGanV2Generator(generator, mapping_network, fan, output_dim=output_dim)


def create_generator_from_checkpoint(args, checkpoint_path, output_dim=None):
    """
    Convenience function to create StarGAN v2 generator from checkpoint.

    Args:
        args: Model configuration arguments
        checkpoint_path: Path to trained checkpoint

    Returns:
        StarGanV2Generator instance
    """

    # Build the networks
    _, nets_ema = build_model(args)

    # Load checkpoint
    ckptios = [CheckpointIO(checkpoint_path, **nets_ema)]
    for ckptio in ckptios:
        ckptio.load(step=None)

    # Extract the networks
    generator = nets_ema.generator
    mapping_network = nets_ema.mapping_network
    fan = nets_ema.fan if hasattr(nets_ema, "fan") else None

    return StarGanV2Generator(generator, mapping_network, fan, output_dim=output_dim)


def create_gpu_efficient_generator(args, checkpoint_path=None, output_dim=None):
    """
    Create a GPU-efficient StarGAN v2 generator where only the generator network
    is on GPU while mapping network stays on CPU for style code generation.

    This function implements your specific workflow:
    1. Build nets_ema
    2. Use create_fixed_domain_style_codes to get style codes on CPU
    3. Only put generator on GPU

    Args:
        args: Model configuration arguments containing num_domains, latent_dim, etc.
        checkpoint_path: Optional path to trained checkpoint

    Returns:
        tuple: (generator_instance, domain_style_codes)
            - generator_instance: StarGanV2Generator with generator_only_gpu=True
            - domain_style_codes: Dict of pre-computed style codes for all domains
    """

    # Step 1: Build the networks
    _, nets_ema = build_model(args)

    # Load checkpoint if provided
    if checkpoint_path:
        ckptios = [CheckpointIO(checkpoint_path, **nets_ema)]
        for ckptio in ckptios:
            ckptio.load(step=None)

    # Extract the networks
    generator = nets_ema.generator
    mapping_network = nets_ema.mapping_network
    fan = nets_ema.fan if hasattr(nets_ema, "fan") else None

    # Step 2: Create fixed domain style codes on GPU, then move mapping network back to CPU
    # First, temporarily move mapping network to GPU for style code generation
    gpu_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mapping_network.to(gpu_device)

    # Generate style codes on GPU
    domain_style_codes = create_fixed_domain_style_codes(
        mapping_network=mapping_network,
        num_domains=args.num_domains,
        latent_dim=args.latent_dim,
        device=gpu_device,  # Use GPU for style code generation
        seed=777,
        force_cpu=False,  # Allow GPU usage
    )

    # Move mapping network back to CPU to save GPU memory
    mapping_network.to(torch.device("cpu"))

    # Step 3: Create generator with only generator on GPU
    generator_instance = StarGanV2Generator(
        generator=generator,
        mapping_network=mapping_network,
        fan=fan,
        generator_only_gpu=True,  # This ensures only generator goes to GPU
        output_dim=output_dim,
    )

    return generator_instance, domain_style_codes
