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
        self.embedding_dim = embedding_dim
        self.classifier = nn.Linear(
            embedding_dim, num_classes
        )  # flexible number of classes

    def forward(self, x):
        with torch.no_grad():  # freeze backbone
            features = self.backbone(x)
        # Normalize features before classification
        features = F.normalize(features, p=2, dim=1)
        out = self.classifier(features)
        # Normalize output as well
        out = F.normalize(out, p=2, dim=1)
        return out


class OneLayerClassifier(nn.Module):
    def __init__(self, input_dim=384, num_classes=2):
        super().__init__()
        self.classifier = nn.Linear(input_dim, num_classes)

    def forward(self, x):
        out = self.classifier(x)
        # Normalize output along feature dimension
        out = F.normalize(out, p=2, dim=1)
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

        # Initialize resize transform if output_dim is specified

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

        # Keep in [0, 1] range for pretrained models (DINO, ResNet, etc.)
        # Most pretrained models expect inputs in [0, 1] range, not [-1, 1]

        return fake_images


class BlindDomainPredictor(nn.Module):
    """
    Blind convex model that learns domain weights without image input.
    Contains learnable parameters that are optimized through backpropagation
    to produce convex output (non-negative values that sum to 1).
    """

    def __init__(self, number_domain):
        super().__init__()
        self.number_domain = number_domain

        # Learnable parameters initialized randomly
        # These will be optimized through backpropagation
        self.domain_weights = nn.Parameter(torch.randn(number_domain))

    def forward(self, x=None):
        """
        Forward pass that returns convex output (non-negative, sum=1)

        Args:
            x: Input image tensor (ignored, can be None)

        Returns:
            Domain weights tensor [1, number_domain] where:
            - All values are non-negative
            - Each row sums to 1 (convex combination)
        """
        # Apply softmax to ensure convex output (non-negative, sum=1)
        domain_weights = F.softmax(self.domain_weights, dim=0)

        # Expand to match batch size if input is provided
        if x is not None:
            batch_size = x.size(0)
            domain_weights = domain_weights.unsqueeze(0).expand(batch_size, -1)
        else:
            # Return single batch dimension if no input provided
            domain_weights = domain_weights.unsqueeze(0)

        return domain_weights


class ConvexDomainPredictor(nn.Module):
    """
    Simple single-layer convex model that takes an image as input and returns convex output
    (non-negative values that sum to 1) with dimension equal to number of domains.
    """

    def __init__(self, number_domain, input_channels=3):
        super().__init__()
        self.number_domain = number_domain

        # Simple single convolutional layer followed by global average pooling
        self.conv_layer = nn.Conv2d(input_channels, number_domain, kernel_size=1)

    def forward(self, x):
        """
        Forward pass that returns convex output (non-negative, sum=1)

        Args:
            x: Input image tensor [batch_size, channels, height, width]

        Returns:
            Domain weights tensor [batch_size, number_domain] where:
            - All values are non-negative
            - Each row sums to 1 (convex combination)
        """
        # Pass through single conv layer: [batch_size, number_domain, height, width]
        conv_out = self.conv_layer(x)

        # Global average pooling to get [batch_size, number_domain]
        pooled = F.adaptive_avg_pool2d(conv_out, (1, 1)).squeeze(-1).squeeze(-1)

        # Apply softmax to ensure convex output (non-negative, sum=1)
        domain_weights = F.softmax(pooled, dim=1)

        return domain_weights


class LinearDomainPredictor(nn.Module):
    """
    Simple single linear layer model that takes an image as input and returns convex output
    (non-negative values that sum to 1) with dimension equal to number of domains.
    """

    def __init__(self, number_domain, input_channels=3, image_size=256):
        super().__init__()
        self.number_domain = number_domain
        self.input_size = input_channels * image_size * image_size

        # Simple single linear layer
        self.linear_layer = nn.Linear(self.input_size, number_domain)

    def forward(self, x):
        """
        Forward pass that returns convex output (non-negative, sum=1)

        Args:
            x: Input image tensor [batch_size, channels, height, width]

        Returns:
            Domain weights tensor [batch_size, number_domain] where:
            - All values are non-negative
            - Each row sums to 1 (convex combination)
        """
        # Flatten the image: [batch_size, channels * height * width]
        flattened = x.view(x.size(0), -1)

        # Pass through single linear layer: [batch_size, number_domain]
        logits = self.linear_layer(flattened)

        # Apply softmax to ensure convex output (non-negative, sum=1)
        domain_weights = F.softmax(logits, dim=1)

        return domain_weights


class SimpleAttention(nn.Module):
    """
    Simple attention module with independent normalization for each fake image.
    Each fake image is normalized independently using its own running statistics.
    """

    def __init__(
        self, feature_dim, num_fakes, hidden_dim=128, momentum=0.1, res_on=True
    ):
        super().__init__()
        self.feature_dim = feature_dim
        self.num_fakes = num_fakes
        self.hidden_dim = hidden_dim
        self.momentum = momentum
        self.res_on = res_on

        # Running statistics for normalization - one set per fake image
        self.register_buffer("running_mean", torch.zeros(num_fakes, feature_dim))
        self.register_buffer("running_var", torch.ones(num_fakes, feature_dim))
        self.register_buffer("num_batches_tracked", torch.tensor(0, dtype=torch.long))

        # Neural network to transform query features
        self.query_network = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, feature_dim),
        )

        # self.attention_network = nn.Sequential(
        #     nn.Linear(feature_dim * 2, hidden_dim),  # Concat query + aggregated fakes
        #     nn.ReLU(),
        #     nn.Linear(hidden_dim, feature_dim)
        # )

    def forward(self, real_image_feature, fakes_image_vector):
        """
        Args:
            real_image_feature: [batch_size, feature_dim] - query features
            fakes_image_vector: [batch_size, num_fakes, feature_dim] - key features

        Returns:
            output_feature: [batch_size, feature_dim]
        """

        # Step 1: Normalize each fake image independently
        if self.training:
            # Update running statistics during training
            # Compute statistics for each fake image independently across the batch dimension
            batch_mean = fakes_image_vector.mean(dim=0)  # [num_fakes, feature_dim]
            batch_var = fakes_image_vector.var(
                dim=0, unbiased=False
            )  # [num_fakes, feature_dim]

            # Update running statistics for each fake image independently
            self.num_batches_tracked += 1
            self.running_mean = (
                1 - self.momentum
            ) * self.running_mean + self.momentum * batch_mean
            self.running_var = (
                1 - self.momentum
            ) * self.running_var + self.momentum * batch_var

            # Use batch statistics for normalization
            normalized_fakes = (
                fakes_image_vector - batch_mean.unsqueeze(0)
            ) / torch.sqrt(batch_var.unsqueeze(0) + 1e-8)
        else:
            # Use running statistics during evaluation
            normalized_fakes = (
                fakes_image_vector - self.running_mean.unsqueeze(0)
            ) / torch.sqrt(self.running_var.unsqueeze(0) + 1e-8)

        # Step 2: Pass query (real image features) through neural network
        aggregated_fakes = normalized_fakes.mean(dim=1)
        transformed_query = self.query_network(
            aggregated_fakes
        )  # [batch_size, feature_dim]

        # Step 3: Create output using real image vector and network result
        # Simple combination: element-wise multiplication
        output_feature = (
            real_image_feature * transformed_query
        )  # [batch_size, feature_dim]

        # Step 4: Apply residual connection if enabled
        if self.res_on:
            output_feature = output_feature + real_image_feature

        return output_feature


class AttentionModule(nn.Module):
    """
    Multi-head attention module for combining real and fake image features.
    Supports three different attention modes.
    """

    def __init__(self, feature_dim, num_heads=8, dropout=0.1, use_residual=True):
        super().__init__()
        self.feature_dim = feature_dim
        self.num_heads = num_heads
        self.head_dim = feature_dim // num_heads
        self.use_residual = use_residual

        assert feature_dim % num_heads == 0, (
            "feature_dim must be divisible by num_heads"
        )

        # Linear transformations for Q, K, V
        self.W_q = nn.Parameter(torch.randn(feature_dim, feature_dim))
        self.W_k = nn.Parameter(torch.randn(feature_dim, feature_dim))
        self.W_v = nn.Parameter(torch.randn(feature_dim, feature_dim))

        nn.init.xavier_uniform_(self.W_q)
        nn.init.xavier_uniform_(self.W_k)
        nn.init.xavier_uniform_(self.W_v)

        # Output projection
        self.W_o = nn.Linear(feature_dim, feature_dim)

        # Dropout
        self.dropout = nn.Dropout(dropout)

        # Layer norm - only create if using residual connections
        if self.use_residual:
            self.layer_norm = nn.LayerNorm(feature_dim)

    def forward(self, query_features, key_features, value_features):
        """
        Args:
            query_features: [batch_size, feature_dim]
            key_features: [batch_size, num_images, feature_dim]
            value_features: [batch_size, num_images, feature_dim]

        Returns:
            attended_features: [batch_size, feature_dim]
        """
        batch_size = query_features.size(0)

        # Linear transformations
        Q = torch.matmul(query_features, self.W_q)  # [batch_size, feature_dim]
        K = torch.matmul(
            key_features, self.W_k
        )  # [batch_size, num_images, feature_dim]
        V = torch.matmul(
            value_features, self.W_v
        )  # [batch_size, num_images, feature_dim]
        # Reshape for multi-head attention
        Q = Q.view(batch_size, 1, self.num_heads, self.head_dim).transpose(
            1, 2
        )  # [batch_size, num_heads, 1, head_dim]
        K = K.view(batch_size, -1, self.num_heads, self.head_dim).transpose(
            1, 2
        )  # [batch_size, num_heads, num_images, head_dim]
        V = V.view(batch_size, -1, self.num_heads, self.head_dim).transpose(
            1, 2
        )  # [batch_size, num_heads, num_images, head_dim]

        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (
            self.head_dim**0.5
        )  # [batch_size, num_heads, 1, num_images]
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)

        # Apply attention to values
        attended = torch.matmul(
            attention_weights, V
        )  # [batch_size, num_heads, 1, head_dim]

        # Concatenate heads and project
        attended = (
            attended.transpose(1, 2).contiguous().view(batch_size, self.feature_dim)
        )  # [batch_size, feature_dim]
        output = self.W_o(attended)

        # Optional residual connection and layer norm
        if self.use_residual:
            output = self.layer_norm(output + query_features)

        return output


class MultiplePipeline(nn.Module):
    def __init__(
        self,
        generator,
        style_codes,
        number_domain,
        feature_extractor,
        feature_extractor_embedding,
        backbone_input_size,
        conv_type="linear",
        number_label=2,
        all_domain=False,
        number_convex=5,
        include_image=True,
        use_residual=True,
        mode=None,  # New parameter to control output format
    ):
        super().__init__()
        self.generator = generator
        self.number_domain = number_domain
        self.include_image = include_image
        self.feature_extractor = feature_extractor
        self.all_domain = all_domain
        self.number_convex = number_convex
        self.backbone_input_size = backbone_input_size
        self.classifier = OneLayerClassifier(
            input_dim=feature_extractor_embedding, num_classes=number_label
        )
        self.mode = mode  # 'ensemble', 'vector_ensemble', 'matrix_ensemble', or None

        # For vector_ensemble: learnable weights for each logit vector
        if self.mode in ["vector_ensemble", "affine_vector_ensemble"]:
            print(f"++++++++++{self.mode}+++++++++++++++")
            self.vector_weights = nn.Parameter(
                torch.ones(number_convex + int(include_image))
            )
        # For matrix_ensemble: learnable weights for each scalar in each logit vector
        elif self.mode in ["matrix_ensemble", "affine_matrix_ensemble"]:
            print(f"++++++++++{self.mode}+++++++++++++++")
            self.matrix_weights = nn.Parameter(
                torch.ones(
                    number_convex + int(include_image), feature_extractor_embedding
                )
            )
        elif "attention" in self.mode:
            print(f"++++++++++{self.mode}+++++++++++++++")
            self.attention_module = AttentionModule(
                feature_dim=feature_extractor_embedding,
                num_heads=8,
                use_residual=use_residual,
            )
        elif "fake_guide" == self.mode:
            print(f"++++++++++{self.mode}+++++++++++++++")
            self.fake_guide = SimpleAttention(
                feature_dim=feature_extractor_embedding,
                num_fakes=number_convex,
                hidden_dim=128,
                momentum=0.1,
                res_on=use_residual,
            )

        self.resize_transform = (
            None
            if backbone_input_size is None
            else transforms.Resize(
                (self.backbone_input_size, self.backbone_input_size),
                interpolation=transforms.InterpolationMode.BILINEAR,
                antialias=True,
            )
        )

        # Convert style_codes dict from create_fixed_domain_style_codes to tensor once
        # and register as buffer so it moves with the model to different devices
        style_codes_tensor = torch.stack([style_codes[i] for i in range(number_domain)])
        self.register_buffer("style_codes_tensor", style_codes_tensor)

        # Create convex models that output domain weights (non-negative, sum=1)
        if conv_type == "conv":
            self.convex_models = nn.ModuleList(
                [ConvexDomainPredictor(number_domain) for _ in range(number_convex)]
            )
        elif conv_type == "linear":
            self.convex_models = nn.ModuleList(
                [LinearDomainPredictor(number_domain) for _ in range(number_convex)]
            )
        elif conv_type == "blind":
            self.convex_models = nn.ModuleList(
                [BlindDomainPredictor(number_domain) for _ in range(number_convex)]
            )
        else:
            raise ValueError(f"Unknown conv_type: {conv_type}")

    # ...existing code...
    def extract(self, x):
        with torch.no_grad():  # freeze backbone
            return self.feature_extractor(x)

    def size_fixer(self, x):
        if self.resize_transform is not None:
            if self.backbone_input_size != x.size(2):
                x = self.resize_transform(x)
        return x

    def forward(self, x):
        # Extract domain weights from input image

        # Store all generated images
        xs = []

        if self.all_domain is False:
            for model in self.convex_models:
                domain_weights = model(x)
                weighted_style_code = torch.matmul(
                    domain_weights, self.style_codes_tensor
                )

                fake_images = self.generator.generate_with_style_code(
                    x, weighted_style_code
                )
                fake_images = self.size_fixer(fake_images)
                xs.append(fake_images)
        else:
            for i in range(self.number_domain):
                fake_images = self.generator.generate_with_style_code(
                    x, self.style_codes_tensor[i]
                )
                fake_images = self.size_fixer(fake_images)
                xs.append(fake_images)

        if self.include_image:
            # Convert x from [-1, 1] to [0, 1] range to match fake_images
            original_images = (self.size_fixer(x) + 1.0) / 2.0
            xs.append(original_images)

        # Extract features from all images
        all_logits = []
        for image in xs:
            logits = self.extract(image)
            all_logits.append(logits)
        if self.mode == "fake_guide":
            if self.include_image:
                # Last image is real (original), others are fake
                real = all_logits[-1]  # ✅ CORRECT - Last is real
                fake_logits = all_logits[:-1]  # ✅ CORRECT - All others are fake
            else:
                # If no real image included, use first fake as query
                real = all_logits[0]
                fake_logits = all_logits
            fake_features_stack = torch.stack(
                fake_logits, dim=1
            )  # [batch_size, num_fakes, feature_dim]

            return self.classifier(self.fake_guide(real, fake_features_stack))
        elif "attention" in self.mode:
            # Stack fake features: [batch_size, num_fake_images, feature_dim]
            if self.include_image:
                # Last image is real (original), others are fake
                query_logits = all_logits[-1]  # ✅ CORRECT - Last is real
                fake_logits = all_logits[:-1]  # ✅ CORRECT - All others are fake
            else:
                # If no real image included, use first fake as query
                query_logits = all_logits[0]
                fake_logits = all_logits
            fake_features_stack = torch.stack(fake_logits, dim=1)

            attended_features = None

            if self.mode == "attention_ff":
                # Mode 1: Real image as query, fake images as both key and value
                attended_features = self.attention_module(
                    query_features=query_logits,
                    key_features=fake_features_stack,
                    value_features=fake_features_stack,
                )

            elif self.mode == "attention_fr":
                # Mode 2: Real image as query and value, fake images as key
                real_features_stack = query_logits.unsqueeze(
                    1
                )  # [batch_size, 1, feature_dim]
                attended_features = self.attention_module(
                    query_features=query_logits,
                    key_features=fake_features_stack,
                    value_features=real_features_stack,
                )

            elif self.mode == "attention_bb":
                # Mode 3: Real image as query, both fake and real as key and value
                all_features_stack = torch.cat(
                    [fake_features_stack, query_logits.unsqueeze(1)], dim=1
                )
                attended_features = self.attention_module(
                    query_features=query_logits,
                    key_features=all_features_stack,
                    value_features=all_features_stack,
                )
            elif self.mode == "attention_br":
                # Mode 4: Real image as query, both fake and real as key, real as value
                all_features_stack = torch.cat(
                    [fake_features_stack, query_logits.unsqueeze(1)], dim=1
                )
                real_features_stack = query_logits.unsqueeze(
                    1
                )  # [batch_size, 1, feature_dim]
                attended_features = self.attention_module(
                    query_features=query_logits,
                    key_features=all_features_stack,
                    value_features=real_features_stack,
                )
            return self.classifier(attended_features)

        elif self.mode == "ensemble":
            # Simple average (sum) over all logits
            ensemble_logits = torch.stack(all_logits, dim=0).mean(dim=0)
            return self.classifier(ensemble_logits)
        elif self.mode in ["vector_ensemble", "affine_vector_ensemble"]:
            # Weighted sum with learnable vector weights
            if self.mode == "vector_ensemble":
                weights = torch.softmax(
                    self.vector_weights, dim=0
                )  # Convex: softmax normalization
            else:  # affine_vector_ensemble
                weights = self.vector_weights / self.vector_weights.sum(
                    dim=0, keepdim=True
                )  # Affine: simple normalization
            logits_stack = torch.stack(all_logits, dim=0)  # [N, batch, logit]
            # Weighted sum over N pipelines/images
            ensemble_logits = (logits_stack * weights.unsqueeze(1).unsqueeze(2)).sum(
                dim=0
            )
            return self.classifier(ensemble_logits)
        elif self.mode in ["matrix_ensemble", "affine_matrix_ensemble"]:
            # Weighted sum with learnable matrix weights (per logit scalar)
            if self.mode == "matrix_ensemble":
                # Convex: softmax normalization - weights across all pipelines sum to 1
                weights = torch.softmax(self.matrix_weights, dim=0)  # [N, logit]
            else:  # affine_matrix_ensemble
                # Affine: simple normalization - weights can be negative
                weights = self.matrix_weights / self.matrix_weights.sum(
                    dim=0, keepdim=True
                )  # [N, logit]
            logits_stack = torch.stack(all_logits, dim=0)  # [N, batch, logit]
            # Apply element-wise weights to each scalar in each logit vector, then sum over N
            weighted_logits = logits_stack * weights.unsqueeze(1)  # [N, batch, logit]
            ensemble_logits = weighted_logits.sum(dim=0)  # [batch, logit]
            return self.classifier(ensemble_logits)
        else:
            # Return list of logits (for custom training loops that can handle multiple outputs)
            return [self.classifier(logits) for logits in all_logits]


class Pipeline(nn.Module):
    def __init__(
        self,
        generator,
        style_codes,
        number_domain,
        feature_extractor,
        backbone_input_size,
        mix_up=False,
        mix_up_start=0.0,
        mix_up_end=1.0,
        mix_up_growth=0.0001,
        fast_forward=False,
        fake_guide=False,
        fake_guide_epsilon=0.1,
    ):
        super().__init__()
        self.generator = generator
        self.number_domain = number_domain
        self.color_extractor = (
            ColorStyleExtractor(self.number_domain) if not fast_forward else None
        )
        self.feature_extractor = feature_extractor
        self.fast_forward = fast_forward
        self.fake_guide = fake_guide
        self.backbone_input_size = backbone_input_size
        self.mix_up = mix_up
        self.mix_up_start = mix_up_start
        self.mix_up_end = mix_up_end
        self.mix_up_growth = mix_up_growth
        self.mix_up_current = 0
        self.fake_guide_epsilon = fake_guide_epsilon
        self.resize_transform = (
            None
            if backbone_input_size is None
            else transforms.Resize(
                (self.backbone_input_size, self.backbone_input_size),
                interpolation=transforms.InterpolationMode.BILINEAR,
                antialias=True,
            )
        )

        # Convert style_codes dict from create_fixed_domain_style_codes to tensor once
        # and register as buffer so it moves with the model to different devices
        if not fast_forward and style_codes is not None:
            style_codes_tensor = torch.stack(
                [style_codes[i] for i in range(number_domain)]
            )
            self.register_buffer("style_codes_tensor", style_codes_tensor)
        else:
            self.register_buffer("style_codes_tensor", None)

    def size_fixer(self, x):
        if self.resize_transform is not None:
            if self.backbone_input_size != x.size(2):
                x = self.resize_transform(x)
        return x

    def forward(self, x):
        # Extract domain weights from input image
        if not self.fast_forward:
            domain_weights = self.color_extractor(x)  # [batch_size, number_domain]

            # Weight the style codes: [batch_size, number_domain] @ [number_domain, style_dim] = [batch_size, style_dim]
            weighted_style_code = torch.matmul(domain_weights, self.style_codes_tensor)

            # Generate fake image using the generator with weighted style code
            fake_images = self.generator.generate_with_style_code(
                x, weighted_style_code
            )
            fake_images = self.size_fixer(fake_images)
            if self.mix_up is True:
                alpha = (
                    self.mix_up_start
                    + (self.mix_up_end - self.mix_up_start) * self.mix_up_current
                )
                x = self.size_fixer(x)
                # Convert x from [-1, 1] to [0, 1] range to match fake_images
                x = (x + 1.0) / 2.0
                x = (1 - alpha) * x + alpha * fake_images
                if self.mix_up_current < 1:
                    self.mix_up_current += self.mix_up_growth

            if self.fake_guide:
                x = self.size_fixer(x)
                logits = self.feature_extractor(x)
                fake_logits = self.feature_extractor(fake_images)
                diff = torch.abs(logits - fake_logits)
                # Set only those logit values to zero where diff is greater than epsilon
                mask = diff > self.fake_guide_epsilon
                logits = torch.where(mask, torch.zeros_like(logits), logits)
                return logits
            else:
                x = fake_images
        else:
            x = (x + 1.0) / 2.0

        # Extract features and classify
        x = self.size_fixer(x)
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
