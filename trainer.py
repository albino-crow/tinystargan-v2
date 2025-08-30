import torch
import os
from tqdm import tqdm
from core.checkpoint import CheckpointIO
from torchvision import transforms
from torch.utils.data import DataLoader
from robust_dataset import RobustImageFolder
import torch.nn.functional as F
from enum import Enum


class ProcessingMode(Enum):
    """
    Enum for different processing modes when handling multiple outputs.
    """

    NORMAL = "normal"  # Standard single output processing
    AVERAGE = "average"  # Average all logits from multiple outputs
    MAJORITY = "majority"  # Use majority voting among predictions
    LEASTRISK = "leastrisk"  # Select prediction with highest confidence (lowest risk)

    @classmethod
    def get_valid_modes(cls):
        """Return list of valid mode strings."""
        return [mode.value for mode in cls]

    @classmethod
    def is_valid_mode(cls, mode):
        """Check if a mode string is valid."""
        if isinstance(mode, cls):
            return True
        return mode in cls.get_valid_modes()


def process_multiple_outputs(outputs_list, mode=ProcessingMode.AVERAGE):
    """
    Process multiple outputs based on different modes.

    Args:
        outputs_list: List of output tensors from model
        mode: Processing mode - ProcessingMode enum or string

    Returns:
        final_outputs: Processed outputs for loss calculation
        final_preds: Final predictions for accuracy calculation
    """
    # Convert string to enum if needed
    if isinstance(mode, str):
        mode_str = mode.lower()
        if mode_str == "normal":
            mode = ProcessingMode.NORMAL
        elif mode_str == "average":
            mode = ProcessingMode.AVERAGE
        elif mode_str == "majority":
            mode = ProcessingMode.MAJORITY
        elif mode_str == "leastrisk":
            mode = ProcessingMode.LEASTRISK
        else:
            raise ValueError(
                f"Unknown mode: {mode_str}. Valid modes: {ProcessingMode.get_valid_modes()}"
            )

    if mode == ProcessingMode.AVERAGE:
        # Average all logits
        final_outputs = torch.stack(outputs_list, dim=0).mean(dim=0)
        _, final_preds = torch.max(final_outputs, 1)

    elif mode == ProcessingMode.MAJORITY:
        # Get predictions from each output and use majority voting
        all_preds = []
        final_outputs = torch.stack(outputs_list, dim=0).mean(
            dim=0
        )  # For loss calculation

        for output in outputs_list:
            _, pred = torch.max(output, 1)
            all_preds.append(pred)

        # Stack predictions and use majority voting
        all_preds = torch.stack(all_preds, dim=0)  # [num_models, batch_size]

        # Majority voting for each sample
        final_preds = []
        for i in range(all_preds.size(1)):  # For each sample in batch
            sample_preds = all_preds[:, i]  # Get all predictions for this sample
            # Find most common prediction
            values, counts = torch.unique(sample_preds, return_counts=True)
            majority_pred = values[torch.argmax(counts)]
            final_preds.append(majority_pred)

        final_preds = torch.stack(final_preds)

    elif mode == ProcessingMode.LEASTRISK:
        # Select the prediction with highest confidence (lowest risk)
        max_confidences = []

        for output in outputs_list:
            # Get maximum confidence (probability) for each sample
            probs = F.softmax(output, dim=1)
            max_conf, _ = torch.max(probs, dim=1)
            max_confidences.append(max_conf)

        # Stack confidences: [num_models, batch_size]
        max_confidences = torch.stack(max_confidences, dim=0)

        # For each sample, find which model has highest confidence
        best_model_indices = torch.argmax(max_confidences, dim=0)

        # Select outputs and predictions from best models
        final_outputs = []
        final_preds = []

        for i, best_idx in enumerate(best_model_indices):
            final_outputs.append(outputs_list[best_idx][i])
            _, pred = torch.max(outputs_list[best_idx][i : i + 1], 1)
            final_preds.append(pred)

        final_outputs = torch.stack(final_outputs)
        final_preds = torch.cat(final_preds)

    else:
        raise ValueError(
            f"Unknown mode: {mode}. Valid modes: {ProcessingMode.get_valid_modes()}"
        )

    return final_outputs, final_preds


def train_one_epoch_with_mode(
    model,
    train_loader,
    criterion,
    optimizer,
    device,
    epoch,
    num_epochs,
    mode=ProcessingMode.NORMAL,
):
    """
    Enhanced train_one_epoch function that supports different processing modes.

    Args:
        mode: ProcessingMode enum or string - "normal", "average", "majority", "leastrisk"
    """
    # Convert to string for display purposes
    mode_str = mode.value if isinstance(mode, ProcessingMode) else mode
    model.train()
    loop = tqdm(
        train_loader, desc=f"Epoch [{epoch + 1}/{num_epochs}] ({mode_str})", leave=False
    )

    running_loss = 0.0
    running_corrects = 0
    num_batches = 0
    total_samples = 0

    for images, labels in loop:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)

        # Convert string to enum if needed
        if isinstance(mode, str):
            if mode.lower() == "normal":
                current_mode = ProcessingMode.NORMAL
            else:
                current_mode = mode  # Will be handled by process_multiple_outputs
        else:
            current_mode = mode

        if current_mode == ProcessingMode.NORMAL:
            # Standard processing for single output
            loss = criterion(outputs, labels)
            _, preds = torch.max(outputs, 1)
        else:
            # Multiple outputs processing
            if not isinstance(outputs, list):
                raise ValueError(
                    f"Expected list of outputs for mode '{mode_str}', but got single tensor"
                )

            # Process multiple outputs based on mode
            final_outputs, preds = process_multiple_outputs(outputs, current_mode)
            loss = criterion(final_outputs, labels)

        loss.backward()
        optimizer.step()

        # Update running loss
        running_loss += loss.item()

        # Compute batch accuracy
        running_corrects += torch.sum(preds == labels).item()
        total_samples += labels.size(0)

        num_batches += 1

        # Update tqdm bar with running loss and accuracy
        loop.set_postfix(
            loss=running_loss / num_batches, accuracy=running_corrects / total_samples
        )

        # Clear GPU memory cache after each batch
        torch.cuda.empty_cache()

    epoch_loss = running_loss / num_batches
    epoch_acc = running_corrects / total_samples
    print(
        f"Train Epoch [{epoch + 1}/{num_epochs}] ({mode_str}) Loss: {epoch_loss:.4f}, Acc: {epoch_acc:.4f}"
    )
    return epoch_loss, epoch_acc


@torch.no_grad()
def validate_one_epoch_with_mode(
    model, val_loader, criterion, device, epoch, num_epochs, mode=ProcessingMode.NORMAL
):
    """
    Enhanced validate_one_epoch function that supports different processing modes.

    Args:
        mode: ProcessingMode enum or string - "normal", "average", "majority", "leastrisk"
    """
    # Convert to string for display purposes
    mode_str = mode.value if isinstance(mode, ProcessingMode) else mode
    model.eval()
    running_loss = 0.0
    running_corrects = 0
    num_batches = 0
    total_samples = 0

    loop = tqdm(
        val_loader, desc=f"Val [{epoch + 1}/{num_epochs}] ({mode_str})", leave=False
    )
    for images, labels in loop:
        images, labels = images.to(device), labels.to(device)

        outputs = model(images)

        # Convert string to enum if needed
        if isinstance(mode, str):
            if mode.lower() == "normal":
                current_mode = ProcessingMode.NORMAL
            else:
                current_mode = mode  # Will be handled by process_multiple_outputs
        else:
            current_mode = mode

        if current_mode == ProcessingMode.NORMAL:
            # Standard processing for single output
            loss = criterion(outputs, labels)
            _, preds = torch.max(outputs, 1)
        else:
            # Multiple outputs processing
            if not isinstance(outputs, list):
                raise ValueError(
                    f"Expected list of outputs for mode '{mode_str}', but got single tensor"
                )

            # Process multiple outputs based on mode
            final_outputs, preds = process_multiple_outputs(outputs, current_mode)
            loss = criterion(final_outputs, labels)

        running_loss += loss.item()
        running_corrects += torch.sum(preds == labels).item()
        total_samples += labels.size(0)
        num_batches += 1

        loop.set_postfix(
            loss=running_loss / num_batches, accuracy=running_corrects / total_samples
        )

        # Clear GPU memory cache after each batch
        torch.cuda.empty_cache()

    epoch_loss = running_loss / num_batches
    epoch_acc = running_corrects / total_samples
    print(
        f"Val Epoch [{epoch + 1}/{num_epochs}] ({mode_str}) Loss: {epoch_loss:.4f}, Acc: {epoch_acc:.4f}"
    )
    return epoch_loss, epoch_acc


@torch.no_grad()
def test_model_with_mode(
    model, test_loader, criterion, device, mode=ProcessingMode.NORMAL
):
    """
    Enhanced test function that supports different processing modes.

    Args:
        mode: ProcessingMode enum or string - "normal", "average", "majority", "leastrisk"
    """
    # Convert to string for display purposes
    mode_str = mode.value if isinstance(mode, ProcessingMode) else mode
    model.eval()
    running_loss = 0.0
    running_corrects = 0
    num_batches = 0
    total_samples = 0

    print(f"Testing model with mode: {mode_str}...")
    loop = tqdm(test_loader, desc=f"Testing ({mode_str})", leave=True)

    for images, labels in loop:
        images, labels = images.to(device), labels.to(device)

        outputs = model(images)

        # Convert string to enum if needed
        if isinstance(mode, str):
            if mode.lower() == "normal":
                current_mode = ProcessingMode.NORMAL
            else:
                current_mode = mode  # Will be handled by process_multiple_outputs
        else:
            current_mode = mode

        if current_mode == ProcessingMode.NORMAL:
            # Standard processing for single output
            loss = criterion(outputs, labels)
            _, preds = torch.max(outputs, 1)
        else:
            # Multiple outputs processing
            if not isinstance(outputs, list):
                raise ValueError(
                    f"Expected list of outputs for mode '{mode_str}', but got single tensor"
                )

            # Process multiple outputs based on mode
            final_outputs, preds = process_multiple_outputs(outputs, current_mode)
            loss = criterion(final_outputs, labels)

        running_loss += loss.item()
        running_corrects += torch.sum(preds == labels).item()
        total_samples += labels.size(0)
        num_batches += 1

        loop.set_postfix(
            loss=running_loss / num_batches, accuracy=running_corrects / total_samples
        )

        # Clear GPU memory cache after each batch
        torch.cuda.empty_cache()

    test_loss = running_loss / num_batches
    test_acc = running_corrects / total_samples
    print(f"Test Results ({mode}) - Loss: {test_loss:.4f}, Accuracy: {test_acc:.4f}")

    return test_loss, test_acc


def create_data_loader(
    data_dir,
    image_size=96,
    num_workers=4,
    train_batch_size=32,
    val_batch_size=32,
    test_batch_size=32,
):
    train_dir = f"{data_dir}/train"
    valid_dir = f"{data_dir}/valid"
    test_dir = f"{data_dir}/test"

    # this mean mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    # Transforms (example: resize, normalize, etc.)
    transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),  # resize images
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )

    # Datasets
    train_dataset = RobustImageFolder(train_dir, transform=transform)
    valid_dataset = RobustImageFolder(valid_dir, transform=transform)
    test_dataset = RobustImageFolder(test_dir, transform=transform)

    # DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=train_batch_size,
        shuffle=True,
        num_workers=num_workers,
    )
    valid_loader = DataLoader(
        valid_dataset, batch_size=val_batch_size, shuffle=False, num_workers=num_workers
    )
    test_loader = DataLoader(
        test_dataset, batch_size=test_batch_size, shuffle=False, num_workers=num_workers
    )
    return train_loader, valid_loader, test_loader


def train_one_epoch(
    model, train_loader, criterion, optimizer, device, epoch, num_epochs
):
    model.train()
    loop = tqdm(train_loader, desc=f"Epoch [{epoch + 1}/{num_epochs}]", leave=False)

    running_loss = 0.0
    running_corrects = 0
    num_batches = 0
    total_samples = 0

    for images, labels in loop:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # Update running loss
        running_loss += loss.item()

        # Compute batch accuracy
        _, preds = torch.max(outputs, 1)
        running_corrects += torch.sum(preds == labels).item()
        total_samples += labels.size(0)

        num_batches += 1

        # Update tqdm bar with running loss and accuracy
        loop.set_postfix(
            loss=running_loss / num_batches, accuracy=running_corrects / total_samples
        )

        # Clear GPU memory cache after each batch
        torch.cuda.empty_cache()

    epoch_loss = running_loss / num_batches
    epoch_acc = running_corrects / total_samples
    print(
        f"Train Epoch [{epoch + 1}/{num_epochs}] Loss: {epoch_loss:.4f}, Acc: {epoch_acc:.4f}"
    )
    return epoch_loss, epoch_acc


@torch.no_grad()
def validate_one_epoch(model, val_loader, criterion, device, epoch, num_epochs):
    model.eval()
    running_loss = 0.0
    running_corrects = 0
    num_batches = 0
    total_samples = 0

    loop = tqdm(val_loader, desc=f"Val [{epoch + 1}/{num_epochs}]", leave=False)
    for images, labels in loop:
        images, labels = images.to(device), labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        running_loss += loss.item()

        _, preds = torch.max(outputs, 1)
        running_corrects += torch.sum(preds == labels).item()
        total_samples += labels.size(0)
        num_batches += 1

        loop.set_postfix(
            loss=running_loss / num_batches, accuracy=running_corrects / total_samples
        )

        # Clear GPU memory cache after each batch
        torch.cuda.empty_cache()

    epoch_loss = running_loss / num_batches
    epoch_acc = running_corrects / total_samples
    print(
        f"Val Epoch [{epoch + 1}/{num_epochs}] Loss: {epoch_loss:.4f}, Acc: {epoch_acc:.4f}"
    )
    return epoch_loss, epoch_acc


@torch.no_grad()
def test_model(model, test_loader, criterion, device):
    """
    Test function to evaluate model performance on test dataset.

    Args:
        model: The model to test
        test_loader: DataLoader for test data
        criterion: Loss function
        device: Device to run on (cuda/cpu)

    Returns:
        test_loss: Average test loss
        test_acc: Test accuracy
    """
    model.eval()
    running_loss = 0.0
    running_corrects = 0
    num_batches = 0
    total_samples = 0

    print("Testing model...")
    loop = tqdm(test_loader, desc="Testing", leave=True)

    for images, labels in loop:
        images, labels = images.to(device), labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        running_loss += loss.item()

        _, preds = torch.max(outputs, 1)
        running_corrects += torch.sum(preds == labels).item()
        total_samples += labels.size(0)
        num_batches += 1

        loop.set_postfix(
            loss=running_loss / num_batches, accuracy=running_corrects / total_samples
        )

        # Clear GPU memory cache after each batch
        torch.cuda.empty_cache()

    test_loss = running_loss / num_batches
    test_acc = running_corrects / total_samples
    print(f"Test Results - Loss: {test_loss:.4f}, Accuracy: {test_acc:.4f}")

    return test_loss, test_acc


class Trainer:
    """
    Advanced trainer class that handles epoch management, model saving, and training loop.
    """

    def __init__(
        self,
        model,
        train_loader,
        val_loader,
        test_loader,
        criterion,
        optimizer,
        device,
        checkpoint_dir="checkpoints",
        save_every_n_epochs=10,
        name="Trainer",
        mode="normal",  # New parameter for processing mode
        description="Image data loader",
    ):
        """
        Initialize the trainer.

        Args:
            model: The model to train
            train_loader: DataLoader for training data
            val_loader: DataLoader for validation data
            criterion: Loss function
            optimizer: Optimizer
            device: Device to run on (cuda/cpu)
            checkpoint_dir: Directory to save checkpoints
            save_every_n_epochs: Save current model every n epochs
            mode: Processing mode - "normal", "average", "majority", "leastrisk"
        """
        self.name = name
        self.mode = mode  # Store the processing mode
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.checkpoint_dir = checkpoint_dir
        self.save_every_n_epochs = save_every_n_epochs
        self.description = description

        # Create checkpoint directory
        os.makedirs(checkpoint_dir, exist_ok=True)

        # Initialize checkpoint IO
        self.checkpoint_io = CheckpointIO(
            os.path.join(checkpoint_dir, "{:06d}_nets.ckpt"),
            model=model,
            optimizer=optimizer,
        )

        # Track best model
        self.best_val_acc = 0.0
        self.best_val_loss = float("inf")
        self.best_epoch = 0

        # Training history
        self.train_history = {
            "train_loss": [],
            "train_acc": [],
            "val_loss": [],
            "val_acc": [],
        }

    def save_best_model(self, val_loss, val_acc, epoch):
        """Save model if it's the best so far."""
        is_best = val_acc > self.best_val_acc

        if is_best:
            self.best_val_acc = val_acc
            self.best_val_loss = val_loss
            self.best_epoch = epoch

            # Save best model
            best_path = os.path.join(self.checkpoint_dir, "best_model.ckpt")
            print(f"New best model! Saving to {best_path}")
            print(
                f"Best Val Acc: {self.best_val_acc:.4f}, Best Val Loss: {self.best_val_loss:.4f}"
            )

            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": self.model.state_dict(),
                    "optimizer_state_dict": self.optimizer.state_dict(),
                    "best_val_acc": self.best_val_acc,
                    "best_val_loss": self.best_val_loss,
                    "train_history": self.train_history,
                },
                best_path,
            )

    def save_current_model(self, epoch):
        """Save current model state."""
        self.checkpoint_io.save(epoch)
        print(f"Saved current model at epoch {epoch}")

    def load_checkpoint(self, epoch):
        """Load model from checkpoint."""
        try:
            self.checkpoint_io.load(epoch)
            print(f"Loaded checkpoint from epoch {epoch}")
            return True
        except Exception as e:
            print(f"Failed to load checkpoint from epoch {epoch}: {e}")
            return False

    def load_best_model(self):
        """Load the best model."""
        best_path = os.path.join(self.checkpoint_dir, "best_model.ckpt")

        if os.path.exists(best_path):
            print(f"Loading best model from {best_path}")
            checkpoint = torch.load(best_path, map_location=self.device)

            self.model.load_state_dict(checkpoint["model_state_dict"])
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            self.best_val_acc = checkpoint["best_val_acc"]
            self.best_val_loss = checkpoint["best_val_loss"]
            self.best_epoch = checkpoint["epoch"]

            if "train_history" in checkpoint:
                self.train_history = checkpoint["train_history"]

            print(f"Loaded best model from epoch {self.best_epoch}")
            print(
                f"Best Val Acc: {self.best_val_acc:.4f}, Best Val Loss: {self.best_val_loss:.4f}"
            )
            return True
        else:
            print(f"Best model not found at {best_path}")
            return False

    def train_epoch(self, epoch, num_epochs):
        """Train for one epoch using the configured mode."""
        print(self.description)
        return train_one_epoch_with_mode(
            self.model,
            self.train_loader,
            self.criterion,
            self.optimizer,
            self.device,
            epoch,
            num_epochs,
            self.mode,
        )

    def validate_epoch(self, epoch, num_epochs):
        """Validate for one epoch using the configured mode."""
        return validate_one_epoch_with_mode(
            self.model,
            self.val_loader,
            self.criterion,
            self.device,
            epoch,
            num_epochs,
            self.mode,
        )

    def train(self, num_epochs, resume_epoch=0):
        """
        Main training loop.

        Args:
            num_epochs: Total number of epochs to train
            resume_epoch: Epoch to resume from (0 for fresh start)

        Returns:
            train_history: Dictionary containing training history
        """
        print(f"Starting training for {num_epochs} epochs")
        print(f"Saving current model every {self.save_every_n_epochs} epochs")
        print(f"Checkpoints will be saved to: {self.checkpoint_dir}")

        # Resume from checkpoint if specified
        if resume_epoch > 0:
            if not self.load_checkpoint(resume_epoch):
                print(
                    f"Could not load checkpoint for epoch {resume_epoch}, starting fresh"
                )
                resume_epoch = 0

        start_epoch = resume_epoch

        for epoch in range(start_epoch, num_epochs):
            print(f"\n{'=' * 50}")
            print(f"Epoch {epoch + 1}/{num_epochs}")
            print(f"{'=' * 50}")

            # Training phase
            train_loss, train_acc = self.train_epoch(epoch, num_epochs)

            # Validation phase
            val_loss, val_acc = self.validate_epoch(epoch, num_epochs)
            test_loss, test_acc = self.test()

            # Update history
            self.train_history["train_loss"].append(train_loss)
            self.train_history["train_acc"].append(train_acc)
            self.train_history["val_loss"].append(val_loss)
            self.train_history["val_acc"].append(val_acc)

            # Save best model
            self.save_best_model(val_loss, val_acc, epoch + 1)

            # Save current model every n epochs
            if (epoch + 1) % self.save_every_n_epochs == 0:
                self.save_current_model(epoch + 1)

            # Print epoch summary
            print(f"\nname = {self.name},Epoch {epoch + 1} Summary:")
            print(f"Train - Loss: {train_loss:.4f}, Acc: {train_acc:.4f}")
            print(f"Val   - Loss: {val_loss:.4f}, Acc: {val_acc:.4f}")
            print(f"cheat test - Loss: {test_loss:.4f}, Acc: {test_acc:.4f}")
            print(
                f"Best Val Acc so far: {self.best_val_acc:.4f} (Epoch {self.best_epoch})"
            )

        # Save final model
        self.save_current_model(num_epochs)

        print(f"\n{'=' * 50}")
        print("Training completed!")
        print("Best model achieved:")
        print(f"  - Epoch: {self.best_epoch}")
        print(f"  - Val Accuracy: {self.best_val_acc:.4f}")
        print(f"  - Val Loss: {self.best_val_loss:.4f}")
        print(f"{'=' * 50}")

        return self.train_history

    def test(self):
        """
        Test the current model using the configured mode.

        Returns:
            test_loss: Test loss
            test_acc: Test accuracy
        """
        return test_model_with_mode(
            self.model, self.test_loader, self.criterion, self.device, self.mode
        )

    def test_best_model(self):
        """
        Test the best saved model.

        Args:
            test_loader: DataLoader for test data

        Returns:
            test_loss: Test loss
            test_acc: Test accuracy
        """
        # Save current model state
        current_state = {
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
        }

        # Load best model
        if self.load_best_model():
            # Test best model
            test_loss, test_acc = self.test()

            # Restore current model state
            self.model.load_state_dict(current_state["model"])
            self.optimizer.load_state_dict(current_state["optimizer"])

            return test_loss, test_acc
        else:
            print("No best model found, testing current model instead")
            return self.test()

    def get_training_summary(self):
        """Get a summary of the training process."""
        if not self.train_history["train_loss"]:
            print("No training history available")
            return

        print(f"\n{'=' * 50}")
        print("Training Summary")
        print(f"{'=' * 50}")
        print(f"Total epochs trained: {len(self.train_history['train_loss'])}")
        print(f"Final train loss: {self.train_history['train_loss'][-1]:.4f}")
        print(f"Final train accuracy: {self.train_history['train_acc'][-1]:.4f}")
        print(f"Final val loss: {self.train_history['val_loss'][-1]:.4f}")
        print(f"Final val accuracy: {self.train_history['val_acc'][-1]:.4f}")
        print(f"Best val accuracy: {self.best_val_acc:.4f} (Epoch {self.best_epoch})")
        print(f"Best val loss: {self.best_val_loss:.4f}")
        print(f"{'=' * 50}")


# Example usage function
def train_model_example(
    model,
    train_loader,
    val_loader,
    test_loader,
    criterion,
    optimizer,
    device,
    num_epochs=100,
    checkpoint_dir="checkpoints",
    save_every_n_epochs=10,
    mode="normal",  # New parameter for processing mode
):
    """
    Example function showing how to use the Trainer class.

    Args:
        model: The model to train
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
        test_loader: DataLoader for test data
        criterion: Loss function
        optimizer: Optimizer
        device: Device to run on
        num_epochs: Number of epochs to train
        checkpoint_dir: Directory to save checkpoints
        save_every_n_epochs: Save model every n epochs
        mode: Processing mode - "normal", "average", "majority", "leastrisk"
    """
    # Create trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        checkpoint_dir=checkpoint_dir,
        save_every_n_epochs=save_every_n_epochs,
        mode=mode,  # Pass the mode parameter
    )

    # Train the model
    print("Starting training...")
    history = trainer.train(num_epochs=num_epochs)

    # Get training summary
    trainer.get_training_summary()

    # Test current model
    print("\nTesting current model...")
    current_test_loss, current_test_acc = trainer.test()

    # Test best model
    print("\nTesting best model...")
    best_test_loss, best_test_acc = trainer.test_best_model()

    print("\nFinal Results:")
    print(
        f"Current model - Test Loss: {current_test_loss:.4f}, Test Acc: {current_test_acc:.4f}"
    )
    print(
        f"Best model    - Test Loss: {best_test_loss:.4f}, Test Acc: {best_test_acc:.4f}"
    )

    return trainer, history


# Usage Examples:
#
# For normal single output models:
# trainer = Trainer(model, train_loader, val_loader, test_loader, criterion, optimizer, device, mode="normal")
#
# For MultiplePipeline with ensemble averaging:
# pipeline = MultiplePipeline(..., ensemble_mode=True)  # Returns single averaged output
# trainer = Trainer(pipeline, train_loader, val_loader, test_loader, criterion, optimizer, device, mode="normal")
#
# For MultiplePipeline with multiple outputs and average processing:
# pipeline = MultiplePipeline(..., ensemble_mode=False)  # Returns list of outputs
# trainer = Trainer(pipeline, train_loader, val_loader, test_loader, criterion, optimizer, device, mode="average")
#
# For MultiplePipeline with multiple outputs and majority voting:
# pipeline = MultiplePipeline(..., ensemble_mode=False)  # Returns list of outputs
# trainer = Trainer(pipeline, train_loader, val_loader, test_loader, criterion, optimizer, device, mode="majority")
#
# For MultiplePipeline with multiple outputs and least risk (highest confidence) selection:
# pipeline = MultiplePipeline(..., ensemble_mode=False)  # Returns list of outputs
# trainer = Trainer(pipeline, train_loader, val_loader, test_loader, criterion, optimizer, device, mode="leastrisk")
