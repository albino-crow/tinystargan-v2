import os
from PIL import Image
from torchvision import datasets


class RobustImageFolder(datasets.ImageFolder):
    """
    A robust version of ImageFolder that skips corrupted images
    instead of crashing the training process.
    """

    def __init__(self, *args, **kwargs):
        self.skip_corrupted = kwargs.pop("skip_corrupted", True)
        self.corrupted_files = []
        super().__init__(*args, **kwargs)

        if self.skip_corrupted:
            self._validate_and_filter_samples()

    def _validate_and_filter_samples(self):
        """Pre-validate all samples and remove corrupted ones"""
        valid_samples = []
        corrupted_count = 0

        for i, (path, class_index) in enumerate(self.samples):
            try:
                # Try to open and convert the image
                with Image.open(path) as img:
                    img.convert("RGB")
                valid_samples.append((path, class_index))
            except Exception as e:
                corrupted_count += 1
                self.corrupted_files.append(path)
                print(f"Skipping corrupted image: {path} - Error: {str(e)}")

        self.samples = valid_samples
        self.imgs = self.samples  # For compatibility

        if corrupted_count > 0:
            print(f"Found and skipped {corrupted_count} corrupted images")
            print(f"Valid images remaining: {len(valid_samples)}")

    def __getitem__(self, index):
        """
        Override getitem to handle any remaining corrupted images
        that might have passed initial validation
        """
        max_retries = 5
        current_index = index

        for attempt in range(max_retries):
            try:
                path, target = self.samples[current_index]
                sample = self.loader(path)

                if self.transform is not None:
                    sample = self.transform(sample)
                if self.target_transform is not None:
                    target = self.target_transform(target)

                return sample, target

            except Exception as e:
                print(f"Error loading image at index {current_index}: {e}")
                # Try next image
                current_index = (current_index + 1) % len(self.samples)
                if current_index == index:
                    # We've tried all images, raise the error
                    raise RuntimeError(
                        f"Unable to load any valid image after {max_retries} attempts"
                    )

        raise RuntimeError(f"Failed to load image after {max_retries} retries")


def scan_for_corrupted_images(data_dir):
    """
    Scan a directory for corrupted images and return a list of corrupted files
    """
    corrupted_files = []
    total_files = 0

    print(f"Scanning directory: {data_dir}")

    for root, dirs, files in os.walk(data_dir):
        for file in files:
            if file.lower().endswith(
                (".png", ".jpg", ".jpeg", ".gif", ".bmp", ".tiff")
            ):
                total_files += 1
                file_path = os.path.join(root, file)
                try:
                    with Image.open(file_path) as img:
                        img.convert("RGB")
                except Exception as e:
                    corrupted_files.append((file_path, str(e)))
                    print(f"Corrupted: {file_path} - {str(e)}")

    print(
        f"Scan complete. Total images: {total_files}, Corrupted: {len(corrupted_files)}"
    )
    return corrupted_files


def remove_corrupted_images(corrupted_files, backup=True):
    """
    Remove corrupted images from the dataset
    """
    if backup:
        backup_dir = "corrupted_images_backup"
        os.makedirs(backup_dir, exist_ok=True)

    for file_path, error in corrupted_files:
        try:
            if backup:
                backup_path = os.path.join(backup_dir, os.path.basename(file_path))
                os.rename(file_path, backup_path)
                print(f"Moved to backup: {file_path} -> {backup_path}")
            else:
                os.remove(file_path)
                print(f"Removed: {file_path}")
        except Exception as e:
            print(f"Failed to remove {file_path}: {e}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Scan for corrupted images")
    parser.add_argument("--data_dir", required=True, help="Directory to scan")
    parser.add_argument("--remove", action="store_true", help="Remove corrupted images")
    parser.add_argument(
        "--no_backup", action="store_true", help="Don't backup when removing"
    )

    args = parser.parse_args()

    corrupted = scan_for_corrupted_images(args.data_dir)

    if corrupted:
        print(f"\nFound {len(corrupted)} corrupted images:")
        for file_path, error in corrupted:
            print(f"  {file_path}: {error}")

        if args.remove:
            remove_corrupted_images(corrupted, backup=not args.no_backup)
    else:
        print("No corrupted images found!")
