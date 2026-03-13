from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

class ImageFolder(Dataset):

    def __init__(self, roots, im_exts, transform=None):

        self.samples = []
        for current_dir in roots:
            current_dir = Path(current_dir)
            if not current_dir.is_dir():
                raise RuntimeError(f'Invalid directory "{current_dir}"')
            for ext in im_exts:
                self.samples.extend(sorted(str(x) for x in current_dir.rglob(f'*.{ext}')))

        # splitdir = Path(roots)
        # self.samples = [f for f in splitdir.iterdir() if f.is_file()]

        self.transform = transform
        

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            img: `PIL.Image.Image` or transformed `PIL.Image.Image`.
        """
        img = Image.open(self.samples[index]).convert("RGB")
        if self.transform:
            return self.transform(img)
        return img

    def __len__(self):
        return len(self.samples)
