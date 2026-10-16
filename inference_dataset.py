"""Minimal VITON inputs for parser-free inference (no pose/agnostic files)."""

from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class VITONInferenceDataset(Dataset):
    def __init__(self, opt):
        self.root = Path(opt.dataset_imgpath)
        self.mode = opt.mode
        # Match the existing dataset's aspect-preserving PIL resize exactly.
        self.transform = transforms.Compose([
            transforms.Resize(opt.load_width, interpolation=2),
            transforms.ToTensor(),
            transforms.Normalize((0.5,) * 3, (0.5,) * 3),
        ])
        self.pairs = []
        with open(opt.dataset_list, encoding="utf-8") as stream:
            for number, line in enumerate(stream, 1):
                if not line.strip():
                    continue
                fields = line.split()
                if len(fields) != 2:
                    raise ValueError("{}:{}: expected person and clothing filenames".format(
                        opt.dataset_list, number))
                if any(Path(name).name != name or name in (".", "..")
                       or "\\" in name for name in fields):
                    raise ValueError("{}:{}: filenames must not contain directories".format(
                        opt.dataset_list, number))
                self.pairs.append(tuple(fields))
        if not self.pairs:
            raise ValueError("No image pairs found in {}".format(opt.dataset_list))

    def _read(self, path):
        with Image.open(path) as image:
            return self.transform(image.convert("RGB"))

    def __getitem__(self, index):
        person, cloth = self.pairs[index]
        cloth = cloth.replace("_0.jpg", "_1.jpg")
        return {
            "img_name": person,
            "img": self._read(self.root / (self.mode + "_img") / person),
            "cloth": self._read(self.root / "clothes" / cloth),
        }

    def __len__(self):
        return len(self.pairs)
