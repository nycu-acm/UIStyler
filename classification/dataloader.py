import os
from PIL import Image
from torch.utils.data import Dataset


class ImageList(Dataset):
    """Dataset for loading images and corresponding labels from a directory.

    The root directory should contain subdirectories with names
    representing labels. Only images with extensions .jpg, .jpeg,
    .png, .bmp will be processed.
    """

    def __init__(self, root_dir, transform_w=None, transform_str=None):
        """
        Initialize the ImageList dataset.

        Args:
            root_dir (str): Root directory with subdirectories for each label.
            transform_w (callable, optional): Weak transformation function.
            transform_str (callable, optional): Strong transformation function.
        """
        if "results" not in root_dir:
            self.root_dir = os.path.join(root_dir, "imgs")
        else:
            self.root_dir = root_dir
        
        self.transform_w = transform_w
        self.transform_str = transform_str

        self.image_files = [f for f in os.listdir(self.root_dir)
                            if f.lower().endswith(('.png', '.jpg', '.jpeg'))] #  and "normal" not in f.lower()

        # Find unique class names from the filenames (splitting by underscore)
        self.classes = sorted(set([f.split("_")[0] for f in self.image_files])) # ["benign", "malignant"]
        # Create a mapping from class name to integer id
        self.class_to_id = {cls: idx for idx, cls in enumerate(self.classes)}
        self.num_classes = len(self.classes)

    def __len__(self):
        """Return the total number of images."""
        return len(self.image_files)

    def __getitem__(self, idx):
        output = {}

        file_name = self.image_files[idx]
        img_path = os.path.join(self.root_dir, file_name)
        image = Image.open(img_path).convert("RGB")

        # Apply weak transformation if available.
        if self.transform_w:
            image_w = self.transform_w(image)
            output["img_w"] = image_w
        # Apply strong transformation if available.
        if self.transform_str:
            image_str = self.transform_str(image)
            output["img_str"] = image_str
        
        output["target"] = self.class_to_id[file_name.split("_")[0]]        
        output["path"] = img_path

        return output
