from pathlib import Path
from torch.utils.data import Dataset, DataLoader
import torchvision
from sklearn.preprocessing import LabelEncoder


class CustomDataset(Dataset):
    """
    Dataset to images, expects images to be organized into class subdirectories
    """

    def __init__(self, _file_paths, labels):
        self.file_paths = _file_paths
        self.labels = labels

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        file_path = self.file_paths[idx]
        data = torchvision.io.decode_image(file_path).float()  # Load data from file
        data /= 255.0
        data = 2 * data - 1

        if self.labels is not None:
            return (data, self.labels[idx] + 1)

        return (data, 0)


def read_dataset(train_directory, conditional, batch_size, workers):
    data_root = train_directory

    file_paths = [str(file) for file in Path(data_root).rglob("*") if file.is_file()]

    labels = None
    label_cnt = 0

    if conditional:
        label_encoder = LabelEncoder()
        label_encoder.fit([p.name for p in Path(data_root).iterdir() if p.is_dir()])
        labels = list(map(lambda x: x.split("/")[-2], file_paths))
        labels = label_encoder.transform(labels)
        label_cnt = len(label_encoder.classes_)

    dataset = CustomDataset(file_paths, labels)

    return (
        DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=workers,
        ),
        label_cnt,
    )
