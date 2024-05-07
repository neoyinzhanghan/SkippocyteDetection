import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from torchvision import transforms, models, datasets
import albumentations as A
from albumentations.pytorch import ToTensorV2
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from torchmetrics import Accuracy, AUROC, F1Score


def get_feat_extract_augmentation_pipeline(image_size):
    transform_shape = A.Compose(
        [
            A.ShiftScaleRotate(p=0.8),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.Affine(shear=(-10, 10), p=0.3),
            A.ISONoise(
                color_shift=(0.01, 0.02),
                intensity=(0.05, 0.01),
                always_apply=False,
                p=0.2,
            ),
        ]
    )
    transform_color = A.Compose(
        [
            A.RandomBrightnessContrast(
                contrast_limit=0.4, brightness_by_max=0.4, p=0.5
            ),
            A.CLAHE(p=0.3),
            A.ColorJitter(p=0.2),
            A.RandomGamma(p=0.2),
        ]
    )
    return A.Compose(
        [
            A.Resize(image_size, image_size),
            A.OneOf([transform_shape, transform_color]),
            ToTensorV2(),
        ]
    )


class BinaryResNet(LightningModule):
    def __init__(self):
        super().__init__()
        self.model = models.resnet50(pretrained=True)
        num_features = self.model.fc.in_features
        self.model.fc = torch.nn.Linear(num_features, 1)
        self.accuracy = Accuracy(task="binary")
        self.auroc = AUROC(num_classes=1, task="binary")
        self.f1 = F1Score(num_classes=1, threshold=0.5, task="binary")
        self.transform = get_feat_extract_augmentation_pipeline(96)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.binary_cross_entropy_with_logits(y_hat.view(-1), y.float())
        self.log("train_loss", loss)
        self.log("train_acc", self.accuracy(torch.sigmoid(y_hat), y.int()))
        self.log("train_auc", self.auroc(torch.sigmoid(y_hat), y.int()))
        self.log("train_f1", self.f1(torch.sigmoid(y_hat), y.int()))
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.binary_cross_entropy_with_logits(y_hat.view(-1), y.float())
        self.log("val_loss", loss)
        self.log("val_acc", self.accuracy(torch.sigmoid(y_hat), y.int()))
        self.log("val_auc", self.auroc(torch.sigmoid(y_hat), y.int()))
        self.log("val_f1", self.f1(torch.sigmoid(y_hat), y.int()))
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)

    def prepare_data(self):
        # Make sure to replace 'path_to_your_data' with the actual path to your dataset
        dataset = datasets.ImageFolder(
            root="/media/hdd3/neo/skippocyte_data", transform=self.transform
        )
        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size

        self.train_dataset, self.val_dataset = random_split(
            dataset, [train_size, val_size]
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=32,
            shuffle=True,
            num_workers=4,
            collate_fn=self.collate_fn,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset, batch_size=32, num_workers=4, collate_fn=self.collate_fn
        )

    def collate_fn(self, batch):
        imgs, targets = zip(*batch)
        imgs = torch.stack([self.transform(image=img)["image"] for img in imgs])
        targets = torch.tensor(targets)
        return imgs, targets


if __name__ == "__main__":
    model = BinaryResNet()
    model.prepare_data()
    logger = TensorBoardLogger("tb_logs", name="binary_resnet50")
    checkpoint_callback = ModelCheckpoint(monitor="val_loss", mode="min")
    trainer = Trainer(max_epochs=20, logger=logger, callbacks=[checkpoint_callback])
    trainer.fit(model)
