"""The Siamese ResNet used for biometric pretraining."""

import torch
from biomecg.models.resnet import SiameseResNet1d
from biomecg.paths import ROOT


class ClassificationNetwork:
    def __init__(self, config):
        if config["network"] != "SiameseResNet1d":
            raise ValueError("Only SiameseResNet1d is supported")
        a = config["architecture"]
        self.network = SiameseResNet1d(
            embedding_dim=a["embedding_dim"],
            input_channels=len(a["leads"]),
            fix_feature_dim=a["fix_feature_dim"],
            kernel_size=a["kernel_size"],
            dilation_rate=a["dilation_rate"],
            dropout=config["training"]["dropout_p"],
        )
        if config["phase"] == "test":
            path = ROOT / "checkpoints/biometric" / config["name"] / "network.pt"
            self.network.load_state_dict(
                torch.load(path, map_location="cpu", weights_only=True), strict=True
            )
        self.network.to(config["training"]["device"])
