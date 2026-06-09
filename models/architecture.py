"""
Orbital Logis — GAIE: Arquiteturas CNN para Classificação de Imagens Satelitais
Global Solution 2026 | FIAP | Applied Computer Vision

Modelos treinados do zero (sem transfer learning) sobre o dataset EuroSAT (10 classes, 64x64 RGB).
"""

import torch
import torch.nn as nn


class OrbitalVision_Baseline(nn.Module):
    """CNN Baseline para classificação de imagens satelitais EuroSAT.

    Arquitetura simples com 3 blocos convolucionais Conv→BN→ReLU→MaxPool
    seguidos de camadas totalmente conectadas com Dropout.

    Parâmetros: ~2.2M
    Scheduler:  StepLR (step=10, gamma=0.5)
    Épocas:     30
    """

    def __init__(self, num_classes: int = 10):
        super(OrbitalVision_Baseline, self).__init__()

        self.features = nn.Sequential(
            # Bloco 1: 3 → 32 canais | 64x64 → 32x32
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Bloco 2: 32 → 64 canais | 32x32 → 16x16
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Bloco 3: 64 → 128 canais | 16x16 → 8x8
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.classifier = nn.Sequential(
            nn.Dropout(p=0.4),
            nn.Linear(128 * 8 * 8, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),
            nn.Linear(256, num_classes),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = x.view(x.size(0), -1)   # Flatten: [B, 128, 8, 8] → [B, 8192]
        x = self.classifier(x)
        return x


class DoubleConvBlock(nn.Module):
    """Bloco de dupla convolução: [Conv→BN→ReLU] × 2 → MaxPool → Dropout2d."""

    def __init__(self, in_channels: int, out_channels: int, dropout: float = 0.0):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.dropout = nn.Dropout2d(p=dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.block(x))


class OrbitalVision_Deep(nn.Module):
    """CNN Profunda para classificação de imagens satelitais EuroSAT.

    4 blocos de dupla convolução com GlobalAvgPool e classificador
    com BatchNorm e múltiplos Dropouts para regularização robusta.

    Parâmetros: ~3.5M
    Scheduler:  CosineAnnealingLR (T_max=40, eta_min=1e-6)
    Épocas:     40
    """

    def __init__(self, num_classes: int = 10):
        super(OrbitalVision_Deep, self).__init__()

        self.block1 = DoubleConvBlock(3,   32,  dropout=0.10)  # 64→32
        self.block2 = DoubleConvBlock(32,  64,  dropout=0.20)  # 32→16
        self.block3 = DoubleConvBlock(64,  128, dropout=0.25)  # 16→8
        self.block4 = DoubleConvBlock(128, 512, dropout=0.25)  # 8→4

        self.global_avg_pool = nn.AdaptiveAvgPool2d(output_size=(1, 1))

        self.classifier = nn.Sequential(
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),
            nn.Linear(256, num_classes),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.block1(x)           # (B,  32, 32, 32)
        x = self.block2(x)           # (B,  64, 16, 16)
        x = self.block3(x)           # (B, 128,  8,  8)
        x = self.block4(x)           # (B, 512,  4,  4)
        x = self.global_avg_pool(x)  # (B, 512,  1,  1)
        x = x.view(x.size(0), -1)   # (B, 512)
        x = self.classifier(x)       # (B, num_classes)
        return x


def load_model(model_name: str, weights_path: str, num_classes: int = 10, device: str = "cpu"):
    """Carrega um modelo com os pesos salvos.

    Args:
        model_name:   'baseline' ou 'deep'
        weights_path: caminho para o arquivo .pth
        num_classes:  número de classes (padrão 10 para EuroSAT)
        device:       'cpu' ou 'cuda'

    Returns:
        Modelo em modo eval() pronto para inferência.
    """
    if model_name == "baseline":
        model = OrbitalVision_Baseline(num_classes=num_classes)
    elif model_name == "deep":
        model = OrbitalVision_Deep(num_classes=num_classes)
    else:
        raise ValueError(f"model_name deve ser 'baseline' ou 'deep', recebeu '{model_name}'")

    state = torch.load(weights_path, map_location=device)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model
