import torch.nn as nn

class VariationalAutoencoder(nn.Module):
    def __init__(self):
        super(VariationalAutoencoder, self).__init__()

        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, 3, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, 3, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),

        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 3, stride=2, padding=1, output_padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1, output_padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(16, 3, 3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()
        )
    def forward(self, x):
        xe = self.encoder(x)
        x = self.decoder(xe)
        return x,xe

class HighCompressionAutoencoder(nn.Module):
    def __init__(self):
        super(HighCompressionAutoencoder, self).__init__()

        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, 3, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.MaxPool2d(2, stride=2),  # 14x14
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1,output_padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=2),
            nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1,output_padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=2),
            nn.ConvTranspose2d(16, 3, 3, stride=2, padding=1,output_padding=1),
            nn.Sigmoid()
        )
    def forward(self, x):
        xe = self.encoder(x)
        x = self.decoder(xe)
        return x,xe



class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        # Encoder layers
        # Encoder
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),  # Output: 32 x 112 x 112
            nn.GELU(),  # GELU activation to avoid gradient decay
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),  # Output: 64 x 56 x 56
            nn.GELU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),  # Output: 128 x 28 x 28
            nn.GELU(),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),  # Output: 256 x 14 x 14
            nn.GELU(),
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),  # Output: 128 x 28 x 28
            nn.GELU(),
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),  # Output: 64 x 56 x 56
            nn.GELU(),
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),  # Output: 32 x 112 x 112
            nn.GELU(),
            nn.ConvTranspose2d(32, 3, kernel_size=3, stride=2, padding=1, output_padding=1),  # Output: 3 x 224 x 224
            nn.Sigmoid()  # Sigmoid for final layer to normalize the output
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


class CustomAutoEncoder(nn.Module):
    def __init__(self):
        super(CustomAutoEncoder, self).__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=5, stride=2, padding=2),  # Output: 64 x 112 x 112
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1),  # LeakyReLU to avoid gradient decay
            nn.Conv2d(64, 128, kernel_size=5, stride=2, padding=2), # Output: 128 x 56 x 56
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1),
            nn.Conv2d(128, 256, kernel_size=5, stride=2, padding=2), # Output: 256 x 28 x 28
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1),
            nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1), # Output: 256 x 14 x 14
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1)
        )

        # Internal representation layer
        self.int_repr = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=1), # 1x1 convolution for internal representation
            nn.BatchNorm2d(256),
            nn.Dropout(.2),
            nn.LeakyReLU(0.1)
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 256, kernel_size=3, stride=2, padding=1, output_padding=1), # Output: 256 x 28 x 28
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1),
            nn.ConvTranspose2d(256, 128, kernel_size=5, stride=2, padding=2, output_padding=1), # Output: 128 x 56 x 56
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1),
            nn.ConvTranspose2d(128, 64, kernel_size=5, stride=2, padding=2, output_padding=1),  # Output: 64 x 112 x 112
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1),
            nn.ConvTranspose2d(64, 3, kernel_size=5, stride=2, padding=2, output_padding=1),    # Output: 3 x 224 x 224
            nn.Sigmoid()  # Tanh for output normalization
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.int_repr(x)
        x = self.decoder(x)
        return x
