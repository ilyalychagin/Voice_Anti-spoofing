import torch
from torch import nn
from torch.nn import Sequential
import torchaudio

class LogMelspec(nn.Module):
    """
    Log Mel-Spectrogram feature extractor.
    """

    def __init__(self):
        """
        Initialize Log Mel-Spectrogram transform.
        """
        super().__init__()
        self.melspec = torchaudio.transforms.MelSpectrogram(
                sample_rate=16000,
                n_fft=400,
                win_length=400,
                hop_length=160,
                n_mels=80
        )


    def __call__(self, batch):
        """
        Convert audio waveforms to log mel-spectrogram features.

        Args:
            batch (Tensor): Input audio waveforms
        
        Returns:
            output (Tensor): Log mel-spectrogram features
        """

        x = torch.log(self.melspec(batch).clamp_(min=1e-9, max=1e9))
        
        return x

class MFM(nn.Module):
    """
    Max-Feature-Map activation layer
    """
    
    def __init__(self):
        super().__init__()

    def forward(self, x):
        """
        Forward pass of MFM layer.

        Args:
            x (Tensor): input tensor of shape (batch, channels, height, width)
        
        Returns:
            output (Tensor): output tensor of shape (batch, channels//2, height, width)
        """

        chnk1, chnk2 = torch.split(x, x.shape[1] // 2, dim=1)
        return torch.maximum(chnk1, chnk2)


class ConvBlock(nn.Module):
    """
    Convolutional block with MFM activation, optional batch normalization and pooling.
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: tuple[int, int],
        stride: tuple[int, int] = (1, 1),
        use_batchnorm: bool = True,
        use_pooling: bool = False,
        pool_kernel: tuple[int, int] = (2, 2),
        pool_stride: tuple[int, int] = (2, 2)
    ):
        """
        Initialize ConvBlock.

        Args:
            in_channels (int): number of input channels
            out_channels (int): number of output channels (before MFM)
            kernel_size (tuple): convolution kernel size (height, width)
            stride (tuple): convolution stride (height, width)
            use_batchnorm (bool): whether to use batch normalization
            use_pooling (bool): whether to add max pooling
            pool_kernel (tuple): pooling kernel size if use_pooling is True
            pool_stride (tuple): pooling stride if use_pooling is True
        """

        super().__init__()
        
        layers = [
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride
            ),
            MFM()
        ]
        
        if use_batchnorm:
            layers.append(nn.BatchNorm2d(out_channels // 2))
        
        if use_pooling:
            layers.append(
                nn.MaxPool2d(
                    kernel_size=pool_kernel,
                    stride=pool_stride
                )
            )
        
        self.block = nn.Sequential(*layers)
    
    def forward(self, x):
        """
        Forward pass of ConvBlock.

        Args:
            x (Tensor): input tensor
        
        Returns:
            output (Tensor): processed output tensor
        """
        return self.block(x)


class LCNNModel(nn.Module):
    """
    Light CNN for audio classification tasks.
    """
    
    def __init__(self):
        """
        Initialize LCNN model.
        """

        super().__init__()
        
        self.mel_spec = LogMelspec()

        self.feature_extractor = nn.Sequential(
            
            ConvBlock(
                in_channels=1,
                out_channels=96,
                kernel_size=(9, 9),
                use_pooling=True
            ),
            
            ConvBlock(
                in_channels=48,
                out_channels=192,
                kernel_size=(1, 1),
                use_batchnorm=True,
                use_pooling=True
            ),
            
            ConvBlock(
                in_channels=96,
                out_channels=256,
                kernel_size=(5, 5),
                use_batchnorm=True,
                use_pooling=True
            ),
            
            ConvBlock(
                in_channels=128,
                out_channels=384,
                kernel_size=(4, 4),
                use_batchnorm=True,
                use_pooling=True
            )
        )

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(192, 512),
            MFM(),
            nn.Dropout(0.5),
            nn.Linear(256, 2)
        )

    def forward(self, **batch):
        """
        Forward pass of LCNN model.

        Args:
            x (Tensor): input audio waveform tensor
        
        Returns:
            output (Dict[str, Tensor]): output dictionary containing logits
        """
        x = self.mel_spec(batch["data_object"])
        #x = x.unsqueeze(1)
        
        x = self.feature_extractor(x)

        logits = self.classifier(x)

        return {"logits": logits}
