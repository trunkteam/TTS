import torch
import torchaudio


class MelSpecEncoder(object):
    def __init__(self,
                 sample_rate=16000,
                 fft_size=1024,
                 win_length=1024,
                 hop_length=256,
                 num_mels=128,
                 ) -> None:
        super().__init__()
        self.sample_rate = sample_rate
        self.mel_spec = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=fft_size,
            win_length=win_length,
            hop_length=hop_length,
            n_mels=num_mels
        )

    def encode(self, aud: torch.Tensor) -> torch.Tensor:
        return self.mel_spec(aud)

    def encode_af(self, src_af: str) -> torch.Tensor:
        aud, sr = torchaudio.load(src_af)
        if aud.shape[0] > 1:
            aud = torch.mean(aud, dim=0).unsqueeze(0)
        if sr != self.sample_rate:
            aud = torchaudio.functional.resample(aud, sr, self.sample_rate)

        return self.encode(aud)
