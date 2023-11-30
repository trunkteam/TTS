import torch
import torchaudio
from denoiser import pretrained


class Denoiser(object):
    def __init__(self) -> None:
        super().__init__()
        self.den_model = pretrained.master64()
        self.den_model = self.den_model.cuda() if torch.cuda.is_available() else self.den_model

    def denoise(self, aud: torch.Tensor, sr: int, rep: int = 5) -> torch.Tensor:
        with torch.no_grad():
            aud_16k = torchaudio.functional.resample(aud, sr, 16000)
            aud_16k = aud_16k.cuda()
            for i in range(rep):
                aud_16k = self.den_model(aud_16k[None])[0]

            return torchaudio.functional.resample(aud_16k, 16000, sr).detach().cpu()

    def denoise_af(self, src_af):
        aud, sr = torchaudio.load(src_af)
        if aud.shape[0] > 1:
            aud = torch.mean(aud, dim=0).unsqueeze(0)

        return self.denoise(aud, sr)
