import logging

import torch
import torchaudio
from transformers import Wav2Vec2FeatureExtractor, WavLMForXVector

logging.basicConfig(level=logging.INFO)
_log = logging.getLogger("CH")


class SpeakerEncoder(object):
    def __init__(self) -> None:
        super().__init__()
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained('microsoft/wavlm-base-plus-sv')
        self.se_model = WavLMForXVector.from_pretrained('microsoft/wavlm-base-plus-sv').to(self.device)

    def encode(self, aud: torch.Tensor, sr: int) -> torch.Tensor:
        aud_16k = torchaudio.functional.resample(aud, sr, 16000)
        aud_feat = self.feature_extractor(aud_16k.squeeze(0),
                                          padding=True,
                                          return_tensors="pt",
                                          sampling_rate=16000).to(self.device)

        spk_emb = self.se_model(**aud_feat).embeddings
        spk_emb = torch.nn.functional.normalize(spk_emb, dim=-1).detach().cpu().squeeze(0)
        return spk_emb

    def encode_af(self, src_af: str) -> torch.Tensor:
        aud, sr = torchaudio.load(src_af)
        if aud.shape[0] > 1:
            aud = torch.mean(aud, dim=0).unsqueeze(0)
        return self.encode(aud=aud, sr=sr)
