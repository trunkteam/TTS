import json
import logging
import os

import torch
import torchaudio
# from denoiser.audio import convert_audio
import torchaudio.functional as taf
from denoiser import pretrained

logging.basicConfig(level=logging.INFO)
_log = logging.getLogger("RCH")

den_model = pretrained.master64()
den_model = den_model.cuda() if torch.cuda.is_available() else den_model

src_man = "data/tts/v3/hi/manifest_spk_cluster.json"
tgt_man = "data/tts/v3/hi/manifest_spk_cluster_cln.json"

cln_dir = "data/tts/v3/hi/cln"

os.makedirs(cln_dir, exist_ok=True)

with open(src_man, encoding="ytf-8") as sm:
    with open(tgt_man, encoding="ytf-8", mode="w") as tm:
        for line in sm:
            jd = json.loads(line.strip("\n").strip())
            src_af = jd["audio_filepath"]
            aud, sr = torchaudio.load(src_af)
            if aud.shape[0] > 1:
                aud = torch.mean(aud, dim=0).unsqueeze(0)

            aud = taf.resample(aud, sr, 16000).cuda()

            with torch.no_grad():
                # den_wav = convert_audio(aud.cuda(), sr, sample_rate, 1)
                _log.info(f"aud_bf: {aud.shape}")
                aud = den_model(aud.squeeze(0))[0]
                _log.info(f"aud_af: {aud.shape}")
