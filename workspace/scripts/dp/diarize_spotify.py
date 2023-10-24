import json
import logging

import torch
import torchaudio
from pyannote.audio import Pipeline

logging.basicConfig(level=logging.INFO)
_log = logging.getLogger("DS")

api_token = "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"
pipeline = Pipeline.from_pretrained(
    "pyannote/speaker-diarization-3.0",
    use_auth_token=api_token).to(torch.device("cuda"))

for lang in ["en", "ar"]:

    src_manifest = f"/data/asr/workspace/audio/tts/data/tts/{lang}/spotify/manifest/manifest_srt.json"
    tgt_manifest = f"/data/asr/workspace/audio/tts/data/tts/{lang}/spotify/manifest/manifest_diar.json"

    with open(src_manifest, encoding="utf-8") as sm:
        with open(tgt_manifest, encoding="utf-8", mode="w") as tm:
            for line in sm:
                jd = json.loads(line.strip("\n").strip())
                src_af = jd['audio_filepath']
                aud, sr = torchaudio.load(src_af)
                diarization = pipeline({"waveform": aud, "sample_rate": sr}, min_speakers=1, max_speakers=10)
                _log.info(diarization)
