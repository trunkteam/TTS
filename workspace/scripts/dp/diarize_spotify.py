import json
import logging
import os
import uuid

import pysubs2 as pysubs2
import torch
import torchaudio
import torchaudio.functional as taf
from denoiser import pretrained
from denoiser.dsp import convert_audio
from pyannote.audio import Pipeline

logging.basicConfig(level=logging.INFO)
_log = logging.getLogger("DS")

api_token = "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"
pipeline = Pipeline.from_pretrained(
    "pyannote/speaker-diarization-3.0",
    use_auth_token=api_token).to(torch.device("cuda"))

den_model = pretrained.master64()
den_model = den_model.cuda() if torch.cuda.is_available() else den_model

for lang in ["en", "ar"]:

    src_manifest = f"/data/asr/workspace/audio/tts/data/tts/{lang}/spotify/manifest/manifest_srt.json"
    tgt_manifest = f"/data/asr/workspace/audio/tts/data/tts/{lang}/spotify/manifest/manifest_diar.json"

    tgt_wav_dir = f"/data/asr/workspace/audio/tts/data/tts/{lang}/spotify/wav/44k"
    os.makedirs(tgt_wav_dir, exist_ok=True)

    with open(src_manifest, encoding="utf-8") as sm:
        with open(tgt_manifest, encoding="utf-8", mode="w") as tm:
            for line in sm:
                jd = json.loads(line.strip("\n").strip())
                src_af = jd['audio_filepath']
                src_srt = jd['srt_filepath']
                subs = pysubs2.load(src_srt, encoding="utf-8")

                for s_index, sub in enumerate(subs):

                    start = sub.start / 1000
                    end = sub.end / 1000

                    effects = [
                        ['trim', f'{start}', f'{end - start}'],
                    ]
                    aud, sr = torchaudio.sox_effects.apply_effects_file(src_af, effects)
                    if aud.shape[0] > 1:
                        aud = torch.mean(aud, dim=0).unsqueeze(0)

                    diarization = pipeline({"waveform": aud, "sample_rate": sr}, min_speakers=1, max_speakers=10)

                    spk_sets = set()
                    for turn, spk_id, speaker in diarization.itertracks(yield_label=True):
                        spk_sets.add(speaker)

                    _log.info(spk_sets)

                    if len(spk_sets) == 1:
                        with torch.no_grad():
                            aud = convert_audio(aud.cuda(), sr, 16000, 1)
                            aud = den_model(aud[None])[0]

                        aud = aud.detach().cpu()

                        aud = taf.resample(aud, 16000, 44100)
                        u_fid = str(uuid.uuid4())
                        tgt_wav_fp = os.path.join(tgt_wav_dir, f"{u_fid}.wav")
                        torchaudio.save(tgt_wav_fp, aud, 44100)
                        meta = {
                            "audio_filepath": tgt_wav_fp,
                            "text": sub.text,
                            "duration": float(end - start),
                            "u_fid": str(uuid.uuid4()),
                        }

                        json.dump(meta, tm)
                        tm.write("\n")

                        _log.info(f"{meta}")

                    _log.info(f"{sub} :{s_index}\n")
