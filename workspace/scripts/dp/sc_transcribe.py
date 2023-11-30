import json
import logging
import os
import uuid

import pysubs2 as pysubs2
import stable_whisper
import torch
import torchaudio
import torchaudio.functional as taf
from denoiser import pretrained
from pyannote.audio import Pipeline

logging.basicConfig(level=logging.INFO)
_log = logging.getLogger("CH")

model = stable_whisper.load_model('large-v3')

api_token = "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"
pipeline = Pipeline.from_pretrained(
    "pyannote/speaker-diarization-3.0",
    use_auth_token=api_token).to(torch.device("cuda"))

den_model = pretrained.master64()
den_model = den_model.cuda() if torch.cuda.is_available() else den_model

timestamp_adjustment = 370
for lang in ["en", "ar"]:
    base_dir = f"data/tts/vo/{lang}"
    os.makedirs(base_dir, exist_ok=True)
    src_manifest = os.path.join(base_dir, "manifest_dwl.json")
    tgt_manifest = os.path.join(base_dir, "manifest_trn_2.json")
    tgt_wav_dir = os.path.join(base_dir, "wav_44k")
    os.makedirs(tgt_wav_dir, exist_ok=True)

    count = 0
    with open(tgt_manifest, encoding="utf-8", mode="w") as tm:
        with open(src_manifest, encoding="utf-8") as sm:
            for line in sm:
                try:
                    jd = dict(json.loads(line.strip("\n").strip()))
                    dwl_af = str(jd["dwl_af"])
                    srt = dwl_af.replace(".mp3", ".srt")
                    if not os.path.exists(srt):
                        result = model.transcribe(dwl_af, language=f"{lang}")
                        result = result.split_by_punctuation(punctuation=['!', '.', '؟', '?'])
                        result.to_srt_vtt(srt, word_level=False)
                        _log.info(f"Created: {srt}")

                    subs = pysubs2.load(srt, encoding="utf-8")
                    for s_index, sub in enumerate(subs):
                        start = sub.start + timestamp_adjustment
                        end = sub.end + timestamp_adjustment
                        duration = end - start

                        start = float(start / 1000)
                        end = float(end / 1000)
                        duration = float(duration / 1000)

                        effects = [
                            ['trim', f'{start}', f'{duration}'],
                        ]
                        aud, sr = torchaudio.sox_effects.apply_effects_file(dwl_af, effects)
                        if aud.shape[0] > 1:
                            aud = torch.mean(aud, dim=0).unsqueeze(0)
                        aud_16k = taf.resample(aud, sr, 16000)
                        diarization = pipeline({"waveform": aud_16k, "sample_rate": 16000}, min_speakers=1,
                                               max_speakers=10)
                        spk_sets = set()
                        for turn, spk_id, speaker in diarization.itertracks(yield_label=True):
                            spk_sets.add(speaker)

                        _log.info(spk_sets)

                        if len(spk_sets) == 1:
                            with torch.no_grad():
                                aud_16k = aud_16k.cuda()
                                for i in range(5):
                                    aud_16k = den_model(aud_16k[None])[0]
                                aud_44k = taf.resample(aud_16k, 16000, 44100)
                                aud_44k = aud_44k.detach().cpu()
                                u_fid = str(uuid.uuid4())
                                tgt_wav_fp = os.path.join(tgt_wav_dir, f"{u_fid}.wav")
                                torchaudio.save(tgt_wav_fp, aud_44k, 44100)

                                jd_copy = jd.copy()
                                jd_copy["srt"] = srt
                                jd_copy["audio_filepath"] = tgt_wav_fp
                                jd_copy["text"] = str(sub.text).strip().replace("  ", " ")
                                jd_copy["u_fid"] = u_fid
                                jd_copy["duration"] = duration

                                json.dump(jd_copy, tm)
                                tm.write("\n")

                                count = count + 1
                                _log.info(f"jd: {jd_copy}, lang: {lang}, index: {count}")
                except Exception as e:
                    _log.exception(e)
