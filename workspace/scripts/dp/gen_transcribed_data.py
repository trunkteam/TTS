import json
import logging
import os
import uuid

import stable_whisper
import torch
import torchaudio
import torchaudio.functional as taf

logging.basicConfig(level=logging.INFO)
_log = logging.getLogger("GTD")

model = stable_whisper.load_model('large-v2')
src_maps = [
    {
        "language": "en",
        "audio_file": "/data/asr/workspace/audio/tts/data/english/lh_spotify/KF",
        "d_set_name": "spot_kf",
    },
    {
        "language": "en",
        "audio_file": "/data/asr/workspace/audio/tts/data/english/lh_spotify/dwl",
        "d_set_name": "spot_lh",
    },
    {
        "language": "ar",
        "audio_file": "/data/asr/workspace/audio/tts/data/arabic_spotify/spt_arabic_data/sky_news_spt",
        "d_set_name": "spot_sk",
    },
    {
        "language": "ar",
        "audio_file": "/data/asr/workspace/audio/tts/data/arabic_spotify/spt_arabic_data/dubai_future_foundation",
        "d_set_name": "spot_df",
    },
]

for src_map in src_maps:
    language = src_map["language"]
    base_dir = src_map["audio_file"]
    d_set_name = src_map["d_set_name"]
    tgt_audio_dir = f"/data/asr/workspace/audio/tts/data/tts/{language}/spotify/audio"
    tgt_transcript_dir = f"/data/asr/workspace/audio/tts/data/tts/{language}/spotify/transcript"
    tgt_manifest_dir = f"/data/asr/workspace/audio/tts/data/tts/{language}/spotify/manifest"

    os.makedirs(tgt_audio_dir, exist_ok=True)
    os.makedirs(tgt_transcript_dir, exist_ok=True)
    os.makedirs(tgt_manifest_dir, exist_ok=True)

    tgt_manifest = os.path.join(tgt_manifest_dir, "manifest_srt.json")
    with open(tgt_manifest, encoding="utf-8", mode="w") as tm:
        for index, fn in enumerate(os.listdir(base_dir)):
            if fn not in [".DS_Store"] and (fn.endswith(".mp3") or fn.endswith(".wav")):
                src_af = os.path.join(base_dir, fn)
                aud, sr = torchaudio.load(src_af)
                if aud.shape[0] > 1:
                    aud = torch.mean(aud, dim=0).unsqueeze(0)
    
                aud = taf.resample(aud, sr, 44100)
                
                u_fid = str(uuid.uuid4())
                
                fn_dir_wav = os.path.join(tgt_audio_dir, f"{d_set_name}_{index}")
                os.makedirs(fn_dir_wav, exist_ok=True)
                tgt_wav_fp = os.path.join(fn_dir_wav, f"{u_fid}.wav")
                torchaudio.save(tgt_wav_fp, aud, 44100)

                fn_dir_srt = os.path.join(tgt_transcript_dir, f"{d_set_name}_{index}")
                os.makedirs(fn_dir_srt, exist_ok=True)
                tgt_srt_fp = os.path.join(fn_dir_srt, f"{u_fid}.srt")
    
                if not os.path.exists(tgt_srt_fp):
                    _log.info(f"Transcribing: {tgt_wav_fp}")
                    result = model.transcribe(tgt_wav_fp, language=language)
                    result.to_srt_vtt(tgt_srt_fp, word_level=False)
                    _log.info(f"Created: {tgt_srt_fp}")
                    
                jd = {"audio_filepath": tgt_wav_fp, "srt_filepath": tgt_srt_fp}
                json.dump(jd, tm)
                tm.write("\n")
                _log.info(f"Manifest entry: {jd}\n")
