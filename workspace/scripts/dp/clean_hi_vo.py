import json
import logging
import os

import torch
import torchaudio
import torchaudio.functional as taf
from transformers import Wav2Vec2FeatureExtractor, WavLMForXVector

from TTS.tts.utils.managers import save_file, load_file

logging.basicConfig(level=logging.INFO)
_log = logging.getLogger("CH")

src_dir = "/data/asr/workspace/audio/tts/data/tts/hi/fiver_hin/hin_vo"
tgt_dir = "data/tts/v3/hi/raw"
emb_dir = "data/tts/v3/hi/emb"
cln_dir = "data/tts/v3/hi/cln"

tgt_man = "data/tts/v3/hi/manifest_spk_cluster.json"

os.makedirs(tgt_dir, exist_ok=True)
os.makedirs(emb_dir, exist_ok=True)
os.makedirs(cln_dir, exist_ok=True)

tgt_sr = 22050

device = "cuda:0" if torch.cuda.is_available() else "cpu"

feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained('microsoft/wavlm-base-plus-sv')
se_model = WavLMForXVector.from_pretrained('microsoft/wavlm-base-plus-sv')
se_model = se_model.to(device)

vad_model, vad_utils = torch.hub.load(repo_or_dir='snakers4/silero-vad',
                                      model='silero_vad',
                                      force_reload=True)

get_speech_timestamps = vad_utils[0]

spk_emb_map = {}
spk_count = 0
chunk_duration = 3.0

with open(tgt_man, encoding="utf-8", mode="w") as tm:
    for f_index, fl in enumerate(os.listdir(src_dir)):
        try:
            if fl.endswith(".mp3") or fl.endswith(".mp4") or fl.endswith(".wav"):
                src_af = os.path.join(src_dir, fl)
                tgt_af = os.path.join(tgt_dir, f"hi_vo_{f_index}_{tgt_sr}k.wav")

                aud, sr = torchaudio.load(src_af)
                if aud.shape[0] > 1:
                    aud = torch.mean(aud, dim=0).unsqueeze(0)

                # _log.info(f"aud: {aud.shape}")

                aud_22k = taf.resample(aud, sr, tgt_sr)
                torchaudio.save(tgt_af, aud_22k, tgt_sr)
                # _log.info(f"aud_22k: {aud_22k.shape}")

                aud_16k = taf.resample(aud, sr, 16000)
                # _log.info(f"aud_16k: {aud_16k.shape}")
                speech_ts = get_speech_timestamps(aud_16k.squeeze(0), vad_model, sampling_rate=16000, return_seconds=True)
                for t_index, ts in enumerate(speech_ts):
                    start = ts["start"]
                    end = ts["end"]
                    duration = end - start

                    # _log.info({"start": start, "end": end, "duration": duration})

                    if duration >= chunk_duration:
                        effects = [
                            ['trim', f'{start}', f'{duration}'],
                        ]
                        try:
                            aud_ch_22k, sr_22k = torchaudio.sox_effects.apply_effects_tensor(aud_22k, tgt_sr, effects)
                            aud_ch_16k, sr_16k = torchaudio.sox_effects.apply_effects_tensor(aud_16k, 16000, effects)
                            _log.info(f"aud_ch_16k: {aud_ch_16k.shape}")
                            tgt_af_ch = os.path.join(tgt_dir, f"hi_vo_{f_index}_{tgt_sr}k_{t_index}.wav")
                            torchaudio.save(tgt_af_ch, aud_ch_22k, tgt_sr)
                            aud_feat = feature_extractor(aud_ch_16k.squeeze(0),
                                                         padding=True,
                                                         return_tensors="pt",
                                                         sampling_rate=16000).to(device)
                            spk_emb = se_model(**aud_feat).embeddings
                            spk_emb = torch.nn.functional.normalize(spk_emb, dim=-1).cpu().squeeze(0)
                            tgt_emb_file = os.path.join(emb_dir, f"hi_vo_{f_index}_{t_index}.pth")
                            save_file(spk_emb, tgt_emb_file)
                            largest_similarity = {"spk": None, "score": 0.0}
                            _log.info(f"speakers: {len(spk_emb_map)}")
                            for spk, emb_file in spk_emb_map.items():
                                ref_emb = load_file(emb_file)
                                # src_spk_emb = torch.from_numpy(spk_emb)
                                cos = torch.cosine_similarity(ref_emb, spk_emb, dim=0)
                                if cos > largest_similarity["score"]:
                                    largest_similarity = {"spk": spk, "score": cos}
                                    _log.debug(f"src_af: {src_af}, LS: {largest_similarity}")

                            if largest_similarity["spk"] is None:
                                spk_count = spk_count + 1
                                spk_name = f"spk_hi_vo_{spk_count}"
                                spk_emb_map[spk_name] = tgt_emb_file
                            elif 0.87 <= largest_similarity["score"] <= 1.0:
                                spk_name = largest_similarity["spk"]
                                _log.info(f"src_af: {src_af}, ELS: {largest_similarity}")
                            else:
                                spk_count = spk_count + 1
                                spk_name = f"spk_hi_vo_{spk_count}"
                                spk_emb_map[spk_name] = tgt_emb_file

                            jd = {"audio_filepath": tgt_af_ch, "speaker": spk_name}

                            json.dump(jd, tm)
                            tm.write("\n")

                            _log.info(f"tgt_af_ch: {tgt_af_ch}, speaker: {spk_name}\n")
                        except Exception as e:
                            _log.exception(e)

        except Exception as e:
            _log.exception(e)
