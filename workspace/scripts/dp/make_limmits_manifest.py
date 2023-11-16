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

device = "cuda:0" if torch.cuda.is_available() else "cpu"

feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained('microsoft/wavlm-base-plus-sv')
se_model = WavLMForXVector.from_pretrained('microsoft/wavlm-base-plus-sv')
se_model = se_model.to(device)

src_dirs = ["data/tts/en/v7/limmits/English_F", "data/tts/en/v7/limmits/English_M",
            "data/tts/hi/v7/limmits/Hindi_F", "data/tts/hi/v7/limmits/Hindi_M"]

spk_emb_map = {}
spk_count = 0

for src_dir in src_dirs:

    src_wav_dir = os.path.join(src_dir, "wav")
    src_txt_dir = os.path.join(src_dir, "txt")
    emb_dir = os.path.join(src_dir, "emb")

    src_wav_dir_44k = os.path.join(src_wav_dir, "44k")
    src_wav_dir_22k = os.path.join(src_wav_dir, "22k")

    os.makedirs(src_wav_dir_44k, exist_ok=True)
    os.makedirs(src_wav_dir_22k, exist_ok=True)
    os.makedirs(emb_dir, exist_ok=True)

    tgt_man_44k = os.path.join(src_dir, "manifest_44k.json")
    tgt_man_22k = os.path.join(src_dir, "manifest_22k.json")

    with open(tgt_man_44k, encoding="utf-8", mode="w") as tm_44k:
        with open(tgt_man_22k, encoding="utf-8", mode="w") as tm_22k:
            for fl in os.listdir(src_wav_dir):
                if fl.endswith(".wav"):
                    src_af = os.path.join(src_wav_dir, fl)
                    src_tf = os.path.join(src_txt_dir, fl.replace(".wav", ".txt"))
                    if os.path.exists(src_af) and os.path.exists(src_tf):
                        with open(src_tf, encoding="utf-8") as st:
                            text = st.read()
                            text = text.strip("\n").replace("  ", " ").strip()
                            aud, sr = torchaudio.load(src_af)
                            if aud.shape[0] > 1:
                                aud = torch.mean(aud, dim=0).unsqueeze(0)

                            if sr != 44100:
                                tgt_af_44k = os.path.join(src_wav_dir_44k, fl)
                                aud_44k = taf.resample(aud, sr, 44100)
                                torchaudio.save(tgt_af_44k, aud_44k, 44100)
                            else:
                                tgt_af_44k = src_af

                            audio_meta = torchaudio.info(tgt_af_44k)
                            duration = audio_meta.num_frames / audio_meta.sample_rate

                            tgt_af_22k = os.path.join(src_wav_dir_22k, fl)
                            aud_22k = taf.resample(aud, sr, 22050)
                            torchaudio.save(tgt_af_22k, aud_22k, 22050)

                            jd_44k = {"audio_filepath": tgt_af_44k, "text": text, "duration": duration}
                            jd_22k = {"audio_filepath": tgt_af_22k, "text": text, "duration": duration}

                            aud_16k = taf.resample(aud, sr, 16000)
                            aud_feat = feature_extractor(aud_16k.squeeze(0),
                                                         padding=True,
                                                         return_tensors="pt",
                                                         sampling_rate=16000).to(device)
                            spk_emb = se_model(**aud_feat).embeddings
                            spk_emb = torch.nn.functional.normalize(spk_emb, dim=-1).cpu().squeeze(0)
                            tgt_emb_file = os.path.join(emb_dir, fl.replace('.wav', '.pth'))
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
                                spk_name = f"spk_lim_{spk_count}"
                                spk_emb_map[spk_name] = tgt_emb_file
                            elif 0.87 <= largest_similarity["score"] <= 1.0:
                                spk_name = largest_similarity["spk"]
                                _log.info(f"src_af: {src_af}, ELS: {largest_similarity}")
                            else:
                                spk_count = spk_count + 1
                                spk_name = f"spk_lim_{spk_count}"
                                spk_emb_map[spk_name] = tgt_emb_file

                            jd_44k["speaker"] = spk_name
                            jd_22k["speaker"] = spk_name

                            json.dump(jd_44k, tm_44k)
                            tm_44k.write("\n")

                            json.dump(jd_22k, tm_22k)
                            tm_22k.write("\n")

                            _log.info(jd_44k)
                            _log.info(f"{jd_22k}\n")
