import json
import logging
import os

import torch
import torchaudio
import torchaudio.functional as taf
from transformers import Wav2Vec2FeatureExtractor, WavLMForXVector

from TTS.tts.utils.managers import save_file, load_file

# from TTS.tts.utils.speakers import SpeakerManager

logging.basicConfig(level=logging.INFO)
_log = logging.getLogger("CS")

device = "cuda:0" if torch.cuda.is_available() else "cpu"

feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained('microsoft/wavlm-base-plus-sv')
se_model = WavLMForXVector.from_pretrained('microsoft/wavlm-base-plus-sv')
se_model = se_model.to(device)

spk_emb_map = {}

for lang in ["en", "ar"]:
    src_manifest = f"/data/asr/workspace/audio/tts/data/tts/{lang}/spotify/manifest/manifest_diar.json"
    tgt_manifest = f"/data/asr/workspace/audio/tts/data/tts/{lang}/spotify/manifest/manifest_spk_cluster.json"

    tgt_spk_emb_dir = f"/data/asr/workspace/audio/tts/data/tts/{lang}/spotify/emb"
    os.makedirs(tgt_spk_emb_dir, exist_ok=True)
    spk_count = 0
    with open(src_manifest, encoding="utf-8") as sm:
        with open(tgt_manifest, encoding="utf-8", mode="w") as tm:
            for line in sm:
                try:
                    jd = json.loads(line.strip("\n").strip())
                    src_af = jd['audio_filepath']
                    u_fid = jd['u_fid']
                    aud, sr = torchaudio.load(src_af)
                    aud = taf.resample(aud, sr, 16000)
                    aud = aud.squeeze(0).to(device)
                    aud_feat = feature_extractor(aud, padding=True, return_tensors="pt", sampling_rate=16000)
                    aud_feat = aud_feat.to(device)
                    spk_emb = se_model(**aud_feat).embeddings
                    spk_emb = torch.nn.functional.normalize(spk_emb, dim=-1).cpu()
                    spk_emb = spk_emb.squeeze(0)
                    tgt_emb_file = os.path.join(tgt_spk_emb_dir, f"{u_fid}.pth")
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

                    _log.info(f"largest_similarity: {largest_similarity}")

                    if largest_similarity["spk"] is None:
                        spk_count = spk_count + 1
                        spk_name = f"spk_sp_{lang}_{spk_count}"
                        spk_emb_map[spk_name] = tgt_emb_file
                    elif 0.87 <= largest_similarity["score"] <= 1.0:
                        spk_name = largest_similarity["spk"]
                        _log.info(f"src_af: {src_af}, ELS: {largest_similarity}")
                    else:
                        spk_count = spk_count + 1
                        spk_name = f"spk_sp_{lang}_{spk_count}"
                        spk_emb_map[spk_name] = tgt_emb_file

                    jd["speaker"] = spk_name

                    json.dump(jd, tm)
                    tm.write("\n")

                    _log.info(f"src_af: {src_af}, speaker: {spk_name}\n")
                except Exception as e:
                    _log.exception(e)
                    # raise e
