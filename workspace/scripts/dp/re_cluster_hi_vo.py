import json
import logging
import os
import uuid

import torch
import torchaudio
# from denoiser.audio import convert_audio
import torchaudio.functional as taf
from denoiser import pretrained
from transformers import Wav2Vec2FeatureExtractor, WavLMForXVector

from TTS.tts.utils.managers import save_file, load_file

logging.basicConfig(level=logging.INFO)
_log = logging.getLogger("RCH")

den_model = pretrained.master64()
den_model = den_model.cuda() if torch.cuda.is_available() else den_model
device = "cuda:0" if torch.cuda.is_available() else "cpu"

feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained('microsoft/wavlm-base-plus-sv')
se_model = WavLMForXVector.from_pretrained('microsoft/wavlm-base-plus-sv').to(device)

src_man = "data/tts/v3/hi/manifest_spk_cluster.json"
tgt_man = "data/tts/v3/hi/manifest_spk_cluster_cln.json"

cln_dir = "data/tts/v3/hi/cln"
emb_dir = "data/tts/v3/hi/emb"

os.makedirs(cln_dir, exist_ok=True)
os.makedirs(emb_dir, exist_ok=True)

spk_emb_map = {}
spk_count = 0
with open(src_man, encoding="utf-8") as sm:
    with open(tgt_man, encoding="utf-8", mode="w") as tm:
        for line in sm:
            try:
                jd = json.loads(line.strip("\n").strip())
                src_af = jd["audio_filepath"]
                aud, sr = torchaudio.load(src_af)
                if aud.shape[0] > 1:
                    aud = torch.mean(aud, dim=0).unsqueeze(0)

                aud = taf.resample(aud, sr, 16000).cuda()

                with torch.no_grad():
                    aud = den_model(aud)[0]
                    u_fid = str(uuid.uuid4())

                    aud_feat = feature_extractor(aud.squeeze(0),
                                                 padding=True,
                                                 return_tensors="pt",
                                                 sampling_rate=16000).to(device)
                    spk_emb = se_model(**aud_feat).embeddings
                    spk_emb = torch.nn.functional.normalize(spk_emb, dim=-1).cpu().squeeze(0)
                    tgt_emb_file = os.path.join(emb_dir, f"{u_fid}.pth")
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

                    aud = taf.resample(aud.detach().cpu(), 16000, sr)
                    tgt_af = src_af.replace("/raw/", "/cln/")
                    torchaudio.save(tgt_af, aud, sr)

                    audio_meta = torchaudio.info(tgt_af)
                    duration = audio_meta.num_frames / audio_meta.sample_rate

                    jd = {"audio_filepath": tgt_af, "speaker": spk_name, "duration": duration, "u_fid": u_fid}

                    json.dump(jd, tm)
                    tm.write("\n")

                    _log.info(f"{jd}\n")

            except Exception as e:
                _log.exception(e)
