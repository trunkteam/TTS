import json
import logging
import os

import torchaudio

logging.basicConfig(level=logging.INFO)
_log = logging.getLogger("SDD")

BASE_DIR = "/data/asr/workspace/audio/tts2"
MANIFEST_DIR = os.path.join(BASE_DIR, "data/tts2/manifest")

os.makedirs(MANIFEST_DIR, exist_ok=True)

src_manifests_maps = [
    {
        "manifest": os.path.join(BASE_DIR, "data/tts/ar/quran/manifest/manifest_44k.json"),
        "language": "ar",
        "d_set_name": "ar_qu_gen_v1"
    },
    {
        "manifest": os.path.join(BASE_DIR, "data/tts/ar/manifest/v2/manifest_ar_emotion_el_44k.json"),
        "language": "ar",
        "d_set_name": "ar_el_gen_v2"
    },
    {
        "manifest": os.path.join(BASE_DIR, "data/tts/ar/manifest/v3/manifest_ar_emotion_el_44k.json"),
        "language": "ar",
        "d_set_name": "ar_el_gen_v3"
    },
    {
        "manifest": os.path.join(BASE_DIR, "data/tts/ar/manifest/v5/manifest_ar_emotion_el_44k.json"),
        "language": "ar",
        "d_set_name": "ar_el_gen_v5"
    },
    {
        "manifest": os.path.join(BASE_DIR, "data/tts/en/manifest/v2/manifest_en_emotion_el_44k.json"),
        "language": "en",
        "d_set_name": "en_el_gen_v2"
    },
    {
        "manifest": os.path.join(BASE_DIR, "data/tts/en/manifest/v3/manifest_en_emotion_v3_el_44k.json"),
        "language": "en",
        "d_set_name": "en_el_gen_v3"
    },
    {
        "manifest": os.path.join(BASE_DIR, "data/tts/spk_enc/manifest_multi_lang_se_ar_8_44k_clean.json"),
        "language": "ar",
        "d_set_name": "ar_se_v1"
    },
    {
        "manifest": os.path.join(BASE_DIR, "data/tts/spk_enc/manifest_multi_lang_se_en_8_44k_clean.json"),
        "language": "en",
        "d_set_name": "en_se_v1"
    },
    {
        "manifest": os.path.join(BASE_DIR, "data/tts/en/v5/manifest/manifest_en_azure_gen_sent_clean.json"),
        "language": "en",
        "d_set_name": "en_az_gen_v1"
    },
]

for manifest_map in src_manifests_maps:
    src_manifest = manifest_map['manifest']
    language = manifest_map['language']
    d_set_name = manifest_map['d_set_name']

    manifest_dir = os.path.join(MANIFEST_DIR, language)
    manifest_dir = os.path.join(manifest_dir, d_set_name)

    os.makedirs(manifest_dir, exist_ok=True)

    tgt_manifest_train = os.path.join(manifest_dir, "manifest.json")
    tgt_manifest_eval = os.path.join(manifest_dir, "manifest_eval.json")

    with open(tgt_manifest_train, encoding="utf-8", mode="w") as tmt:
        with open(tgt_manifest_eval, encoding="utf-8", mode="w") as tmv:
            with open(src_manifest, encoding="utf-8") as sm:
                for line in sm:
                    jd = json.loads(line.strip("\n").strip())
                    src_af = jd['audio_filepath']
                    if BASE_DIR not in src_af:
                        src_af = os.path.join(BASE_DIR, src_af)
                    # aud, sr = torchaudio.load(src_af)
                    audio_meta = torchaudio.info(src_af)
                    duration = audio_meta.num_frames / audio_meta.sample_rate

                    if duration > 21.0:
                        json.dump(jd, tmv)
                        tmv.write("\n")
                        _log.info(f"src_af: {src_af}, split: eval")
                    else:
                        json.dump(jd, tmt)
                        tmt.write("\n")
                        _log.info(f"src_af: {src_af}, split: train")
