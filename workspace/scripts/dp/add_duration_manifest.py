import json
import os

import torchaudio

BASE_PATH = "/data/asr/workspace/audio/tts2"
DATA_PATH_AR = os.path.join(BASE_PATH, "data/tts2/manifest/ar")
DATA_PATH_EN = os.path.join(BASE_PATH, "data/tts2/manifest/en")

manifests = [
    os.path.join(os.path.join(DATA_PATH_AR, "ar_qu_v1"), "manifest.json"),
    os.path.join(os.path.join(DATA_PATH_AR, "ar_qu_v1"), "manifest_eval.json"),
    os.path.join(os.path.join(DATA_PATH_AR, "ar_el_gen_v2"), "manifest.json"),
    os.path.join(os.path.join(DATA_PATH_AR, "ar_el_gen_v2"), "manifest_eval.json"),
    os.path.join(os.path.join(DATA_PATH_AR, "ar_el_gen_v3"), "manifest.json"),
    os.path.join(os.path.join(DATA_PATH_AR, "ar_el_gen_v3"), "manifest_eval.json"),
    os.path.join(os.path.join(DATA_PATH_AR, "ar_el_gen_v5"), "manifest.json"),
    os.path.join(os.path.join(DATA_PATH_AR, "ar_el_gen_v5"), "manifest_eval.json"),
    os.path.join(os.path.join(DATA_PATH_AR, "ar_se_v1"), "manifest.json"),
    os.path.join(os.path.join(DATA_PATH_AR, "ar_se_v1"), "manifest_eval.json"),
    os.path.join(os.path.join(DATA_PATH_EN, "en_el_gen_v2"), "manifest.json"),
    os.path.join(os.path.join(DATA_PATH_EN, "en_el_gen_v2"), "manifest_eval.json"),
    os.path.join(os.path.join(DATA_PATH_EN, "en_el_gen_v3"), "manifest.json"),
    os.path.join(os.path.join(DATA_PATH_EN, "en_el_gen_v3"), "manifest_eval.json"),
    os.path.join(os.path.join(DATA_PATH_EN, "en_se_v1"), "manifest.json"),
    os.path.join(os.path.join(DATA_PATH_EN, "en_se_v1"), "manifest_eval.json"),
    os.path.join(os.path.join(DATA_PATH_EN, "en_az_gen_v1"), "manifest.json"),
    os.path.join(os.path.join(DATA_PATH_EN, "en_az_gen_v1"), "manifest_eval.json"),
]

for manifest in manifests:
    man_tgt = manifest.replace(".json", "_dur.json")
    with open(manifest, encoding="utf-8") as sm:
        with open(man_tgt, encoding="utf-8", mode="w") as tm:
            for line in sm:
                jd = json.loads(line.strip("\n").strip())
                if "duration" not in jd:
                    wav_file = jd["audio_filepath"]
                    wav_file = os.path.join(BASE_PATH, wav_file)
                    audio_meta = torchaudio.info(wav_file)
                    duration = audio_meta.num_frames / audio_meta.sample_rate
                    jd["duration"] = duration
                json.dump(jd, tm)
                tm.write("\n")
