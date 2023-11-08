import json

from scripts.diacritizer.arabic.diacritizer_msa import MSADiacritizer

diacritizer = MSADiacritizer()

run_index = 3
src = f"/data/asr/workspace/audio/tts/data/tts/ar/spotify/manifest/manifest_spk_cluster_{run_index}.json"
tgt = f"/data/asr/workspace/audio/tts/data/tts/ar/spotify/manifest/manifest_spk_cluster_diac_{run_index}.json"

t_count = 0
with open(src, encoding="utf-8") as sm:
    with open(tgt, encoding="utf-8", mode="w") as tm:
        for l_index, line in enumerate(sm):
            jd = dict(json.loads(str(line).strip("\n").strip()))
            text = str(jd["text"]).strip().replace("  ", " ").strip()
            text = diacritizer.diacritize(text)

            jd_new = jd.copy()
            jd_new["text"] = text

            json.dump(jd_new, tm)
            tm.write("\n")
            t_count = t_count + 1
            if t_count % 100 == 0:
                print(f"t_count: {t_count}, l_index: {l_index}")
