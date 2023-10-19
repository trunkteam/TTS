import json
import uuid

base_path = "/data/asr/workspace/audio/tts/"
# src_man = "/data/asr/workspace/audio/tts/data/tts/en/v5/manifest/manifest_en_azure_gen_sent.json"
# tgt_man = "/data/asr/workspace/audio/tts/data/tts/en/v5/manifest/manifest_en_azure_gen_sent_clean.json"

src_man = "/data/asr/workspace/audio/tts/data/tts/ar/manifest/v5/manifest_ar_el_cmon_diac_el_44k.json"
tgt_man = "/data/asr/workspace/audio/tts/data/tts/ar/manifest/v5/manifest_ar_emotion_el_44k.json"

with open(tgt_man, encoding="utf-8", mode="w") as tm:
    with open(src_man, encoding="utf-8") as sm:
        for line in sm:
            try:
                jd = json.loads(line.strip("\n").strip())
                src_af = str(jd['audio_filepath'])
                if base_path in src_af:
                    src_af = src_af.replace(base_path, "")
                if ".mp3" in src_af and "/tmp/" in src_af:
                    src_af = src_af.replace("/tmp/", "/44k/")
                    src_af = src_af.replace(".mp3", ".wav")

                jd['audio_filepath'] = src_af
                if "u_fid" not in jd:
                    jd['u_fid'] = str(uuid.uuid4())

                json.dump(jd, tm)
                tm.write("\n")
            except Exception as e:
                print(e)
