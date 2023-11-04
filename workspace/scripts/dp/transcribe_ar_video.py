import logging

import stable_whisper
import torchaudio

logging.basicConfig(level=logging.INFO)
_log = logging.getLogger("GTD")

model = stable_whisper.load_model('large-v2')

src_af = "/data/asr/workspace/audio/tts2/spl/The_Molhem_Show_Epi_3_V2.mov"
tgt_af = "/data/asr/workspace/audio/tts2/spl/The_Molhem_Show_Epi_3_V2.wav"
tgt_srt_en = "/data/asr/workspace/audio/tts2/spl/The_Molhem_Show_Epi_3_V2_en.srt"
tgt_srt_ar = "/data/asr/workspace/audio/tts2/spl/The_Molhem_Show_Epi_3_V2_ar.srt"

aud, sr = torchaudio.load(src_af)
_log.info(f"Loaded audio, sr: {sr}")

torchaudio.save(tgt_af, aud, sr)
result_en = model.transcribe(tgt_af, language="en")
result_ar = model.transcribe(tgt_af, language="ar")
result_en.to_srt_vtt(tgt_srt_en, word_level=False)
_log.info(f"Created: {tgt_srt_en}")
result_ar.to_srt_vtt(tgt_srt_ar, word_level=False)
_log.info(f"Created: {tgt_srt_ar}")
