import os

import torch
import torchaudio
import torchaudio.functional as taf
from denoiser import pretrained
from pyannote.audio import Pipeline

src_vid = "data/vid/vid_08.mp4"
tgt_vid_dir = "data/vid/proc"

os.makedirs(tgt_vid_dir, exist_ok=True)

api_token = "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"
pipeline = Pipeline.from_pretrained(
    "pyannote/speaker-diarization-3.0",
    use_auth_token=api_token)

pipeline = pipeline.to(torch.device("cuda")) if torch.cuda.is_available() else pipeline

den_model = pretrained.master64()
den_model = den_model.cuda() if torch.cuda.is_available() else den_model

with torch.no_grad():
    aud, sr = torchaudio.load(src_vid)
    if aud.shape[0] > 1:
        aud = torch.mean(aud, dim=0).unsqueeze(0)

    torchaudio.save(os.path.join(tgt_vid_dir, "vid_08_original.wav"), aud, sr)

    aud = taf.resample(aud, sr, 16000)

    diarization = pipeline({"waveform": aud, "sample_rate": sr}, min_speakers=1, max_speakers=10)

    seg_count = 0
    spk_sets = set()
    if torch.cuda.is_available():
        aud = aud.cuda()
    aud = den_model(aud)[0]
    aud = aud.detach().cpu()

    for turn, spk_id, speaker in diarization.itertracks(yield_label=True):
        spk_sets.add(speaker)
        start = turn.start
        end = turn.end

        print(f"turn: {turn}, spk_id: {spk_id}, speaker:{speaker}")
        effects = [
            ['trim', f'{start}', f'{end - start}'],
        ]
        print(effects)
        aud_seg, _ = torchaudio.sox_effects.apply_effects_tensor(aud, 16000, effects)
        seg_count = seg_count + 1

        if torch.cuda.is_available():
            aud_seg = aud_seg.detach().cpu()

        aud_seg_22k = taf.resample(aud_seg, 16000, 22050)
        aud_seg_44k = taf.resample(aud_seg, 16000, 44100)

        tgt_af_22k = os.path.join(tgt_vid_dir, f"{spk_id}_{seg_count}_22k.wav")
        tgt_af_44k = os.path.join(tgt_vid_dir, f"{spk_id}_{seg_count}_44k.wav")

        torchaudio.save(tgt_af_22k, aud_seg_22k, 22050)
        torchaudio.save(tgt_af_44k, aud_seg_44k, 44100)
