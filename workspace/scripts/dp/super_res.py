from audiosr import build_model, save_wave, make_batch_for_super_resolution,super_resolution

input_file = "data/tts/vo/en/wav/fc73123d-cd32-46ef-9aec-0446b298f2c5.wav"
random_seed = 37
sample_rate = 48000
latent_t_per_second = 12.8
guidance_scale = 3.5
ddim_steps = 50

model = build_model(model_name="speech", device="cpu")
print(model)

waveform = super_resolution(
    model,
    input_file,
    seed=random_seed,
    guidance_scale=guidance_scale,
    ddim_steps=ddim_steps,
    latent_t_per_second=latent_t_per_second
)

save_wave(waveform, inputpath=input_file, savepath="data/tts/vo/en/enhanced",
          name="fc73123d-cd32-46ef-9aec-0446b298f2c5", samplerate=sample_rate)


class SuperRes(object):
    def __init__(self) -> None:
        self.model = build_model(model_name="speech", device=None)
        super().__init__()

    def upsample(self, arc_af, "", )


