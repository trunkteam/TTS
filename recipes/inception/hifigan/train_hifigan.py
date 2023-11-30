import os

from trainer import Trainer, TrainerArgs

from TTS.config import BaseAudioConfig
from TTS.utils.audio import AudioProcessor
from TTS.vocoder.configs.hifigan_config import HifiganConfig
from TTS.vocoder.datasets.preprocess import load_wav_data
from TTS.vocoder.models.gan import GAN

BASE_PATH = "/data/asr/workspace/audio/tts2"

output_path = os.path.dirname(os.path.abspath(__file__))

config = HifiganConfig(
    batch_size=128,
    eval_batch_size=32,
    num_loader_workers=8,
    num_eval_loader_workers=8,
    run_eval=True,
    test_delay_epochs=5,
    epochs=1000,
    seq_len=16384,
    pad_short=2000,
    use_noise_augment=True,
    eval_split_size=10,
    print_step=25,
    print_eval=False,
    mixed_precision=False,
    lr_gen=1e-4,
    lr_disc=1e-4,
    data_path=os.path.join(output_path, "../LJSpeech-1.1/wavs/"),
    output_path=output_path,
    audio=BaseAudioConfig(
        sample_rate=48000,
        fft_size=2048,
        hop_length=512,
        win_length=2048,
        num_mels=128,
    ),
    generator_model_params={
        "upsample_factors": [8, 8, 4, 2],
        "upsample_kernel_sizes": [16, 16, 8, 4],
        "upsample_initial_channel": 512,
        "resblock_kernel_sizes": [3, 7, 11, 13],
        "resblock_dilation_sizes": [[1, 3, 5], [1, 3, 5], [1, 3, 5], [1, 3, 5]],
        "resblock_type": "1",
        "cond_channels": 512,
    },

)

# init audio processor
ap = AudioProcessor(**config.audio.to_dict())

# load training samples
eval_samples, train_samples = load_wav_data(config.data_path, config.eval_split_size)

# init model
model = GAN(config, ap)

# init the trainer and 🚀
trainer = Trainer(
    TrainerArgs(), config, output_path, model=model, train_samples=train_samples, eval_samples=eval_samples
)
trainer.fit()
