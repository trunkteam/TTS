import json
import os

import torch
from trainer import Trainer, TrainerArgs

from TTS.bin.compute_embeddings_v2 import compute_embeddings
from TTS.config.shared_configs import BaseDatasetConfig
from TTS.tts.configs.vits_config import VitsConfig
from TTS.tts.datasets import load_tts_samples
from TTS.tts.models.vits import Vits, VitsArgs, VitsAudioConfig

torch.set_num_threads(24)
RUN_NAME = "YTTS-ML-EL"
EXP_ID = "v5_ML_EL"
REF_EXP_ID = "v5_ML_EL"

SPK_EMBEDDING_VERSION = "v1"
PHN_CACHE_VERSION = "v1"
LNG_EMBEDDING_VERSION = "v1"
BASE_PATH = "/data/asr/workspace/audio/tts2"
BASE_PATH_DATA = "/data/asr/workspace/audio/tts"
DATA_PATH_AR = "data/tts2/manifest/ar"
DATA_PATH_EN = "data/tts2/manifest/en"
EXPMT_PATH = os.path.join(BASE_PATH, f"expmt/ytts/{EXP_ID}")
REF_EXPMT_PATH = os.path.join(BASE_PATH, f"expmt/ytts/{REF_EXP_ID}")
PHN_CACHE_PATH = os.path.join(REF_EXPMT_PATH, f"phn_cache_{PHN_CACHE_VERSION}")
SPK_EMB_CACHE_PATH = os.path.join(REF_EXPMT_PATH, f"spk_emb_{SPK_EMBEDDING_VERSION}")
LNG_EMB_CACHE_PATH = os.path.join(REF_EXPMT_PATH, f"lng_emb_{LNG_EMBEDDING_VERSION}")
RESTORE_PATH = os.path.join(BASE_PATH, "models/ytts/en/model_file.pth")

os.makedirs(PHN_CACHE_PATH, exist_ok=True)
# os.makedirs(SPK_EMB_CACHE_PATH, exist_ok=True)
os.makedirs(LNG_EMB_CACHE_PATH, exist_ok=True)

LNG_EMB = {
    "ar": 0,
    "en": 1,
}

LNG_EMB_FILE = os.path.join(LNG_EMB_CACHE_PATH, "language_ids.json")
with open(LNG_EMB_FILE, mode="w") as lef:
    json.dump(LNG_EMB, lef)

SKIP_TRAIN_EPOCH = False
BATCH_SIZE = 16
EVAL_BATCH_SIZE = 32
SAMPLE_RATE = 44100
MAX_AUDIO_LEN_IN_SECONDS = 21
MIN_AUDIO_LEN_IN_SECONDS = 0
NUM_RESAMPLE_THREADS = 10


def get_dataset(manifest_train: str, manifest_eval: str, d_name: str, lang: str = "ar", base_path=BASE_PATH_DATA,
                data_path=DATA_PATH_AR):
    data_path = os.path.join(data_path, d_name)
    return BaseDatasetConfig(
        formatter="iiai_tts",
        dataset_name=f"{lang}_{d_name}",
        meta_file_train=os.path.join(data_path, manifest_train),
        meta_file_val=os.path.join(data_path, manifest_eval),
        path=base_path,
        language=lang,
    )


DATASETS_CONFIG_LIST = [
    get_dataset(manifest_train="manifest_dur.json",
                manifest_eval="manifest_eval_dur.json",
                d_name="ar_qu_v1",
                lang="ar"),
    get_dataset(manifest_train="manifest_dur.json",
                manifest_eval="manifest_eval_dur.json",
                d_name="ar_el_gen_v2",
                lang="ar"),
    get_dataset(manifest_train="manifest_dur.json",
                manifest_eval="manifest_eval_dur.json",
                d_name="ar_el_gen_v3",
                lang="ar"),
    get_dataset(manifest_train="manifest_dur.json",
                manifest_eval="manifest_eval_dur.json",
                d_name="ar_el_gen_v5",
                lang="ar"),
    # get_dataset(manifest_train="manifest_dur.json",
    #             manifest_eval="manifest_eval_dur.json",
    #             d_name="ar_se_v1",
    #             lang="ar"),
    get_dataset(manifest_train="manifest_dur.json",
                manifest_eval="manifest_eval_dur.json",
                d_name="en_el_gen_v2",
                lang="en",
                data_path=DATA_PATH_EN),
    get_dataset(manifest_train="manifest_dur.json",
                manifest_eval="manifest_eval_dur.json",
                d_name="en_el_gen_v3",
                lang="en",
                data_path=DATA_PATH_EN),
    # get_dataset(manifest_train="manifest_dur.json",
    #             manifest_eval="manifest_eval_dur.json",
    #             d_name="en_se_v1",
    #             lang="en",
    #             data_path=DATA_PATH_EN),
    # get_dataset(manifest_train="manifest_dur.json",
    #             manifest_eval="manifest_eval_dur.json",
    #             d_name="en_az_gen_v1",
    #             lang="en",
    #             data_path=DATA_PATH_EN),

]

# D_VECTOR_FILES = [os.path.join(BASE_PATH, "expmt/ytts/v1_ML_EL/spk_emb_v1/spk_emb_el.pth")]
D_VECTOR_FILES = []

for dataset_conf in DATASETS_CONFIG_LIST:
    embeddings_file = os.path.join(SPK_EMB_CACHE_PATH, f"speakers_{dataset_conf.dataset_name}.pth")
    if not os.path.isfile(embeddings_file):
        print(f">>> Computing the speaker embeddings for the {dataset_conf.dataset_name} dataset")
        compute_embeddings(
            embeddings_file,
            config_dataset_path=None,
            formatter_name=dataset_conf.formatter,
            dataset_name=dataset_conf.dataset_name,
            dataset_path=dataset_conf.path,
            meta_file_train=dataset_conf.meta_file_train,
            meta_file_val=dataset_conf.meta_file_val,
            disable_cuda=False,
            no_eval=False,
        )
    D_VECTOR_FILES.append(embeddings_file)

# print(f" > D_VECTOR_FILES: {D_VECTOR_FILES}")

# Audio config used in training.
audio_config = VitsAudioConfig(
    sample_rate=SAMPLE_RATE,
    hop_length=512,
    win_length=4096,
    fft_size=4096,
    num_mels=480,
)

# Init VITSArgs setting the arguments that are needed for the YourTTS model
model_args = VitsArgs(
    use_sdp=True,
    spec_segment_size=64,
    hidden_channels=192,
    dp_hidden_channels=256,
    d_vector_file=D_VECTOR_FILES,
    use_d_vector_file=True,
    d_vector_dim=512,
    out_channels=2049,
    num_heads_text_encoder=4,
    num_layers_text_encoder=10,
    dropout_p_text_encoder=0.3,
    dropout_p_duration_predictor=0.3,
    num_layers_dp_flow=32,
    num_layers_flow=32,
    num_layers_posterior_encoder=32,
    upsample_rates_decoder=[8, 8, 4, 2],
    # speaker_encoder_model_path=SPEAKER_ENCODER_CHECKPOINT_PATH,
    # speaker_encoder_config_path=SPEAKER_ENCODER_CONFIG_PATH,
    # use_speaker_encoder_as_loss=True,
    # encoder_sample_rate=44100,
    # use_language_embedding=True,
)

# General training config, here you can change the batch size and others useful parameters
config = VitsConfig(
    epochs=10000,
    output_path=EXPMT_PATH,
    lr_gen=0.00023,
    lr_disc=0.00023,
    model_args=model_args,
    run_name=RUN_NAME,
    project_name=RUN_NAME,
    run_description=RUN_NAME,
    dashboard_logger="tensorboard",
    audio=audio_config,
    batch_size=BATCH_SIZE,
    batch_group_size=48,
    eval_batch_size=EVAL_BATCH_SIZE,
    num_loader_workers=24,
    print_step=100,
    plot_step=100,
    log_model_step=100,
    save_step=6305,
    save_all_best=True,
    save_n_checkpoints=7,
    save_checkpoints=True,
    print_eval=True,
    compute_input_seq_cache=True,
    add_blank=True,
    text_cleaner="gen_cleaners",
    start_by_longest=True,
    datasets=DATASETS_CONFIG_LIST,
    cudnn_benchmark=False,
    min_audio_len=SAMPLE_RATE * MIN_AUDIO_LEN_IN_SECONDS,
    max_audio_len=SAMPLE_RATE * MAX_AUDIO_LEN_IN_SECONDS,
    mixed_precision=False,
    use_d_vector_file=True,
    d_vector_dim=2560,
    d_vector_file=D_VECTOR_FILES,
    use_bpe_tokenizer=True,
    bpe_tokenizer_vocab=os.path.join(BASE_PATH, "data/tts/token/v1/model/bpe_tokenizer.json"),
    speaker_encoder_loss_alpha=9.0,
    use_language_weighted_sampler=True,
    use_weighted_sampler=True,
    weighted_sampler_attrs={
        "language": 1.0
    },
    weighted_sampler_multipliers={
        "language": {}
    },
    test_sentences=[
        [
            'بِفَضْلِ عَرَبَاتِ التِّلِفْرِيكِ أَوْ " الْقَاطِرَاتِ الْمُعَلَّقَةِ " وَالطَّعَامِ التَّقْلِيدِيِّ وَالْهَوَاءِ الْجَبَلِيِّ النَّظِيفِ ، يَرْغَبُ صُنَّاعُ الْقَرَارِ وَسُكَّانُ مَدِينَةِ عَجْلُونَ شَمَالَ الْأُرْدُنِّ فِي التَّرْوِيجِ لِمِنْطَقَتِهِمْ بِاعْتِبَارِهَا مَرْكَزًا لِلسِّيَاحَةِ الْخَضْرَاءِ فِي الْبِلَادِ، لِإِنْعَاشِ الِاقْتِصَادِ الْمَحَلِّيِّ مِنْ خِلَالِ الْعَائِدَاتِ الْمَالِيَّةِ الْقَادِمَةِ مِنْ زِيَادَةِ عَدَدِ الزُّوَّارِ لِمَحَطَّةِ التِّلِفْرِيكِ الْجَدِيدَةِ فِي جِبَالِ عَجْلُونَ.',
            "spk_ar_el_v2_1",
            None,
            "ar",
        ],
        [
            "Alright folks, you're tuned into ninety-nine-point-nine Dubai's ADDC Morning Mania with RJ Danny and RJ Sam. We're here to make your morning traffic sound like a lullaby...or a comedy show. Whichever you prefer!",
            "spk_en_el_v3_1",
            None,
            "en",
        ],
        [
            "Over at Al Khail Road, we've got some smooth sailing. By 'smooth', I mean slightly less congested than your morning nose after forgetting the night's antihistamine.",
            "spk_en_el_v3_2",
            None,
            "en",
        ],
        [
            "Umm, so, like, IDK what you're, you know, totally getting at, but TBH, it's, uh, kinda hard to, erm, figure out without, like, more deets, LOL.",
            "spk_en_el_v3_4",
            None,
            "en",
        ],
        [
            "a b c d e f g h i j k l m n o p q r s t u v w x y z",
            "spk_en_el_v3_5",
            None,
            "en",
        ],
        [
            "A B C D E F G H I J K L M N O P Q R S T U V W X Y Z",
            "spk_en_el_v3_6",
            None,
            "en",
        ],
        [
            "ADDC or ETH? Sense of abbreviation. Among XXX and ABC or TED and MAS and again TEDx, E L E V E N, ELEVEN",
            "spk_en_el_v3_7",
            None,
            "en",
        ],
        [
            "Sam",
            "spk_en_el_v3_8",
            None,
            "en",
        ],
        [
            "John",
            "spk_en_el_v3_9",
            None,
            "en",
        ],
        [
            "Peter",
            "spk_en_el_v3_9",
            None,
            "en",
        ],
        [
            "حَاكَتْه",
            "spk_ar_el_v3_10",
            None,
            "en",
        ],
    ],
)

# Load all the datasets samples and split training and evaluation sets
# for dataset in config.datasets:
#     print(dataset)

train_samples, eval_samples = load_tts_samples(
    config.datasets,
    eval_split=True,
    eval_split_max_size=config.eval_split_max_size,
    eval_split_size=config.eval_split_size,
)

# Init the model
model = Vits.init_from_config(config)

# Init the trainer and 🚀
trainer = Trainer(
    TrainerArgs(
        restore_path=RESTORE_PATH if RESTORE_PATH else "",
        skip_train_epoch=SKIP_TRAIN_EPOCH,
        grad_accum_steps=4,
    ),
    config,
    output_path=EXPMT_PATH,
    model=model,
    train_samples=train_samples,
    eval_samples=eval_samples,
)
trainer.fit()
