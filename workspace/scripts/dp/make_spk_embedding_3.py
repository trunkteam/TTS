import os

from TTS.bin.compute_embeddings_v2 import compute_embeddings
from TTS.config.shared_configs import BaseDatasetConfig

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
RESTORE_PATH = os.path.join(BASE_PATH, "models/ytts/v3_best_ckpt.pth")


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


dataset = get_dataset(manifest_train="manifest_dur.json",
                      manifest_eval="manifest_eval_dur.json",
                      d_name="ar_el_gen_v5",
                      lang="ar")

DATASETS_CONFIG_LIST = [dataset]

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
