ARG BASE=nvidia/cuda:12.2.0-base-ubuntu22.04
ARG DEBIAN_FRONTEND=noninteractive
FROM ${BASE}
RUN apt-get update && apt-get upgrade -y
RUN apt-get install -y --no-install-recommends gcc g++ make python3 python3-dev python3-pip python3-venv python3-wheel espeak-ng libsndfile1-dev ffmpeg tmux htop ffmpeg sox libsox-dev lame git && rm -rf /var/lib/apt/lists/*
RUN python3 -m pip install -U pip
RUN python3 -m pip install llvmlite pyannote.audio elevenlabs stable-ts nemo-text-processing --ignore-installed

WORKDIR /root
COPY . /root
RUN python3 -m pip install torch torchaudio
RUN rm -rf /root/.cache/pip
RUN make install
ENTRYPOINT ["tts"]
CMD ["--help"]
