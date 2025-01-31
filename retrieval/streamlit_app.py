import argparse
import os
from functools import partial
from pathlib import Path

import numpy as np
import streamlit as st
import torch
import torchaudio
from tqdm import tqdm

from crossmodal_alignment.retrieval_clap_model import (
    # AudioEmbeddingTorchText,
    TransformersModel,
)


@st.cache_resource
def load_model(ckpt: str | os.PathLike | None = None):
    if ckpt is None:
        model = TransformersModel()
    # else:
    #     model = AudioEmbeddingTorchText.load_from_checkpoint(ckpt)
    model.train(False)
    return model


def load_audio_input(audio_path: Path, sampling_rate: int):
    if audio_path.suffix == ".npy":
        return torch.from_numpy(np.load(audio_path))
    else:
        audio, sr = torchaudio.load(audio_path)
        audio = torchaudio.functional.resample(audio, sr, sampling_rate)
        return audio.mean(0)


# @st.cache_data
# def build_audio_index(root_dir: Path, _audio_encoder, pattern: str = "*.wav", **kwargs):
#     file_names = []
#     audios = []
#     for file in tqdm(root_dir.rglob(pattern)):
#         with torch.inference_mode():
#             input_audio = load_audio_input(file, **kwargs)
#             embedded_audio = _audio_encoder(input_audio)
#         audios.append(embedded_audio)
#         file_names.append(file.name)
#     return torch.stack(audios), file_names

@st.cache_data
def build_audio_index(root_dir: Path, _audio_encoder, pattern: str = "*.npy", **kwargs):
    file_names = []
    audios = []
    for file in tqdm(root_dir.rglob(pattern)):
        # with torch.inference_mode():
        embedded_audio = load_audio_input(file, **kwargs)
            # embedded_audio = _audio_encoder(input_audio)
        audios.append(embedded_audio)
        file_names.append(file.name)
    return torch.stack(audios), file_names


def map_file_path(
    path_to_map: Path, source_root: Path, target_root: Path, new_ext: str | None = None
):
    if path_to_map.is_relative_to(source_root):
        sub_path = path_to_map.relative_to(source_root)
    else:
        sub_path = path_to_map
    new_path = target_root / sub_path
    if new_ext:
        return new_path.with_suffix(new_ext)
    return new_path


def main(model, name_to_result_mapping):
    st.title("Cross-modal Search Demo")

    query = st.text_input(
        "Enter your search query:",
        max_chars=128,
        placeholder="e.g. a dog barks and a duck quacks",
    )
    query = query.strip()

    with st.sidebar:
        k = st.number_input(
            "Number of results returned", value=10, min_value=1, max_value=100
        )
        st.write(f"Number of items in dataset {len(ref_names)}")

    if query:
        st.header(f"Top {k} results")
        with torch.inference_mode():
            embedded_query = model.get_text_embedding(query)
        similarities = torch.cosine_similarity(embedded_query, ref_audios)
        matches, match_indices = torch.topk(similarities, k=10)
        for match, idx in zip(matches, match_indices.tolist()):
            result_path = Path(ref_names[idx])
            base_name = os.path.splitext(result_path)[0]
            st.write(f"{base_name}")
            # st.write(f"{result_path}")
            st.audio(str(name_to_result_mapping(result_path)))
            st.caption(f"Score: {match}")


desc = """Run a retrieval interface app to test a text-to-audio search system.

The model loaded from CKPT_PATH indexes preprocessed audio files from DATA_DIR.
Query results are displayed by mapping paths of matching files to AUDIO_DIR.
"""

parser = argparse.ArgumentParser(description=desc)
parser.add_argument(
    "data_dir",
    help="Root directory of all audio inputs to be indexed by the model",
    type=Path,
)
parser.add_argument(
    "audio_dir",
    help="Root directory of all audio inputs used to display as results",
    type=Path,
)
parser.add_argument(
    "--ckpt_path", help="Path to a checkpoint to load the model from", type=Path
)
try:
    args = parser.parse_args()
except:
    st.code(parser.format_help())
    raise

model = load_model(args.ckpt_path)
ref_audios, ref_names = build_audio_index(
    args.data_dir, model.get_audio_embedding, sampling_rate=model.sampling_rate
)
name_to_result_mapping = partial(
    map_file_path, source_root=args.data_dir, target_root=args.audio_dir, new_ext=".wav"
)

main(model, name_to_result_mapping)
