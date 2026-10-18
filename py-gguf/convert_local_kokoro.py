#!/usr/bin/env python3
"""Convert a local kokoro-ru model dir to GGUF via TTS.cpp py-gguf encoder.

Patches hf_hub_download (kokoro.model, kokoro.pipeline, encoder) to resolve
files from a local directory instead of HuggingFace, and registers the local
dir in KModel.MODEL_NAMES so the .pth checkpoint is found locally. No network.
"""
import os
import sys
import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--model-file", required=True)
    parser.add_argument("--voices", required=True)
    parser.add_argument("--save-path", required=True)
    args = parser.parse_args()

    model_dir = os.path.abspath(args.model_dir)
    model_file = args.model_file
    voices = [v.strip() for v in args.voices.split(",") if v.strip()]
    save_path = os.path.abspath(args.save_path)

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from huggingface_hub import hf_hub_download as _orig

    def local_aware(repo_id, filename, *a, **kw):
        if isinstance(repo_id, str) and os.path.isdir(repo_id):
            p = os.path.join(repo_id, filename)
            if os.path.isfile(p):
                return p
        return _orig(repo_id, filename, *a, **kw)

    import kokoro.model
    import kokoro.pipeline
    kokoro.model.hf_hub_download = local_aware
    kokoro.pipeline.hf_hub_download = local_aware

    from kokoro.model import KModel
    KModel.MODEL_NAMES[model_dir] = model_file

    import tts_encoders.kokoro_gguf_encoder as ke
    ke.hf_hub_download = local_aware

    from tts_encoders import KokoroEncoder
    encoder = KokoroEncoder(
        model_path=save_path,
        repo_id=model_dir,
        voices=voices,
        use_espeak=True,
    )
    encoder.write()
    print("WROTE", save_path)


if __name__ == "__main__":
    main()
