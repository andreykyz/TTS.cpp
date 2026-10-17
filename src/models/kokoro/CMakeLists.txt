target_sources(tts PRIVATE
        loader.cpp
        model.cpp
        model.h
        phonemizer.cpp
        phonemizer.h
        ru_frontend.cpp
        ru_frontend.h
        ru_tokenizer.cpp
        ru_tokenizer.h
        ru_dict.cpp
        ru_dict.h
        ru_encoder.cpp
        ru_encoder.h
        ru_accent.cpp
        ru_accent.h
)

# The encoder's GEMM uses AVX2/FMA intrinsics (guarded by __AVX2__); the rest
# of the target is built portable.
set_source_files_properties(ru_encoder.cpp PROPERTIES COMPILE_OPTIONS "-mavx2;-mfma")
