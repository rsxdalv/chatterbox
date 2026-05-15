# Chatterbox Multilingual Runtime And Benchmark Notes

Updated: 2026-05-15

## Runtime shape

- API: `http://127.0.0.1:8010`
- Endpoint: `POST /v1/audio/speech`
- Queue: `chatterbox_mtl_3060x2`
- Workers: 2 total, one on physical GPU `2`, one on physical GPU `3`
- GPU policy: RTX 3060 only, with `EXPECTED_GPU_NAME=RTX 3060`
- Lazy behavior: API stays CPU-side; workers load on first request and unload after `300` seconds idle
- VRAM gate: `MIN_FREE_VRAM_MB=4500` before model load
- Chunking: Spanish sentence-boundary split, dynamic Celery dispatch, ordered PCM stitching in the API process
- Result collection: chunk task results are polled with `ready()` and read from `task.result`, avoiding Celery Redis `task.get()` hangs on large fan-out

## 100-sentence Spanish benchmark

Test input: 100 Spanish dot-terminated sentences.

| Metric | Value |
| --- | ---: |
| Audio duration | `541.180s` |
| Client wall time | `173.295s` |
| Client speed | `3.1229x` realtime |
| Server speed | `3.1237x` realtime |
| RTF | `0.3202` |
| Chunks / tasks | `100 / 100` |
| Worker split | `pid:1305231=52`, `pid:1305277=48` |
| ASR speed on Parakeet `:5092/v1` | `22.5024x` realtime |
| ASR WER | `1.0200` |

Output files:

- `/home/op/tts_unified_benchmark_outputs/chatterbox_multilingual_100_sentences.wav`
- `/home/op/tts_unified_benchmark_outputs/chatterbox_multilingual_100_sentences_asr.txt`
- `/home/op/tts_unified_benchmark_outputs/unified_tts_100_sentence_benchmark_results.json`

## Notes

- Distribution across the two workers is balanced; model generation is the bottleneck.
- This path is slower than Turbo on the same 3060 pair but now has the same lazy Celery behavior and VRAM gate.
- vLLM2 correctly refused to load immediately after this benchmark while this model was resident and free VRAM was below its configured gate.
