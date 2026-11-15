# ShortyBird config

- `providers.json`: provider URLs, extension repos, Stremio add-ons and smart-search locations, read by the app at startup.
- Releases: `story-index` (shared story index with vectors), `model-v2` (smart-search models), `model-minilm-v1` (old model).
- `tools/build_shared_index.py`: rebuilds `story_index_shared.db` from a phone's story index (`pip install onnxruntime numpy tokenizers`), then upload it to the `story-index` release.
