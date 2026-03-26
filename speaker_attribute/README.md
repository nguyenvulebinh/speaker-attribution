# MSA-ASR Service

### 1. Build with Custom Image Name


```bash
# Build with version tag
docker build -t spk-attribute:v1.0 .

# Run the named image
docker run --gpus all -p 5024:5024 \
  -d \
  -e CUDA_VISIBLE_DEVICES=2 \
  --name spk-attribute \
  spk-attribute:v1.0
```

### 2. Test the Service

```bash
curl -X POST "http://0.0.0.0:5024/asr/infer/en,None" \
  -F "pcm_s16le=@test_audio.wav" \
  -F "transcript=@test_audio.txt"
```

Note: Ensure audio is 16Khz, pcm_s16le codec.

```json
{
  "hypo": "<text transcript>", 
  "lid": "en", 
  "saasr": [[<word-0>, <word-1>, ...], [[<spk-embedding-word-0>], [<spk-embedding-word-1>], ...]],
}
```

### 3. Run inference without flask interface

Follow the `inference.py` script:

```python
# Initialize models
asr_model, asr_processor, spk_attribute_model = initialize_model()

# Load audio file
audio_file = 'test_audio.wav'
audio_transcript = "I have never withheld instruction from any. And I don't think anybody's ever criticized me for being shy about my opinions. I've never really had a problem saying anything to anyone as an adult. I have withheld teaching from no one. Eight."

# Set input language
language = "en"

# Run inference
hypos, lids, saasr_outputs = infer_batch(
    audio_wavs=[load_audio(audio_file)],
    transcripts=[audio_transcript],
    languages=[language],
    asr_model=asr_model,
    asr_processor=asr_processor,
    spk_attribute_model=spk_attribute_model
)

# Print result
result = {"hypo": hypos[0].strip(), "lid": lids[0], "saasr": saasr_outputs[0]}
for word, embed in zip(result['saasr'][0], result['saasr'][1]):
    print(f"Word: {word}\n\tEmbedding size: {len(embed)}, head: {embed[:5]}")
```