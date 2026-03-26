import torch
from transformers import WhisperForConditionalGeneration, WhisperProcessor
from modeling_sa import ConditionalSpeakerGeneration
import time
import numpy as np
import torchaudio
from io import BytesIO

def load_audio(audio_path=None, audio_bytes=None, target_sample_rate=16000):
    # print(f"Loading audio from {audio_path if audio_path else 'bytes'}")
    assert audio_path is not None or audio_bytes is not None, "Either audio_path or audio_bytes must be provided"
    assert audio_path is None or audio_bytes is None, "Only one of audio_path or audio_bytes should be provided"
    if audio_bytes is not None:
        audio, sample_rate = torchaudio.load(BytesIO(audio_bytes))
    elif audio_path is not None:
        audio, sample_rate = torchaudio.load(audio_path)

    if sample_rate != target_sample_rate:
        print(f"Resampling audio from {sample_rate}Hz to {target_sample_rate}Hz")
        resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=target_sample_rate)
        audio = resampler(audio)
    return audio[0]

def initialize_model():
    cache_dir = "./cache"
    asr_model_name = 'openai/whisper-large-v2'
    print(f"Start downloading or reloading ASR model {asr_model_name}")
    asr_model = WhisperForConditionalGeneration.from_pretrained(asr_model_name, cache_dir=cache_dir).eval()
    asr_processor = WhisperProcessor.from_pretrained(asr_model_name, cache_dir=cache_dir)
    print(f"ASR model {asr_model_name} downloaded successfully")
    spk_attribute_name = "nguyenvulebinh/MSA-ASR"
    print(f"Start downloading or reloading speaker attribute model {spk_attribute_name}")
    spk_attribute_model = ConditionalSpeakerGeneration.from_pretrained(spk_attribute_name, cache_dir=cache_dir).eval()
    print(f"Speaker attribute model {spk_attribute_name} downloaded successfully")
    if torch.cuda.is_available():
        spk_attribute_model = spk_attribute_model.cuda().half()
        asr_model = asr_model.cuda().half()

    print("Models initialized successfully")
    return asr_model, asr_processor, spk_attribute_model

def extract_word_speaker_embedding(spk_attribute_model, processor, audios, audio_features, acoustic_features, words_batch, languages, use_attention_mask=True):
    device = next(spk_attribute_model.parameters()).device
    with torch.no_grad():
        if words_batch is not None:
            decoder_input_ids = processor.tokenizer.pad(processor.tokenizer.batch_encode_plus(
                [
                    f"<|startoftranscript|><|{language}|><|transcribe|><|notimestamps|> "+' '.join(words) + "<|endoftext|>" for language, words in zip(languages, words_batch)
                ], 
                add_special_tokens=False
            ), return_tensors="pt")['input_ids'].to(device)
        else:
            decoder_input_ids = None

        if use_attention_mask:
            input_lengths = torch.tensor([len(audio) for audio in audios]).to(device)
        else:
            input_lengths = None

        spk_embedding = spk_attribute_model(
            input_features=audio_features,
            acoustic_features=acoustic_features, 
            decoder_input_ids=decoder_input_ids,
            input_lengths=input_lengths
        ).logits
    
    output_ids = decoder_input_ids[:, 1:]
    spk_embedding = spk_embedding[:, :-1]
    
    # Convert all embeddings to CPU once
    spk_embedding_cpu = spk_embedding.detach().cpu().numpy()
    
    # Batch decode all tokens at once for each sequence
    batch_output = []
    for idx in range(len(output_ids)):
        # Decode all tokens at once
        tokens = processor.tokenizer.convert_ids_to_tokens(output_ids[idx].tolist())
        
        sample_words = []
        sample_word_spk_embedding = []
        current_word_tokens = []
        current_word_embeddings = []
        
        for token_idx, token in enumerate(tokens):
            # Skip empty tokens
            if not token or token in processor.tokenizer.all_special_tokens:
                continue
                
            # Check if token starts a new word (starts with space or special prefix)
            if token.startswith(' ') or token.startswith('Ġ'):  # Ġ is used by some tokenizers
                # Finalize previous word if exists
                if current_word_tokens:
                    word = processor.tokenizer.convert_tokens_to_string(current_word_tokens).strip()
                    if word:  # Only add non-empty words
                        sample_words.append(word)
                        # Average embeddings for the word
                        word_embedding = np.mean(current_word_embeddings, axis=0).tolist()
                        sample_word_spk_embedding.append(word_embedding)
                    current_word_tokens = []
                    current_word_embeddings = []
            
            current_word_tokens.append(token)
            current_word_embeddings.append(spk_embedding_cpu[idx, token_idx])
        
        # Handle last word
        if current_word_tokens:
            word = processor.tokenizer.convert_tokens_to_string(current_word_tokens).strip()
            if word:
                sample_words.append(word)
                word_embedding = np.mean(current_word_embeddings, axis=0).tolist()
                sample_word_spk_embedding.append(word_embedding)
        
        batch_output.append([sample_words, sample_word_spk_embedding])
    
    ## Check if the output is correct
    if words_batch is not None:
        for i in range(len(batch_output)):
            try:
                # print(f"{i}: {len(words_batch[i])}, {batch_output[i][0]}")
                assert len(batch_output[i]) == len(words_batch[i]), f"{len(batch_output[i])} != {len(words_batch[i])}"
                assert [w[0] for w in batch_output[i]] == words_batch[i], f"{[w[0] for w in batch_output[i]]} != {words_batch[i]}"
            except:
                pass
    return batch_output

def infer_batch(audio_wavs, transcripts, languages, asr_model, asr_processor, spk_attribute_model):
    print(f"batch: {len(languages)}; languages: {languages}; length: {[len(item) for item in transcripts]}")
    start_time = time.time()
    # get device based on the model parameters
    device = next(spk_attribute_model.parameters()).device
    dtype = next(spk_attribute_model.parameters()).dtype
    
    # Process audio features using ASR processor
    audios = [item.numpy() for item in audio_wavs]
    audio_features = asr_processor(audios, sampling_rate=16000, return_tensors='pt')
    audio_features = audio_features['input_features'].to(device, dtype)
    
    # Extract acoustic features using ASR model encoder
    with torch.no_grad():
        acoustic_features = asr_model.model.encoder(audio_features).last_hidden_state
    
    # Prepare words batch
    words_batch = [t.split() for t in transcripts]
        
    print("Audio extract feature time: {:.2f}s".format(time.time()-start_time))
    start_time = time.time()
    saasr_output = extract_word_speaker_embedding(
        spk_attribute_model=spk_attribute_model,
        processor=asr_processor,
        audios=audios,
        audio_features=audio_features,
        acoustic_features=acoustic_features,
        words_batch=words_batch,
        languages=languages
    )
    print("SAASR time: {:.2f}s".format(time.time()-start_time))
    
    return transcripts, languages, saasr_output

if __name__ == "__main__":
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