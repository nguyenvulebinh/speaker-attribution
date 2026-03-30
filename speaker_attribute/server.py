# import os
# os.environ["CUDA_VISIBLE_DEVICES"]="7"
from flask import Flask, request
import os
import torch
import numpy as np
import math
import sys
import json
import threading
import queue
import uuid
import traceback
from transformers import WhisperForConditionalGeneration, WhisperProcessor
from modeling_sa import ConditionalSpeakerGeneration
import time

host = os.getenv("HOST", "0.0.0.0")
port = int(os.getenv("PORT", "5024"))

app = Flask(__name__)
trusted_hosts = os.getenv("FLASK_TRUSTED_HOSTS")
app.config["TRUSTED_HOSTS"] = trusted_hosts.split(",") if trusted_hosts else None

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

        # Handle the case where the decoder input ids is too long
        max_decoder_input_length = spk_attribute_model.config.max_length
        current_decoder_input_length = decoder_input_ids.size(1)
        truncate_decoder_input = False
        if current_decoder_input_length > max_decoder_input_length:
            print(f"WARNING: Current decoder input length {current_decoder_input_length} is greater than max decoder input length {max_decoder_input_length}. Truncating the decoder input ids.")
            truncated_decoder_input_ids = decoder_input_ids[:, :max_decoder_input_length]
            truncate_decoder_input = True
        else:
            truncated_decoder_input_ids = decoder_input_ids

        spk_embedding = spk_attribute_model(
            input_features=audio_features,
            acoustic_features=acoustic_features, 
            decoder_input_ids=truncated_decoder_input_ids,
            input_lengths=input_lengths
        ).logits


    if truncate_decoder_input:
        # duplicate the spk_embedding to the padding length
        number_of_duplicates = math.ceil(current_decoder_input_length / max_decoder_input_length)
        spk_embedding = spk_embedding.repeat(1, number_of_duplicates, 1)[:, :current_decoder_input_length, :]
        print(f"WARNING: Duplicated the spk_embedding to {number_of_duplicates} times to match the decoder input length {current_decoder_input_length}.")

    
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


def create_unique_list(my_list):
    my_list = list(set(my_list))
    return my_list

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
    max_batch_size = 32
    print("Models initialized successfully")
    return (asr_model, asr_processor, spk_attribute_model), max_batch_size

def add_prefix_tokens(processor, prefix, forced_decoder_ids):
    if len(prefix) > 0:
        prompt_ids = processor.get_prompt_ids(prefix).tolist()[1:]
        for wid in prompt_ids:
            forced_decoder_ids.append((len(forced_decoder_ids) + 1, wid))

def infer_batch(audio_wavs, transcripts, languages):
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

def use_model(reqs):
    if len(reqs) == 1:
        req = reqs[0]
        audio_tensor, prefix, transcript, input_language, output_language = req.get_data()
        
        # For single request, just use the input language
        hypos, lids, saasr_outputs = infer_batch(
            audio_wavs=[audio_tensor],
            transcripts=[transcript],
            languages=[input_language]
        )
            
        result = {"hypo": hypos[0].strip(), "lid": lids[0], "saasr": saasr_outputs[0]}
        req.publish(result)
    else:
        audio_tensors = []
        transcripts = []
        input_languages = []

        for req in reqs:
            audio_tensor, prefix, transcript, input_language, output_language = req.get_data()
            audio_tensors.append(audio_tensor)
            transcripts.append(transcript)
            input_languages.append(input_language)

        # Process all requests in batch
        hypos, lids, saasr_outputs = infer_batch(
            audio_wavs=audio_tensors,
            transcripts=transcripts,
            languages=input_languages
        )

        for req, hypo, lid, saasr_output in zip(reqs, hypos, lids, saasr_outputs):
            result = {"hypo": hypo.strip(), "lid": lid, "saasr": saasr_output}
            req.publish(result)

def run_decoding():
    while True:
        reqs = [queue_in.get()]
        while not queue_in.empty() and len(reqs) < max_batch_size:
            req = queue_in.get()
            reqs.append(req)
            if req.priority >= 1:
                break

        print("Batch size:",len(reqs),"Queue size:",queue_in.qsize())

        try:
            use_model(reqs)
        except Exception as e:
            print("An error occured during model inference")
            traceback.print_exc()
            for req in reqs:
                req.publish({"hypo":"", "status":400})

class Priority:
    next_index = 0

    def __init__(self, priority, id, condition, data):
        self.index = Priority.next_index

        Priority.next_index += 1

        self.priority = priority
        self.id = id
        self.condition = condition
        self.data = data

    def __lt__(self, other):
        return (-self.priority, self.index) < (-other.priority, other.index)

    def get_data(self):
        return self.data

    def publish(self, result):
        dict_out[self.id] = result
        try:
            with self.condition:
                self.condition.notify()
        except:
            print("ERROR: Count not publish result")

def pcm_s16le_to_tensor(pcm_s16le):
    audio_tensor = np.frombuffer(pcm_s16le, dtype=np.int16)
    audio_tensor = torch.from_numpy(audio_tensor)
    audio_tensor = audio_tensor.float() / math.pow(2, 15)
    audio_tensor = audio_tensor.unsqueeze(1)  # shape: frames x 1 (1 channel)
    return audio_tensor

# corresponds to an asr_server "http://$host:$port/asr/infer/en,en" in StreamASR.py
# use None when no input- or output language should be specified
@app.route("/asr/infer/<input_language>,<output_language>", methods=["POST"])
def inference(input_language, output_language):
    try:
        pcm_file = request.files.get("pcm_s16le")
        if pcm_file is None:
            raise KeyError("pcm_s16le missing from multipart")
        pcm_s16le: bytes = pcm_file.read()
    except Exception as e:
        files_keys = list(request.files.keys())
        print(
            "[speaker_attribute] 400 missing/unreadable pcm_s16le "
            f"files={files_keys} content_length={request.content_length} "
            f"content_type={request.content_type} err={type(e).__name__}: {e}"
        )
        return json.dumps({
            "hypo": "",
            "status": 400,
            "error": "missing_or_invalid_pcm_s16le",
            "files": files_keys,
            "content_length": request.content_length,
            "content_type": request.content_type,
            "exception": type(e).__name__,
        }), 400
    prefix = request.files.get("prefix") # can be None
    if prefix is not None:
        prefix: str = prefix.read().decode("utf-8")
        
    transcript = request.files.get("transcript") # can be None
    if transcript is not None:
        transcript: str = transcript.read().decode("utf-8")

    # calculate features corresponding to a torchaudio.load(filepath) call
    audio_tensor = pcm_s16le_to_tensor(pcm_s16le).squeeze()

    priority = request.files.get("priority") # can be None
    try:
        priority = int(priority.read()) # used together with priority queue
    except:
        priority = 0

    condition = threading.Condition()
    with condition:
        id = str(uuid.uuid4())
        data = (audio_tensor,prefix,transcript,input_language,output_language)

        queue_in.put(Priority(priority,id,condition,data))

        condition.wait()

    result = dict_out.pop(id)
    status = 200
    if status in result:
        status = result.pop(status)

    # result has to contain a key "hypo" with a string as value (other optional keys are possible)
    return json.dumps(result), status

# called during automatic evaluation of the pipeline to store worker information
@app.route("/asr/version", methods=["GET","POST"])
def version():
    # return dict or string (as first argument)
    return "Whisper large v2", 200

@app.route("/asr/available_languages", methods=["GET","POST"])
def languages():
    langs = [x[2:-2] for x in asr_processor.tokenizer.additional_special_tokens if len(x)==6]
    return langs

(asr_model, asr_processor, spk_attribute_model), max_batch_size = initialize_model()

queue_in = queue.PriorityQueue()
dict_out = {}

decoding = threading.Thread(target=run_decoding)
decoding.daemon = True
decoding.start()

if __name__ == "__main__":
    app.run(host=host, port=port)
