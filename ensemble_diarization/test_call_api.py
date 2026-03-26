import requests
import traceback
from typing import List, Optional

from embedding.verification import (
    format_confusion_matrix,
    verify_cluster_consistency,
)

def structure_response(output):
    # {
    #     "segments": [
    #         {
    #         "start": 0,
    #         "duration": 3.563625,
    #         "text": "Good morning, good afternoon, good evening.",
    #         "language": "en",
    #         "words": [
    #             {
    #             "word": "Good",
    #             "speaker": "SPEAKER_00",
    #             "confidence": "sent",
    #             "embedding": [
    #                 -1.923828125,
    #                 -2.044921875,
    #                 -1.3251953125,
    #                 ..
    #             ]
    #             },
    #             ...
    #         ]
    #         },
    #         ...,
    #     ],
    #     "speakers": {
    #         "SPEAKER_00": {
    #         "embeddings": [
    #             [
    #             1.2236328125,
    #             1.1669921875,
    #             -2.169921875,
    #             ...
    #             ],
    #             ...
    #         ]
    #         },
    #         "SPEAKER_01": {
    #         "embeddings": [
    #             [
    #             1.5859375,
    #             -0.08013916015625,
    #             -1.427734375,
    #             ...
    #             ],
    #             ...
    #         ]
    #         }
    #     }
    #     }

    segments = output.get("segments", [])
    speakers_map = output.get("speakers", {})
    print(f"\nReceived {len(segments)} segments, {len(speakers_map)} speakers")

    # for seg in output['segments']:
    #     for word in seg['words']:
    #         print(f"Word: {word['word']}, Speaker: {word['speaker']}, head embedding: {word['embedding'][:5]}")
    
    # for spk, embeddings in output['speakers'].items():
    #     print(f"Speaker: {spk}, Embeddings: {len(embeddings['embeddings'])}")

    # --- flatten across all segments for verification ---
    words_flat: List[str] = []
    emb_flat: List[Optional[List[float]]] = []
    spk_flat: List[Optional[str]] = []

    for seg in segments:
        for entry in seg.get("words", []):
            words_flat.append(entry["word"])
            emb_flat.append(entry.get("embedding"))
            spk_flat.append(entry.get("speaker"))

    # --- verify cluster consistency ---
    vr = verify_cluster_consistency(
        words=words_flat,
        embeddings=emb_flat,
        diarization_speakers=spk_flat,
        top_k=10,
        prototype_use="topk_mean",
    )
    print("\n=== Verification (prototype reassignment vs diarization speakers) ===")
    print(f"Evaluated words: {vr.total}")
    print(f"Correct: {vr.correct}")
    print(f"Error rate: {vr.error_rate:.4f}")
    print("\nConfusion matrix:")
    print(format_confusion_matrix(vr.confusion, vr.speakers))

def test_call():
    """Test function that records individual request timing"""
    
    audio_file = '/home/tbnguyen/workspaces/pyannote/bak/examples/session_1_output/session_1_concat.wav'
    segments_json = '/home/tbnguyen/workspaces/pyannote/bak/examples/session_1_output/segments.json'
    
    try:
        with open(audio_file,"rb") as f, open(segments_json,"r") as s:
            files = {
                "audio_wav": ("audio.wav", f.read(), "audio/wav"),
                "segments_json": ("segments.json", s.read(), "application/json"),
            }
    
        response = requests.post(f"http://0.0.0.0:5025/ensemble_diarize", files=files, headers={"accept": "application/json"})
        
        structure_response(response.json())
    except Exception as e:
        traceback.print_exc()

if __name__ == "__main__":
    test_call()