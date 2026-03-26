import requests
import traceback

def structure_response(output):
    print("Total words output: ", len(output['saasr'][0]))
    # for word, embed in zip(output['saasr'][0], output['saasr'][1]):
    #     print(f"Word: {word}\n\tEmbedding size: {len(embed)}, head: {embed[:5]}")
    

def test_call():
    """Test function that records individual request timing"""
    
    audio_file = 'test_audio.wav'
    audio_transcript = "I have never withheld instruction from any. And I don't think anybody's ever criticized me for being shy about my opinions. I've never really had a problem saying anything to anyone as an adult. I have withheld teaching from no one. Eight." * 10
    language = "en"
    print("Total words input: ", len(audio_transcript.split()))
    try:
        with open(audio_file,"rb") as f:
            d = {'pcm_s16le': f.read(), 'transcript': audio_transcript}
        response = requests.post(f"http://192.168.0.66:6024/asr/infer/{language},None", files=d)
        
        structure_response(response.json())
    except Exception as e:
        traceback.print_exc()

if __name__ == "__main__":
    test_call()