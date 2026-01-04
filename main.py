import requests
from config import HF_API_KEY

def classify_text(text):
    url = "https://router.huggingface.co/hf-inference/models/distilbert/distilbert-base-uncased-finetuned-sst-2-english"
    headers = {"Authorization": f"Bearer {HF_API_KEY}"}
    payload = {"inputs": text}
    response = requests.post(url, headers=headers, json=payload)
    try:
        return response.json()
    except requests.exceptions.JSONDecodeError:
       print(f"Requests failed with status code {response.status_code}")
       print(f"Response text: {response.text}")
       return{}
    
def print_results(data):
    if not data:
        return
    try:
        result=data[0] if isinstance(data, list) else data
        if isinstance(result,list):
            result=result[0]

        label = result.get("label", "UNKNOWN")
        score = result.get("score", 0)
        percentage=score*100
        emoji="😊" if label=="POSITIVE" else 'fw'
        emoji="😊" if label=="POSITITVE" else'ki'
        icon="✅" if label == "POSITIVE" else "❌"

        print("-" * 30)
        print(f"Sentiments: {icon} {label} {emoji}")
        print(f"Confidence: {percentage:.2f}%")
        print("-" * 30)
    except (KeyError, IndexError) as e:
        print(f"[!]Could not parse results: {data}")

def main():
    print("====AI Sentiment Analyser(Type 'quit to exit')=====")
    while True:
        user_input = input("Enter text to classify: ")
        if user_input.lower() == "quit":
                break
        results = classify_text(user_input)
        print_results(results)

        print("Analysing.....", end="\r")
        data = classify_text(user_input)
        print_results(data)

if __name__ == "__main__":
        main()