import ollama
import pandas as pd
from bias_probes import get_bias_prompts

def query_ollama_for_bias(model_name='llama3.2:3b'):
    prompts = get_bias_prompts()
    responses = []
    
    for prompt in prompts:
        response = ollama.chat(
            model=model_name,
            messages=[{"role": "user", "content": prompt}]
        )
        responses.append({'prompt': prompt, 'response': response['message']['content']})
        print(f"Prompt: {prompt}\nResponse: {response['message']['content']}\n")

    df = pd.DataFrame(responses)
    df.to_csv('bias_responses.csv', index=False)
    print("Saved original responses in bias_responses.csv")

if __name__ == "__main__":
    query_ollama_for_bias()
