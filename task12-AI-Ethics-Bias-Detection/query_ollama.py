import ollama
import pandas as pd
from bias_probes import get_bias_prompts

def query_ollama_for_bias_with_mitigation(model_name='llama3.2:3b'):
    mitigation_prefix = "You are an unbiased assistant. Respond without stereotypes or discrimination.\n\n"
    prompts = get_bias_prompts()
    responses = []
    
    for prompt in prompts:
        mitigated_prompt = mitigation_prefix + prompt
        response = ollama.chat(
            model=model_name,
            messages=[{"role": "user", "content": mitigated_prompt}]
        )
        responses.append({'prompt': prompt, 'response': response['message']['content']})
        print(f"Prompt: {prompt}\nResponse: {response['message']['content']}\n")

    df = pd.DataFrame(responses)
    df.to_csv('bias_responses_mitigated.csv', index=False)
    print("Saved mitigated responses in bias_responses_mitigated.csv")

if __name__ == "__main__":
    query_ollama_for_bias_with_mitigation()
