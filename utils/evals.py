from transformers import AutoTokenizer, AutoModelForCausalLM
import torch    

def load_model(dir, device = 'cuda'):
    tokenizer = AutoTokenizer.from_pretrained(dir, trust_remote_code = True)
    model = AutoModelForCausalLM.from_pretrained(dir, trust_remote_code = True)
    return tokenizer, model