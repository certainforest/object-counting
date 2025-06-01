from transformers import AutoTokenizer, AutoModelForCausalLM  
from jinja2 import Template 

def load_model(dir, device = 'cuda'):
    tokenizer = AutoTokenizer.from_pretrained(dir, trust_remote_code = True)
    model = AutoModelForCausalLM.from_pretrained(dir, trust_remote_code = True)
    model = model.to(device)
    return tokenizer, model

# templating + instruct formatting 
def create_counting_prompt(entity_type, word_list, tokenizer, device):
    """
    Generate the counting prompt. 

    Args: 
        entity_type: a string identifying the type of entity to count (e.g. 'fruit' or 'animal')
        word_list: list of words or string representation

    Returns: 
        str: formatted prompt 
    """

    user_template = """Count the number of words in the following  list that match the given type, and put the numerical answer in parentheses.
        Type: {{ entity_type }}
        List: [{{ word_list | list }}]
        """ 
    template = Template(user_template)
    user_content = template.render(entity_type = entity_type, word_list = word_list)

    base_prompt = [
        {'role': 'user', 'content': user_content},
        {'role': 'assistant', 'content': 'Answer: ('}

    ]

    output = tokenizer.apply_chat_template(
        base_prompt, 
        # return_tensors = 'pt',
        tokenize = False,
        add_generation_prompt = False,
        continue_final_message = True)

    return output


