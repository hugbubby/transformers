from transformers import GPT2TokenizerFast

ends = ['!', '.', '?', '."']

quotation_chars = ['"', "'"]

brace_chars = {'{': '}', '(': ')', '[': ']'}

exception_ends = [*map(lambda end: end + ".", 
        [
            '.' #Unfinished Ellipsis
            'Dr', 'Mr', 'Ms', 'Mrs',  #No titles
            *map(lambda i: str(i), range(10))]) #Or numbers, which can come with decimal points.
        ]

#TODO: ML
default_tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')

def is_sentence_tokens(tokens, tokenizer=default_tokenizer):
    text: str = tokenizer.decode(tokens)
    out = not any( #No exception ending
        map(lambda ending: text.endswith(ending), exception_ends) #No non-end-of-sentence-periods
    ) and not any(
        map(lambda qc: text.count(qc) % 2 != 0, quotation_chars) #No unfinished quotations
    ) and not any(
        map(lambda bck: bck in text and not brace_chars[bck] in text, brace_chars) #No unfinished brace characters.
    ) and any(
        map(lambda ending: text.endswith(ending), ends),
    )
    return out
