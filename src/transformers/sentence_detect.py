from transformers import GPT2TokenizerFast

ends = ['!', '.', '?']
ends.extend(map(lambda end: end + '"', ends))

quotation_chars = ['"', "'"]

brace_chars = {'{': '}', '(': ')', '[': ']'}

exception_ends = [*map(lambda end: end + ".", 
        [
            '.' #Unfinished Ellipsis
            'Dr', 'Mr', 'Ms', 'Mrs',  #No titles
            *map(lambda i: str(i), range(10))]) #Or numbers, which can come with decimal points.
        ]

default_tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')

def is_sentence_tokens(tokens, tokenizer=default_tokenizer):
    text: str = tokenizer.decode(tokens)
    print("Token accum (sentence check): ", text)
    out = not any(
        map(lambda ending: text.endswith(ending), exception_ends) #No non-end-of-sentence-periods
    ) and any(
        map(lambda ending: text.endswith(ending), ends),
    )
    return out
