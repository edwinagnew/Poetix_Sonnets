import gpt_revised

gpt = gpt_revised.gpt_gen()

b = 5

seeds = ["A passion toiled by a strong summertime\n"] # TODO: replace with our corpus

for input_seed in seeds:

    input_ids = gpt.tokenizer(input_seed, return_tensors="pt").input_ids
    outputs = gpt.model.generate(input_ids=input_ids, num_beams=b, num_return_sequences=1,
                                max_length=175,  no_repeat_ngram_size=4)


    lines = gpt.tokenizer.batch_decode(outputs, skip_special_tokens=True)

    print(lines[0])
