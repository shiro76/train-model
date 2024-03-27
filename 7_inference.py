from data_prep import alpaca_prompt
from load_model import tokenizer
from train_model import model
from unsloth import FastLanguageModel

# alpaca_prompt = Copied from above
FastLanguageModel.for_inference(model) # Enable native 2x faster inference
inputs = tokenizer(
[
    alpaca_prompt.format(
        "structure pour les texte de Dragon Ball Online.", # instruction
        "", # input
        "", # output - leave this blank for generation!
    )
], return_tensors = "pt").to("cuda")

outputs = model.generate(**inputs, max_new_tokens = 64, use_cache = True)

# Decode and print the output
decoded_output = tokenizer.batch_decode(outputs)
print(decoded_output)
