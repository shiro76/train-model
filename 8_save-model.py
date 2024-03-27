# Sauvegarde du modèle avec les adaptateurs LoRA
model.save_pretrained("lora_model")

# Sauvegarde du modèle avec la quantification Q4_K_M
model.save_pretrained_gguf("q4_k_m_model", tokenizer, quantization_method="q4_k_m")


# model.save_pretrained("lora_model") # Local saving


# # Merge to 16bit
# if False: model.save_pretrained_merged("model", tokenizer, save_method = "merged_16bit",)
# if False: model.push_to_hub_merged("hf/model", tokenizer, save_method = "merged_16bit", token = "")

# # Merge to 4bit
# if False: model.save_pretrained_merged("model", tokenizer, save_method = "merged_4bit",)
# if False: model.push_to_hub_merged("hf/model", tokenizer, save_method = "merged_4bit", token = "")

# # Just LoRA adapters
# if False: model.save_pretrained_merged("model", tokenizer, save_method = "lora",)
# if False: model.push_to_hub_merged("hf/model", tokenizer, save_method = "lora", token = "")

# # Save to 8bit Q8_0
# if False: model.save_pretrained_gguf("model", tokenizer,)
# if False: model.push_to_hub_gguf("hf/model", tokenizer, token = "")

# # Save to 16bit GGUF
# if False: model.save_pretrained_gguf("model", tokenizer, quantization_method = "f16")
# if False: model.push_to_hub_gguf("hf/model", tokenizer, quantization_method = "f16", token = "")

# # Save to q4_k_m GGUF
# if False: model.save_pretrained_gguf("model", tokenizer, quantization_method = "q4_k_m")
# if False: model.push_to_hub_gguf("hf/model", tokenizer, quantization_method = "q4_k_m", token = "")