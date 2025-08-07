import torch

def generate_response(fsdp_model, raw_instruction_text, preprocessor, device, max_new_tokens = 100, temperature=0.0, top_k=None):
    
    system_text = "Below is an instruction that describes a task. Write a response that completes the request.\n\n"
    instruction_text = f"###Instruction:\n{raw_instruction_text}\n\n###Response:\n"
    input_text_with_format = f"{system_text}{instruction_text}"
    input_ids_formatted = preprocessor.encode(
        input_text_with_format, return_tensors="pt"
    ).to(device)
    print("Shape: ", input_ids_formatted.shape)

    output_ids = generate_token_ids_batch(
            fsdp_model=fsdp_model, 
            input_ids=input_ids_formatted, 
            eos_token_id = preprocessor.get_eos_token_id(), 
            device=device,
            context_size = preprocessor.get_max_length(), 
            max_new_tokens = 100, 
            temperature=0.0, 
            top_k=None)
    
    num_instruction_tokens = input_ids_formatted.shape[1]
    response_token_ids = output_ids.squeeze()[num_instruction_tokens:]
    print(response_token_ids.shape)
    return preprocessor.decode(response_token_ids)
    

def generate_token_ids_batch(fsdp_model, input_ids, eos_token_id, device,
                             context_size, max_new_tokens = 100, temperature=0.0, top_k=None): 
    
    fsdp_model.eval()
    
    for _ in range(max_new_tokens):
        # Cut beginning off prompt to fit rest in context
        input_ids = input_ids[:, -context_size:].to(device)
        
        with torch.no_grad():
            # Parameters are automatically gathered and flushed in forward pass
            logits = fsdp_model(input_ids).logits
            # For each seq in batch, select logits of last token, which is predicted token
            logits = logits[:, -1, :]
            
            # Filter logits with top_k sampling
            if top_k is not None:
                # Sets logit value of all non top_k logits to -infinity => restricts sampling to top_k tokens
                top_logits, _ = torch.topk(logits, top_k)
                min_val = top_logits[:, -1]
                logits = torch.where(
                        logits < min_val, 
                        torch.tensor(float("-inf")).to(logits.device), 
                        logits)
            
            # Smooth logit distribution by temperature
            if temperature > 0.0:
                logits = logits / temperature # dim: (batch_size, vocab_size)
                probs = torch.softmax(logits, dim = -1) # perform along vocab_size dimension (last one)
                next_id = torch.multinomial(probs, num_samples=1) # dims: (batch_size, 1)

            else:
                next_id = torch.argmax(logits, dim=-1, keepdim=True) # dim: (batch_size, 1)
            
            if torch.all(next_id == eos_token_id): # stop if all are done
                break #stop when model decides sequence is finished

            input_ids = torch.cat((input_ids, next_id), dim=1) # (batch_size, num_tokens+1)
    
    fsdp_model.train()
    
    return input_ids
