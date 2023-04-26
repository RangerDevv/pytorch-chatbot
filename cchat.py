from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("OpenAssistant/stablelm-7b-sft-v7-epoch-3")
model = AutoModelForCausalLM.from_pretrained("OpenAssistant/stablelm-7b-sft-v7-epoch-3")


# get the user input
user_input = input("You: ")

# encode the input and get the generated response
input_ids = tokenizer.encode(user_input, return_tensors="pt")
output = model.generate(input_ids, max_length=50, do_sample=True, top_p=0.95, top_k=60)


for i in range(output.shape[0]):
    print("ChatBot {}: {}".format(i+1, tokenizer.decode(output[i], skip_special_tokens=True)))
