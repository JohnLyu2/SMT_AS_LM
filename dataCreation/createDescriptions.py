from openai import OpenAI

# API configuration
api_key = '46b575d32e86d56355dc59dbdcc497c0'  # Replace with your API key
base_url = "https://chat-ai.academiccloud.de/v1"
model = "meta-llama-3.1-8b-instruct"  # Choose any available model

# Start OpenAI client
client = OpenAI(
    api_key=api_key,
    base_url=base_url
)
content = """  You are a helpful assistant and an expert in SMT Instances. Here is the content of a smt file. Please epxlain it as thoroughly as possible."""
# Get response
chat_completion = client.chat.completions.create(
    messages=[{"role": "system", "content": content},],
    model=model,
)

# Print full response as JSON
print(chat_completion)  #