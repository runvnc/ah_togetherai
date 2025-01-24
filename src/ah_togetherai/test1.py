from together import Together

client = Together()

response = client.chat.completions.create(
    model="deepseek-ai/DeepSeek-R1",
    messages=[{"role": "user", "content": "Hello there"}], # "What are some fun things to do in New York?"}],
)
print(response.choices[0].message.content)
