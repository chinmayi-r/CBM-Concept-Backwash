from openai import OpenAI

client = OpenAI(
    api_key="hHXRYSsR5B/hP7jVdYpxihi7Pr8L",
    base_url="https://api.portkey.ai/v1",  # confirm this with OIT if it fails
)

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "Say hello in one sentence."}],
    max_tokens=50,
)

print(response.choices[0].message.content)
print("Tokens used:", response.usage.total_tokens)
