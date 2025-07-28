from openai import OpenAI
client = OpenAI(api_key=open(r"C:\Users\artio\OneDrive\Desktop\aiskills\OPENAI_KEY.txt").read().strip())

def openai_embed(to_embed):
    if isinstance(to_embed, str):
        inputs = [to_embed]
    elif isinstance(to_embed, list) and all(isinstance(x, str) for x in to_embed):
        inputs = to_embed
    else:
        raise ValueError("openai_embed input must be a string or list of strings")

    response = client.embeddings.create(
        input=inputs,
        model="text-embedding-3-large"
    )

    embeddings = [d.embedding for d in response.data]
    return embeddings[0] if isinstance(to_embed, str) else embeddings