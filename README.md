# Mr Secret Agent

This is a cli chat client for enriching AI context utilizing RAG, faiss and
Langchain.

## Background and Goal

Some classes are extremely textbook heavy and sometimes those textbooks are
filled with nonsense (filler) so I decided to do a very basic RAG setup, where I
generate a faiss index for an entire textbook then allow you to chat with AI
with this added context.

The entire setup is completely local using lmstudio, you could probably make some minor modifications to use ollama as well, required models for this are:
    - openai/gpt-oss-20b
    - text-embedding-nomic-embed-text-v1.5@q4_k_s
ensure that both models are loaded at the same time in lmstudio.