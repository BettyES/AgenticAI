# AgenticAI
Build my own research agent based on the course deeplearningAI AgenticAI

To adapt the DeepLearning.AI "Agentic AI" course notebooks to use a local LLM via Ollama instead of the default OpenAI API, you can take advantage of Ollama's native compatibility with the OpenAI completions format. The process involves setting up Ollama, downloading your desired local model, and modifying the API calls within the course notebooks. 

## Step 1: Install and set up Ollama 
- **Install Ollama:** Follow the installation instructions for your operating system by downloading the application from the official website, ollama.com.\
- **Pull a model:** Open your terminal and download a model from Ollama's library. For an agentic workflow, a more capable instruction-following model like Llama 3 is a good choice.

```bash
ollama run llama3
```

This command will download the model and start an interactive chat. To exit, type /bye or press Ctrl+D.\
- **Start the Ollama server:** Ensure the Ollama server is running in the background. On macOS, this happens automatically when the app is running. On other systems, you may need to start it with ollama serve. 

## Step 2: Configure the Ollama client 
Ollama exposes a server that can be treated as a drop-in replacement for the OpenAI API endpoint. 
- **Install the OpenAI Python library:** If you haven't already, install the openai library in your local environment.

```bash
pip install openai
```

- **Modify the API client:** In your Jupyter notebooks, you'll need to change how the client is initialized. Replace the standard client initialization with one that points to the local Ollama server.

```python
# Before (using default OpenAI client)
# from openai import OpenAI
# client = OpenAI()

# After (using Ollama server as an OpenAI-compatible endpoint)
from openai import OpenAI
client = OpenAI(
    base_url="http://localhost:11434/v1",
    api_key="ollama" # api_key is not required, but this is a common placeholder
)
```
## Step 3: Update LLM calls in the notebooks 
For agentic workflows, the model's ability to output structured data (like JSON) is crucial for tool use and reflection. Ollama can be configured to produce JSON responses. \
- **Modify LLM calls:** Update any calls to client.chat.completions.create to specify your local Ollama model and request JSON output where needed.

```python
# Original API call example
# chat_response = client.chat.completions.create(
#     model="gpt-4-turbo",
#     messages=[...],
#     response_format={"type": "json_object"},
# )

# Modified API call for Ollama
chat_response = client.chat.completions.create(
    model="llama3", # Use the name of your local model
    messages=[...],
    response_format={"type": "json_object"},
)
```

**Adjust tool definitions and prompts:** Review the course notebooks to see how agent tools are defined and what format they expect. You may need to adapt your prompts slightly to ensure the local model correctly follows instructions, particularly when complex tool calls are involved. 

## Step 4: Handle external dependencies and compatibility issues 
Running locally presents unique challenges compared to the course's cloud-based setup. 
- **Search tools:** The notebooks often rely on external tools like Tavily for web searches. You will still need to provide an API key for this service, as it can't be run locally.\
- **Model performance:** Local LLMs, especially smaller 7B–13B models, are much faster than their large counterparts but may be less capable. This could affect the agent's ability to reason, plan, and use tools accurately. You may need to use a smaller, faster model for prototyping and a larger, more capable one for later experiments.\
- **Structured output:** Not all local models are equally proficient at reliably generating JSON output. Some open-source models may require more explicit prompting to ensure they produce the correct format for tool use. If a model is consistently failing, try pulling a different one from Ollama's library. 


test


