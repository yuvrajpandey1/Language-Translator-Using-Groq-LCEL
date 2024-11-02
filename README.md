
# Language Translator using ChatGroq and ChatOpnAI





# Documentation 

[ChatOpenAI](https://js.langchain.com/v0.2/docs/integrations/text_embedding/)

Embedding models create a vector representation of a piece of text.This page documents integrations with various model providers that allow you to use embeddings in LangChain.

Embedding models are often used in retrieval-augmented generation (RAG) flows, both as part of indexing data as well as later retrieving it.You can embed queries for search with embedQuery. This generates a vector representation specific to the query

[ChatGroq](https://python.langchain.com/docs/integrations/vectorstores/faiss/)

Facebook AI Similarity Search (FAISS) is a library for efficient similarity search and clustering of dense vectors. It contains algorithms that search in sets of vectors of any size, up to ones that possibly do not fit in RAM. It also includes supporting code for evaluation and parameter tuning.

Faiss is written in C++ with complete wrappers for Python. Some of the most useful algorithms are implemented on the GPU. It is developed primarily at FAIR, the fundamental AI research team of Meta.












 








## Important Libraries Used


 - [PromptTemplate](https://python.langchain.com/v0.1/docs/modules/model_io/prompts/quick_start/)


- [StrOutputParser](https://python.langchain.com/api_reference/core/output_parsers/langchain_core.output_parsers.string.StrOutputParser.html)








## Plateform or Providers

 - [langchain_groq](https://python.langchain.com/docs/integrations/chat/groq/)


## Model

 - LLM - Gemma2-9b-It


## Installation

Install below libraries

```bash
  pip install langchain
  pip install langchain_community
  pip install langchain-core
  pip install langchain_openai
  pip install langchain-groq
 


```
    
## Tech Stack

**Client:** Python, LangChain PromptTemplate, OpenAI,ChatOpenAI,ChatGroq

**Server:** Anaconda Navigator, Jupyter Notebook


## Environment Variables

To run this project, you will need to add the following environment variables to your .env file

`API_KEY`

`GROQ_API_KEY`
`OPENAI_API_KEY`



## Examples

Instantiate our model object and generate chat completions

```javascript
from langchain_groq import ChatGroq

llm = ChatGroq(
    model="mixtral-8x7b-32768",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
    # other params...
)
```

We can chain our model with a prompt template like so 
```javascript
from langchain_core.prompts import ChatPromptTemplate

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful assistant that translates {input_language} to {output_language}.",
        ),
        ("human", "{input}"),
    ]
)

chain = prompt | llm
chain.invoke(
    {
        "input_language": "English",
        "output_language": "German",
        "input": "I love programming.",
    }
)
```

## Invocation

```javascript
messages = [
    (
        "system",
        "You are a helpful assistant that translates English to French. Translate the user sentence.",
    ),
    ("human", "I love programming."),
]
ai_msg = llm.invoke(messages)
ai_msg
```
