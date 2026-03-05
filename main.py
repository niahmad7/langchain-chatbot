from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate 
from langchain_core.output_parsers import StrOutputParser



llm = ChatOllama(
    model="minimax-m2.5:cloud",
    temperature=0.7,
    num_ctx=2048,
    num_predict=1024,
    # other params...
)


prompt = ChatPromptTemplate.from_messages([
("system", "You are an expert in, give concise answers."),
("human", "{question}")
])

chain = prompt | llm | StrOutputParser() # Pipeline or LCEL for structured output (used pipe operator to chain the prompt, llm and output parser together)

# response = chain.invoke({
#     "topic": "AI and Machine Learning",
#     "question": "What is RAG?"
# })
# print(response)

for chunks in chain.stream({"question": "What is LangChain?"}):
     print(chunks, end="", flush=True)
