# llm_setup.py

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import AIMessage, HumanMessage
from typing import List, Tuple

# LLM setup
llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo-0125")

# Chat prompt template
_template = """Given the following conversation and a follow-up question, rephrase the follow-up question to be a standalone question in its original language.
Chat History:
{chat_history}
Follow-Up Input: {question}
Standalone question:"""

CONDENSE_QUESTION_PROMPT = PromptTemplate.from_template(_template)

def format_chat_history(chat_history: List[Tuple[str, str]]) -> List:
    buffer = []
    for human, ai in chat_history:
        buffer.append(HumanMessage(content=human))
        buffer.append(AIMessage(content=ai))
    return buffer
