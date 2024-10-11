# main.py
import os
from config import setup_env
from graph_operations import Neo4jHandler
from text_preprocessing import (
    load_and_split_documents,
    create_vector_index,
    generate_full_text_query
)
from llm_setup import llm, format_chat_history, CONDENSE_QUESTION_PROMPT
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain_core.runnables import (
    RunnableBranch,
    RunnableLambda,
    RunnablePassthrough,
    RunnableParallel
)
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from typing import List, Tuple
from langchain_core.messages import AIMessage, HumanMessage
from pydantic import BaseModel, Field
# Step 1: Setup environment variables
setup_env()

# Step 2: Initialize Neo4j handler
neo4j_handler = Neo4jHandler(
    uri=os.environ["NEO4J_URI"],
    username=os.environ["NEO4J_USERNAME"],
    password=os.environ["NEO4J_PASSWORD"]
)

# Step 3: Load and split documents
documents = load_and_split_documents("Elizabeth I")

# Step 4: Transform documents into graph format
llm_transformer = LLMGraphTransformer(llm=llm)
graph_documents = llm_transformer.convert_to_graph_documents(documents)
neo4j_handler.add_documents(graph_documents)

# Step 5: Create vector index
vector_index = create_vector_index()

# Step 6: Define entity extraction chain
class Entities(BaseModel):
    """Identifying information about entities."""
    names: List[str] = Field(
        ...,
        description="All the person, organization, or business entities that appear in the text",
    )

entity_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are extracting organization and person entities from the text.",
        ),
        (
            "human",
            "Use the given format to extract information from the following input: {question}",
        ),
    ]
)

entity_chain = entity_prompt | llm.with_structured_output(Entities)
def structured_retriever(question: str) -> str:
    result = ""
    entities = entity_chain.invoke({"question": question})
    for entity in entities.names:
        response = neo4j_handler.graph.query(
            """CALL db.index.fulltext.queryNodes('entity', $query, {limit:2})
            YIELD node, score
            CALL {
              WITH node
              MATCH (node)-[r:!MENTIONS]->(neighbor)
              RETURN node.id + ' - ' + type(r) + ' -> ' + neighbor.id AS output
              UNION ALL
              WITH node
              MATCH (node)<-[r:!MENTIONS]-(neighbor)
              RETURN neighbor.id + ' - ' + type(r) + ' -> ' +  node.id AS output
            }
            RETURN output LIMIT 50
            """,
            {"query": generate_full_text_query(entity)},
        )
        result += "\n".join([el['output'] for el in response])
    return result

def retriever(question: str):
    print(f"Search query: {question}")
    structured_data = structured_retriever(question)
    unstructured_data = [el.page_content for el in vector_index.similarity_search(question)]
    final_data = f"""Structured data:
{structured_data}
Unstructured data:
{"#Document ".join(unstructured_data)}
    """
    return final_data

# Step 7: Define the main chain
search_query = RunnableBranch(
    # If input includes chat_history, condense it with the follow-up question
    (
        RunnableLambda(lambda x: bool(x.get("chat_history"))).with_config(
            run_name="HasChatHistoryCheck"
        ),
        RunnablePassthrough.assign(
            chat_history=lambda x: format_chat_history(x["chat_history"])
        )
        | CONDENSE_QUESTION_PROMPT
        | llm
        | StrOutputParser(),
    ),
    # Else, pass through the question
    RunnableLambda(lambda x: x["question"]),
)

template = """Answer the question based only on the following context:
{context}

Question: {question}
Use natural language and be concise.
Answer:"""

final_prompt = ChatPromptTemplate.from_template(template)

chain = (
    RunnableParallel(
        {
            "context": search_query | retriever,
            "question": RunnablePassthrough(),
        }
    )
    | final_prompt
    | llm
    | StrOutputParser()
)

# Step 8: Run the chain
if __name__ == "__main__":
    question = "Which house did Elizabeth I belong to?"
    result = chain.invoke({"question": question})
    print("Answer:")
    print(result)
