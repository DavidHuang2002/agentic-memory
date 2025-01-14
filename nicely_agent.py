# Dependencies
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from qdrant_client import QdrantClient
from qdrant_client.http import models
from sentence_transformers import SentenceTransformer

# Initialize LLM
llm = ChatOpenAI(
    temperature=0.7,
    model="gpt-4o",
)

# Constants
collection_name = "nicely_episodic_memory"

# Initialize Qdrant Client
qdrant_client = QdrantClient(
    url="https://205ac0d3-de1e-4cad-8071-57bbddf23c04.us-east4-0.gcp.cloud.qdrant.io",
    api_key="ivaF1cwbPeZ-qWpw7Gq42zW_VoHcJitqCFejcHk7E1EtENcawrn2gA",
)

# Initialize embedding model
embedding_model = SentenceTransformer('nomic-ai/nomic-embed-text-v1', trust_remote_code=True)

# Create collection
qdrant_client.recreate_collection(
    collection_name=collection_name,
    vectors_config=models.VectorParams(
        size=768,  # Dimension size for nomic-embed-text
        distance=models.Distance.COSINE
    )
)

# Define reflection prompt template
reflection_prompt_template = """
You are analyzing conversations about research papers to create memories that will help guide future interactions. Your task is to extract key elements that would be most helpful when encountering similar academic discussions in the future.

Review the conversation and create a memory reflection following these rules:

1. For any field where you don't have enough information or the field isn't relevant, use "N/A"
2. Be extremely concise - each string should be one clear, actionable sentence
3. Focus only on information that would be useful for handling similar future conversations
4. Context_tags should be specific enough to match similar situations but general enough to be reusable

Output valid JSON in exactly this format:
{
    "context_tags": [              // 2-4 keywords that would help identify similar future conversations
        string,                    // Use field-specific terms like "deep_learning", "methodology_question", "results_interpretation"
        ...
    ],
    "conversation_summary": string, // One sentence describing what the conversation accomplished
    "what_worked": string,         // Most effective approach or strategy used in this conversation
    "what_to_avoid": string        // Most important pitfall or ineffective approach to avoid
}

Do not include any text outside the JSON object in your response.

Here is the prior conversation:

{conversation}
"""

reflection_prompt = ChatPromptTemplate.from_template(reflection_prompt_template)
reflect = reflection_prompt | llm | JsonOutputParser()

def format_conversation(messages):
    """Format conversation by removing system prompt."""
    conversation = []
    for message in messages[1:]:
        conversation.append(f"{message.type.upper()}: {message.content}")
    return "\n".join(conversation)

def add_episodic_memory(messages, qdrant_client):
    """Add conversation to episodic memory."""
    conversation = format_conversation(messages)
    reflection = reflect.invoke({"conversation": conversation})
    embedding = embedding_model.encode(conversation)
    
    qdrant_client.upsert(
        collection_name=collection_name,
        points=[
            models.PointStruct(
                id=abs(hash(conversation)),
                vector=embedding.tolist(),
                payload={
                    "conversation": conversation,
                    "context_tags": reflection['context_tags'],
                    "conversation_summary": reflection['conversation_summary'],
                    "what_worked": reflection['what_worked'],
                    "what_to_avoid": reflection['what_to_avoid'],
                }
            )
        ]
    )

def episodic_recall(query, qdrant_client):
    """Retrieve relevant memory based on query."""
    query_embedding = embedding_model.encode(query)
    search_result = qdrant_client.search(
        collection_name=collection_name,
        query_vector=query_embedding.tolist(),
        limit=1
    )
    if search_result:
        return search_result[0].payload
    return None

def procedural_memory_update(what_worked, what_to_avoid):
    """Update procedural memory based on conversation feedback."""
    with open("./procedural_memory.txt", "r") as content:
        current_takeaways = content.read()

    procedural_prompt = f"""You are maintaining a continuously updated list of the most important procedural behavior instructions for an AI assistant. Your task is to refine and improve a list of key takeaways based on new conversation feedback while maintaining the most valuable existing insights.

    CURRENT TAKEAWAYS:
    {current_takeaways}

    NEW FEEDBACK:
    What Worked Well:
    {what_worked}

    What To Avoid:
    {what_to_avoid}

    Please generate an updated list of up to 10 key takeaways that combines:
    1. The most valuable insights from the current takeaways
    2. New learnings from the recent feedback
    3. Any synthesized insights combining multiple learnings

    Requirements for each takeaway:
    - Must be specific and actionable
    - Should address a distinct aspect of behavior
    - Include a clear rationale
    - Written in imperative form (e.g., "Maintain conversation context by...")

    Format each takeaway as:
    [#]. [Instruction] - [Brief rationale]

    The final list should:
    - Be ordered by importance/impact
    - Cover a diverse range of interaction aspects
    - Focus on concrete behaviors rather than abstract principles
    - Preserve particularly valuable existing takeaways
    - Incorporate new insights when they provide meaningful improvements

    Return up to but no more than 10 takeaways, replacing or combining existing ones as needed to maintain the most effective set of guidelines.
    Return only the list, no preamble or explanation.
    """

    procedural_memory = llm.invoke(procedural_prompt)
    with open("./procedural_memory.txt", "w") as content:
        content.write(procedural_memory.content)

def episodic_system_prompt(query, qdrant_client):
    """Generate system prompt with episodic and procedural memory."""
    memory = episodic_recall(query, qdrant_client)
    
    with open("./procedural_memory.txt", "r") as content:
        procedural_memory = content.read()
    
    current_conversation = memory['conversation']
    if current_conversation not in conversations:
        conversations.append(current_conversation)
    what_worked.update(memory['what_worked'].split('. '))
    what_to_avoid.update(memory['what_to_avoid'].split('. '))
    
    previous_convos = [conv for conv in conversations[-4:] if conv != current_conversation][-3:]
    
    episodic_prompt = f"""You are a helpful AI Assistant. Answer the user's questions to the best of your ability.
    You recall similar conversations with the user, here are the details:
    
    Current Conversation Match: {current_conversation}
    Previous Conversations: {' | '.join(previous_convos)}
    What has worked well: {' '.join(what_worked)}
    What to avoid: {' '.join(what_to_avoid)}
    
    Use these memories as context for your response to the user.
    
    Additionally, here are 10 guidelines for interactions with the current user: {procedural_memory}"""
    
    return SystemMessage(content=episodic_prompt)

def main():
    # Initialize memory stores
    global conversations, what_worked, what_to_avoid
    conversations = []
    what_worked = set()
    what_to_avoid = set()
    messages = []

    while True:
        user_input = input("\nUser: ")
        user_message = HumanMessage(content=user_input)
        
        system_prompt = episodic_system_prompt(user_input, qdrant_client)
        messages = [
            system_prompt,
            *[msg for msg in messages if not isinstance(msg, SystemMessage)]
        ]
        
        if user_input.lower() == "exit":
            add_episodic_memory(messages, qdrant_client)
            print("\n == Conversation Stored in Episodic Memory ==")
            procedural_memory_update(what_worked, what_to_avoid)
            print("\n== Procedural Memory Updated ==")
            break
        if user_input.lower() == "exit_quiet":
            print("\n == Conversation Exited ==")
            break
        
        response = llm.invoke([*messages, user_message])
        print("\nAI Message: ", response.content)
        messages.extend([user_message, response])

if __name__ == "__main__":
    main()