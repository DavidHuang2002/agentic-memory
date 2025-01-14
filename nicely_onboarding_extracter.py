'''
I am trying to build an AI companion for people in therapy that will help them receive therapy-like experience when they need it between therapy sessions they have with their therapist
'''
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
collection_name = "nicely_user_profile"

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


user_profile_prompt_template  = '''
You are analyzing a conversation as an experienced therapist with deep insight into human behavior. Your task is to extract key elements that will help guide extremely personalized therapy and form the basis for future therapeutic insights and interventions.

Review the conversation and create a memory reflection following these rules:

1. **Focus on Therapeutic Relevance**  
   - Only include information that will meaningfully guide future therapeutic conversations.

2. **Four Categories**  
   - Each item must have a "type" among: "goal", "struggle", "insight", or "next_step".  
   - Choose the category that best represents the point’s function in therapy.

3. **Context Tags**  
   - Provide 2-4 keywords that capture the emotional, behavioral, or thematic essence of the point
   - Examples: ["self_esteem", "relationship_conflict", "perfectionism"], ["grief", "coping_skills"]  
   - Avoid broad or generic tags like ["therapy", "session", "conversation"].

4. **Concise Summary**  
   - Write a single-sentence summary that pinpoints the essence of the user’s statement.  
   - Example: "Feels anxious about an upcoming job interview due to fear of failure."

5. **Original Quote**  
   - Include a short direct quote from the user that prompted you to create this entry.  
   - This helps preserve the user’s own words and context.

6. **Maintain Therapeutic Utility**  
   - Ensure each entry is actionable or reflective enough to be revisited in later sessions.  
   - Capture emotional tone, motivation, or specific details relevant to therapeutic progress.

7. **Output Format**  
   - Your output must be valid JSON in **exactly** this array format:  
     ```
     [
       {{
         "type": string,
         "context_tags": [
           string,
           ...
         ],
         "summary": string,
         "original_quote": string
       }},
       ...
     ]
     ```

**Do not** include any text outside of this JSON array.  
**Do not** add any additional fields.

Here is the prior conversation:

{conversation}
'''
user_profile_prompt = ChatPromptTemplate.from_template(user_profile_prompt_template)
user_profile_model = user_profile_prompt | llm | JsonOutputParser()

def format_conversation(messages):
    """Format conversation by removing system prompt."""
    conversation = []
    for message in messages[1:]:
        conversation.append(f"{message.type.upper()}: {message.content}")
    return "\n".join(conversation)



# def add_episodic_memory(messages, qdrant_client):
#     """Add conversation to episodic memory."""
#     conversation = format_conversation(messages)
#     reflection = reflect.invoke({"conversation": conversation})
#     embedding = embedding_model.encode(conversation)
    
#     qdrant_client.upsert(
#         collection_name=collection_name,
#         points=[
#             models.PointStruct(
#                 id=abs(hash(conversation)),
#                 vector=embedding.tolist(),
#                 payload={
#                     "conversation": conversation,
#                     "context_tags": reflection['context_tags'],
#                     "conversation_summary": reflection['conversation_summary'],
#                     "what_worked": reflection['what_worked'],
#                     "what_to_avoid": reflection['what_to_avoid'],
#                 }
#             )
#         ]
#     )


def extract_user_profile(conversation):
    """Extract user profile from conversation."""
    user_profile = user_profile_model.invoke({"conversation": conversation})
    
    print(user_profile)

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

def interaction_loop():
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
            # procedural_memory_update(what_worked, what_to_avoid)
            # print("\n== Procedural Memory Updated ==")
            break
        if user_input.lower() == "exit_quiet":
            print("\n == Conversation Exited ==")
            break
        
        response = llm.invoke([*messages, user_message])
        print("\nAI Message: ", response.content)
        messages.extend([user_message, response])

def main():
    messages = [
        AIMessage(content='''Now let’s dive into your goals:  

**1. What are you currently working on in therapy?**  
**2. Are there specific goals you’d like to achieve, such as managing anxiety, improving relationships, or building confidence?**  
**3. What does progress or success look like for you?**  '''),
        
        HumanMessage(content='''well it was to gain more clarity in my decision as well as my emotion but right now it has been a while since my last therapy session and I think right now what I really want to do is help me manage emotions regarding relationships particularly help me gain more clarity on how I should interact with girls and like whether I should pursue relationship whether I should be actively seeking relationship or not this has been causing me some confusion and a little bit anxiety recently and progress or success will look like me making a confident decision on whether I should get out there and seek relationship or I should just work on myself and not care too much about that aspect'''),
        
        AIMessage(content='''Current Challenges
1. What’s been challenging for you lately?
2. Are there specific situations or emotions that feel overwhelming?
3. What’s something you’d like to better understand about yourself?'''),
        
        HumanMessage(content='''One challenge I've been facing lately was the challenge of balancing my important task of building a startup with my social life. Perhaps out of habit, I sometimes want to get more social interaction, not for any purpose or anything. For example, tonight I went out to the cafeteria to find some friends because the thought of eating dinner alone didn't feel great to me. And that came at the cost of my productivity because I spent a lot of time talking with my friends. Sometimes I know I need to drill down and lock in, but another part of me really craves social interaction. And because of that temptation of social interaction, I went out and started interacting with others. And that distracted me from focusing on my task, and the task did not get completed on time. So that is one challenge I have been facing. The second challenge is relationship. This is also an area I want to get a better understanding of myself in. One of the relationships is particularly a relationship with a female, like with girls. So on one hand, I really have this big crush on a girl that is not currently seeking a relationship. And I just really like her, and I want to create more ways I can interact with her. Although I know I probably shouldn't fantasize, but sometimes I just couldn't help it. I like her so much, I really want to be with her. And one problem that is confusing me right now is, should I be active in pursuing relationships with other people? Now that the possibility of me being with that girl is so low. The girl I had a crush on, her name was Lucy. And now that the possibility of me actually being with her is very low. Logically, I should be open to other relationships. But at the same time, I like her so much, I feel like even if there is just an iota of possibility, I really want to give it a try. And that is something that has been stuck in my mind. And another aspect of it is that I am not sure whether I should pursue relationships at all. I am quite busy. Although part of me really craves this kind of experience, and I think this kind of experience can bring me a lot of growth. But also at the same time, there are other things I want to focus on. Like building a startup, and working on myself. And that will take a lot of time, and I don't know whether getting into a relationship will take out too much time from my current schedule and whether I can afford it. And that is another thing that has been stuck in my mind as well.''')
    ]
    extract_user_profile(messages)

if __name__ == "__main__":
    main()