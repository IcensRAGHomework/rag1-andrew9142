import json
import traceback

from model_configurations import get_model_configuration

from langchain_openai import AzureChatOpenAI
from langchain_core.messages import HumanMessage

gpt_chat_version = 'gpt-4o'
gpt_config = get_model_configuration(gpt_chat_version)

def generate_hw01(question):
    llm = AzureChatOpenAI(
        model = gpt_config['model_name'],
        deployment_name = gpt_config['deployment_name'],
        openai_api_key = gpt_config['api_key'],
        openai_api_version = gpt_config['api_version'],
        azure_endpoint = gpt_config['api_base'],
        temperature= gpt_config['temperature']
    )

    system = """
            You are a helpful assistant.
            Please respond in JSON format.
            The top-level key must be 'Result', and its value must be a list of objects.
            Each object should contain two keys: 'date' (the date of the holiday) and 'name' (the name of the holiday).
            """
    try:
        answer_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system),
                ("human", "Answer the question based on the below description and only use English or traditional Chinese. Please respond in JSON format."),
                ("human", f"Question: {question} \n "),
            ]
        )

        json_llm = llm.bind(response_format={"type": "json_object"})
        
        rag_chain = LLMChain(
            llm=json_llm,
            prompt=answer_prompt,
            output_parser=StrOutputParser()
        )

        # RAG generation
        answer = rag_chain.invoke({"question": question})
        print(answer)
        text_content = answer.get('text')

        return text_content
    except Exception as e:
        traceback_info = traceback.format_exc()
        print(traceback_info) 
    
def generate_hw02(question):
    pass
    
def generate_hw03(question2, question3):
    pass
    
def generate_hw04(question):
    pass
    
def demo(question):
    llm = AzureChatOpenAI(
            model=gpt_config['model_name'],
            deployment_name=gpt_config['deployment_name'],
            openai_api_key=gpt_config['api_key'],
            openai_api_version=gpt_config['api_version'],
            azure_endpoint=gpt_config['api_base'],
            temperature=gpt_config['temperature']
    )
    message = HumanMessage(
            content=[
                {"type": "text", "text": question},
            ]
    )
    response = llm.invoke([message])
    
    return response
