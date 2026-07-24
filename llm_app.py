def llm_app(topic):

 from langchain_core.prompts import PromptTemplate
 from langchain_groq import ChatGroq
 # 1. Initialize your LLM
 groq_api = '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'
 llm = ChatGroq(model='openai/gpt-oss-120b', api_key=groq_api, temperature=0.1)

 prompt=PromptTemplate(
    input_variables=['topic'],
    
    template='You are a animal expert.\
    provide five import lines coverng about {topic} in english,hindi and sanskrit.'
 )

 chain=prompt | llm

 #topic=input('Enter a topic')

 
 output=chain.invoke(topic)
 #print('Generated Blog Title ', output.content)
 return output.content