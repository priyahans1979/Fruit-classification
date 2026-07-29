def llm_app(topic, groq_api):

 from langchain_core.prompts import PromptTemplate
 from langchain_groq import ChatGroq
 # 1. Initialize your LLM
 
 llm = ChatGroq(model='openai/gpt-oss-120b', api_key=groq_api, temperature=0.1)

 prompt=PromptTemplate(
    input_variables=['topic'],
    
    template='You are a animal expert.\
    provide five import lines coverng about {topic}in thre languageshindi,english,sanskrit, provide one Fun Fact!, which fruit help in which disease, also their season, nutritions in different columns only in english language with emojis.'
 )

 chain=prompt | llm

 #topic=input('Enter a topic')

 
 output=chain.invoke(topic)
 #print('Generated Blog Title ', output.content)
 return output.content
