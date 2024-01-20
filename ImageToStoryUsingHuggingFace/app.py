from dotenv import load_dotenv, find_dotenv
import streamlit as st
from transformers import pipeline
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_community.llms import HuggingFaceHub
import requests
import os
import numpy as np



load_dotenv(find_dotenv())
HUGGINGFACEHUB_API_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")

def img2Text(url):
    img_to_text = pipeline("image-to-text", model="Salesforce/blip-image-captioning-base")
    
    text = img_to_text(url)[0]["generated_text"]
    
    #print(text)
    return text


def generate_story(scenario):
    # template = """
    # You are a story teller:
    # You can generate a short story based on a simple narrative. the story should be no more than 20 words:
    
    # CONTEXT:{scenario}
    # STORY:
    # """
    
    question = scenario
    
    template = """Question: {question}

    You are a story teller:
    # You can generate a short story based on a simple narrative. the story should be no more than 128 words:"""

    prompt = PromptTemplate(template=template, input_variables=["question"])
    
    repo_id = "mistralai/Mixtral-8x7B-Instruct-v0.1"
    
    llm = HuggingFaceHub(
    repo_id=repo_id, model_kwargs={"temperature": 0.5, "max_length": 128}
)
    llm_chain = LLMChain(prompt=prompt, llm=llm)

    story = llm_chain.invoke(question)
    #print(story['text'])

    # prompt = PromptTemplate(template=template, input_variables={"scenario"})
    
    # story_llm = LLMChain(llm=OpenAI(model_name="gpt-3.5-turbo", temperature=1), prompt=prompt, verbose=True)
    
    # story = story_llm.predict(scenario=scenario)
    
    return story

def text2Speech(message):
    
    string = HUGGINGFACEHUB_API_TOKEN 

    API_URL = "https://api-inference.huggingface.co/models/espnet/kan-bayashi_ljspeech_vits"
    headers = {"Authorization": f"Bearer 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"} 
   
    payloads = {
        "inputs":message['text']
    }
    print(message['text'])
    response = requests.post(API_URL, headers=headers, json=payloads)
    with open('audio.flac','wb') as file:
        file.write(response.content)
    
    
def main():
    st.set_page_config(page_title="img 2 audio story", page_icon="")
    
    st.header("Turn img into audio story")
    uploaded_file = st.file_uploader("Choose and image...", type="jpg")
    
    if uploaded_file is not None:
        print(uploaded_file)
        bytes_data = uploaded_file.getvalue()
        with open(uploaded_file.name, "wb") as file:
            file.write(bytes_data)
        st.image(uploaded_file, caption='Uploaded Image.', use_column_width=True)
        scenario = img2Text(uploaded_file.name)
        story = generate_story(scenario)
        text2Speech(story)
        
        with st.expander("scenario"):
            st.write(scenario)
        with st.expander("story"):
            st.write(story['text'])
        

        audio_file = open('audio.flac', 'rb')
        audio_bytes = audio_file.read()

        st.audio(audio_bytes)

    

if __name__ == '__main__':
    main()