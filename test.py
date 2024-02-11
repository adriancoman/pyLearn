from dotenv import find_dotenv, load_dotenv
from transformers import pipeline

from langchain_community.llms import OpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
import requests
import os
import streamlit as st

load_dotenv(find_dotenv())
HF_API_TOKEN = os.getenv("HF_TOKEN")


def img2text(url):
    image_to_text = pipeline("image-to-text", model="Salesforce/blip-image-captioning-large")
    # image_to_text = pipeline("text-generation", model="LanguageBind/MoE-LLaVA-Phi2-2.7B-4e-384", trust_remote_code=True)

    text = image_to_text(url)[0]["generated_text"]
    return text


def generate_story(scenario):
    template = """
    Embellish the features of the picture. Limit to 50 words.

    CONTEXT: {scenario}
    STORY:    
    """

    prompt = PromptTemplate(template=template, input_variables=["scenario"])
    story_llm = LLMChain(
        llm=OpenAI(
            model_name="gpt-3.5-turbo",
            temperature=1
        ),
        prompt=prompt,
        verbose=True
    )

    story = story_llm.predict(scenario=scenario)
    print(story)
    return story


def text2speech(message):
    api_url = "https://api-inference.huggingface.co/models/espnet/kan-bayashi_ljspeech_vits"
    headers = {"Authorization": "Bearer " + HF_API_TOKEN}

    payloads = {
        "inputs": message,
    }

    response = requests.post(api_url, headers=headers, json=payloads).content

    with open('audio.flac', 'wb') as file:
        file.write(response)


def main():
    st.set_page_config(page_title="My app")

    st.header("Turn the image into a story")

    uploaded_file = st.file_uploader("Choose an image", type="jpeg")

    if uploaded_file is not None:
        print(uploaded_file)
        bytes_data = uploaded_file.getvalue()
        with open(uploaded_file.name, 'wb') as file:
            file.write(bytes_data)
        st.image(
            uploaded_file, caption="Uploaded image.",
            use_column_width=True
        )

        scenario = img2text(uploaded_file.name)
        story = generate_story(scenario)
        text2speech(story)

        with st.expander("scenario"):
            st.write(scenario)
        with st.expander("story"):
            st.write(story)
        st.audio("audio.flac")


if __name__ == '__main__':
    main()

# local runs
# scenario = img2text("img2.jpeg")
# print("---")
# print("Scenario is ready:")
# print(scenario)
# story = generate_story(scenario)
# text2speech(story)