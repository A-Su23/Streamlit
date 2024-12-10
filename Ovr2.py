import re
import streamlit as st
import time
from openai import OpenAI
from crewai import Agent, Task, Crew
import os
from dotenv import load_dotenv
import logging
from typing import Dict
from sentence_transformers import SentenceTransformer
from bert_score import BERTScorer
from sklearn.metrics.pairwise import cosine_similarity

load_dotenv() 


OPENAI_API_KEY = st.secrets["OPENAI_API"]
HUGGINGFACE_API_KEY = st.secrets["HUGGINGFACE_API"]
# API Client Initialization
client = OpenAI(
    base_url="https://e9gim5wxyuboirrq.us-east-1.aws.endpoints.huggingface.cloud/v1/", 
	api_key=HUGGINGFACE_API_KEY
)


import streamlit as st

class Overview:
    def func():
        # Inject CSS for larger font sizes
        st.markdown(
            """
            <style>
                /* General font size for all markdown text */
                div[data-testid="stMarkdownContainer"] p {
                    font-size: 22px !important; /* Adjust for paragraph text */
                }
                
                /* Larger font for titles */
                .stTitle {
                    font-size: 50px !important;
                }

                /* Larger font for subheaders */
                .stSubtitle, h2 {
                    font-size: 40px !important;
                }

                /* Adjust font size for list items specifically */
                div[data-testid="stMarkdownContainer"] ul, 
                div[data-testid="stMarkdownContainer"] ol {
                    font-size: 16px !important;
                    margin-left: 1.5em; /* Ensure proper indentation */
                }

                /* Adjust list item font size directly */
                div[data-testid="stMarkdownContainer"] li {
                    font-size: 22px !important;
                }
            </style>
            """,
            unsafe_allow_html=True
        )


        # Page Content
        st.title("🤖 A3LLM Tool Overview")

        st.markdown("Welcome! Let us do the heavy lifting—turning lengthy texts into crisp, clear summaries so you can focus on what matters most.")

        st.markdown("---")

        st.subheader("What is This Page?")
        st.markdown("""
        This page introduces a powerful summarization tool that helps users distill large blocks of text into concise, meaningful summaries. It leverages advanced Natural Language Processing (NLP) techniques to simplify complex information, enabling users to quickly grasp the core ideas without reading every detail.
        """)

        st.markdown("---")
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("A3LLM")
            st.markdown("""
            We developed an Arabic abstractive summarization Model that transforms lengthy Arabic texts into concise, context-rich summaries, offering a seamless way to grasp essential information while preserving meaning and tone.
            """)

            st.subheader("Why A3LLM?")
            st.markdown("""
            - **Time-Saving Benefits**         
                - Reduces reading time by condensing long texts into accessible summaries.
                - Extracts essential information in seconds, supporting informed decisions and efficient learning.
            - **Use Cases**         
                - **Research Papers**: Digest lengthy academic texts efficiently.
                - **News Summaries**: Stay updated with quick overviews of detailed articles.
                - **Business Reports**: Extract actionable insights from comprehensive reports.
                - **Personal Use**: Simplify web content, documents, or reading material.
            """)
        with col2:
            st.subheader("Agentic Summarization")
            st.markdown("""
            Our second Model, the **Agentic Summarization**, leverages multi-agent AI collaboration to deliver concise, context-aware, and accurate summaries by specializing in tasks like key point extraction and refinement.        """)
            st.subheader("Why Agentic Summarization?")
            st.markdown("""
                        
            - **Use cases include**: 

                - **Research and Academia**: Precisely summarize complex papers while preserving technical details.  
                - **Business Reports**: Extract actionable insights from lengthy reports with accuracy.  
                - **Content Curation**: Create concise, context-aware summaries for news and media.  
            - **Why Agentic Summarization?**

                - **Precision and Context**: Specialized agents ensure deeper understanding and accurate summaries.  
                - **Dynamic and Adaptive**: Adjusts to content complexity, outperforming static models.  
                - **Iterative Refinement**: Multi-agent collaboration reduces errors and enhances clarity.  
            """)
        st.markdown("---")
        col1, col2 = st.columns(2)
        with col1:

            st.subheader("A3LLM vs Agentic Summarization.")
            st.markdown("""
            This page allows you to compare the summarization outputs of the **A3LLM Model** and the **Agentic Summarization Model** using the same input text. It highlights differences in their approaches to summarization, including clarity, brevity, and content relevance, providing a clear side-by-side analysis to evaluate their performance and suitability for various use cases.
            """)
        with col2:
            st.subheader("Why to Compare?")
            st.markdown("""
            - **Performance Evaluation**: Assess the quality and relevance of summaries generated by different models.
            - **Use Case Suitability**: Determine which model best meets your summarization needs based on output comparison.
            - **Feature Comparison**: Understand the strengths and limitations of each model for specific applications.
            """)
        st.markdown("---")
        
        st.subheader("How to  Use the Model")
        col1, col2 = st.columns(2)
        with col1:
            
            st.markdown("""
            1. **Input Requirements**
                - Text must be in a readable format, such as plain text or supported document types.
                - Ensure clarity and relevance to the desired summary.
                - Recommended input size: up to 1,000 words for optimal results.
            2. **Model Parameters**
            Users can tailor the summarization to their preferences by adjusting:
                - **Temperature**: Makes responses more creative or focused.  
            """)
        with col2:
            st.markdown("""       
            3. **Output Explanation**
                - The model generates a summary that concisely presents the main points.
                - Delivered as sentences or paragraphs, the output ensures context and meaning are preserved.
            4. **Getting Started**
                - Input the Text: Paste or upload your content.
                - Select Parameters: Adjust settings like summary length and style.
                - Generate Summary: Click "Generate" to instantly receive your summary.
            """)


        


# Streamlit App Configuration
st.set_page_config(
    page_title="Arabic Text Summarization",
    layout="wide",
    initial_sidebar_state="collapsed"
)

def evaluate_arabic_summary(reference: str, candidate: str) -> Dict[str, float]:
    if not isinstance(candidate, str):
        candidate = str(candidate)
    # Initialize models
    sent_bert_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
    bert_scorer = BERTScorer(
        model_type="asafaya/bert-base-arabic",
        num_layers=9,
        lang="ar",
        rescale_with_baseline=False
    )

    # Preprocess text
    def preprocess_arabic(text: str) -> str:
        return re.sub(r'/s+', ' ', text).strip()

    reference = preprocess_arabic(reference)
    candidate = preprocess_arabic(candidate)

    # Calculate SentenceBert score
    ref_embedding = sent_bert_model.encode([reference])
    cand_embedding = sent_bert_model.encode([candidate])
    sent_bert_score = cosine_similarity(ref_embedding, cand_embedding)[0][0]

    # Calculate BERTScore
    P, R, F1 = bert_scorer.score([candidate], [reference])

    return {
        'SentenceBert': sent_bert_score,
        'Precision': P.item(),
        'Recall': R.item(),
        'F1-Score': F1.item()
    }


def calculate_summary_statistics(text, summary):
    if not isinstance(summary, str):
        summary = str(summary)
    # Calculate lengths
    text_length = len(text)
    summary_length = len(summary)
    
    # Calculate summary-to-text percentage
    summary_to_text_percentage = (summary_length / text_length) * 100 if text_length > 0 else 0
    
    return text_length, summary_length, summary_to_text_percentage


def arabic_summarization_workflow(input_text, user_token):
    """
    Executes the Arabic summarization workflow with a focus on brevity and outputs the results.

    Args:
        input_text (str): The Arabic text to summarize.
        user_token (str): The OpenAI API token provided by the user.

    Returns:
        dict: A dictionary containing the outputs from each step:
              - 'summarizer': Output from the summarizer
              - 'reviewer': Output from the reviewer
              - 'editor': Output from the editor
    """
    # Set the OpenAI API key and model name globally using the user's token
    os.environ["OPENAI_API_KEY"] = user_token
    os.environ["OPENAI_MODEL_NAME"] = 'gpt-4'

    # Calculate the maximum word count for the summary (40% of the original text)
    original_word_count = len(input_text.split())
    max_summary_word_count = int(0.4 * original_word_count)

    # Define Agents
    summarizer = Agent(
        role="Arabic Abstractive Summarization Expert",
        goal=(
            "Read and understand the following text and write an abstractive summary in Arabic: {text}. "
            "Ensure the summary is concise and captures the core ideas and tone of the original text."
        ),
        backstory=(
            "You are a professional in Arabic abstractive summarization with expertise in analyzing complex Arabic texts "
            "and generating high-quality summaries. Focus on distilling the key ideas while maintaining the text's tone and intent."
        ),
        allow_delegation=False,
        verbose=True
    )

    reviewer = Agent(
        role="Arabic Abstractive Summary Reviewer",
        goal=(
            "Review the Arabic summary provided by the Summarization Expert. "
            "Focus only on ensuring alignment with the original text ({text}) and removing any major redundancies. "
            "Provide brief feedback to improve clarity and conciseness."
        ),
        backstory=(
            "You are an expert in reviewing Arabic summaries. Your task is to provide constructive, high-level feedback, "
            "focusing only on critical issues or redundancies that affect clarity or brevity. Avoid overly detailed feedback."
        ),
        allow_delegation=False,
        verbose=True
    )

    editor = Agent(
        role="Arabic Summary Editor",
        goal=(
            "Refine the summary based on the reviewer's feedback, ensuring it is concise, clear, and aligned with the original text ({text}). "
            "The final summary should not exceed {max_word_count} words while preserving the key points."
        ),
        backstory=(
            "You are an experienced editor specializing in Arabic texts. Your task is to finalize the summary, ensuring it communicates the main ideas "
            "effectively and concisely, while adhering to the specified word count limit."
        ),
        allow_delegation=False,
        verbose=True
    )

    # Define Tasks
    create_summary = Task(
        description=(
            "1. Read and understand the given Arabic text.\n"
            "2. Extract the most important ideas and themes from the text.\n"
            "3. Write a concise, high-quality abstractive summary in Arabic that captures the core ideas.\n"
            "4. Ensure the summary is original, coherent, and free from grammatical errors."
        ),
        expected_output="A high-quality Arabic abstractive summary that retains the core ideas and tone of the original text.",
        agent=summarizer
    )

    review_summary = Task(
        description=(
            "1. Review the Arabic abstractive summary and compare it with the original text ({text}).\n"
            "2. Identify any critical issues or redundancies that significantly impact clarity or conciseness.\n"
            "3. Provide high-level feedback to improve the summary.\n"
            "4. Forward the original text, the unmodified summary, and your notes to the editor."
        ),
        expected_output="Brief reviewer notes highlighting critical issues and suggestions for improvement.",
        agent=reviewer
    )

    final_edit = Task(
        description=(
            "1. Receive the original text, the unmodified summary, and the reviewer’s notes.\n"
            "2. Refine the summary based on feedback, ensuring brevity and clarity.\n"
            "3. Ensure the final summary does not exceed {max_word_count} words.\n"
            "4. Deliver a polished, concise summary ready for publication."
        ),
        expected_output=(
            "A finalized Arabic abstractive summary that is concise, coherent, and does not exceed the specified word count."
        ),
        agent=editor
    )

    # Define the Crew
    arabic_summary_crew = Crew(
        agents=[summarizer, reviewer, editor],
        tasks=[create_summary, review_summary, final_edit],
        verbose=True
    )

    # Execute the Crew
    logging.info("Starting Arabic summarization workflow.")
    result = arabic_summary_crew.kickoff(inputs={"text": input_text, "max_word_count": max_summary_word_count})

    # Validate and return outputs
    outputs = {
        "summarizer": create_summary.output.exported_output,
        "reviewer": review_summary.output.exported_output,
        "editor": final_edit.output.exported_output
    }
    for step, output in outputs.items():
        logging.info(f"{step} output: {output}")

    return outputs






# Helper Functions
def filter_arabic_text(text):
    """
    Filters text to retain only Arabic characters, punctuation, and spaces.
    """
    arabic_pattern = r"[^\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF\s،؛؟.!]"
    return re.sub(arabic_pattern, "", text).strip()

def summarize_text_with_api(input_text, temp):
    """
    Sends a summarization request to the API endpoint.
    """
    try:
        chat_completion = client.chat.completions.create(
            model="tgi",
            messages=[
                {"role": "system", "content": (
                    "You are an assistant specialized in Arabic language processing. "
                    "Perform abstractive summarization of the following Arabic text. "
                    "The output should be in Arabic ONLY. DO NOT WRITE IN NON-ARABIC LANGUAGE."
                    "The summary should be abstractive and concise and capture the main ideas."
                    "Please ensure the summary is coherent and maintains the original context."
                    "The summary should be in Arabic, and has the length of 20% to 50% Of the original text."
                    "You are not allowed to copy the original text. Use your own words."
                    "You should'nt include any personal opinion or additional information."
                    "You are not allowed to ansewr any thing other than the requested summary."
                )},
                {"role": "user", "content": f"قم بتلخيص النص التالي: {input_text}"}
            ],
        	top_p=None,
            temperature=temp,
            max_tokens=None,
            stream=True,
            seed=None,
            frequency_penalty=None,
            presence_penalty=None
        )
        response_text = ""
        for message in chat_completion:
            # Check for None before concatenating
            delta_content = message.choices[0].delta.content
            if delta_content:
                response_text += delta_content
        return response_text
    except Exception as e:
        print(f"Error connecting to the API: {e}")
        return ""

# Sidebar: Model Parameters
def sidebar_model_parameters():
    """
    Renders model parameter controls in the sidebar.
    """

    st.sidebar.title("A3LLM Model Tuning")
    st.sidebar.subheader("Adjust Model Parameters:")
    st.sidebar.image("A3LLMgry.png")

    temperature = st.sidebar.slider("Temperature", 0.0, 1.0, 0.25, 0.01)
    
    st.sidebar.markdown(f"""
    **Selected Parameters:**
    - **Temperature:** {temperature}
    """)
    return temperature

temperature = sidebar_model_parameters()

def render_a3llm_tab():
    """
    Renders the content for the A3LLM tab, including the main summarization functionality.
    """
    st.title("A3LLM Arabic Summarization")
    st.markdown("""
    **A3LLM** is a large language model for Arabic abstractive summarization, designed to condense lengthy texts into clear and meaningful summaries while preserving context and linguistic richness.
    """)
    
    # Apply RTL styling for text input
    st.markdown("""
    <style>
    input, .rtl {
        unicode-bidi: bidi-override;
        direction: RTL;
        font-size: 24px !important;
    }
    </style>
    """, unsafe_allow_html=True)

    # Inject custom CSS for RTL support in the text area
    st.markdown("""
    <style>
        input, textarea {
            unicode-bidi: bidi-override;
            direction: RTL; /* Right-to-left text direction */
            text-align: right; /* Align text to the right */
            font-size: 24px !important;
            padding: 10px; /* Add padding for better spacing */
            border-radius: 5px; /* Rounded corners for aesthetics */
            border: 1px solid #ccc; /* Subtle border for better visibility */
            background-color: #f9f9f9; /* Light background for readability */
        }
        div[data-testid="stChatInput"] textarea {
            direction: rtl;
            font-size: 24px !important;
            text-align: right;
        }
        
        /* Apply RTL direction to all chat messages */
        div[data-testid="stChatMessage"] {
            direction: rtl;
            font-size: 24px !important;
            text-align: right;
        }        
    </style>
    """, unsafe_allow_html=True)

    # Text area for input
    input_text = st.chat_input("أدخل النص هنا:", key="a3llm_input")

    # Sidebar for model parameters    
    # Summarize Button
    if input_text:  # Automatically triggered when user submits input
        if not input_text:
            st.error("Please enter valid Arabic text only.")
        else:
            with st.spinner("Summarizing..."):
                summary = summarize_text_with_api(input_text, temperature)
                time.sleep(1)  # Simulating processing delay
            if summary:
                st.success("Summarization Complete!")
                
                # Display Original Text and Summary in RTL format with larger bold headlines
                with st.chat_message("user"):
                    st.markdown(
                            "<div style='font-size:28px; font-weight:bold; direction:rtl; text-align:right;'> النص الأصلي: </div>",
                            unsafe_allow_html=True
                        )
                    st.markdown(
                            f"<div style='font-size:24px; direction:rtl; text-align:right;'>{input_text}</div>",
                            unsafe_allow_html=True
                        )
                with st.chat_message("assistant"): 
                    st.markdown('<div class="rtl"><h2 style="font-weight: bold;">الملخص:</h2><p>%s</p></div>' % summary, unsafe_allow_html=True)
                try:
                    outpusstat = calculate_summary_statistics(input_text, summary)
                    text_length = outpusstat[0]
                    summary_length = outpusstat[1]
                    summary_to_text_percentage = outpusstat[2]
                    score=evaluate_arabic_summary(input_text, summary)
                    bert = score['F1-Score']
                    sentence = score['SentenceBert']
                    st.warning(
                            f"**BERT F1-Score**: {bert:.3f}"
                            f"\n**SentenceBert Similarity**: {sentence:.3f}"
                            f"\n**Text Length**: {text_length} words"
                            f"\n**Summary Length**: {summary_length} words"
                            f"\n**Summary to Text Percentage**: {summary_to_text_percentage:.2f}%")
                except:
                    st.error("Failed to evaluate the summary.")

def render_agentic_tab():
    """
    Renders the content for the Agentic Summarization tab.
    """
    st.title("Agentic Summarization")
    st.markdown("""
This page shows the **Agentic Summarization**, an agent designed to supervise another agent for producing tailored summaries. It focuses on crafting summaries to meet specific objectives, including:

- Research and analysis.
- Decision-making.
- Personalized understanding.

The model ensures that summaries are goal-oriented, enhancing clarity, precision, and relevance based on the intended application.
    """)
    
    st.markdown("""
        <style>
            input, textarea {
                unicode-bidi: bidi-override;
                direction: RTL; /* Right-to-left text direction */
                text-align: right; /* Align text to the right */
                font-size: 24px !important;
                padding: 10px; /* Add padding for better spacing */
                border-radius: 5px; /* Rounded corners for aesthetics */
                border: 1px solid #ccc; /* Subtle border for better visibility */
                background-color: #f9f9f9; /* Light background for readability */
            }
            div[data-testid="stChatInput"] textarea {
                direction: rtl;
                font-size: 24px !important;
                text-align: right;
            }
            /* Removed global RTL styling for chat messages */
        </style>
        """, unsafe_allow_html=True)
    input_text = st.chat_input("أدخل النص هنا:", key="Agentic_input")
    api_key = OPENAI_API_KEY
    if input_text:
        if not input_text:
            st.error("Please enter valid Arabic text.")
        else:
            with st.spinner("Generating summary..."):
                outputs = arabic_summarization_workflow(input_text, api_key)
                time.sleep(1)  # Simulating processing delay
            if outputs:
                st.success("Multi-Agent Summarization Complete!")
                st.markdown("<div style='flex: 1;'></div>", unsafe_allow_html=True)
                with st.chat_message("user"):
                    st.markdown(
                            "<div style='font-size:28px; font-weight:bold; direction:rtl; text-align:right;'> النص الأصلي: </div>",
                            unsafe_allow_html=True
                        )
                    st.markdown(
                            f"<div style='font-size:24px; direction:rtl; text-align:right;'>{input_text}</div>",
                            unsafe_allow_html=True
                        )
                # First Message (التلخيص المبدئي)
                with st.chat_message("assistant"):
                    st.markdown(
                        "<div style='font-size:28px; font-weight:bold; direction:rtl; text-align:right;'> التلخيص المبدئي:</div>",
                        unsafe_allow_html=True
                    )
                    st.markdown(
                        f"<div style='font-size:24px; direction:rtl; text-align:right;'>{outputs['summarizer']}</div>",
                        unsafe_allow_html=True
                    )
                
                # Second Message (التلخيص المراجع)
                with st.chat_message("user", avatar='🤖'):
                    st.markdown(
                        "<div style='font-size:28px; font-weight:bold; direction:rtl; text-align:right;'> رأي المراجع:</div>",
                        unsafe_allow_html=True
                    )
                    st.markdown(
                        f"<div style='font-size:24px; direction:ltr; text-align:left;'>{outputs['reviewer']}</div>",
                        unsafe_allow_html=True
                    )
                
                # Third Message (التلخيص النهائي)
                with st.chat_message("assistant"):
                    st.markdown(
                        "<div style='font-size:28px; font-weight:bold; direction:rtl; text-align:right;'> التلخيص النهائي:</div>",
                        unsafe_allow_html=True
                    )
                    st.markdown(
                        f"<div style='font-size:24px; direction:rtl; text-align:right;'>{outputs['editor']}</div>",
                        unsafe_allow_html=True
                    )
                try:
                    outpusstat = calculate_summary_statistics(input_text, outputs['editor'])
                    text_length = outpusstat[0]
                    summary_length = outpusstat[1]
                    summary_to_text_percentage = outpusstat[2]
                    score=evaluate_arabic_summary(input_text, outputs['editor'])
                    bert = score['F1-Score']
                    sentence = score['SentenceBert']
                    st.info(
                            f"**BERT F1-Score**: {bert:.3f}"
                            f"\n**SentenceBert Similarity**: {sentence:.3f}"
                            f"\n**Text Length**: {text_length} words"
                            f"\n**Summary Length**: {summary_length} words"
                            f"\n**Summary to Text Percentage**: {summary_to_text_percentage:.2f}%")
                except:
                    st.error("Failed to evaluate the summary.")
    
            else:
                st.error("Failed to generate summary. Please try again.")

def render_a3llm_vs_agentic():
    st.title("A3LLM vs Agentic Summarization")
    st.markdown("""
This page compares the summarization outputs of the **A3LLM Model** and the **Agentic Summarization Model** using the same input text. It highlights differences in their approaches to summarization, including clarity, brevity, and content relevance, providing a clear side-by-side analysis to evaluate their performance and suitability for various use cases.    """)

    st.markdown("""
        <style>
            input, textarea {
                unicode-bidi: bidi-override;
                direction: RTL; /* Right-to-left text direction */
                text-align: right; /* Align text to the right */
                font-size: 24px !important;
                padding: 10px; /* Add padding for better spacing */
                border-radius: 5px; /* Rounded corners for aesthetics */
                border: 1px solid #ccc; /* Subtle border for better visibility */
                background-color: #f9f9f9; /* Light background for readability */
            }
            div[data-testid="stChatInput"] textarea {
                direction: rtl;
                font-size: 24px !important;
                text-align: right;
            }
            /* Removed global RTL styling for chat messages */
        </style>
        """, unsafe_allow_html=True)
    input_text = st.chat_input("أدخل النص هنا:", key="Comparsiion_input")
    api_key = OPENAI_API_KEY

    if input_text:

        with st.spinner("Generating summary..."):
            outputs = arabic_summarization_workflow(input_text, api_key)
            summary = summarize_text_with_api(input_text, temperature)
            if summary and outputs:
                st.success("Both Summarizations Complete!")
            time.sleep(1)  # Simulating processing delay
        with st.chat_message("user"):
            st.markdown(
                    "<div style='font-size:28px; font-weight:bold; direction:rtl; text-align:right;'> النص الأصلي: </div>",
                    unsafe_allow_html=True
                )
            st.markdown(
                    f"<div style='font-size:24px; direction:rtl; text-align:right;'>{input_text}</div>",
                    unsafe_allow_html=True
                )
        st.markdown("---")
        col1, col2 = st.columns(2)

        # Column 1: Agentic Summarization
        with col1:
            st.header("Agentic Summarization")
            with st.chat_message("user", avatar='🤖'):
                st.markdown(
                    "<div style='font-size:28px; font-weight:bold; direction:rtl; text-align:right;'></div>",
                    unsafe_allow_html=True
                )
                st.markdown(
                    f"<div style='font-size:24px; direction:rtl; text-align:right;'>{outputs['editor']}</div>",
                    unsafe_allow_html=True
                )
                try:
                    outpusstat = calculate_summary_statistics(input_text, outputs['editor'])
                    text_length = outpusstat[0]
                    summary_length = outpusstat[1]
                    summary_to_text_percentage = outpusstat[2]
                    score=evaluate_arabic_summary(input_text, outputs['editor'])
                    bert = score['F1-Score']
                    sentence = score['SentenceBert']
                    st.info(
                            f"**BERT F1-Score**: {bert:.3f}"
                            f"\n**SentenceBert Similarity**: {sentence:.3f}"
                            f"\n**Text Length**: {text_length} words"
                            f"\n**Summary Length**: {summary_length} words"
                            f"\n**Summary to Text Percentagee**: {summary_to_text_percentage:.2f}%")
                except:
                    st.error("Failed to evaluate the summary.")

        # Column 2: A3LLM Summarization
        with col2:
            st.header("A3LLM Summarization")
            with st.chat_message("user", avatar='assistant'):
                st.markdown(
                    "<div style='font-size:28px; font-weight:bold; direction:rtl; text-align:right;'></div>",
                    unsafe_allow_html=True
                )
                st.markdown(
                    f"<div style='font-size:24px; direction:rtl; text-align:right;'>{summary}</div>",
                    unsafe_allow_html=True
                )
                try:
                    outpusstat = calculate_summary_statistics(input_text, summary)
                    text_length = outpusstat[0]
                    summary_length = outpusstat[1]
                    summary_to_text_percentage = outpusstat[2]
                    score=evaluate_arabic_summary(input_text, summary)
                    bert = score['F1-Score']
                    sentence = score['SentenceBert']
                    st.warning(
                            f"**BERT F1-Score**: {bert:.3f}"
                            f"\n**SentenceBert Similarity**: {sentence:.3f}"
                            f"\n**Text Length**: {text_length} words"
                            f"\n**Summary Length**: {summary_length} words"
                            f"\n**Summary to Text Percentage**: {summary_to_text_percentage:.2f}%")
                except:
                    st.error("Failed to evaluate the summary.")
            


    

# Main Application
def main():
    """
    Main function to render the application.
    """
    tabs = st.tabs(["Overview", "A3LLM", "Agentic Summarization", "Comparison"])
    
    # Render content based on the active tab
    with tabs[0]:
        Overview.func()
    
    with tabs[1]:
        render_a3llm_tab()
    
    with tabs[2]:
        render_agentic_tab()
    with tabs[3]:
        render_a3llm_vs_agentic()

if __name__ == "__main__":
    main()
