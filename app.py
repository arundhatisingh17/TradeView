
import streamlit as st
import pandas as pd
import pytz
from datetime import datetime
import requests
import time
from haystack import Pipeline
from haystack_integrations.document_stores.elasticsearch import ElasticsearchDocumentStore
from haystack_integrations.components.retrievers.elasticsearch import ElasticsearchBM25Retriever
from haystack.components.builders.prompt_builder import PromptBuilder
from haystack.components.generators import HuggingFaceAPIGenerator
from haystack.components.converters import PyPDFToDocument
from haystack.components.writers import DocumentWriter
from haystack.components.preprocessors import DocumentSplitter
from elasticsearch import Elasticsearch, NotFoundError
from huggingface_hub import login
import tempfile
import os
from dotenv import load_dotenv

load_dotenv('/content/drive/MyDrive/my_secrets.env')
elastic_password = os.getenv('elastic_password')
huggingface_token = os.getenv('HF_token')


st.set_page_config(layout="wide")
st.title("📈 Tech Stock Performance Dashboard")


# Elastic config
elastic_username = "elastic"
endpoint = "https://a6f3f2a9bd694e308e3da03571231683.us-central1.gcp.cloud.es.io"
index_name = "stocks_data"


client = Elasticsearch(hosts=endpoint, basic_auth=(elastic_username, elastic_password))


try:
    client.indices.delete(index=index_name)
except NotFoundError:
    pass


client.indices.create(index=index_name)

document_store = ElasticsearchDocumentStore(
    hosts=endpoint,
    basic_auth=(elastic_username, elastic_password),
    index=index_name
)


login(token=huggingface_token)
retriever = ElasticsearchBM25Retriever(document_store=document_store)
generator = HuggingFaceAPIGenerator(
    api_type="serverless_inference_api",
    api_params={"model": "microsoft/Phi-3.5-mini-instruct", "max_new_tokens": 200}
)


template = """
<|system|>
You are a helpful assistant.<|end|>
<|user|>
Given the following information, answer the question.

Context: 
{% for document in documents %}
    {{ document.content }}
{% endfor %}

Question: {{ query }}?<|end|>
<|assistant|>"""


rag_pipeline = Pipeline()
rag_pipeline.add_component("retriever", retriever)
rag_pipeline.add_component("prompt_builder", PromptBuilder(template=template, required_variables=["documents", "query"]))
rag_pipeline.add_component("llm", generator)
rag_pipeline.connect("retriever", "prompt_builder.documents")
rag_pipeline.connect("prompt_builder", "llm")


def generate_summary_from_pdf(uploaded_file, query):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name

    index_pipeline = Pipeline()
    index_pipeline.add_component("converter", PyPDFToDocument())
    index_pipeline.add_component("splitter", DocumentSplitter(split_by="word", split_length=150))
    index_pipeline.add_component("writer", DocumentWriter(document_store))
    index_pipeline.connect("converter", "splitter")
    index_pipeline.connect("splitter", "writer")

    index_pipeline.run({"converter": {"sources": [tmp_path]}})

    result = rag_pipeline.run({
        "retriever": {"query": query},
        "prompt_builder": {"query": query}
    })

    return result["llm"]["replies"][0]



tech_tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA", "AMD", "CRM", "ADBE", "INTC", "ORCL"]
API_KEY = "d0idvshr01qrfsafijd0d0idvshr01qrfsafijdg"
central = pytz.timezone("US/Central")

def fetch_stock_quote(ticker):
    time.sleep(1.5) 
    url = f"https://finnhub.io/api/v1/quote?symbol={ticker}&token={API_KEY}"
    resp = requests.get(url)
    return resp.json()

col1, col2 = st.columns(2)

with col1:
    st.subheader("📊 Stock Snapshot Viewer")
    tickers = ["Select a company: "] + tech_tickers
    selected = st.selectbox("Company:", tickers)

    if selected != "Select a company: ":
        quote = fetch_stock_quote(selected)
        change = quote["c"] - quote["o"]
        percent_change = (change / quote["o"]) * 100 if quote["o"] != 0 else 0

        ts = pd.to_datetime(quote["t"], unit="s").tz_localize("UTC").astimezone(central)
        formatted_time = ts.strftime("%Y-%m-%d %I:%M %p %Z")

        st.metric("Close Price", f"${quote['c']:.2f}", f"{change:+.2f} ({percent_change:+.2f}%)")

        st.write("**Details**")
        st.write({
            "Open": f"${quote['o']:.2f}",
            "High": f"${quote['h']:.2f}",
            "Low": f"${quote['l']:.2f}",
            "Close": f"${quote['c']:.2f}",
            "Volume": quote.get("v", "N/A"),
            "Timestamp": formatted_time
        })

with col2:
    if selected != "Select a company: ":
        selected_quote = fetch_stock_quote(selected)
        selected_close = selected_quote["c"]

        st.subheader("📊 Compare with Peer Company")
        peer_candidates = [
            t for t in tech_tickers
            if t != selected and abs(fetch_stock_quote(t)["c"] - selected_close) / selected_close <= 0.05
        ]

        if peer_candidates:
            selected_peer = st.selectbox("Peer Company:", ["Select peer..."] + peer_candidates)

            if selected_peer != "Select peer...":
                peer_quote = fetch_stock_quote(selected_peer)
                peer_change = peer_quote["c"] - peer_quote["o"]
                peer_pct = (peer_change / peer_quote["o"]) * 100 if peer_quote["o"] != 0 else 0

                ts = pd.to_datetime(peer_quote["t"], unit="s").tz_localize("UTC").astimezone(central)
                formatted_time = ts.strftime("%Y-%m-%d %I:%M %p %Z")

                st.metric("Close Price", f"${peer_quote['c']:.2f}", f"{peer_change:+.2f} ({peer_pct:+.2f}%)")

                st.write("**Details**")
                st.write({
                    "Open": f"${peer_quote['o']:.2f}",
                    "High": f"${peer_quote['h']:.2f}",
                    "Low": f"${peer_quote['l']:.2f}",
                    "Close": f"${peer_quote['c']:.2f}",
                    "Volume": peer_quote.get("v", "N/A"),
                    "Timestamp": formatted_time
                })
        else:
            st.info("No peers found within 5% of the selected company's close price.")



if selected != "Select a company: ":
    st.subheader(f"📄 Upload {selected}'s Financial Report to Estimate Investment Timing")

    uploaded_file = st.file_uploader("Upload a PDF financial report", type=["pdf"])

    if uploaded_file:
        user_query = st.text_input(
            "Ask a question about the financial report:",
            value=f"Summarize {selected}'s overall financial performance."
        )

        if st.button("Generate Summary"):
            with st.spinner("Analyzing document..."):
              response = generate_summary_from_pdf(uploaded_file, user_query)
              st.success("Generated Summary:")
              st.write(response)
                      
        else:
            st.info("Please select a company to upload its financial report.")

