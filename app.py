import os
import sys
from flask import Flask, render_template, request, jsonify
from dotenv import load_dotenv

from src.prompt import system_prompt
from src.helper import download_hugging_face_embeddings

load_dotenv()

app = Flask(__name__)

PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')
OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')
INDEX_NAME = os.environ.get('PINECONE_INDEX_NAME', 'medical-chatbot')

rag_chain = None
init_error = None


def init_rag_system():
    global rag_chain, init_error
    
    # Reload environment in case .env was edited
    load_dotenv(override=True)
    pinecone_key = os.environ.get('PINECONE_API_KEY', '').strip()
    openai_key = os.environ.get('OPENAI_API_KEY', '').strip()

    if not pinecone_key or pinecone_key.startswith("xxxx"):
        init_error = "PINECONE_API_KEY is not configured in .env file."
        return False

    if not openai_key or openai_key.startswith("xxxx"):
        init_error = "OPENAI_API_KEY is not configured in .env file."
        return False

    os.environ["PINECONE_API_KEY"] = pinecone_key
    os.environ["OPENAI_API_KEY"] = openai_key

    try:
        from langchain_pinecone import PineconeVectorStore
        from langchain_openai import ChatOpenAI
        from langchain.chains import create_retrieval_chain
        from langchain.chains.combine_documents import create_stuff_documents_chain
        from langchain_core.prompts import ChatPromptTemplate

        print("[INFO] Loading HuggingFace embedding model...")
        embeddings = download_hugging_face_embeddings()

        print(f"[INFO] Connecting to Pinecone index: '{INDEX_NAME}'...")
        docsearch = PineconeVectorStore.from_existing_index(
            index_name=INDEX_NAME,
            embedding=embeddings
        )

        retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k": 3})
        model_name = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
        base_url = os.environ.get("OPENAI_BASE_URL", "").strip() or None
        chat_model = ChatOpenAI(model=model_name, base_url=base_url)

        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "{input}"),
        ])

        question_answer_chain = create_stuff_documents_chain(chat_model, prompt)
        rag_chain = create_retrieval_chain(retriever, question_answer_chain)
        init_error = None
        print("[SUCCESS] RAG system initialized and ready!")
        return True

    except Exception as e:
        init_error = str(e)
        print(f"[WARNING] Could not initialize RAG chain: {e}")
        return False


# Attempt initialization at start
init_rag_system()


@app.route("/")
def index():
    return render_template('chat.html')


@app.route("/get", methods=["GET", "POST"])
def chat():
    global rag_chain, init_error

    if request.method == "POST":
        msg = request.form.get("msg", "").strip()
    else:
        msg = request.args.get("msg", "").strip()

    if not msg:
        return "Please type a question."

    # Try initializing if not yet ready
    if rag_chain is None:
        init_rag_system()

    if rag_chain is not None:
        try:
            response = rag_chain.invoke({"input": msg})
            answer = response.get("answer", "No answer found.")
            return str(answer)
        except Exception as e:
            err_str = str(e)
            if "insufficient_quota" in err_str or "credit_balance_exhausted" in err_str:
                return (
                    "<strong>OpenAI Notice:</strong><br>"
                    "Your OpenAI API key has 0 remaining credits (account balance exhausted).<br>"
                    "To fix this:<br>"
                    "1. Add credits to your OpenAI account at <a href='https://platform.openai.com/settings/organization/billing' target='_blank'>OpenAI Billing</a>.<br>"
                    "2. <em>Or</em> use a completely free provider like <strong>Groq</strong> (Free Llama 3) by setting <code>OPENAI_BASE_URL=https://api.groq.com/openai/v1</code> in <code>.env</code>."
                )
            return f"Error executing query: {err_str}"
    else:
        return (
            "<strong>Medical Assistant:</strong><br>"
            "Hello! The web application is working properly.<br><br>"
            "To connect to the live AI Medical knowledge base, please configure your API credentials:<br>"
            "1. Open <code>.env</code> file in your project directory.<br>"
            "2. Add your <code>PINECONE_API_KEY</code> and <code>OPENAI_API_KEY</code>.<br>"
            "3. Run <code>python store_index.py</code> once to index the medical encyclopedia.<br><br>"
            f"<em>Current Status: {init_error or 'Keys missing in .env'}</em>"
        )


@app.route("/status")
def status():
    return jsonify({
        "status": "online",
        "rag_ready": rag_chain is not None,
        "error": init_error,
        "index_name": INDEX_NAME
    })


if __name__ == '__main__':
    port = int(os.environ.get("PORT", 8080))
    print(f"\n==============================================")
    print(f" Medical Chatbot running at: http://localhost:{port}")
    print(f"==============================================\n")
    app.run(host="0.0.0.0", port=port, debug=True)
