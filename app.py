__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import streamlit as st
from groq import Groq
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="METU IE Staj Danışmanı",
    page_icon="🎓",
    layout="centered"
)

st.title("🎓 METU IE Sanal Staj Danışmanı")
st.caption("IE 300 / IE 400 yaz stajı süreçleri hakkında sorularınızı yanıtlamak için buradayım.")

# --- SECURITY AND API SETUP ---
# We never hardcode the API key; we fetch it securely via Streamlit Secrets.
try:
    GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
except KeyError:
    st.error("🚨 GROQ_API_KEY not found! Please check your Streamlit secrets.toml file.")
    st.stop()

client = Groq(api_key=GROQ_API_KEY)

# --- VECTOR DATABASE LOADING ---
# Using cache to prevent Streamlit from reloading the database on every user interaction.
@st.cache_resource
def load_vector_db():
    try:
        # IMPORTANT NOTE: You must use the exact same embedding model here 
        # that you used when initially creating the vectors.
        # Defaulting to SentenceTransformers as it is standard and open-source.
        # Update the model_name if you used OpenAI or a different HF model.
        embeddings = HuggingFaceEmbeddings(model_name="intfloat/multilingual-e5-small")


        # Using only relative file paths to ensure seamless deployment on Streamlit Cloud.
        db = Chroma(persist_directory="./vector_db", embedding_function=embeddings)
        return db


    except Exception as e:
        st.error(f"🚨 An error occurred while loading the vector database: {e}")
        st.stop()

vector_store = load_vector_db()

# --- SYSTEM PROMPT (STRICT GUARDRAILS) ---
SYSTEM_PROMPT = """Sen, ODTÜ Endüstri Mühendisliği öğrencilerinin yaz stajı (IE 300 / IE 400) süreçleri için resmi "Sanal Danışman"ısın. 

KESİN KURALLAR:
1. SADECE kullanıcıya sağlanan DÖKÜMAN BAĞLAMI'nı kullanarak cevap ver.
2. DÖKÜMAN BAĞLAMI'nda sorunun cevabı YOKSA veya soru stajla TAMAMEN ALAKASIZSA (örn: hava durumu, yemek vb.): "Bu soru ODTÜ IE staj danışmanlığı kapsamı dışındadır veya elimdeki güncel kılavuzlarda cevabı bulunmamaktadır." de ve DUR. Asla genel bilgi kullanma.
3. Nazik, profesyonel ve Türkçe yanıt ver."""

# --- CHAT HISTORY MANAGEMENT ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# Render previous messages on the screen
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- USER INPUT AND RAG LOGIC ---
if prompt := st.chat_input("Staj başvurumu ne zamana kadar yapmalıyım?"):
    
    # 1. Display the user's message on the screen and save it to the session history
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 2. Fetch relevant documents from the Vector DB via semantic search (Top 3 docs)
    results = vector_store.similarity_search(prompt, k=3)
    
    # Combine the retrieved documents to construct the context string
    retrieved_context = "\n\n---\n\n".join([doc.page_content for doc in results])

    # 3. Prepare the message payload for the Groq API
    api_messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    
    # Append previous messages (user/assistant) so the LLM remembers the conversational context.
    # We slice the last 4 messages (excluding the current one) to manage token limits effectively.
    for msg in st.session_state.messages[-4:-1]:
        api_messages.append({"role": msg["role"], "content": msg["content"]})

    # Model için en kritik adım: Son kullanıcının sorusunu ve veritabanı bağlamını (RAG) harmanla
    rag_enriched_prompt = f"Kullanıcının Sorusu: {prompt}\n\nArama Sonucunda Gelen DÖKÜMAN BAĞLAMI:\n{retrieved_context}"
    api_messages.append({"role": "user", "content": rag_enriched_prompt})

    # 4. Groq API Call and Streaming the Response to the UI
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        
        try:
            # Llama-3-8b-8192 is highly efficient and capable for RAG use cases.
            stream = client.chat.completions.create(
                model="llama3-8b-8192",
                messages=api_messages,
                temperature=0.0, # ZERO temperature prevents hallucinations and ensures deterministic RAG outputs
                stream=True
            )
            
            # Stream the response chunks to create a typing effect
            for chunk in stream:
                if chunk.choices[0].delta.content is not None:
                    full_response += chunk.choices[0].delta.content
                    message_placeholder.markdown(full_response + "▌")
            
            # Remove the blinking cursor once the stream is complete
            message_placeholder.markdown(full_response)
            
        except Exception as e:
            full_response = f"🚨 An error occurred while communicating with the Groq API: {e}"
            message_placeholder.error(full_response)

    # 5. Append the assistant's final response to the chat history
    st.session_state.messages.append({"role": "assistant", "content": full_response})