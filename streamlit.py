import time
from tenacity import retry, stop_after_attempt, wait_exponential
import streamlit as st
import google.generativeai as genai
import toml

# Set page config
st.set_page_config(
    page_title="Eapen's  Chatbot",
    page_icon="🤖",
    layout="wide"
)

# --- Configuration ---
try:
    # Load configuration
    secrets = toml.load(".streamlit/secrets.toml")
    GEMINI_API_KEY = secrets.get("GEMINI_API_KEY")
    
    if not GEMINI_API_KEY:
        st.error("GEMINI_API_KEY not found in secrets.toml")
        st.stop()
        
    genai.configure(api_key=GEMINI_API_KEY)
    
    # Get list of available models
    available_models = [m.name for m in genai.list_models() 
                       if 'generateContent' in m.supported_generation_methods]
    
    if not available_models:
        st.error("No available generative models found. Check your API access.")
        st.stop()
        
except Exception as e:
    st.error(f"🔐 Configuration failed: {str(e)}")
    st.stop()

# --- App Title and Description ---
st.title("Eapen's Gemini Chatbot 🤖")
st.caption("Powered by Google's Gemini")

# --- Model Selection ---
# Ordered by priority (lightest models first)
model_options = {
    "Gemini 1.0 Pro": "models/gemini-1.0-pro",
    "Gemini 1.5 Flash": "models/gemini-1.5-flash-latest",
    "Gemini 1.5 Pro": "models/gemini-1.5-pro-latest"
}

# Filter to only show available models and prioritize less quota-intensive models
available_options = {k:v for k,v in model_options.items() if v in available_models}
if not available_options:
    st.error("No supported models available in your region")
    st.stop()

# Select first available model by default (prioritizing lighter models)
default_model_index = 0
selected_model = st.sidebar.selectbox(
    "Choose Model",
    list(available_options.keys()),
    index=default_model_index
)

# --- Retry Configuration ---
@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    reraise=True
)
def send_message_with_retry(chat_session, prompt):
    """Send message with retry logic for rate limits"""
    try:
        return chat_session.send_message(prompt, stream=True)
    except Exception as e:
        if "quota" in str(e).lower() or "429" in str(e):
            time.sleep(5)  # Additional delay for quota errors
            raise
        raise

# --- Chat History Management ---
if "messages" not in st.session_state:
    st.session_state.messages = []
    
if "gemini_model" not in st.session_state:
    st.session_state["gemini_model"] = available_options[selected_model]

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- Chat Input ---
if prompt := st.chat_input("Ask me anything..."):
    # Add user message to chat
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Generate response
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        
        try:
            # Initialize model
            model = genai.GenerativeModel(st.session_state["gemini_model"])
            
            # Start or continue chat session
            if "chat_session" not in st.session_state:
                st.session_state.chat_session = model.start_chat(history=[])
            
            # Stream the response with retry
            response = send_message_with_retry(st.session_state.chat_session, prompt)
            
            # Display streamed response
            for chunk in response:
                full_response += chunk.text
                message_placeholder.markdown(full_response + "▌")
            
            message_placeholder.markdown(full_response)
            
        except Exception as e:
            error_msg = str(e)
            if "quota" in error_msg.lower() or "429" in error_msg:
                error_msg = "⚠️ Rate limit exceeded. Please try again later or switch to a different model."
            else:
                error_msg = f"⚠️ Error: {error_msg}"
            
            st.error(error_msg)
            full_response = "Sorry, I couldn't process your request. Please try again."
            message_placeholder.markdown(full_response)
    
    # Add assistant response to history
    st.session_state.messages.append({"role": "assistant", "content": full_response})

# --- Sidebar Controls ---
with st.sidebar:
    st.header("Controls")
    
    if st.button("Clear Chat History"):
        st.session_state.messages = []
        if "chat_session" in st.session_state:
            del st.session_state["chat_session"]
        st.rerun()
    
    st.divider()
    st.markdown("### Available Models:")
    for model_name, model_path in available_options.items():
        st.markdown(f"- {model_name}")
    
    st.divider()
    st.markdown("### Tips:")
    st.markdown("1. Start with Gemini 1.0 Pro or 1.5 Flash for better availability")
    st.markdown("2. Longer responses may hit rate limits faster")
    st.markdown("3. Clear history if you switch models")
    
    st.divider()
    st.markdown("### Quota Status:")
    st.markdown("Free tier has limited requests per minute")
    st.markdown("[View quotas](https://ai.google.dev/gemini-api/docs/rate-limits)")