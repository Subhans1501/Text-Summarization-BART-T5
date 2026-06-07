import streamlit as st
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
import os
import time
from ensemble import calculate_ensemble_winner

st.set_page_config(
    page_title="Ensemble AI Summarizer", 
    page_icon="🧠", 
    layout="wide",
    initial_sidebar_state="expanded"
)
@st.cache_resource(show_spinner=False)
def load_models():
    models = {}
    device = "cuda" if torch.cuda.is_available() else "cpu"

    base_dir = os.path.dirname(__file__)
    model_paths = {
        "BART": os.path.abspath(os.path.join(base_dir, '..', 'models', 'Model_BART')),
        "T5": os.path.abspath(os.path.join(base_dir, '..', 'models', 'Model_T5')),
        "PEGASUS": os.path.abspath(os.path.join(base_dir, '..', 'models', 'Model_Pegasus'))
    }
    
    for name, path in model_paths.items():
        if os.path.exists(path):
            try:
                tokenizer = AutoTokenizer.from_pretrained(path)
                model = AutoModelForSeq2SeqLM.from_pretrained(path).to(device)

                if name == "PEGASUS":
                    model.config.use_cache = True
                    
                models[name] = {"tokenizer": tokenizer, "model": model, "device": device}
            except Exception as e:
                pass 
    return models

with st.spinner("Initializing Neural Networks..."):
    active_models = load_models()
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2103/2103832.png", width=80)
    st.title("System Status")
    st.markdown("### Active Models:")
    
    if "BART" in active_models: st.success("🟢 BART (facebook/bart-large)")
    else: st.error("🔴 BART (Offline)")
        
    if "T5" in active_models: st.success("🟢 T5 (t5-base)")
    else: st.error("🔴 T5 (Offline)")
        
    if "PEGASUS" in active_models: st.success("🟢 PEGASUS (google/pegasus-xsum)")
    else: st.error("🔴 PEGASUS (Offline)")
        
    st.markdown("---")
    st.markdown("**Hardware:**")
    st.info(f"Processing via: **{'GPU (CUDA)' if torch.cuda.is_available() else 'CPU'}**")
    
st.title("Research Paper Title Generator")
st.markdown("""
Welcome to the Generative Ensemble Engine. Paste a complex academic abstract below. 
The system will deploy **multiple fine-tuned transformers** simultaneously and use an algorithmic judge to select the optimal scientific title.
""")
st.divider()

if not active_models:
    st.error("Critical Error: No models found. Please ensure your fine-tuned weights are inside the /models folder.")
    st.stop()

input_text = st.text_area(
    "Enter Abstract:", 
    height=250, 
    placeholder="Example: Machine learning is a field of inquiry devoted to understanding and building methods that 'learn'..."
)

col_spacer1, col_btn, col_spacer2 = st.columns([1, 1, 1])
with col_btn:
    generate_pressed = st.button("Generate Ensemble Summary", use_container_width=True, type="primary")

if generate_pressed:
    if len(input_text.split()) < 10:
        st.warning("Please enter a longer, valid academic abstract.")
    else:
        st.divider()
        st.markdown("### Transformer Outputs")

        cols = st.columns(len(active_models))
        generated_results = {}

        for index, (model_name, components) in enumerate(active_models.items()):
            with cols[index]:
                with st.spinner(f"{model_name} is synthesizing..."):
                    tokenizer = components["tokenizer"]
                    model = components["model"]
                    device = components["device"]

                    process_text = "summarize: " + input_text if model_name == "T5" else input_text
                    
                    inputs = tokenizer(process_text, max_length=512, truncation=True, return_tensors="pt").to(device)
                    
                    summary_ids = model.generate(
                        inputs["input_ids"], 
                        max_length=30,      
                        min_length=5, 
                        num_beams=4,        
                        early_stopping=True
                    )
                    
                    title = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
                    generated_results[model_name] = title

                    st.markdown(f"**{model_name}**")
                    st.info(f"{title}")

        time.sleep(0.5)

        st.divider()

        winning_model, winning_title = calculate_ensemble_winner(input_text, generated_results)

        st.balloons()
        
        st.markdown("<h2 style='text-align: center;'> Algorithmic Ensemble Winner</h2>", unsafe_allow_html=True)

        st.success(f"**Selected Model:** {winning_model}")
        st.markdown(f"""
        > ### {winning_title}
        """)
        
        st.caption(f"Reasoning: The ensemble judge determined that **{winning_model}** retained the highest density of critical keywords from the original source text.")