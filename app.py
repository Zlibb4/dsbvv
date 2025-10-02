# streamlit_smolLM2_app.py
"""
Streamlit deployment for JustToTryModels/sssss (SmolLM2 135M Instruct) with
- OOD classifier integration (IamPradeep/Query_Classifier_DistilBERT)
- Placeholder handling (static + dynamic via spaCy NER)
- UI state-management (disable inputs while generating)

Usage:
- Put this file at the root of your Streamlit repo (the repo connected to Streamlit Cloud / Streamlit for Teams).
- Add the requirements.txt (included below) to the repo.
- If models are private, add your Hugging Face token to Streamlit secrets as `HF_TOKEN`.

Notes:
- The code tries to load models onto GPU (device_map='auto') when available. If that fails, it falls back to CPU.
- en_core_web_trf is heavy. If not available in the runtime we fall back to en_core_web_sm. If you prefer trf, install the model in your environment or include the wheel/tar.gz in requirements.
- This implementation generates full text then streams it out word-by-word in the UI (reliable across different model/tokenizer implementations).

"""

import streamlit as st
import torch
import time
import random
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSequenceClassification
import spacy
from typing import Tuple, Dict

# -----------------------------
# CONFIG
# -----------------------------
SMOLLM_ID = "JustToTryModels/sssss"
CLASSIFIER_ID = "IamPradeep/Query_Classifier_DistilBERT"

# Fallback responses (same as in your example)
FALLBACK_RESPONSES = [
    "I’m sorry, but I am unable to assist with this request. If you need help regarding event tickets, I’d be happy to support you.",
    "Apologies, but I am not able to provide assistance on this matter. Please let me know if you require help with event tickets.",
    "Unfortunately, I cannot assist with this. However, I am here to help with any event ticket-related concerns you may have.",
    "Regrettably, I am unable to assist with this request. If there's anything I can do regarding event tickets, feel free to ask.",
    "I regret that I am unable to assist in this case. Please reach out if you need support related to event tickets."
]

# -----------------------------
# HELPERS: placeholders
# -----------------------------
STATIC_PLACEHOLDERS = {
    "{{WEBSITE_URL}}": "[website](https://github.com/MarpakaPradeepSai)",
    "{{SUPPORT_TEAM_LINK}}": "[support team](https://github.com/MarpakaPradeepSai)",
    "{{CONTACT_SUPPORT_LINK}}": "[support team](https://github.com/MarpakaPradeepSai)",
    "{{EVENT_ORGANIZER_OPTION}}": "<b>Event Organizer</b>",
    "{{CANCEL_TICKET_BUTTON}}": "<b>Cancel Ticket</b>",
}


def replace_placeholders(response: str, dynamic_placeholders: Dict[str, str], static_placeholders: Dict[str, str]) -> str:
    for placeholder, value in static_placeholders.items():
        response = response.replace(placeholder, value)
    for placeholder, value in dynamic_placeholders.items():
        response = response.replace(placeholder, value)
    return response


def extract_dynamic_placeholders(user_question: str, nlp) -> Dict[str, str]:
    doc = nlp(user_question)
    dynamic_placeholders = {}
    for ent in doc.ents:
        # Choose labels that match your spaCy model; adjust if needed
        if ent.label_ == "EVENT":
            event_text = ent.text.title()
            dynamic_placeholders['{{EVENT}}'] = f"<b>{event_text}</b>"
        elif ent.label_ in ("GPE", "LOC"):
            city_text = ent.text.title()
            dynamic_placeholders['{{CITY}}'] = f"<b>{city_text}</b>"
    if '{{EVENT}}' not in dynamic_placeholders:
        dynamic_placeholders['{{EVENT}}'] = "event"
    if '{{CITY}}' not in dynamic_placeholders:
        dynamic_placeholders['{{CITY}}'] = "city"
    return dynamic_placeholders

# -----------------------------
# MODEL LOADING (cached resources)
# -----------------------------

@st.cache_resource
def load_spacy_model():
    # Prefer the transformer pipeline if available; otherwise use the small model
    try:
        nlp = spacy.load("en_core_web_trf")
        return nlp
    except Exception:
        # Fallback: try the small model
        try:
            nlp = spacy.load("en_core_web_sm")
            return nlp
        except Exception:
            # Last resort: create a blank English pipeline (NER won't be available)
            nlp = spacy.blank("en")
            return nlp

@st.cache_resource(show_spinner=False)
def load_classifier_model():
    hf_token = None
    if 'HF_TOKEN' in st.secrets:
        hf_token = st.secrets['HF_TOKEN']

    try:
        tokenizer = AutoTokenizer.from_pretrained(CLASSIFIER_ID, use_auth_token=hf_token)
        model = AutoModelForSequenceClassification.from_pretrained(CLASSIFIER_ID, use_auth_token=hf_token)
        return model, tokenizer
    except Exception as e:
        st.error(f"Failed to load classifier model: {e}")
        return None, None

@st.cache_resource(show_spinner=False)
def load_smol_model_and_tokenizer():
    """Try loading on available accelerators (GPU) with device_map='auto' and fall back to CPU gracefully."""
    hf_token = None
    if 'HF_TOKEN' in st.secrets:
        hf_token = st.secrets['HF_TOKEN']

    # First try: device_map='auto' with reduced dtype (if supported)
    try:
        model = AutoModelForCausalLM.from_pretrained(
            SMOLLM_ID,
            torch_dtype=torch.bfloat16,
        )
        tokenizer = AutoTokenizer.from_pretrained(SMOLLM_ID, use_auth_token=hf_token)
        return model, tokenizer
    except Exception as e:
        # Second attempt: try loading to CPU (safe)
        try:
            st.warning("Could not load model to accelerator with device_map='auto'. Falling back to CPU. This will be slower.")
            model = AutoModelForCausalLM.from_pretrained(SMOLLM_ID, torch_dtype=torch.float32, device_map=None, low_cpu_mem_usage=False, use_auth_token=hf_token)
            tokenizer = AutoTokenizer.from_pretrained(SMOLLM_ID, use_auth_token=hf_token)
            return model, tokenizer
        except Exception as e2:
            st.error(f"Failed to load SmolLM model: {e2}")
            return None, None

# -----------------------------
# UTILITY: OOD check
# -----------------------------

def is_ood(query: str, clf_model, clf_tokenizer) -> bool:
    if clf_model is None or clf_tokenizer is None:
        return False
    device = torch.device("cuda" if torch.cuda.is_available() and next(clf_model.parameters()).is_cuda else "cpu")
    try:
        inputs = clf_tokenizer(query, return_tensors="pt", truncation=True, padding=True, max_length=256)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        clf_model.to(device)
        clf_model.eval()
        with torch.no_grad():
            outputs = clf_model(**inputs)
        pred_id = torch.argmax(outputs.logits, dim=1).item()
        return pred_id == 1
    except Exception:
        # On any failure, treat as in-domain (to avoid false OOD when classifier fails)
        return False

# -----------------------------
# TEXT GENERATION
# -----------------------------

def build_prompt(instruction: str, tokenizer) -> str:
    # Some tokenizers provide a chat template (as you used in the CLI version).
    # We'll try to use it if available; otherwise form a simple instruction->response prompt.
    if hasattr(tokenizer, 'apply_chat_template'):
        try:
            chat_format = [{"role": "user", "content": instruction}]
            return tokenizer.apply_chat_template(chat_format, tokenize=False, add_generation_prompt=True)
        except Exception:
            pass
    # Fallback prompt
    prompt = f"User: {instruction}\nAssistant:" 
    return prompt


def generate_response_smol(model, tokenizer, instruction: str, max_new_tokens=256) -> str:
    if model is None or tokenizer is None:
        return "Sorry — the language model is not available right now."

    device = next(model.parameters()).device if any(True for _ in model.parameters()) else torch.device('cpu')

    prompt_text = build_prompt(instruction, tokenizer)
    inputs = tokenizer(prompt_text, return_tensors="pt", truncation=True, padding=True).to(device)

    # Use generate (non-streaming) then we stream to UI word-by-word
    with torch.no_grad():
        outputs = model.generate(
            input_ids=inputs['input_ids'],
            attention_mask=inputs['attention_mask'],
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.4,
            top_p=0.95,
            pad_token_id=tokenizer.eos_token_id
        )

    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # If we used a template that includes the prompt, strip the prompt prefix
    if decoded.startswith(prompt_text):
        decoded = decoded[len(prompt_text):]
    # Some tokenizers include 'Assistant:' label — try to remove it
    if decoded.strip().lower().startswith('assistant:'):
        decoded = decoded.strip()[len('assistant:'):].strip()

    return decoded.strip()

# -----------------------------
# STREAMLIT UI
# -----------------------------

def main():
    st.set_page_config(page_title="SmolLM2 Event Ticketing Assistant", layout='wide')

    st.markdown("""
    <style>
    .stButton>button { background: linear-gradient(90deg, #ff8a00, #e52e71); color: white !important; border: none; border-radius: 12px; padding: 8px 14px; }
    </style>
    """, unsafe_allow_html=True)

    st.title("Advanced Event Ticketing Assistant — SmolLM2")

    # Example queries
    example_queries = [
        "How do I buy a ticket?",
        "How can I upgrade my ticket for the upcoming event in Hyderabad?",
        "How do I change my personal details on my ticket?",
        "How do I get a refund?",
    ]

    # Initialize session state
    if 'models_loaded' not in st.session_state:
        st.session_state.models_loaded = False
    if 'generating' not in st.session_state:
        st.session_state.generating = False
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []

    # Load models
    if not st.session_state.models_loaded:
        with st.spinner("Loading models and resources... this may take a minute"):
            nlp = load_spacy_model()
            smol_model, smol_tokenizer = load_smol_model_and_tokenizer()
            clf_model, clf_tokenizer = load_classifier_model()

            if smol_model is None or smol_tokenizer is None:
                st.error("SmolLM model failed to load. Please check logs and model id or HF token if model is private.")
            else:
                st.session_state.models_loaded = True
                st.session_state.nlp = nlp
                st.session_state.smol_model = smol_model
                st.session_state.smol_tokenizer = smol_tokenizer
                st.session_state.clf_model = clf_model
                st.session_state.clf_tokenizer = clf_tokenizer
                st.experimental_rerun()

    # Main interface
    if st.session_state.models_loaded:
        st.write("Ask me about ticket bookings, cancellations, refunds, or any event-related inquiries!")

        selected_query = st.selectbox("Choose a query from examples:", ["Choose your question"] + example_queries, disabled=st.session_state.generating)
        process_query_button = st.button("Ask this question", disabled=st.session_state.generating)

        nlp = st.session_state.nlp
        smol_model = st.session_state.smol_model
        smol_tokenizer = st.session_state.smol_tokenizer
        clf_model = st.session_state.clf_model
        clf_tokenizer = st.session_state.clf_tokenizer

        # Render chat history
        last_role = None
        for message in st.session_state.chat_history:
            with st.chat_message(message['role'], avatar=message.get('avatar', None)):
                st.markdown(message['content'], unsafe_allow_html=True)
            last_role = message['role']

        def handle_prompt(prompt_text: str):
            if not prompt_text or not prompt_text.strip():
                st.toast("⚠️ Please enter or select a question.")
                return
            st.session_state.generating = True
            prompt_text = prompt_text[0].upper() + prompt_text[1:]
            st.session_state.chat_history.append({"role": "user", "content": prompt_text, "avatar": "👤"})
            # rerun to render the user message and lock UI
            st.experimental_rerun()

        def process_generation():
            last_message = st.session_state.chat_history[-1]["content"]

            with st.chat_message("assistant", avatar="🤖"):
                message_placeholder = st.empty()
                full_response = ""

                # OOD check
                if is_ood(last_message, clf_model, clf_tokenizer):
                    full_response = random.choice(FALLBACK_RESPONSES)
                else:
                    with st.spinner("Generating response..."):
                        dynamic_placeholders = extract_dynamic_placeholders(last_message, nlp)
                        response_smol = generate_response_smol(smol_model, smol_tokenizer, last_message)
                        full_response = replace_placeholders(response_smol, dynamic_placeholders, STATIC_PLACEHOLDERS)

                # Stream to UI word-by-word
                streamed = ""
                for word in full_response.split():
                    streamed += word + " "
                    message_placeholder.markdown(streamed + "⬤", unsafe_allow_html=True)
                    time.sleep(0.03)
                message_placeholder.markdown(full_response, unsafe_allow_html=True)

            st.session_state.chat_history.append({"role": "assistant", "content": full_response, "avatar": "🤖"})
            st.session_state.generating = False

        # Triggers
        if process_query_button:
            if selected_query != "Choose your question":
                handle_prompt(selected_query)
            else:
                st.error("⚠️ Please select your question from the dropdown.")

        if prompt := st.chat_input("Enter your own question:", disabled=st.session_state.generating):
            handle_prompt(prompt)

        # If we are in generating state then call the generation function (after rerun showed the UI locked)
        if st.session_state.generating:
            process_generation()
            # after processing, rerun to re-enable the UI and show final state
            st.experimental_rerun()

        # Clear chat button
        if st.session_state.chat_history:
            if st.button("Clear Chat", disabled=st.session_state.generating):
                st.session_state.chat_history = []
                st.session_state.generating = False
                st.experimental_rerun()


if __name__ == '__main__':
    main()
