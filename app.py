# streamlit_app.py
import streamlit as st
import torch
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    AutoModelForCausalLM
)
import spacy
import time
import random

# =============================
# MODEL AND CONFIGURATION SETUP
# =============================

# Hugging Face model IDs
SMOLLM_MODEL_ID = "HuggingFaceTB/SmolLM2-135M-Instruct"
CLASSIFIER_ID = "IamPradeep/Query_Classifier_DistilBERT"

# Random OOD Fallback Responses
fallback_responses = [
    "I'm sorry, but I am unable to assist with this request. If you need help regarding event tickets, I'd be happy to support you.",
    "Apologies, but I am not able to provide assistance on this matter. Please let me know if you require help with event tickets.",
    "Unfortunately, I cannot assist with this. However, I am here to help with any event ticket-related concerns you may have.",
    "Regrettably, I am unable to assist with this request. If there's anything I can do regarding event tickets, feel free to ask.",
    "I regret that I am unable to assist in this case. Please reach out if you need support related to event tickets.",
    "Apologies, but this falls outside the scope of my support. I'm here if you need any help with event ticket issues.",
    "I'm sorry, but I cannot assist with this particular topic. If you have questions about event tickets, I'd be glad to help.",
    "I regret that I'm unable to provide assistance here. Please let me know how I can support you with event ticket matters.",
    "Unfortunately, I am not equipped to assist with this. If you need help with event tickets, I am here for that.",
    "I apologize, but I cannot help with this request. However, I'd be happy to assist with anything related to event tickets.",
    "I'm sorry, but I'm unable to support this request. If it's about event tickets, I'll gladly help however I can.",
    "This matter falls outside the assistance I can offer. Please let me know if you need help with event ticket-related inquiries.",
    "Regrettably, this is not something I can assist with. I'm happy to help with any event ticket questions you may have.",
    "I'm unable to provide support for this issue. However, I can assist with concerns regarding event tickets.",
    "I apologize, but I cannot help with this matter. If your inquiry is related to event tickets, I'd be more than happy to assist.",
    "I regret that I am unable to offer help in this case. I am, however, available for any event ticket-related questions.",
    "Unfortunately, I'm not able to assist with this. Please let me know if there's anything I can do regarding event tickets.",
    "I'm sorry, but I cannot assist with this topic. However, I'm here to help with any event ticket concerns you may have.",
    "Apologies, but this request falls outside of my support scope. If you need help with event tickets, I'm happy to assist.",
    "I'm afraid I can't help with this matter. If there's anything related to event tickets you need, feel free to reach out.",
    "This is beyond what I can assist with at the moment. Let me know if there's anything I can do to help with event tickets.",
    "Sorry, I'm unable to provide support on this issue. However, I'd be glad to assist with event ticket-related topics.",
    "Apologies, but I can't assist with this. Please let me know if you have any event ticket inquiries I can help with.",
    "I'm unable to help with this matter. However, if you need assistance with event tickets, I'm here for you.",
    "Unfortunately, I can't support this request. I'd be happy to assist with anything related to event tickets instead.",
    "I'm sorry, but I can't help with this. If your concern is related to event tickets, I'll do my best to assist.",
    "Apologies, but this issue is outside of my capabilities. However, I'm available to help with event ticket-related requests.",
    "I regret that I cannot assist with this particular matter. Please let me know how I can support you regarding event tickets.",
    "I'm sorry, but I'm not able to help in this instance. I am, however, ready to assist with any questions about event tickets.",
    "Unfortunately, I'm unable to help with this topic. Let me know if there's anything event ticket-related I can support you with."
]

# =============================
# MODEL LOADING FUNCTIONS
# =============================

@st.cache_resource
def load_spacy_model():
    """Load spaCy NER model for extracting dynamic placeholders"""
    nlp = spacy.load("en_core_web_trf")
    return nlp

@st.cache_resource(show_spinner=False)
def load_smollm_model_and_tokenizer():
    """Load SmolLM2 model and tokenizer with multiple fallback strategies"""
    try:
        # Strategy 1: Try with minimal parameters (most compatible)
        st.info("Loading SmolLM2 model... (Method 1)")
        tokenizer = AutoTokenizer.from_pretrained(
            SMOLLM_MODEL_ID,
            trust_remote_code=True
        )
        
        model = AutoModelForCausalLM.from_pretrained(
            SMOLLM_MODEL_ID,
            trust_remote_code=True,
            torch_dtype = torch.bfloat16
        )
        
        # Set pad token if not already set
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        st.success("✅ Model loaded successfully!")
        return model, tokenizer
        
    except Exception as e1:
        st.warning(f"Method 1 failed: {str(e1)[:100]}...")
        
        try:
            # Strategy 2: Try with explicit torch_dtype=float32
            st.info("Trying alternative loading method... (Method 2)")
            tokenizer = AutoTokenizer.from_pretrained(
                SMOLLM_MODEL_ID,
                trust_remote_code=True
            )
            
            model = AutoModelForCausalLM.from_pretrained(
                SMOLLM_MODEL_ID,
                trust_remote_code=True,
                torch_dtype=torch.float32,
                low_cpu_mem_usage=True
            )
            
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            
            st.success("✅ Model loaded successfully!")
            return model, tokenizer
            
        except Exception as e2:
            st.warning(f"Method 2 failed: {str(e2)[:100]}...")
            
            try:
                # Strategy 3: Try without low_cpu_mem_usage
                st.info("Trying alternative loading method... (Method 3)")
                tokenizer = AutoTokenizer.from_pretrained(
                    SMOLLM_MODEL_ID,
                    trust_remote_code=True
                )
                
                model = AutoModelForCausalLM.from_pretrained(
                    SMOLLM_MODEL_ID,
                    trust_remote_code=True
                )
                
                if tokenizer.pad_token is None:
                    tokenizer.pad_token = tokenizer.eos_token
                
                st.success("✅ Model loaded successfully!")
                return model, tokenizer
                
            except Exception as e3:
                st.error(f"Failed to load SmolLM2 model. All methods failed.")
                st.error(f"Error details: {str(e3)}")
                st.info("**Troubleshooting tips:**")
                st.info("1. Check if the model 'JustToTryModels/sssss' exists on Hugging Face")
                st.info("2. Verify you have access to the model (it might be private)")
                st.info("3. Try upgrading transformers: `pip install --upgrade transformers`")
                return None, None

@st.cache_resource(show_spinner=False)
def load_classifier_model():
    """Load binary classifier for OOD detection"""
    try:
        tokenizer = AutoTokenizer.from_pretrained(CLASSIFIER_ID)
        model = AutoModelForSequenceClassification.from_pretrained(CLASSIFIER_ID)
        return model, tokenizer
    except Exception as e:
        st.error(f"Failed to load classifier model from Hugging Face Hub. Error: {e}")
        return None, None

def is_ood(query: str, model, tokenizer):
    """Check if query is out-of-domain"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    inputs = tokenizer(query, return_tensors="pt", truncation=True, padding=True, max_length=256)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
    pred_id = torch.argmax(outputs.logits, dim=1).item()
    return pred_id == 1  # True if OOD (label 1)

# =============================
# PLACEHOLDER HANDLING FUNCTIONS
# =============================

static_placeholders = {
    "{{WEBSITE_URL}}": "[website](https://github.com/MarpakaPradeepSai)",
    "{{SUPPORT_TEAM_LINK}}": "[support team](https://github.com/MarpakaPradeepSai)",
    "{{CONTACT_SUPPORT_LINK}}": "[support team](https://github.com/MarpakaPradeepSai)",
    "{{SUPPORT_CONTACT_LINK}}": "[support team](https://github.com/MarpakaPradeepSai)",
    "{{CANCEL_TICKET_SECTION}}": "<b>Ticket Cancellation</b>",
    "{{CANCEL_TICKET_OPTION}}": "<b>Cancel Ticket</b>",
    "{{GET_REFUND_OPTION}}": "<b>Get Refund</b>",
    "{{UPGRADE_TICKET_INFORMATION}}": "<b>Upgrade Ticket Information</b>",
    "{{TICKET_SECTION}}": "<b>Ticketing</b>",
    "{{CANCELLATION_POLICY_SECTION}}": "<b>Cancellation Policy</b>",
    "{{CHECK_CANCELLATION_POLICY_OPTION}}": "<b>Check Cancellation Policy</b>",
    "{{APP}}": "<b>App</b>",
    "{{CHECK_CANCELLATION_FEE_OPTION}}": "<b>Check Cancellation Fee</b>",
    "{{CHECK_REFUND_POLICY_OPTION}}": "<b>Check Refund Policy</b>",
    "{{CHECK_PRIVACY_POLICY_OPTION}}": "<b>Check Privacy Policy</b>",
    "{{SAVE_BUTTON}}": "<b>Save</b>",
    "{{EDIT_BUTTON}}": "<b>Edit</b>",
    "{{CANCELLATION_FEE_SECTION}}": "<b>Cancellation Fee</b>",
    "{{CHECK_CANCELLATION_FEE_INFORMATION}}": "<b>Check Cancellation Fee Information</b>",
    "{{PRIVACY_POLICY_LINK}}": "<b>Privacy Policy</b>",
    "{{REFUND_SECTION}}": "<b>Refund</b>",
    "{{REFUND_POLICY_LINK}}": "<b>Refund Policy</b>",
    "{{CUSTOMER_SERVICE_SECTION}}": "<b>Customer Service</b>",
    "{{DELIVERY_PERIOD_INFORMATION}}": "<b>Delivery Period</b>",
    "{{EVENT_ORGANIZER_OPTION}}": "<b>Event Organizer</b>",
    "{{FIND_TICKET_OPTION}}": "<b>Find Ticket</b>",
    "{{FIND_UPCOMING_EVENTS_OPTION}}": "<b>Find Upcoming Events</b>",
    "{{CONTACT_SECTION}}": "<b>Contact</b>",
    "{{SEARCH_BUTTON}}": "<b>Search</b>",
    "{{SUPPORT_SECTION}}": "<b>Support</b>",
    "{{EVENTS_SECTION}}": "<b>Events</b>",
    "{{EVENTS_PAGE}}": "<b>Events</b>",
    "{{TYPE_EVENTS_OPTION}}": "<b>Type Events</b>",
    "{{PAYMENT_SECTION}}": "<b>Payment</b>",
    "{{PAYMENT_OPTION}}": "<b>Payment</b>",
    "{{CANCELLATION_SECTION}}": "<b>Cancellation</b>",
    "{{CANCELLATION_OPTION}}": "<b>Cancellation</b>",
    "{{REFUND_OPTION}}": "<b>Refund</b>",
    "{{TRANSFER_TICKET_OPTION}}": "<b>Transfer Ticket</b>",
    "{{REFUND_STATUS_OPTION}}": "<b>Refund Status</b>",
    "{{DELIVERY_SECTION}}": "<b>Delivery</b>",
    "{{SELL_TICKET_OPTION}}": "<b>Sell Ticket</b>",
    "{{CANCELLATION_FEE_INFORMATION}}": "<b>Cancellation Fee Information</b>",
    "{{CUSTOMER_SUPPORT_PAGE}}": "<b>Customer Support</b>",
    "{{PAYMENT_METHOD}}": "<b>Payment</b>",
    "{{VIEW_PAYMENT_METHODS}}": "<b>View Payment Methods</b>",
    "{{VIEW_CANCELLATION_POLICY}}": "<b>View Cancellation Policy</b>",
    "{{SUPPORT_ SECTION}}": "<b>Support</b>",
    "{{CUSTOMER_SUPPORT_SECTION}}": "<b>Customer Support</b>",
    "{{HELP_SECTION}}": "<b>Help</b>",
    "{{TICKET_INFORMATION}}": "<b>Ticket Information</b>",
    "{{UPGRADE_TICKET_BUTTON}}": "<b>Upgrade Ticket</b>",
    "{{CANCEL_TICKET_BUTTON}}": "<b>Cancel Ticket</b>",
    "{{GET_REFUND_BUTTON}}": "<b>Get Refund</b>",
    "{{PAYMENTS_HELP_SECTION}}": "<b>Payments Help</b>",
    "{{PAYMENTS_PAGE}}": "<b>Payments</b>",
    "{{TICKET_DETAILS}}": "<b>Ticket Details</b>",
    "{{TICKET_INFORMATION_PAGE}}": "<b>Ticket Information</b>",
    "{{REPORT_PAYMENT_PROBLEM}}": "<b>Report Payment</b>",
    "{{TICKET_OPTIONS}}": "<b>Ticket Options</b>",
    "{{SEND_BUTTON}}": "<b>Send</b>",
    "{{PAYMENT_ISSUE_OPTION}}": "<b>Payment Issue</b>",
    "{{CUSTOMER_SUPPORT_PORTAL}}": "<b>Customer Support</b>",
    "{{UPGRADE_TICKET_OPTION}}": "<b>Upgrade Ticket</b>",
    "{{TICKET_AVAILABILITY_TAB}}": "<b>Ticket Availability</b>",
    "{{TRANSFER_TICKET_BUTTON}}": "<b>Transfer Ticket</b>",
    "{{TICKET_MANAGEMENT}}": "<b>Ticket Management</b>",
    "{{TICKET_STATUS_TAB}}": "<b>Ticket Status</b>",
    "{{TICKETING_PAGE}}": "<b>Ticketing</b>",
    "{{TICKET_TRANSFER_TAB}}": "<b>Ticket Transfer</b>",
    "{{CURRENT_TICKET_DETAILS}}": "<b>Current Ticket Details</b>",
    "{{UPGRADE_OPTION}}": "<b>Upgrade</b>",
    "{{CONNECT_WITH_ORGANIZER}}": "<b>Connect with Organizer</b>",
    "{{TICKETS_TAB}}": "<b>Tickets</b>",
    "{{ASSISTANCE_SECTION}}": "<b>Assistance Section</b>",
}

def replace_placeholders(response, dynamic_placeholders, static_placeholders):
    """Replace all placeholders in the response"""
    for placeholder, value in static_placeholders.items():
        response = response.replace(placeholder, value)
    for placeholder, value in dynamic_placeholders.items():
        response = response.replace(placeholder, value)
    return response

def extract_dynamic_placeholders(user_question, nlp):
    """Extract EVENT and CITY entities using spaCy NER"""
    doc = nlp(user_question)
    dynamic_placeholders = {}
    
    for ent in doc.ents:
        if ent.label_ == "EVENT":
            event_text = ent.text.title()
            dynamic_placeholders['{{EVENT}}'] = f"<b>{event_text}</b>"
        elif ent.label_ == "GPE":
            city_text = ent.text.title()
            dynamic_placeholders['{{CITY}}'] = f"<b>{city_text}</b>"
    
    # Set defaults if not found
    if '{{EVENT}}' not in dynamic_placeholders:
        dynamic_placeholders['{{EVENT}}'] = "event"
    if '{{CITY}}' not in dynamic_placeholders:
        dynamic_placeholders['{{CITY}}'] = "city"
    
    return dynamic_placeholders

# =============================
# RESPONSE GENERATION FUNCTION
# =============================

def generate_response(model, tokenizer, instruction, max_length=256):
    """Generate response using SmolLM2 with chat template format"""
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    try:
        # Try using chat template
        chat_format = [{"role": "user", "content": instruction}]
        input_text = tokenizer.apply_chat_template(
            chat_format, 
            tokenize=False, 
            add_generation_prompt=True
        )
    except Exception as e:
        # Fallback to simple instruction format
        input_text = f"<|im_start|>user\n{instruction}<|im_end|>\n<|im_start|>assistant\n"
    
    # Tokenize
    inputs = tokenizer(input_text, return_tensors="pt", padding=True).to(device)
    
    # Generate
    with torch.no_grad():
        outputs = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_length=max_length,
            num_return_sequences=1,
            temperature=0.4,
            top_p=0.95,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
    
    # Decode and extract response
    full_response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extract only the assistant's response
    if "<|im_start|>assistant" in full_response:
        response = full_response.split("<|im_start|>assistant")[-1].strip()
        response = response.replace("<|im_end|>", "").strip()
    elif "assistant\n" in full_response:
        response = full_response.split("assistant\n")[-1].strip()
    else:
        # Fallback: try to remove the input text
        response = full_response.replace(input_text, "").strip()
    
    return response

# =============================
# CSS AND UI SETUP
# =============================

st.set_page_config(page_title="Event Ticketing Chatbot", page_icon="🎫", layout="centered")

st.markdown(
    """
<style>
.stButton>button { 
    background: linear-gradient(90deg, #ff8a00, #e52e71); 
    color: white !important; 
    border: none; 
    border-radius: 25px; 
    padding: 10px 20px; 
    font-size: 1.2em; 
    font-weight: bold; 
    cursor: pointer; 
    transition: transform 0.2s ease, box-shadow 0.2s ease; 
    display: inline-flex; 
    align-items: center; 
    justify-content: center; 
    margin-top: 5px; 
    width: auto; 
    min-width: 100px; 
    font-family: 'Times New Roman', Times, serif !important; 
}
.stButton>button:hover { 
    transform: scale(1.05); 
    box-shadow: 0px 5px 15px rgba(0, 0, 0, 0.3); 
    color: white !important; 
}
.stButton>button:active { 
    transform: scale(0.98); 
}
* { 
    font-family: 'Times New Roman', Times, serif !important; 
}
div[data-testid="stHorizontalBlock"] div[data-testid="stButton"] button:nth-of-type(1) { 
    background: linear-gradient(90deg, #29ABE2, #0077B6); 
    color: white !important; 
}
.horizontal-line { 
    border-top: 2px solid #e0e0e0; 
    margin: 15px 0; 
}
div[data-testid="stChatInput"] { 
    box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2); 
    border-radius: 5px; 
    padding: 10px; 
    margin: 10px 0; 
}

.footer {
    position: fixed;
    left: 0;
    bottom: 0;
    width: 100%;
    background: var(--streamlit-background-color);
    color: gray;
    text-align: center;
    padding: 5px 0;
    font-size: 13px;
    z-index: 9999;
}
.main { 
    padding-bottom: 40px; 
}
</style>
    """, unsafe_allow_html=True
)

st.markdown(
    """
    <div class="footer">
        This is not a conversational AI. It is designed solely for <b>event ticketing</b> queries. Responses outside this scope may be inaccurate.
    </div>
    """,
    unsafe_allow_html=True
)

st.markdown("<h1 style='font-size: 43px;'>Advanced Event Ticketing Chatbot</h1>", unsafe_allow_html=True)

# =============================
# SESSION STATE INITIALIZATION
# =============================

if "models_loaded" not in st.session_state:
    st.session_state.models_loaded = False
if "generating" not in st.session_state:
    st.session_state.generating = False
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Example queries
example_queries = [
    "How do I buy a ticket?", 
    "How can I upgrade my ticket for the upcoming event in Hyderabad?",
    "How do I change my personal details on my ticket?", 
    "How can I find details about upcoming events?",
    "How do I contact customer service?", 
    "How do I get a refund?", 
    "What is the ticket cancellation fee?",
    "How can I track my ticket cancellation status?", 
    "How can I sell my ticket?"
]

# =============================
# MODEL LOADING
# =============================

if not st.session_state.models_loaded:
    with st.spinner("Loading models and resources... Please wait..."):
        try:
            nlp = load_spacy_model()
            smollm_model, smollm_tokenizer = load_smollm_model_and_tokenizer()
            clf_model, clf_tokenizer = load_classifier_model()

            if all([nlp, smollm_model, smollm_tokenizer, clf_model, clf_tokenizer]):
                st.session_state.models_loaded = True
                st.session_state.nlp = nlp
                st.session_state.model = smollm_model
                st.session_state.tokenizer = smollm_tokenizer
                st.session_state.clf_model = clf_model
                st.session_state.clf_tokenizer = clf_tokenizer
                st.rerun()
            else:
                st.error("Failed to load one or more models. Please refresh the page.")
        except Exception as e:
            st.error(f"Error loading models: {str(e)}")

# =============================
# MAIN CHAT INTERFACE
# =============================

if st.session_state.models_loaded:
    st.write("Ask me about ticket bookings, cancellations, refunds, or any event-related inquiries!")

    # Query selection dropdown
    selected_query = st.selectbox(
        "Choose a query from examples:", 
        ["Choose your question"] + example_queries,
        key="query_selectbox", 
        label_visibility="collapsed",
        disabled=st.session_state.generating
    )
    
    # Process query button
    process_query_button = st.button(
        "Ask this question", 
        key="query_button",
        disabled=st.session_state.generating
    )

    # Load models from session state
    nlp = st.session_state.nlp
    model = st.session_state.model
    tokenizer = st.session_state.tokenizer
    clf_model = st.session_state.clf_model
    clf_tokenizer = st.session_state.clf_tokenizer

    # Display chat history
    last_role = None
    for message in st.session_state.chat_history:
        if message["role"] == "user" and last_role == "assistant":
            st.markdown("<div class='horizontal-line'></div>", unsafe_allow_html=True)
        with st.chat_message(message["role"], avatar=message["avatar"]):
            st.markdown(message["content"], unsafe_allow_html=True)
        last_role = message["role"]

    # =============================
    # PROMPT HANDLING FUNCTIONS
    # =============================

    def handle_prompt(prompt_text):
        """Handle incoming prompt and lock UI"""
        if not prompt_text or not prompt_text.strip():
            st.toast("⚠️ Please enter or select a question.")
            return

        # Lock UI
        st.session_state.generating = True

        # Capitalize first letter
        prompt_text = prompt_text[0].upper() + prompt_text[1:] if prompt_text else prompt_text
        
        # Add user message to history
        st.session_state.chat_history.append({
            "role": "user", 
            "content": prompt_text, 
            "avatar": "👤"
        })

        # Rerun to display user message and lock inputs
        st.rerun()

    def process_generation():
        """Generate response and add to chat history"""
        last_message = st.session_state.chat_history[-1]["content"]

        with st.chat_message("assistant", avatar="🤖"):
            message_placeholder = st.empty()
            full_response = ""

            # Check if OOD
            if is_ood(last_message, clf_model, clf_tokenizer):
                full_response = random.choice(fallback_responses)
            else:
                with st.spinner("Generating response..."):
                    # Extract dynamic placeholders
                    dynamic_placeholders = extract_dynamic_placeholders(last_message, nlp)
                    
                    # Generate response
                    response_raw = generate_response(model, tokenizer, last_message)
                    
                    # Replace placeholders
                    full_response = replace_placeholders(
                        response_raw, 
                        dynamic_placeholders, 
                        static_placeholders
                    )

            # Stream the response word by word
            streamed_text = ""
            for word in full_response.split(" "):
                streamed_text += word + " "
                message_placeholder.markdown(streamed_text + "⬤", unsafe_allow_html=True)
                time.sleep(0.05)
            
            # Final display
            message_placeholder.markdown(full_response, unsafe_allow_html=True)

        # Add assistant message to history
        st.session_state.chat_history.append({
            "role": "assistant", 
            "content": full_response, 
            "avatar": "🤖"
        })
        
        # Unlock UI
        st.session_state.generating = False

    # =============================
    # HANDLE USER INPUT
    # =============================

    # Handle button click
    if process_query_button:
        if selected_query != "Choose your question":
            handle_prompt(selected_query)
        else:
            st.error("⚠️ Please select your question from the dropdown.")

    # Handle chat input
    if prompt := st.chat_input("Enter your own question:", disabled=st.session_state.generating):
        handle_prompt(prompt)

    # If generating, process the generation
    if st.session_state.generating:
        process_generation()
        st.rerun()

    # Clear chat button
    if st.session_state.chat_history:
        if st.button("Clear Chat", key="reset_button", disabled=st.session_state.generating):
            st.session_state.chat_history = []
            st.session_state.generating = False
            last_role = None
            st.rerun()
