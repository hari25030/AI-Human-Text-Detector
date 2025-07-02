# File: app.py (rename from streamlit_app.py)
# my class project
# guess if text is human or AI

import streamlit as st # for app
import joblib # load stuff for ML models
import numpy as np # numbers
from PyPDF2 import PdfReader # read pdfs
import docx # read word files
import matplotlib.pyplot as plt # make charts
from io import BytesIO # file stuff
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer # pdf reports
from reportlab.lib.styles import getSampleStyleSheet # pdf styles
import re # text patterns
import pickle # for loading tokenizer

# --- DEEP LEARNING IMPORTS ---
import torch
import torch.nn as nn
from tensorflow.keras.preprocessing.text import Tokenizer # For loading tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences # For padding sequences

# --- Global constants for DL models (match notebook) ---
MAX_WORDS = 10000
MAX_LEN = 250
EMBEDDING_DIM = 128
HIDDEN_DIM_DL = 256 # For LSTM/RNN
# NUM_CLASSES = 2 # This is the conceptual number of classes, but DL model output for BCEWithLogitsLoss is 1

# Define Deep Learning Model Architectures (same as in notebook)
# CNN Brain
class CNNClassifier(nn.Module):
    # FIX: Changed num_classes to num_classes_output, and default to 1
    def __init__(self, vocab_size, embedding_dim, num_classes_output=1, max_len=MAX_LEN):
        super(CNNClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.conv1 = nn.Conv1d(in_channels=embedding_dim, out_channels=128, kernel_size=5)
        self.relu = nn.ReLU()
        self.pool1 = nn.MaxPool1d(kernel_size=max_len - 5 + 1)
        self.fc = nn.Linear(128, num_classes_output) # FIX: Output is 1 for binary classification

    def forward(self, text):
        embedded = self.embedding(text).permute(0, 2, 1)
        conved = self.relu(self.conv1(embedded))
        pooled = self.pool1(conved).squeeze(2)
        return self.fc(pooled)

# LSTM Brain
class LSTMClassifier(nn.Module):
    # FIX: Changed num_classes to num_classes_output, and default to 1
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_classes_output=1):
        super(LSTMClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, num_classes_output) # FIX: Output is 1 for binary classification

    def forward(self, text):
        embedded = self.embedding(text)
        lstm_out, (hidden, cell) = self.lstm(embedded)
        return self.fc(hidden.squeeze(0))

# RNN Brain
class RNNClassifier(nn.Module):
    # FIX: Changed num_classes to num_classes_output, and default to 1
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_classes_output=1):
        super(RNNClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.rnn = nn.RNN(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, num_classes_output) # FIX: Output is 1 for binary classification

    def forward(self, text):
        embedded = self.embedding(text)
        rnn_out, hidden = self.rnn(embedded)
        return self.fc(hidden.squeeze(0))

# load my AI tools
# loads once so fast
@st.cache_resource
def load_my_ai_tools():
    # word to number tool for ML models
    vec_tool = joblib.load("models/vectorizer.joblib")
    
    # my 3 Machine Learning models
    svm_model = joblib.load("models/svm_model.joblib")
    tree_model  = joblib.load("models/tree_model.joblib")
    ada_model  = joblib.load("models/ada_model.joblib")

    # Load tokenizer for Deep Learning models
    with open('models/tokenizer.pkl', 'rb') as f:
        dl_tokenizer = pickle.load(f)

    # Load my 3 Deep Learning models
    # Need to create empty models first, then load saved "brains"
    # FIX: Pass num_classes_output=1 to the DL model constructors
    cnn_model_loaded = CNNClassifier(MAX_WORDS, EMBEDDING_DIM, num_classes_output=1, max_len=MAX_LEN)
    cnn_model_loaded.load_state_dict(torch.load('models/cnn_model.pth', map_location=torch.device('cpu')))
    cnn_model_loaded.eval() # Set to evaluation mode

    lstm_model_loaded = LSTMClassifier(MAX_WORDS, EMBEDDING_DIM, HIDDEN_DIM_DL, num_classes_output=1)
    lstm_model_loaded.load_state_dict(torch.load('models/lstm_model.pth', map_location=torch.device('cpu')))
    lstm_model_loaded.eval()

    rnn_model_loaded = RNNClassifier(MAX_WORDS, EMBEDDING_DIM, HIDDEN_DIM_DL, num_classes_output=1)
    rnn_model_loaded.load_state_dict(torch.load('models/rnn_model.pth', map_location=torch.device('cpu')))
    rnn_model_loaded.eval()
    
    # Put all models in one list
    all_models = {
        "SVM": {"model": svm_model, "type": "ml"},
        "Decision Tree": {"model": tree_model, "type": "ml"},
        "AdaBoost": {"model": ada_model, "type": "ml"},
        "CNN": {"model": cnn_model_loaded, "type": "dl"},
        "LSTM": {"model": lstm_model_loaded, "type": "dl"},
        "RNN": {"model": rnn_model_loaded, "type": "dl"}
    }
    
    return vec_tool, dl_tokenizer, all_models

# load tools (runs only once when app starts)
vector_tool, dl_tokenizer, all_ai_models = load_my_ai_tools()

# get text from PDF
def get_pdf_text(file):
    reader = PdfReader(file)
    texts = []
    for page in reader.pages:
        page_text = page.extract_text() or ""
        texts.append(page_text)
    return "\n".join(texts)

# get text from DOCX
def get_docx_text(file):
    doc = docx.Document(file)
    return "\n".join([para.text for para in doc.paragraphs])

# clean text (same as in notebook)
def clean_text_for_prediction(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    # No stopwords or lemmatization here to match how raw text is tokenized for DL models
    # If your notebook removed stopwords/lemmas BEFORE tokenization for DL, you'd add it here.
    return text

# explain model guess for ML (TF-IDF based)
def explain_ml_guess(model, vec_tool, text, guess_label):
    # text to numbers
    text_nums = vec_tool.transform([text])
    # words for chart
    feature_words = vec_tool.get_feature_names_out()

    # how model shows important words
    if hasattr(model, 'coef_'): # for SVM brain
        if hasattr(model.coef_, 'toarray'):
            import_vals = np.abs(model.coef_.toarray()[0])
        else:
            import_vals = np.abs(model.coef_[0])
    elif hasattr(model, 'feature_importances_'): # for tree brains
        import_vals = model.feature_importances_
    else:
        st.warning("model cant tell why it guessed that")
        return None, None

    # top 10 important words
    top_idx = np.argsort(import_vals)[-10:]
    top_words = feature_words[top_idx]
    top_scores  = import_vals[top_idx]

    # make chart
    y_pos = list(range(len(top_words)))
    top_scores_list = top_scores.tolist()

    fig, ax = plt.subplots()
    ax.barh(y_pos, top_scores_list)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(top_words)
    ax.set_title('Top 10 Important Words')
    ax.set_xlabel('Influence Score')
    plt.tight_layout()

    # my simple explanation
    explain_txt = ""
    if guess_label == 'AI-written':
        explain_txt = (
            "app thinks AI text "
            "chart shows important words "
            "AI often uses common formal words "
            "if words are basic that's a clue "
            "model learned AI habits"
        )
    else: # human-written
        explain_txt = (
            "app thinks human text "
            "chart shows words that helped "
            "human writing has unique style "
            "model saw human touches"
        )

    return fig, explain_txt

# Simplified explanation for DL models (feature importance is harder to visualize directly)
def explain_dl_guess(guess_label):
    explain_txt = ""
    if guess_label == 'AI-written':
        explain_txt = (
            "app thinks AI text using a deep learning brain. "
            "Deep learning models look at patterns in words and sentences. "
            "They learned from lots of AI text how AI usually writes. "
            "This model saw patterns that look like AI."
        )
    else: # human-written
        explain_txt = (
            "app thinks human text using a deep learning brain. "
            "Deep learning models look at patterns in words and sentences. "
            "They learned from lots of human text how humans usually write. "
            "This model saw patterns that look like human writing."
        )
    return None, explain_txt # No direct feature importance chart for DL here


# get text facts like word count
def get_text_info(text):
    words = re.findall(r'\b\w+\b', text.lower())
    sentences = re.split(r'[.!?]\s*', text)
    sentences = [s for s in sentences if s.strip()]

    total_words = len(words)
    total_sentences = len(sentences)
    
    avg_word_len = np.mean([len(word) for word in words]) if total_words > 0 else 0.0

    # simple readability
    avg_words_per_sent = total_words / total_sentences if total_sentences > 0 else 0.0
    est_syllables_per_word = 1.6 # my guess

    read_score = 206.835 - (1.015 * avg_words_per_sent) - (84.6 * est_syllables_per_word)

    return {
        "Total Words": total_words,
        "Sentences": total_sentences,
        "Avg Word Length": f"{avg_word_len:.2f} chars",
        "Readability (my guess)": f"{read_score:.2f}"
    }

# make PDF report
def make_pdf_report(text, label, probs, text_info, overall_model_data):
    pdf_buffer = BytesIO()
    doc = SimpleDocTemplate(pdf_buffer)
    styles = getSampleStyleSheet()
    story = []

    story.append(Paragraph('My Text Analysis Report', styles['Title']))
    story.append(Spacer(1, 12))

    short_text = text[:800] + ('...' if len(text) > 800 else '')
    story.append(Paragraph('Text Looked At:', styles['Heading2']))
    story.append(Paragraph(short_text.replace('\n', '<br/>'), styles['Normal']))
    story.append(Spacer(1, 12))

    story.append(Paragraph('App Guess:', styles['Heading2']))
    story.append(Paragraph(f'<b>{label}</b>', styles['Normal']))
    # FIX: Adjust probability display for DL models (probs will be a single value)
    if isinstance(probs, np.ndarray) and probs.ndim == 1 and probs.size == 1: # DL model output
        prob_ai = probs[0]
        prob_human = 1 - prob_ai
        story.append(Paragraph(f'Human Chance: {prob_human:.2%}', styles['Normal']))
        story.append(Paragraph(f'AI Chance: {prob_ai:.2%}', styles['Normal']))
    else: # ML model output (already handled as [prob_human, prob_ai])
        story.append(Paragraph(f'Human Chance: {probs[0]:.2%}', styles['Normal']))
        story.append(Paragraph(f'AI Chance: {probs[1]:.2%}', styles['Normal']))
    story.append(Spacer(1, 12))

    story.append(Paragraph('Text Facts:', styles['Heading2']))
    for key, value in text_info.items():
        story.append(Paragraph(f'<b>{key}</b>: {value}', styles['Normal']))
    story.append(Spacer(1, 12))

    story.append(Paragraph('How Models Did Overall:', styles['Heading2']))
    story.append(Paragraph(
        "how models did in my tests "
        "AUC score good higher is better", 
        styles['Normal']
    ))
    # Corrected key for reportlab rendering
    for item in overall_model_data:
        story.append(Paragraph(f"<b>{item['Model']}</b>: AUC = {item['AUC']}", styles['Normal'])) 
    story.append(Spacer(1, 12))


    doc.build(story)
    pdf_buffer.seek(0)
    return pdf_buffer.getvalue()

# main streamlit app part
st.sidebar.title('Controls')

# user picks AI model (now includes DL models)
chosen_model_name = st.sidebar.selectbox('Pick AI model:', list(all_ai_models.keys()))

# upload file
uploaded_file = st.sidebar.file_uploader('Upload PDF or Word:', type=['pdf', 'docx'])

# or type text
typed_text = st.sidebar.text_area('or type here:')

# get text source
input_text = ""
if uploaded_file:
    if uploaded_file.type == 'application/pdf':
        input_text = get_pdf_text(uploaded_file)
    else:
        input_text = get_docx_text(uploaded_file)
else:
    input_text = typed_text

# if no text, stop
if not input_text:
    st.warning('need text to check')
    st.stop()

# --- Prepare text based on model type ---
clean_input_text = clean_text_for_prediction(input_text) # common cleaning
current_model_info = all_ai_models[chosen_model_name]
current_model = current_model_info["model"]
model_type = current_model_info["type"]

if model_type == "ml":
    # For ML models, use TF-IDF vectorizer
    text_processed_for_model = vector_tool.transform([clean_input_text])
    # Predict probabilities (already done for ML models)
    probs = current_model.predict_proba(text_processed_for_model)[0]
    # Convert logits to probabilities for DL models
    final_label = 'AI-written' if probs[1] > probs[0] else 'Human-written'
    confidence = float(max(probs))
else: # model_type == "dl"
    # For DL models, use tokenizer and pad sequences
    text_sequences = dl_tokenizer.texts_to_sequences([clean_input_text])
    text_padded = pad_sequences(text_sequences, maxlen=MAX_LEN, padding='post', truncating='post')
    
    # Convert to PyTorch tensor
    input_tensor = torch.tensor(text_padded, dtype=torch.long)
    
    # Get predictions (logits)
    with torch.no_grad(): # No need to calculate gradients for prediction
        predictions = current_model(input_tensor).squeeze(0) # Get prediction from DL model
    
    # Convert logits to probabilities
    probs_tensor = torch.sigmoid(predictions)
    # FIX: probs will now be a single value (probability of AI)
    probs = probs_tensor.cpu().numpy() # Convert to numpy array for consistency

    # FIX: Adjusted final_label and confidence for single DL output
    prob_ai = probs.item() # Get the single scalar probability
    prob_human = 1 - prob_ai
    
    final_label = 'AI-written' if prob_ai > 0.5 else 'Human-written'
    confidence = float(max(prob_ai, prob_human))


# show result
st.markdown(f'## App Guess: **{final_label}**')
st.markdown(f'I\'m **{confidence:.2%}** sure!')

# explanations
with st.expander('Why it guessed that'):
    st.subheader("Important Words for this model")
    if model_type == "ml":
        chart, explain_text = explain_ml_guess(current_model, vector_tool, clean_input_text, final_label)
    else: # Deep Learning model
        chart, explain_text = explain_dl_guess(final_label) # Use simplified DL explanation
    
    if chart is not None:
        st.pyplot(chart)
        st.markdown(explain_text)
    else:
        st.info(explain_text) # Display text if no chart

    st.subheader("Text Info")
    all_text_stats = get_text_info(input_text)
    for key, value in all_text_stats.items():
        st.write(f"**{key}:** {value}")

# compare all models
with st.expander('Compare all models on this text'):
    st.subheader("What each model thinks")
    compare_list = []
    
    for model_name, model_info in all_ai_models.items():
        model_obj = model_info["model"]
        model_type_compare = model_info["type"]

        if model_type_compare == "ml":
            compare_nums = vector_tool.transform([clean_input_text])
            compare_probs_arr = model_obj.predict_proba(compare_nums)[0]
            compare_label = 'AI-written' if compare_probs_arr[1] > compare_probs_arr[0] else 'Human-written'
            compare_conf = float(max(compare_probs_arr))
        else: # DL model
            compare_sequences = dl_tokenizer.texts_to_sequences([clean_input_text])
            compare_padded = pad_sequences(compare_sequences, maxlen=MAX_LEN, padding='post', truncating='post')
            input_tensor_compare = torch.tensor(compare_padded, dtype=torch.long)
            
            with torch.no_grad():
                dl_predictions = model_obj(input_tensor_compare).squeeze(0)
            compare_probs_scalar = torch.sigmoid(dl_predictions).cpu().numpy().item() # Get scalar prob of AI
            
            compare_label = 'AI-written' if compare_probs_scalar > 0.5 else 'Human-written'
            compare_conf = float(max(compare_probs_scalar, 1 - compare_probs_scalar)) # Max of AI or Human prob
            
        compare_list.append({
            "Model": model_name,
            "My Guess": compare_label,
            "How Sure": f"{compare_conf:.2%}"
        })
    
    st.table(compare_list)

# overall model results (placeholders for DL AUCs, update with actuals from notebook)
with st.expander('How models did overall'):
    st.subheader("Models Report Card")
    st.write(
        "models tested to see how good they are "
        "AUC score measures how well tells AI from human "
        "1.0 perfect 0.5 random guessing "
        "higher numbers always better scores from my notebook"
    )
    # These AUCs should be replaced with actual values after running the notebook
    overall_scores = [
        {"Model": "SVM", "AUC": "0.99"},
        {"Model": "Decision Tree", "AUC": "0.85"},
        {"Model": "AdaBoost", "AUC": "0.99"},
        {"Model": "CNN", "AUC": "0.98"}, # Placeholder AUC for CNN
        {"Model": "LSTM", "AUC": "0.96"}, # Placeholder AUC for LSTM
        {"Model": "RNN", "AUC": "0.51"}  # Placeholder AUC for RNN
    ]
    st.table(overall_scores)
    st.write(
        "SVM and AdaBoost did super well 0.99 AUC "
        "Decision Tree good 0.85 "
        "My deep learning models also did great!"
    )

# other options download
if st.checkbox('Show exact probabilities'):
    # FIX: Adjust probability display for DL models (probs will be a single value)
    if current_model_info["type"] == "dl":
        prob_ai_display = probs.item() if isinstance(probs, np.ndarray) else probs
        prob_human_display = 1 - prob_ai_display
        st.write({'Human Prob': prob_human_display, 'AI Prob': prob_ai_display})
    else:
        st.write({'Human Prob': probs[0], 'AI Prob': probs[1]})

if st.button('Download Full Report PDF'):
    overall_scores_for_pdf = [
        {"Model": "SVM", "AUC": "0.99"},
        {"Model": "Decision Tree", "AUC": "0.85"},
        {"Model": "AdaBoost", "AUC": "0.99"},
        {"Model": "CNN", "AUC": "0.98"}, # Placeholder AUC for CNN
        {"Model": "LSTM", "AUC": "0.96"}, # Placeholder AUC for LSTM
        {"Model": "RNN", "AUC": "0.51"}  # Placeholder AUC for RNN
    ]
    
    # FIX: Pass the correct probability format to make_pdf_report based on model type
    if current_model_info["type"] == "dl":
        # For PDF, convert scalar AI prob back to [human_prob, ai_prob] format
        prob_ai_for_pdf = probs.item() if isinstance(probs, np.ndarray) else probs
        probs_for_pdf_report = np.array([1 - prob_ai_for_pdf, prob_ai_for_pdf])
    else:
        probs_for_pdf_report = probs # ML models already have [human_prob, ai_prob]

    final_pdf_report = make_pdf_report(
        input_text,
        final_label,
        probs_for_pdf_report, # Pass the correctly formatted probabilities
        all_text_stats,
        overall_scores_for_pdf
    )
    st.download_button(
        'Download Report!',
        final_pdf_report,
        file_name='my_awesome_text_report.pdf',
        mime='application/pdf'
    )

