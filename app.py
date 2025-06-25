# File: app.py (rename from streamlit_app.py)
# my class project
# guess if text is human or AI

import streamlit as st # for app
import joblib # load stuff
import numpy as np # numbers
from PyPDF2 import PdfReader # read pdfs
import docx # read word files
import matplotlib.pyplot as plt # make charts
from io import BytesIO # file stuff
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer # pdf reports
from reportlab.lib.styles import getSampleStyleSheet # pdf styles
import re # text patterns

# load my AI tools
# loads once so fast
@st.cache_resource
def load_my_ai_tools():
    # word to number tool
    vec_tool = joblib.load("models/vectorizer.joblib")
    
    # my 3 AI models
    svm_model = joblib.load("models/svm_model.joblib")
    tree_model  = joblib.load("models/tree_model.joblib")
    ada_model  = joblib.load("models/ada_model.joblib")
    
    # put models in a list
    all_models = {"SVM": svm_model, "Decision Tree": tree_model, "AdaBoost": ada_model}
    
    return vec_tool, all_models

# load tools
vector_tool, all_ai_models = load_my_ai_tools()

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

# explain model guess
def explain_guess(model, vec_tool, text, guess_label):
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
    # FIX IS ON THE NEXT LINE: Changed 'Model Name' to 'Model'
    for item in overall_model_data:
        story.append(Paragraph(f"<b>{item['Model']}</b>: AUC = {item['AUC']}", styles['Normal'])) # Corrected key here
    story.append(Spacer(1, 12))


    doc.build(story)
    pdf_buffer.seek(0)
    return pdf_buffer.getvalue()

# main streamlit app part
st.sidebar.title('Controls')

# user picks AI model
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

# make prediction
current_model = all_ai_models[chosen_model_name]
text_to_nums = vector_tool.transform([input_text])
probs = current_model.predict_proba(text_to_nums)[0]
final_label = 'AI-written' if probs[1] > probs[0] else 'Human-written'
confidence = float(max(probs))

# show result
st.markdown(f'## App Guess: **{final_label}**')
st.markdown(f'I\'m **{confidence:.2%}** sure!')

# explanations
with st.expander('Why it guessed that'):
    st.subheader("Important Words for this model")
    chart, explain_text = explain_guess(current_model, vector_tool, input_text, final_label)
    
    if chart is not None:
        st.pyplot(chart)
        st.markdown(explain_text)
    else:
        st.info("cant show word importance")

    st.subheader("Text Info")
    all_text_stats = get_text_info(input_text)
    for key, value in all_text_stats.items():
        st.write(f"**{key}:** {value}")

# compare all models
with st.expander('Compare all models on this text'):
    st.subheader("What each model thinks")
    compare_list = []
    
    for model_name, model_obj in all_ai_models.items():
        compare_nums = vector_tool.transform([input_text])
        compare_probs = model_obj.predict_proba(compare_nums)[0]
        compare_label = 'AI-written' if compare_probs[1] > compare_probs[0] else 'Human-written'
        compare_conf = float(max(compare_probs))
        
        compare_list.append({
            "Model": model_name,
            "My Guess": compare_label,
            "How Sure": f"{compare_conf:.2%}"
        })
    
    st.table(compare_list)

# overall model results
with st.expander('How models did overall'):
    st.subheader("Models Report Card")
    st.write(
        "models tested to see how good they are "
        "AUC score measures how well tells AI from human "
        "1.0 perfect 0.5 random guessing "
        "higher numbers always better scores from my notebook"
    )
    overall_scores = [
        {"Model": "SVM", "AUC": "0.99"},
        {"Model": "Decision Tree", "AUC": "0.85"},
        {"Model": "AdaBoost", "AUC": "0.99"}
    ]
    st.table(overall_scores)
    st.write(
        "SVM and AdaBoost did super well 0.99 AUC "
        "Decision Tree good 0.85 "
        "SVM AdaBoost top students"
    )

# other options download
if st.checkbox('Show exact probabilities'):
    st.write({'Human Prob': probs[0], 'AI Prob': probs[1]})

if st.button('Download Full Report PDF'):
    overall_scores_for_pdf = [
        {"Model": "SVM", "AUC": "0.99"},
        {"Model": "Decision Tree", "AUC": "0.85"},
        {"Model": "AdaBoost", "AUC": "0.99"}
    ]
    
    final_pdf_report = make_pdf_report(
        input_text,
        final_label,
        probs,
        all_text_stats,
        overall_scores_for_pdf
    )
    st.download_button(
        'Download Report!',
        final_pdf_report,
        file_name='my_awesome_text_report.pdf',
        mime='application/pdf'
    )
