import gradio as gr
from rapidfuzz import fuzz
import fitz
import easyocr
import numpy as np
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# ===== GLOBALS =====
reader = easyocr.Reader(['en'])
db = None


# ===== PROCESS PDF =====
def process_pdf(file):
    global db

    try:
        if file is None:
            return "⚠️ Please upload a PDF first."

        # 🔥 HANDLE BOTH CASES (HF + Local)
        if hasattr(file, "name"):
            doc = fitz.open(file.name)
        else:
            doc = fitz.open(file)

        text = ""

        for i, page in enumerate(doc[:50]):
            page_text = page.get_text()

            if len(page_text.strip()) < 50:
                pix = page.get_pixmap()
                img = np.frombuffer(
                    pix.samples, dtype=np.uint8
                ).reshape(pix.height, pix.width, pix.n)

                result = reader.readtext(img)
                page_text = " ".join([r[1] for r in result])

            text += page_text + "\n"

        if not text.strip():
            return "⚠️ No text found in PDF."

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=100
        )
        chunks = splitter.split_text(text)

        if len(chunks) == 0:
            return "⚠️ Failed to process text."

        embeddings = HuggingFaceEmbeddings()
        db = FAISS.from_texts(chunks, embeddings)

        return "✅ PDF processed! Ask your question now."

    except Exception as e:
        print("PROCESS ERROR:", e)
        return f"❌ Error processing PDF: {str(e)}"
    

# ===== ANSWER FUNCTION =====
def get_answer(query):
    global db

    try:
        if db is None:
            return "⚠️ Upload and process a PDF first.", ""

        docs = db.similarity_search(query, k=3)

        best_sentence = ""
        best_score = 0
        source = ""

        for doc in docs:
            sentences = doc.page_content.split(".")

            for sent in sentences:
                sent_clean = sent.strip()

                if len(sent_clean) < 20:
                    continue

                score = fuzz.partial_ratio(query.lower(), sent_clean.lower())

                # boost definition-like lines
                if "is" in sent_clean.lower() or "mode" in sent_clean.lower():
                    score += 10

                if score > best_score:
                    best_score = score
                    best_sentence = sent_clean
                    source = doc.page_content[:200]

        # 🔥 FIX: never return empty
        if not best_sentence:
            best_sentence = "❌ No relevant answer found."

        if not source:
            source = "No source available."

        return best_sentence, source

    except Exception as e:
        print("ANSWER ERROR:", e)
        return "⚠️ Error while generating answer.", ""


# ===== CHAT =====
def chat(user_input, history):
    try:
        if not user_input.strip():
            return "", history

        answer, source = get_answer(user_input)

        if not answer:
            answer = "❌ No answer found."

        if not source:
            source = "No source available."

        # ✅ NEW FORMAT (VERY IMPORTANT)
        history.append({"role": "user", "content": user_input})
        history.append({
            "role": "assistant",
            "content": answer + "\n\n📌 Source: " + source
        })

        return "", history

    except Exception as e:
        print("CHAT ERROR:", e)
        history.append({
            "role": "assistant",
            "content": "⚠️ Something went wrong."
        })
        return "", history


# ===== FEEDBACK =====
def feedback(msg):
    print("Feedback:", msg)
    return "✅ Feedback received"


# ===== UI =====
with gr.Blocks(css="""
.gradio-container {max-width: 100% !important;}
""") as demo:

    gr.Markdown("# 🤖 DocuMind")

    file = gr.File(label="📄 Upload PDF")
    status = gr.Textbox(label="Status")

    process_btn = gr.Button("Process PDF")

    chatbot = gr.Chatbot(height=400)

    with gr.Row():
        txt = gr.Textbox(placeholder="Ask question...")
        send = gr.Button("Send")

    clear = gr.Button("Clear Chat")

    gr.Markdown("### 💬 Feedback")
    fb = gr.Textbox(placeholder="Suggestions...")
    fb_btn = gr.Button("Submit")
    fb_out = gr.Textbox(label="Status")

    process_btn.click(process_pdf, file, status)
    send.click(chat, [txt, chatbot], [txt, chatbot])
    txt.submit(chat, [txt, chatbot], [txt, chatbot])
    clear.click(lambda: [], None, chatbot)
    fb_btn.click(feedback, fb, fb_out)

demo.launch()
