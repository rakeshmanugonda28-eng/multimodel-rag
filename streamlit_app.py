# PDF Lens - AskMyPDF Multimodal RAG System (Streamlit version)
# 1. Upload a PDF  ->  2. Click "Analyse Document"  ->  3. Ask questions

import os
import re
import uuid
import base64
import tempfile
from io import BytesIO
from pathlib import Path

import fitz                      # PyMuPDF - reads PDFs
import torch
import chromadb
import streamlit as st
from PIL import Image

from transformers import CLIPProcessor, CLIPModel
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI

st.set_page_config(page_title="PDF Lens - AskMyPDF", page_icon="🔍", layout="wide")

# API key: from Streamlit "Secrets" when deployed, or from your environment on your laptop
try:
    if "OPENAI_API_KEY" in st.secrets:
        os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
except Exception:
    pass    # no secrets file on your laptop - the key comes from your environment instead

labels = ["a bar chart", "a pie chart", "a line graph", "a flowchart diagram",
          "a photograph", "a logo", "a table", "a page of text"]


# ---------------- 1. Load the models ONCE (cached) ----------------
# Streamlit re-runs this whole file on every click.
# @st.cache_resource makes sure the models load only the first time.
@st.cache_resource(show_spinner="Loading AI models (first time only)...")
def load_models():
    text_embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)   # can read images too
    chroma_client = chromadb.EphemeralClient()      # in-memory database
    return text_embeddings, clip_model, clip_processor, llm, chroma_client


text_embeddings, clip_model, clip_processor, llm, chroma_client = load_models()


# ---------------- 2. CLIP helpers ----------------
def get_features(output):
    if isinstance(output, torch.Tensor):
        return output
    return output.pooler_output


def embed_image(image_bytes):
    image = Image.open(BytesIO(image_bytes)).convert("RGB")
    inputs = clip_processor(images=image, return_tensors="pt")
    with torch.no_grad():
        vector = get_features(clip_model.get_image_features(**inputs))
    vector = vector / vector.norm(dim=-1, keepdim=True)
    return vector[0].tolist()


def embed_question(text):
    inputs = clip_processor(text=[text], return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        vector = get_features(clip_model.get_text_features(**inputs))
    vector = vector / vector.norm(dim=-1, keepdim=True)
    return vector[0].tolist()


# ---------------- 3. Extract text, images and charts ----------------
def extract_pdf(pdf_path, image_dir):
    documents, image_data, visual_pages = [], [], []
    pdf_doc = fitz.open(pdf_path)
    seen_images = set()

    for page_number, page in enumerate(pdf_doc):

        # TEXT
        text = page.get_text().strip()
        if text:
            documents.append(Document(page_content=text, metadata={"page": page_number + 1}))

        # IMAGES
        for image_index, image in enumerate(page.get_images(full=True)):
            if image[0] in seen_images:
                continue
            seen_images.add(image[0])
            image_bytes = pdf_doc.extract_image(image[0])
            if image_bytes["width"] < 100 or image_bytes["height"] < 100:
                continue
            path = image_dir / f"p{page_number + 1}_img{image_index + 1}.{image_bytes['ext']}"
            path.write_bytes(image_bytes["image"])
            image_data.append({"content_type": "image", "page": page_number + 1,
                               "image": image_bytes["image"], "path": str(path)})

        # CHARTS
        areas = [fitz.Rect(info["bbox"]) for info in page.get_image_info()]
        drawings = [d for d in page.get_drawings() if d["rect"].width < 0.8 * page.rect.width]
        if drawings:
            areas += page.cluster_drawings(drawings=drawings)

        for area in areas:
            if area.width < 150 or area.height < 100:
                continue
            pix = page.get_pixmap(matrix=fitz.Matrix(2, 2), clip=area)
            img = Image.open(BytesIO(pix.tobytes("png"))).convert("RGB")
            inputs = clip_processor(text=labels, images=img, return_tensors="pt", padding=True)
            with torch.no_grad():
                probs = clip_model(**inputs).logits_per_image.softmax(dim=1)[0]
            label = labels[probs.argmax().item()]
            if label in labels[:4]:
                path = image_dir / f"p{page_number + 1}_chart{len(visual_pages)}.png"
                img.save(path)
                visual_pages.append({"content_type": "visual", "page": page_number + 1,
                                     "chart_type": label, "image": pix.tobytes("png"), "path": str(path)})
    pdf_doc.close()
    return documents, image_data, visual_pages


# ---------------- 4. Process the uploaded PDF ----------------
def process_pdf(uploaded_file):
    # each visitor gets their own folder and database names
    session_id = uuid.uuid4().hex[:8]
    work_dir = Path(tempfile.mkdtemp(prefix=f"pdflens_{session_id}_"))
    pdf_path = work_dir / "uploaded.pdf"
    pdf_path.write_bytes(uploaded_file.getvalue())
    image_dir = work_dir / "images"
    image_dir.mkdir()

    documents, image_data, visual_pages = extract_pdf(str(pdf_path), image_dir)

    chunks = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200).split_documents(documents)

    text_db = Chroma(client=chroma_client, collection_name=f"text_{session_id}",
                     embedding_function=text_embeddings)
    if chunks:
        text_db.add_documents(chunks)

    image_db = chroma_client.create_collection(name=f"images_{session_id}", metadata={"hnsw:space": "cosine"})
    for i, item in enumerate(image_data + visual_pages):
        image_db.add(ids=[f"pic_{i}"], embeddings=[embed_image(item["image"])],
                     metadatas=[{"content_type": item["content_type"], "page": item["page"],
                                 "chart_type": item.get("chart_type", "image"), "path": item["path"]}])

    # remember everything for this visitor
    st.session_state.text_db = text_db
    st.session_state.image_db = image_db


# ---------------- 5. Retrieval + answer ----------------
NOT_FOUND_MESSAGE = ("I'm sorry, but the uploaded document does not contain information related to your question. "
                     "Please try rephrasing your question or ask about a topic covered in the document.")

SHOW_WORDS = ["show", "display", "view", "see", "give me", "open"]
VISUAL_WORDS = ["image", "picture", "photo", "figure", "chart", "graph", "diagram", "plot", "visual"]


def wants_to_see_images(question):
    # True only when the user asks to SEE a picture, e.g. "show me the bar chart"
    q = question.lower()
    return any(w in q for w in SHOW_WORDS) and any(w in q for w in VISUAL_WORDS)


def make_search_queries(question):
    # Multi-query retrieval: ask the LLM for other ways to say the question,
    # e.g. "drawbacks" -> "disadvantages", "limitations of the existing system"
    prompt = (f"Write 3 different short search queries that mean the same as this question, "
              f"using synonyms a document might use. One per line, no numbering.\n\nQuestion: {question}")
    try:
        text = llm.invoke(prompt).content
        if isinstance(text, list):
            text = "".join(p.get("text", "") if isinstance(p, dict) else str(p) for p in text)
        extra = [line.strip("-• ").strip() for line in text.splitlines() if line.strip()]
    except Exception:
        extra = []
    return [question] + extra[:3]


def retrieve_text(question, k_each=4, max_chunks=8):
    chunks, seen = [], set()
    for query in make_search_queries(question):
        for doc in st.session_state.text_db.similarity_search(query, k=k_each):
            if doc.page_content not in seen:      # skip duplicates
                seen.add(doc.page_content)
                chunks.append(doc)
    return chunks[:max_chunks]


def image_to_base64(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def ask(question):
    image_db = st.session_state.image_db

    # 1. retrieve text (multi-query)
    text_docs = retrieve_text(question)

    # 2. retrieve images / charts (if the question names a page, look on that page first)
    pictures = []
    if image_db.count() > 0:
        page_match = re.search(r"page\s*(\d+)", question.lower())
        if page_match:
            found = image_db.get(where={"page": int(page_match.group(1))})
            pictures = found["metadatas"][:3]
        if not pictures:
            results = image_db.query(query_embeddings=[embed_question(question)],
                                     n_results=min(3, image_db.count()))
            pictures = results["metadatas"][0]

    context = ""
    for doc in text_docs:
        context += f"[Page {doc.metadata['page']}]\n{doc.page_content}\n\n"

    # 3. build the prompt
    message = [{
        "type": "text",
        "text": f"""You are a professional document assistant.
Answer the question using ONLY the text context and the images (figures, charts, graphs) given below.

Rules:
- If the question is about a chart or graph, explain what it represents: its title, axes or categories,
  the main values, and the key trend or insight.
- The document may use different words than the question. Treat similar terms as the same,
  e.g. drawbacks = disadvantages = limitations = problems = issues of the existing system.
- If the context contains related information, answer using it, even if it is only partly relevant.
- Mention page numbers where the information comes from.
- Be clear and concise.
- Only if NOTHING in the context or images is related to the question, reply with exactly this word
  and nothing else: NOT_FOUND

Text context:
{context}

Question: {question}"""
    }]

    for pic in pictures:
        message.append({"type": "text", "text": f"[{pic['chart_type']} from page {pic['page']}]"})
        message.append({"type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{image_to_base64(pic['path'])}"}})

    # 4. ask the LLM
    try:
        answer = llm.invoke([HumanMessage(content=message)]).content
        if isinstance(answer, list):    # some versions return a list of parts
            answer = "".join(p.get("text", "") if isinstance(p, dict) else str(p) for p in answer)
    except Exception as e:
        return f"Sorry, something went wrong while generating the answer: {e}", False, []

    # 5. not in the PDF -> professional message, no pictures
    if "NOT_FOUND" in answer:
        return NOT_FOUND_MESSAGE, False, []

    # 6. show pictures only if the user asked to see them (show the best match)
    pictures_to_show = pictures[:1] if wants_to_see_images(question) else []
    return answer, True, pictures_to_show


# ---------------- 6. The web page (Streamlit UI) ----------------
st.title("🔍 PDF Lens")
st.subheader("AskMyPDF: Multimodal RAG System")
st.write("Chat with your PDF's **text**, **images** and **charts/graphs**.")

# --- Step 1: upload ---
st.header("📄 Step 1: Upload PDF")
uploaded_file = st.file_uploader("PDF document", type=["pdf"])

if st.button("Analyse Document", type="primary"):
    if uploaded_file is None:
        st.warning("Please upload a PDF first.")
    else:
        with st.spinner("Analysing your document..."):
            process_pdf(uploaded_file)

if "text_db" in st.session_state:
    st.success("✅ Document analysed successfully. You can now ask questions about it.")

    st.divider()

    # --- Step 2: ask ---
    st.header("💬 Step 2: Ask a Question")
    question = st.text_input("Your question",
                             placeholder="e.g. What does the bar graph represent?  /  Show me the pie chart")

    if st.button("Get Answer", type="primary") and question.strip():
        with st.spinner("Searching the document..."):
            answer, found, pictures = ask(question)

        st.header("🧠 Answer")
        if found:
            st.markdown(answer)
        else:
            st.warning(answer)

        for pic in pictures:
            st.image(pic["path"], caption=f"Page {pic['page']}", width=500)
else:
    st.info("Upload a PDF and click **Analyse Document** to start.")
