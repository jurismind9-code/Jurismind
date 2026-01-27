import streamlit as st
import streamlit.components.v1 as components
import fitz
import google.generativeai as genai
from sentence_transformers import SentenceTransformer
import numpy as np
from io import BytesIO
from langchain_community.document_loaders import WebBaseLoader
from typing import List, Dict, Tuple
import faiss

# --- Page Config ---
st.set_page_config(
    page_title="JurisMind AI",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Hide sidebar completely
st.markdown("""
<style>
    [data-testid="stSidebar"] { display: none; }
    [data-testid="collapsedControl"] { display: none; }
    .stApp > header { display: none; }
    #MainMenu, footer { visibility: hidden; }

    .main .block-container {
        padding: 0 !important;
        max-width: 100% !important;
    }

    .stApp {
        background: #0a0a0b;
    }
</style>
""", unsafe_allow_html=True)

# --- Classes ---
class FAISSVectorDB:
    def __init__(self, dim: int = 384):
        self.dim = dim
        self.index = faiss.IndexFlatL2(dim)
        self.chunks = []

    def add(self, emb: np.ndarray, chunks: List[str]):
        self.index.add(emb)
        self.chunks.extend(chunks)

    def search(self, emb: np.ndarray, k: int = 5):
        if not self.chunks: return [], []
        k = min(k, len(self.chunks))
        dist, idx = self.index.search(emb, k)
        return [self.chunks[i] for i in idx[0] if i < len(self.chunks)], dist[0].tolist()

    def stats(self): return {"chunks": len(self.chunks), "dim": self.dim}

# --- Functions ---
def extract_text(file) -> str:
    ext = file.name.split('.')[-1].lower()
    file.seek(0)
    try:
        if ext == 'pdf':
            with fitz.open(stream=file.read(), filetype="pdf") as doc:
                return "\n".join([p.get_text() for p in doc])
        elif ext == 'docx':
            from docx import Document
            return "\n".join([p.text for p in Document(BytesIO(file.read())).paragraphs])
        elif ext == 'txt':
            return file.read().decode('utf-8')
    except: pass
    return ""

def chunk_text(text: str, size: int = 1000, overlap: int = 200) -> List[str]:
    words = text.split()
    return [" ".join(words[i:i+size]) for i in range(0, len(words), size-overlap) if words[i:i+size]]

@st.cache_resource
def load_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

def get_emb(chunks, model):
    return np.array(model.encode(chunks)).astype('float32')

def gemini_response(prompt: str) -> str:
    for m in ['gemini-2.0-flash', 'gemini-1.5-flash', 'gemini-1.5-pro']:
        try:
            resp = genai.GenerativeModel(m).generate_content(prompt)
            if resp and resp.text: return resp.text
        except: continue
    return None

# --- Session State ---
for key in ['db', 'msgs', 'ready', 'text', 'api_set']:
    if key not in st.session_state:
        st.session_state[key] = None if key in ['db', 'text'] else ([] if key == 'msgs' else False)

# --- Full Page App HTML ---
app_html = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap" rel="stylesheet">
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        html {
            scroll-behavior: smooth;
        }

        body {
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
            background: #0a0a0b;
            color: #fff;
            line-height: 1.6;
            overflow-x: hidden;
        }

        /* Particles Background */
        .particles {
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            pointer-events: none;
            z-index: 1;
        }

        .particle {
            position: absolute;
            width: 4px;
            height: 4px;
            background: rgba(124, 58, 237, 0.6);
            border-radius: 50%;
            animation: particleFloat 15s infinite;
        }

        @keyframes particleFloat {
            0%, 100% { transform: translateY(100vh) rotate(0deg); opacity: 0; }
            10% { opacity: 1; }
            90% { opacity: 1; }
            100% { transform: translateY(-10vh) rotate(720deg); opacity: 0; }
        }

        /* Grid Background */
        .grid-bg {
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background-image:
                linear-gradient(rgba(124, 58, 237, 0.03) 1px, transparent 1px),
                linear-gradient(90deg, rgba(124, 58, 237, 0.03) 1px, transparent 1px);
            background-size: 50px 50px;
            z-index: 0;
        }

        /* Navigation */
        .nav {
            position: fixed;
            top: 0;
            left: 0;
            right: 0;
            z-index: 1000;
            padding: 1rem 2rem;
            display: flex;
            justify-content: space-between;
            align-items: center;
            background: rgba(10, 10, 11, 0.8);
            backdrop-filter: blur(20px);
            border-bottom: 1px solid rgba(124, 58, 237, 0.1);
        }

        .nav-logo {
            display: flex;
            align-items: center;
            gap: 12px;
            font-size: 1.5rem;
            font-weight: 800;
            text-decoration: none;
            color: #fff;
        }

        .nav-logo span {
            background: linear-gradient(135deg, #7c3aed, #a855f7);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }

        .nav-logo-icon {
            width: 40px;
            height: 40px;
            background: linear-gradient(135deg, #7c3aed, #a855f7);
            border-radius: 10px;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 1.25rem;
            animation: logoGlow 3s ease-in-out infinite;
        }

        @keyframes logoGlow {
            0%, 100% { box-shadow: 0 0 20px rgba(124, 58, 237, 0.3); }
            50% { box-shadow: 0 0 40px rgba(124, 58, 237, 0.6); }
        }

        .nav-links {
            display: flex;
            gap: 2rem;
            align-items: center;
        }

        .nav-links a {
            color: #a1a1aa;
            text-decoration: none;
            font-size: 0.9rem;
            font-weight: 500;
            transition: color 0.3s;
            position: relative;
        }

        .nav-links a:hover {
            color: #fff;
        }

        .nav-links a::after {
            content: '';
            position: absolute;
            bottom: -4px;
            left: 0;
            width: 0;
            height: 2px;
            background: linear-gradient(90deg, #7c3aed, #a855f7);
            transition: width 0.3s;
        }

        .nav-links a:hover::after {
            width: 100%;
        }

        .nav-cta {
            padding: 0.6rem 1.5rem;
            background: linear-gradient(135deg, #7c3aed, #a855f7);
            border-radius: 8px;
            color: #fff !important;
            font-weight: 600;
            transition: transform 0.3s, box-shadow 0.3s;
            animation: ctaPulse 2s ease-in-out infinite;
        }

        @keyframes ctaPulse {
            0%, 100% { box-shadow: 0 0 0 0 rgba(124, 58, 237, 0.4); }
            50% { box-shadow: 0 0 0 10px rgba(124, 58, 237, 0); }
        }

        .nav-cta:hover {
            transform: translateY(-2px);
            box-shadow: 0 10px 30px rgba(124, 58, 237, 0.3);
        }

        .nav-cta::after {
            display: none !important;
        }

        /* Hero Section */
        .hero {
            min-height: 90vh;
            display: flex;
            align-items: center;
            justify-content: center;
            padding: 6rem 2rem 2rem;
            position: relative;
            overflow: hidden;
            z-index: 10;
        }

        /* Animated Orbs */
        .orb {
            position: absolute;
            border-radius: 50%;
            filter: blur(80px);
            z-index: 2;
        }

        .orb-1 {
            width: 600px;
            height: 600px;
            background: radial-gradient(circle, rgba(124, 58, 237, 0.3) 0%, transparent 70%);
            top: -200px;
            left: -200px;
            animation: orbMove1 20s ease-in-out infinite;
        }

        .orb-2 {
            width: 500px;
            height: 500px;
            background: radial-gradient(circle, rgba(168, 85, 247, 0.25) 0%, transparent 70%);
            bottom: -150px;
            right: -150px;
            animation: orbMove2 25s ease-in-out infinite;
        }

        .orb-3 {
            width: 300px;
            height: 300px;
            background: radial-gradient(circle, rgba(139, 92, 246, 0.2) 0%, transparent 70%);
            top: 50%;
            left: 50%;
            transform: translate(-50%, -50%);
            animation: orbMove3 15s ease-in-out infinite;
        }

        @keyframes orbMove1 {
            0%, 100% { transform: translate(0, 0) scale(1); }
            33% { transform: translate(100px, 50px) scale(1.1); }
            66% { transform: translate(50px, 100px) scale(0.9); }
        }

        @keyframes orbMove2 {
            0%, 100% { transform: translate(0, 0) scale(1); }
            33% { transform: translate(-80px, -40px) scale(1.1); }
            66% { transform: translate(-40px, -80px) scale(0.9); }
        }

        @keyframes orbMove3 {
            0%, 100% { transform: translate(-50%, -50%) scale(1); }
            50% { transform: translate(-50%, -50%) scale(1.3); }
        }

        .hero-content {
            text-align: center;
            max-width: 900px;
            position: relative;
            z-index: 100;
        }

        .hero-badge {
            display: inline-flex;
            align-items: center;
            gap: 8px;
            padding: 8px 20px;
            background: rgba(124, 58, 237, 0.1);
            border: 1px solid rgba(124, 58, 237, 0.2);
            border-radius: 50px;
            font-size: 0.85rem;
            color: #a78bfa;
            margin-bottom: 2rem;
            animation: fadeInUp 0.8s ease, badgeShine 3s ease-in-out infinite;
        }

        @keyframes badgeShine {
            0%, 100% { border-color: rgba(124, 58, 237, 0.2); }
            50% { border-color: rgba(124, 58, 237, 0.5); box-shadow: 0 0 20px rgba(124, 58, 237, 0.2); }
        }

        .hero-badge-dot {
            width: 8px;
            height: 8px;
            background: #22c55e;
            border-radius: 50%;
            animation: pulse 2s infinite;
        }

        @keyframes pulse {
            0%, 100% { opacity: 1; transform: scale(1); }
            50% { opacity: 0.5; transform: scale(1.2); }
        }

        .hero-title {
            font-size: clamp(3rem, 8vw, 5rem);
            font-weight: 900;
            line-height: 1.1;
            margin-bottom: 1.5rem;
            animation: fadeInUp 0.8s ease 0.2s both;
        }

        .hero-title-gradient {
            background: linear-gradient(135deg, #7c3aed, #a855f7, #c084fc, #7c3aed);
            background-size: 300% 300%;
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            animation: gradientShift 5s ease infinite;
        }

        @keyframes gradientShift {
            0%, 100% { background-position: 0% 50%; }
            50% { background-position: 100% 50%; }
        }

        /* Typing Animation */
        .typing-container {
            height: 40px;
            margin-bottom: 1rem;
            display: flex;
            justify-content: center;
            align-items: center;
        }

        .typing-text {
            font-size: 1.5rem;
            color: #a78bfa;
            font-weight: 600;
            overflow: hidden;
            white-space: nowrap;
            border-right: 3px solid #7c3aed;
            animation: typing 3s steps(30) infinite, blink 0.7s step-end infinite;
        }

        @keyframes typing {
            0%, 100% { width: 0; }
            50%, 70% { width: 100%; }
        }

        @keyframes blink {
            50% { border-color: transparent; }
        }

        .hero-subtitle {
            font-size: 1.25rem;
            color: #71717a;
            max-width: 600px;
            margin: 0 auto 3rem;
            animation: fadeInUp 0.8s ease 0.4s both;
        }

        .hero-buttons {
            display: flex;
            gap: 1rem;
            justify-content: center;
            flex-wrap: wrap;
            animation: fadeInUp 0.8s ease 0.6s both;
        }

        .btn {
            padding: 1rem 2rem;
            border-radius: 12px;
            font-size: 1rem;
            font-weight: 600;
            text-decoration: none;
            transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
            cursor: pointer;
            border: none;
            position: relative;
            overflow: hidden;
        }

        .btn::before {
            content: '';
            position: absolute;
            top: 0;
            left: -100%;
            width: 100%;
            height: 100%;
            background: linear-gradient(90deg, transparent, rgba(255,255,255,0.2), transparent);
            transition: left 0.5s;
        }

        .btn:hover::before {
            left: 100%;
        }

        .btn-primary {
            background: linear-gradient(135deg, #7c3aed, #a855f7);
            color: #fff;
            box-shadow: 0 8px 30px rgba(124, 58, 237, 0.3);
        }

        .btn-primary:hover {
            transform: translateY(-4px) scale(1.02);
            box-shadow: 0 15px 40px rgba(124, 58, 237, 0.5);
        }

        .btn-secondary {
            background: rgba(255, 255, 255, 0.05);
            color: #fff;
            border: 1px solid rgba(255, 255, 255, 0.1);
        }

        .btn-secondary:hover {
            background: rgba(255, 255, 255, 0.1);
            border-color: rgba(124, 58, 237, 0.5);
            transform: translateY(-4px);
        }

        @keyframes fadeInUp {
            from {
                opacity: 0;
                transform: translateY(30px);
            }
            to {
                opacity: 1;
                transform: translateY(0);
            }
        }

        /* Features Section */
        .features {
            padding: 6rem 2rem;
            position: relative;
        }

        .section-header {
            text-align: center;
            margin-bottom: 4rem;
        }

        .section-tag {
            display: inline-block;
            padding: 6px 16px;
            background: rgba(124, 58, 237, 0.1);
            border: 1px solid rgba(124, 58, 237, 0.2);
            border-radius: 50px;
            font-size: 0.8rem;
            color: #a78bfa;
            text-transform: uppercase;
            letter-spacing: 1px;
            margin-bottom: 1rem;
        }

        .section-title {
            font-size: 2.5rem;
            font-weight: 800;
            margin-bottom: 1rem;
        }

        .section-desc {
            color: #71717a;
            font-size: 1.1rem;
            max-width: 500px;
            margin: 0 auto;
        }

        .features-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 2rem;
            max-width: 1200px;
            margin: 0 auto;
        }

        .feature-card {
            background: linear-gradient(145deg, #141419, #0c0c0f);
            border: 1px solid #27272a;
            border-radius: 20px;
            padding: 2.5rem;
            position: relative;
            overflow: hidden;
            transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
        }

        .feature-card::before {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            height: 3px;
            background: linear-gradient(90deg, #7c3aed, #a855f7);
            transform: scaleX(0);
            transition: transform 0.4s;
        }

        .feature-card:hover::before {
            transform: scaleX(1);
        }

        .feature-card:hover {
            transform: translateY(-10px);
            border-color: rgba(124, 58, 237, 0.3);
            box-shadow: 0 25px 50px rgba(0, 0, 0, 0.5), 0 0 40px rgba(124, 58, 237, 0.1);
        }

        .feature-icon {
            width: 60px;
            height: 60px;
            background: linear-gradient(135deg, #7c3aed, #a855f7);
            border-radius: 16px;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 1.75rem;
            margin-bottom: 1.5rem;
            box-shadow: 0 10px 30px rgba(124, 58, 237, 0.3);
            transition: transform 0.3s;
        }

        .feature-card:hover .feature-icon {
            transform: scale(1.1) rotate(5deg);
        }

        .feature-title {
            font-size: 1.25rem;
            font-weight: 700;
            margin-bottom: 0.75rem;
        }

        .feature-desc {
            color: #71717a;
            font-size: 0.95rem;
            line-height: 1.7;
        }

        /* App Section */
        .app-section {
            padding: 6rem 2rem;
            max-width: 1000px;
            margin: 0 auto;
        }

        .app-container {
            background: linear-gradient(145deg, #141419, #0c0c0f);
            border: 1px solid #27272a;
            border-radius: 24px;
            overflow: hidden;
            position: relative;
        }

        .app-container::before {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            height: 3px;
            background: linear-gradient(90deg, #7c3aed, #a855f7, #7c3aed);
        }

        .app-header {
            padding: 1.5rem 2rem;
            border-bottom: 1px solid #27272a;
            display: flex;
            align-items: center;
            gap: 1rem;
        }

        .app-header-dots {
            display: flex;
            gap: 8px;
        }

        .app-header-dot {
            width: 12px;
            height: 12px;
            border-radius: 50%;
        }

        .app-header-dot:nth-child(1) { background: #ef4444; }
        .app-header-dot:nth-child(2) { background: #f59e0b; }
        .app-header-dot:nth-child(3) { background: #22c55e; }

        .app-header-title {
            color: #71717a;
            font-size: 0.9rem;
        }

        .app-body {
            padding: 2rem;
        }

        .app-tabs {
            display: flex;
            gap: 0.5rem;
            margin-bottom: 2rem;
            background: #0a0a0b;
            padding: 6px;
            border-radius: 12px;
            width: fit-content;
        }

        .app-tab {
            padding: 0.75rem 1.5rem;
            border-radius: 8px;
            font-size: 0.9rem;
            font-weight: 500;
            color: #71717a;
            cursor: pointer;
            transition: all 0.3s;
            border: none;
            background: transparent;
        }

        .app-tab.active {
            background: linear-gradient(135deg, #7c3aed, #a855f7);
            color: #fff;
        }

        .app-tab:hover:not(.active) {
            color: #fff;
            background: rgba(124, 58, 237, 0.1);
        }

        /* Upload Zone */
        .upload-zone {
            border: 2px dashed #27272a;
            border-radius: 16px;
            padding: 3rem 2rem;
            text-align: center;
            transition: all 0.3s;
            cursor: pointer;
            margin-bottom: 2rem;
        }

        .upload-zone:hover {
            border-color: #7c3aed;
            background: rgba(124, 58, 237, 0.05);
        }

        .upload-icon {
            font-size: 3rem;
            margin-bottom: 1rem;
        }

        .upload-title {
            font-size: 1.25rem;
            font-weight: 600;
            margin-bottom: 0.5rem;
        }

        .upload-desc {
            color: #71717a;
            font-size: 0.9rem;
        }

        /* Input Group */
        .input-group {
            margin-bottom: 1.5rem;
        }

        .input-label {
            display: block;
            font-size: 0.85rem;
            font-weight: 500;
            color: #a1a1aa;
            margin-bottom: 0.5rem;
        }

        .input-field {
            width: 100%;
            padding: 1rem 1.25rem;
            background: #0a0a0b;
            border: 1px solid #27272a;
            border-radius: 12px;
            color: #fff;
            font-size: 1rem;
            font-family: inherit;
            transition: all 0.3s;
        }

        .input-field:focus {
            outline: none;
            border-color: #7c3aed;
            box-shadow: 0 0 0 3px rgba(124, 58, 237, 0.1);
        }

        .input-field::placeholder {
            color: #52525b;
        }

        /* Chat Area */
        .chat-area {
            background: #0a0a0b;
            border-radius: 16px;
            padding: 1.5rem;
            min-height: 300px;
            max-height: 400px;
            overflow-y: auto;
            margin-bottom: 1.5rem;
        }

        .chat-message {
            padding: 1rem 1.25rem;
            border-radius: 12px;
            margin-bottom: 1rem;
            animation: fadeInUp 0.3s ease;
        }

        .chat-message.user {
            background: linear-gradient(135deg, #7c3aed, #a855f7);
            margin-left: 20%;
        }

        .chat-message.assistant {
            background: #1a1a1f;
            border: 1px solid #27272a;
            margin-right: 20%;
        }

        .chat-input-container {
            display: flex;
            gap: 1rem;
        }

        .chat-input {
            flex: 1;
            padding: 1rem 1.25rem;
            background: #0a0a0b;
            border: 1px solid #27272a;
            border-radius: 12px;
            color: #fff;
            font-size: 1rem;
            font-family: inherit;
        }

        .chat-input:focus {
            outline: none;
            border-color: #7c3aed;
        }

        .chat-send {
            padding: 1rem 1.5rem;
            background: linear-gradient(135deg, #7c3aed, #a855f7);
            border: none;
            border-radius: 12px;
            color: #fff;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.3s;
        }

        .chat-send:hover {
            transform: scale(1.05);
            box-shadow: 0 10px 30px rgba(124, 58, 237, 0.3);
        }

        /* Footer */
        .footer {
            padding: 4rem 2rem;
            text-align: center;
            border-top: 1px solid #1a1a1f;
        }

        .footer-logo {
            font-size: 1.5rem;
            font-weight: 800;
            margin-bottom: 1rem;
        }

        .footer-logo span {
            background: linear-gradient(135deg, #7c3aed, #a855f7);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }

        .footer-text {
            color: #52525b;
            font-size: 0.9rem;
        }

        /* Scrollbar */
        ::-webkit-scrollbar {
            width: 6px;
        }
        ::-webkit-scrollbar-track {
            background: #0a0a0b;
        }
        ::-webkit-scrollbar-thumb {
            background: #27272a;
            border-radius: 3px;
        }
        ::-webkit-scrollbar-thumb:hover {
            background: #7c3aed;
        }
    </style>
</head>
<body>
    <!-- Grid Background -->
    <div class="grid-bg"></div>

    <!-- Particles -->
    <div class="particles">
        <div class="particle" style="left: 10%; animation-delay: 0s;"></div>
        <div class="particle" style="left: 20%; animation-delay: 2s;"></div>
        <div class="particle" style="left: 30%; animation-delay: 4s;"></div>
        <div class="particle" style="left: 40%; animation-delay: 1s;"></div>
        <div class="particle" style="left: 50%; animation-delay: 3s;"></div>
        <div class="particle" style="left: 60%; animation-delay: 5s;"></div>
        <div class="particle" style="left: 70%; animation-delay: 2.5s;"></div>
        <div class="particle" style="left: 80%; animation-delay: 1.5s;"></div>
        <div class="particle" style="left: 90%; animation-delay: 3.5s;"></div>
        <div class="particle" style="left: 15%; animation-delay: 4.5s;"></div>
        <div class="particle" style="left: 25%; animation-delay: 0.5s;"></div>
        <div class="particle" style="left: 75%; animation-delay: 2.2s;"></div>
        <div class="particle" style="left: 85%; animation-delay: 3.8s;"></div>
        <div class="particle" style="left: 45%; animation-delay: 1.8s;"></div>
        <div class="particle" style="left: 55%; animation-delay: 4.2s;"></div>
    </div>

    <!-- Navigation -->
    <nav class="nav">
        <a href="#home" class="nav-logo">
            <div class="nav-logo-icon">⚖️</div>
            Juris<span>Mind</span>
        </a>
        <div class="nav-links">
            <a href="#home">Home</a>
            <a href="#features">Features</a>
            <a href="#app">App</a>
            <a href="#app" class="nav-cta">Get Started</a>
        </div>
    </nav>

    <!-- Hero Section -->
    <section class="hero" id="home">
        <!-- Animated Orbs -->
        <div class="orb orb-1"></div>
        <div class="orb orb-2"></div>
        <div class="orb orb-3"></div>

        <div class="hero-content">
            <div class="hero-badge">
                <div class="hero-badge-dot"></div>
                AI-Powered Legal Intelligence
            </div>
            <h1 class="hero-title">
                Legal Analysis<br>
                <span class="hero-title-gradient">Reimagined with AI</span>
            </h1>

            <!-- Typing Animation -->
            <div class="typing-container">
                <span class="typing-text" id="typed-text"></span>
            </div>

            <p class="hero-subtitle">
                Transform your legal document workflow with intelligent analysis,
                semantic search, and automated case brief generation powered by Gemini AI.
            </p>

            <div class="hero-buttons">
                <a href="#app" class="btn btn-primary">🚀 Start Analyzing</a>
                <a href="#features" class="btn btn-secondary">📖 Learn More</a>
            </div>
        </div>
    </section>

    <!-- Features Section -->
    <section class="features" id="features">
        <div class="section-header">
            <div class="section-tag">Features</div>
            <h2 class="section-title">Powerful Capabilities</h2>
            <p class="section-desc">Everything you need for intelligent legal document analysis</p>
        </div>
        <div class="features-grid">
            <div class="feature-card">
                <div class="feature-icon">🔍</div>
                <h3 class="feature-title">Semantic Search</h3>
                <p class="feature-desc">FAISS-powered vector search finds relevant content instantly using advanced embeddings</p>
            </div>
            <div class="feature-card">
                <div class="feature-icon">🤖</div>
                <h3 class="feature-title">AI Analysis</h3>
                <p class="feature-desc">Gemini AI provides deep legal reasoning, document understanding, and intelligent responses</p>
            </div>
            <div class="feature-card">
                <div class="feature-icon">📋</div>
                <h3 class="feature-title">Case Briefs</h3>
                <p class="feature-desc">Generate comprehensive, structured case briefs automatically with one click</p>
            </div>
            <div class="feature-card">
                <div class="feature-icon">📄</div>
                <h3 class="feature-title">Multi-Format</h3>
                <p class="feature-desc">Support for PDF, DOCX, and TXT files - upload and analyze any legal document</p>
            </div>
            <div class="feature-card">
                <div class="feature-icon">💬</div>
                <h3 class="feature-title">Chat Interface</h3>
                <p class="feature-desc">Ask questions about your documents in natural language and get accurate answers</p>
            </div>
            <div class="feature-card">
                <div class="feature-icon">🔒</div>
                <h3 class="feature-title">Secure & Private</h3>
                <p class="feature-desc">Your documents are processed securely with no data stored on external servers</p>
            </div>
        </div>
    </section>

    <!-- Footer -->
    <footer class="footer">
        <div class="footer-logo">Juris<span>Mind</span> AI</div>
        <p class="footer-text">Advanced Legal Intelligence Platform • Powered by FAISS & Gemini AI</p>
    </footer>

    <script>
        // Smooth scroll for navigation links
        document.querySelectorAll('a[href^="#"]').forEach(anchor => {
            anchor.addEventListener('click', function (e) {
                e.preventDefault();
                document.querySelector(this.getAttribute('href')).scrollIntoView({
                    behavior: 'smooth'
                });
            });
        });

        // Add scroll effect to nav
        window.addEventListener('scroll', () => {
            const nav = document.querySelector('.nav');
            if (window.scrollY > 50) {
                nav.style.background = 'rgba(10, 10, 11, 0.95)';
            } else {
                nav.style.background = 'rgba(10, 10, 11, 0.8)';
            }
        });

        // Typing Animation
        const phrases = [
            'Analyze Legal Documents',
            'Generate Case Briefs',
            'Semantic Search',
            'AI-Powered Research',
            'Contract Analysis'
        ];
        let phraseIndex = 0;
        let charIndex = 0;
        let isDeleting = false;
        const typedText = document.getElementById('typed-text');

        function type() {
            const currentPhrase = phrases[phraseIndex];

            if (isDeleting) {
                typedText.textContent = currentPhrase.substring(0, charIndex - 1);
                charIndex--;
            } else {
                typedText.textContent = currentPhrase.substring(0, charIndex + 1);
                charIndex++;
            }

            let typeSpeed = isDeleting ? 50 : 100;

            if (!isDeleting && charIndex === currentPhrase.length) {
                typeSpeed = 2000;
                isDeleting = true;
            } else if (isDeleting && charIndex === 0) {
                isDeleting = false;
                phraseIndex = (phraseIndex + 1) % phrases.length;
                typeSpeed = 500;
            }

            setTimeout(type, typeSpeed);
        }
        type();
    </script>
</body>
</html>
"""

# --- Render the landing page ---
components.html(app_html, height=1800, scrolling=True)

# --- Streamlit App Section Below ---
st.markdown("""
<div style='max-width: 1000px; margin: 0 auto; padding: 2rem;'>
    <div style='background: linear-gradient(145deg, #141419, #0c0c0f); border: 1px solid #27272a;
                border-radius: 24px; overflow: hidden; position: relative;'>
        <div style='position: absolute; top: 0; left: 0; right: 0; height: 3px;
                    background: linear-gradient(90deg, #7c3aed, #a855f7, #7c3aed);'></div>
        <div style='padding: 1.5rem 2rem; border-bottom: 1px solid #27272a; display: flex; align-items: center; gap: 1rem;'>
            <div style='display: flex; gap: 8px;'>
                <div style='width: 12px; height: 12px; border-radius: 50%; background: #ef4444;'></div>
                <div style='width: 12px; height: 12px; border-radius: 50%; background: #f59e0b;'></div>
                <div style='width: 12px; height: 12px; border-radius: 50%; background: #22c55e;'></div>
            </div>
            <span style='color: #71717a; font-size: 0.9rem;'>JurisMind AI - Document Analysis</span>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# App container
col1, col2, col3 = st.columns([1, 3, 1])

with col2:
    # API Key input
    st.markdown("<br>", unsafe_allow_html=True)
    api_key = st.text_input("🔑 Gemini API Key", type="password", placeholder="Enter your API key...")

    if api_key:
        genai.configure(api_key=api_key)
        st.session_state.api_set = True
        st.success("✓ API Connected")

    st.markdown("<br>", unsafe_allow_html=True)

    # File upload
    uploaded = st.file_uploader("📄 Upload Legal Documents", type=["pdf", "docx", "txt"], accept_multiple_files=True)

    if uploaded and not st.session_state.ready:
        with st.spinner("Processing documents..."):
            text = "\n".join([extract_text(f) for f in uploaded])
            if text:
                chunks = chunk_text(text)
                model = load_model()
                emb = get_emb(chunks, model)
                db = FAISSVectorDB(emb.shape[1])
                db.add(emb, chunks)
                st.session_state.db = db
                st.session_state.model = model
                st.session_state.text = text
                st.session_state.ready = True
                st.success(f"✅ {len(chunks)} chunks indexed!")

    if st.session_state.ready:
        st.markdown("<br>", unsafe_allow_html=True)

        c1, c2 = st.columns(2)
        with c1:
            if st.button("✨ Generate Case Brief", use_container_width=True):
                with st.spinner("Generating..."):
                    brief = gemini_response(f"Create a comprehensive legal case brief:\n\n{st.session_state.text[:15000]}")
                    if brief:
                        st.session_state.msgs.append({"role": "assistant", "content": brief})
        with c2:
            if st.button("🗑️ Clear Chat", use_container_width=True):
                st.session_state.msgs = []
                st.rerun()

        st.markdown("<br>", unsafe_allow_html=True)

        # Chat messages
        for msg in st.session_state.msgs:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

        # Chat input
        if prompt := st.chat_input("Ask about your documents..."):
            st.session_state.msgs.append({"role": "user", "content": prompt})
            with st.chat_message("assistant"):
                with st.spinner(""):
                    emb = get_emb([prompt], st.session_state.model)
                    chunks, _ = st.session_state.db.search(emb, 5)
                    if chunks:
                        resp = gemini_response(f"Context:\n{chr(10).join(chunks)}\n\nQuestion: {prompt}")
                        if resp:
                            st.markdown(resp)
                            st.session_state.msgs.append({"role": "assistant", "content": resp})

    # Stats
    if st.session_state.db:
        st.markdown("<br>", unsafe_allow_html=True)
        stats = st.session_state.db.stats()
        c1, c2 = st.columns(2)
        c1.metric("📊 Chunks", stats["chunks"])
        c2.metric("🔢 Dimensions", stats["dim"])
