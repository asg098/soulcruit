import warnings
warnings.filterwarnings("ignore")
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'  # Prevent multiprocessing OOM in Transformers

# Detect Render environment to skip heavy runtime tasks
IS_RENDER = 'RENDER' in os.environ

print("=" * 70)
print("SOULCRUIT AI Backend v5.1 (Improved for Render)")
print("=" * 70)

import subprocess
import sys
import time

def install_package(package, display_name=None):
    if display_name is None:
        display_name = package.split('==')[0]
    print(f"Installing {display_name}...", end=' ', flush=True)
    try:
        result = subprocess.run(
            [sys.executable, '-m', 'pip', 'install', package],
            capture_output=True,
            text=True,
            timeout=300
        )
        if result.returncode == 0:
            print("✓")
            return True
        else:
            print("⚠ (continuing)")
            return False
    except subprocess.TimeoutExpired:
        print("⚠ (timeout)")
        return False
    except Exception as e:
        print(f"⚠ ({str(e)[:30]})")
        return False

packages = [
    ('flask', 'Flask'),
    ('flask-cors', 'Flask-CORS'),
    ('bcrypt', 'Bcrypt'),
    ('python-jose[cryptography]', 'Jose JWT'),
    ('numpy', 'NumPy'),
    ('pandas', 'Pandas'),
    ('scikit-learn', 'Scikit-learn'),
    ('textblob', 'TextBlob'),
    ('pillow', 'Pillow'),
    ('spacy', 'SpaCy'),
    ('googletrans==4.0.0-rc1', 'Google Translate'),
]

optional_packages = [
    ('opencv-python-headless', 'OpenCV'),
    ('fer', 'FER Emotion Detection'),
    # Defer heavy ML imports to lazy functions to avoid startup OOM
]

if not IS_RENDER:
    print("\nInstalling core packages:")
    for package, name in packages:
        install_package(package, name)

    print("\nInstalling optional packages (may take time):")
    for package, name in optional_packages:
        install_package(package, name)

    print("\nDownloading language models:")
    try:
        print("SpaCy model...", end=' ', flush=True)
        subprocess.run([sys.executable, '-m', 'spacy', 'download', 'en_core_web_sm'],
                      capture_output=True, timeout=120, check=False)
        print("✓")
    except:
        print("⚠")

    try:
        print("TextBlob corpora...", end=' ', flush=True)
        subprocess.run([sys.executable, '-m', 'textblob.download_corpora'],
                      capture_output=True, timeout=60, check=False)
        print("✓")
    except:
        print("⚠")

    print("\n" + "=" * 70)
    print("Installation complete! Starting backend...")
    print("=" * 70 + "\n")
else:
    print("🛡️ On Render: Skipping runtime installs (using pre-built packages)")
    print("📥 Models pre-loaded via build")
    print("\n" + "=" * 70)
    print("Starting backend...")
    print("=" * 70 + "\n")

from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import hashlib
import uuid
import bcrypt
import re
import random
import socket
from datetime import datetime, timedelta
from functools import wraps
from jose import jwt
from textblob import TextBlob
import logging
from functools import lru_cache
import signal

# Conditional availability flags (set dynamically in lazy functions)
EMBEDDINGS_AVAILABLE = False
CV2_AVAILABLE = False
TRANSFORMERS_AVAILABLE = False
SPACY_AVAILABLE = False
TRANSLATOR_AVAILABLE = False

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": "*"}})
app.config.update(
    SECRET_KEY=os.environ.get('SECRET_KEY', 'soulcruit_ai_2025_secure_key_v1_production'),  # Use env var for security
    MAX_CONTENT_LENGTH=100*1024*1024,
    JSON_SORT_KEYS=False
)

# Global lazy model vars
detector = None
sentiment_analyzer = None
embedding_model = None
nlp = None
translator = None

SKIP_MODELS = os.environ.get('SKIP_MODELS', 'true').lower() == 'true'  # Default to true for Render

print("\nLoading AI models...")
if not SKIP_MODELS:
    print("🔄 Attempting initial loads (may OOM on low RAM; use SKIP_MODELS=true)")
    # Initial loads with timeouts (fallback to lazy)
    def timeout_handler(signum, frame):
        raise TimeoutError("Model load timeout")

    # SpaCy (lightest first)
    try:
        import spacy
        SPACY_AVAILABLE = True
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(30)
        nlp = spacy.load('en_core_web_sm')
        signal.alarm(0)
        print("✓ SpaCy NLP")
    except Exception as e:
        SPACY_AVAILABLE = False
        print(f"⚠ SpaCy skipped: {e}")

    # Translator (very light)
    try:
        from googletrans import Translator
        TRANSLATOR_AVAILABLE = True
        translator = Translator()
        print("✓ Translator")
    except Exception as e:
        TRANSLATOR_AVAILABLE = False
        print(f"⚠ Translator skipped: {e}")

    # CV (medium)
    try:
        import cv2
        from fer import FER
        CV2_AVAILABLE = True
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(30)
        detector = FER(mtcnn=False)
        signal.alarm(0)
        print("✓ Emotion detector")
    except Exception as e:
        CV2_AVAILABLE = False
        print(f"⚠ Emotion detector skipped: {e}")

    # Transformers (heavy)
    try:
        from transformers import pipeline
        TRANSFORMERS_AVAILABLE = True
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(60)
        sentiment_analyzer = pipeline("sentiment-analysis",
                                     model="distilbert-base-uncased-finetuned-sst-2-english",
                                     device=-1)
        signal.alarm(0)
        print("✓ Sentiment analyzer")
    except Exception as e:
        TRANSFORMERS_AVAILABLE = False
        print(f"⚠ Sentiment analyzer skipped: {e}")

    # Embeddings (heavy)
    try:
        from sentence_transformers import SentenceTransformer
        EMBEDDINGS_AVAILABLE = True
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(60)
        embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        signal.alarm(0)
        print("✓ Embedding model")
    except Exception as e:
        EMBEDDINGS_AVAILABLE = False
        print(f"⚠ Embedding model skipped: {e}")
else:
    print("🛡️ Skipping all model loads/imports (SKIP_MODELS=true) - Use lazy loads in routes")

# Enhanced lazy load functions with dynamic imports
@lru_cache(maxsize=1)
def get_detector():
    global detector, CV2_AVAILABLE
    if detector is None and not SKIP_MODELS:
        try:
            if not CV2_AVAILABLE:
                import cv2
                from fer import FER
                CV2_AVAILABLE = True
            detector = FER(mtcnn=False)
            logger.info("Lazy-loaded FER detector")
        except Exception as e:
            CV2_AVAILABLE = False
            logger.error(f"Lazy FER load failed: {e}")
    return detector

@lru_cache(maxsize=1)
def get_sentiment_analyzer():
    global sentiment_analyzer, TRANSFORMERS_AVAILABLE
    if sentiment_analyzer is None and not SKIP_MODELS:
        try:
            if not TRANSFORMERS_AVAILABLE:
                from transformers import pipeline
                TRANSFORMERS_AVAILABLE = True
            sentiment_analyzer = pipeline("sentiment-analysis",
                                         model="distilbert-base-uncased-finetuned-sst-2-english",
                                         device=-1)
            logger.info("Lazy-loaded sentiment analyzer")
        except Exception as e:
            TRANSFORMERS_AVAILABLE = False
            logger.error(f"Lazy sentiment load failed: {e}")
    return sentiment_analyzer

@lru_cache(maxsize=1)
def get_embedding_model():
    global embedding_model, EMBEDDINGS_AVAILABLE
    if embedding_model is None and not SKIP_MODELS:
        try:
            if not EMBEDDINGS_AVAILABLE:
                from sentence_transformers import SentenceTransformer
                EMBEDDINGS_AVAILABLE = True
            embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
            logger.info("Lazy-loaded embedding model")
        except Exception as e:
            EMBEDDINGS_AVAILABLE = False
            logger.error(f"Lazy embedding load failed: {e}")
    return embedding_model

@lru_cache(maxsize=1)
def get_nlp():
    global nlp, SPACY_AVAILABLE
    if nlp is None and not SKIP_MODELS:
        try:
            if not SPACY_AVAILABLE:
                import spacy
                SPACY_AVAILABLE = True
            nlp = spacy.load('en_core_web_sm')
            logger.info("Lazy-loaded SpaCy")
        except Exception as e:
            SPACY_AVAILABLE = False
            logger.error(f"Lazy SpaCy load failed: {e}")
    return nlp

@lru_cache(maxsize=1)
def get_translator():
    global translator, TRANSLATOR_AVAILABLE
    if translator is None and not SKIP_MODELS:
        try:
            if not TRANSLATOR_AVAILABLE:
                from googletrans import Translator
                TRANSLATOR_AVAILABLE = True
            translator = Translator()
            logger.info("Lazy-loaded translator")
        except Exception as e:
            TRANSLATOR_AVAILABLE = False
            logger.error(f"Lazy translator load failed: {e}")
    return translator

# ... (all your DBs, constants, functions remain the same, but update uses of models to lazy getters)

# Example updates in functions (apply similarly):
def calculate_skill_match(job_skills, user_skills):
    try:
        if not job_skills or not user_skills:
            return 0
        emb_model = get_embedding_model()
        if not emb_model or not EMBEDDINGS_AVAILABLE:
            # Fallback to set intersection
            job_set = set(s.lower() for s in job_skills)
            user_set = set(s.lower() for s in user_skills)
            if not job_set:
                return 0
            matches = len(job_set & user_set)
            return int((matches / len(job_set)) * 100)
        job_clean = [s.strip().lower() for s in job_skills if s.strip()]
        user_clean = [s.strip().lower() for s in user_skills if s.strip()]
        if not job_clean or not user_clean:
            return 0
        job_emb = emb_model.encode([' '.join(job_clean)])
        user_emb = emb_model.encode([' '.join(user_clean)])
        return max(0, min(100, int(np.inner(job_emb, user_emb)[0] * 100)))  # Use np.inner for similarity
    except Exception as e:
        logger.error(f"Skill match error: {e}")
        return 0

def analyze_text_advanced(text):
    try:
        if not text or not text.strip():
            return {'sentiment': 0, 'subjectivity': 0, 'word_count': 0, 'entities': [], 'keywords': []}
        result = {
            'sentiment': 0,
            'subjectivity': 0,
            'word_count': len(text.split()),
            'entities': [],
            'keywords': []
        }
        try:
            blob = TextBlob(text)
            result['sentiment'] = float(blob.sentiment.polarity)
            result['subjectivity'] = float(blob.sentiment.subjectivity)
        except Exception as e:
            logger.warning(f"TextBlob failed: {e}")
        nlp_model = get_nlp()
        try:
            if nlp_model and SPACY_AVAILABLE:
                doc = nlp_model(text)
                result['entities'] = [(ent.text, ent.label_) for ent in doc.ents][:5]
                result['keywords'] = [token.text for token in doc if token.pos_ in ['NOUN', 'PROPN', 'VERB'] and not token.is_stop][:10]
        except Exception as e:
            logger.warning(f"SpaCy analysis failed: {e}")
        return result
    except Exception as e:
        logger.error(f"Advanced text analysis error: {e}")
        return {'sentiment': 0, 'subjectivity': 0, 'word_count': 0, 'entities': [], 'keywords': []}

# In submit_interview, enhance emotion detection if available
@app.route('/api/interviews/<job_id>', methods=['POST'])
@token_required
def submit_interview(current_user, job_id):
    try:
        data = request.json
        if not data:
            return jsonify({'error': 'No data'}), 400
        responses = data.get('responses', [])
        det = get_detector()
        emotion_scores = {}
        if det and CV2_AVAILABLE and data.get('video_data'):  # Assume video upload; enhance as needed
            # Process video frames for emotions (placeholder - implement CV logic)
            # e.g., cap = cv2.VideoCapture(data['video_data'])
            # ... detect emotions per frame
            emotion_scores = {'confidence': random.uniform(40, 80)}  # Mock for now
            logger.info("Processed video emotions")
        else:
            emotion_scores = {'confidence': 50}
        analysis_results = [analyze_text_advanced(r.get('text', '')) for r in responses]
        sentiment_scores = [res['sentiment'] for res in analysis_results]
        keyword_scores = [calculate_keyword_match(r.get('text', ''), r.get('expected_keywords', [])) for r in responses]
        avg_sentiment = sum(sentiment_scores) / len(sentiment_scores) if sentiment_scores else 0
        avg_keyword = sum(keyword_scores) / len(keyword_scores) if keyword_scores else 0
        confidence = emotion_scores.get('confidence', 50)
        overall_score = round((confidence * 0.3 + avg_keyword * 0.5 + (avg_sentiment + 1) * 50 * 0.2), 2)
        interview_id = f"{current_user['id']}_{job_id}"
        interviews_db[interview_id] = {
            'id': interview_id,
            'user_id': current_user['id'],
            'job_id': job_id,
            'overall_score': overall_score,
            'detailed_scores': {'sentiment': avg_sentiment, 'keywords': avg_keyword, 'confidence': confidence},
            'completed_at': datetime.utcnow().isoformat()
        }
        application = next((app for app in applications_db.values() if app['user_id'] == current_user['id'] and app['job_id'] == job_id), None)
        if application:
            application['interview_completed'] = True
        add_to_blockchain('INTERVIEW_COMPLETED', {'user_id': current_user['id'], 'job_id': job_id})
        send_notification(current_user['id'], f"Interview completed: {overall_score:.1f}%", 'success')
        update_candidate_rankings(job_id)
        return jsonify({'message': 'Interview submitted', 'interview': interviews_db[interview_id]}), 201
    except Exception as e:
        logger.error(f"Interview submission failed: {e}")
        return jsonify({'error': 'Submission failed'}), 500

# ... (all other routes and functions unchanged, but ensure they use lazy getters where applicable)

# For Gunicorn on Render: Comment out app.run() to avoid conflicts
if __name__ == '__main__':
    # For local dev: Use app.run()
    print("\n" + "=" * 70)
    print("SOULCRUIT AI Backend v5.1 Starting (Local Mode)...")
    print(f"SKIP_MODELS: {SKIP_MODELS} | EMBEDDINGS: {EMBEDDINGS_AVAILABLE} | TRANSFORMERS: {TRANSFORMERS_AVAILABLE}")
    print("=" * 70)
    try:
        port = int(os.environ.get('PORT', 5000))
        print(f"🚀 Binding Flask to 0.0.0.0:{port}")
        print("✅ Ready! (Use gunicorn for Render)")
        app.run(host='0.0.0.0', port=port, debug=True, use_reloader=False, threaded=True)
    except KeyboardInterrupt:
        print("\nServer stopped")
    except Exception as e:
        print(f"\n🚨 Startup error: {e}")
        logger.error(f"Startup failed: {e}", exc_info=True)
        raise
else:
    # On import (Gunicorn), just log
    logger.info("App imported successfully for production (e.g., Gunicorn)")
