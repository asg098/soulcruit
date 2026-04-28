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

users_db = {}
profiles_db = {}
jobs_db = {}
applications_db = {}
assessments_db = {}
interviews_db = {}
blockchain_db = []
notifications_db = {}
rankings_db = {}
contact_shares_db = {}
question_bank_db = {}
interview_questions_db = {}
saved_jobs_db = {}
interview_answers_db = {}
video_recordings_db = {}

JOB_ROLES = {
    'Software Development': ['Software Engineer', 'Backend Developer', 'Frontend Developer', 'Full Stack Developer', 'DevOps Engineer', 'Mobile App Developer'],
    'Data & AI': ['Data Scientist', 'Data Analyst', 'Machine Learning Engineer', 'AI Engineer', 'Data Engineer'],
    'Design': ['UI/UX Designer', 'Graphic Designer', 'Product Designer', 'Web Designer'],
    'Management': ['Product Manager', 'Project Manager', 'Engineering Manager', 'Scrum Master'],
    'Marketing & Sales': ['Digital Marketing Manager', 'Content Writer', 'SEO Specialist', 'Sales Executive'],
    'Quality & Testing': ['QA Engineer', 'Test Automation Engineer', 'Quality Analyst'],
    'Other': ['Business Analyst', 'Technical Writer', 'System Administrator', 'Network Engineer']
}

SKILL_CATEGORIES = {
    'Programming Languages': ['Python', 'Java', 'JavaScript', 'C++', 'TypeScript', 'Go', 'Rust', 'Swift', 'Kotlin'],
    'Web Development': ['React', 'Angular', 'Vue.js', 'Node.js', 'Django', 'Flask', 'Spring', 'Express.js', 'Next.js'],
    'Mobile Development': ['React Native', 'Flutter', 'iOS Development', 'Android Development', 'SwiftUI'],
    'Data Science & ML': ['Machine Learning', 'Deep Learning', 'Data Analysis', 'Pandas', 'NumPy', 'TensorFlow', 'PyTorch'],
    'Database': ['MySQL', 'PostgreSQL', 'MongoDB', 'Redis', 'Firebase', 'SQLite'],
    'Cloud & DevOps': ['AWS', 'Azure', 'Google Cloud', 'Docker', 'Kubernetes', 'Jenkins', 'Git', 'GitHub'],
    'Design Tools': ['Figma', 'Adobe XD', 'Photoshop', 'Sketch', 'Canva'],
    'Business Skills': ['Project Management', 'Agile', 'Scrum', 'Product Management', 'Marketing'],
    'Soft Skills': ['Communication', 'Leadership', 'Teamwork', 'Problem Solving', 'Time Management']
}

DEFAULT_QUESTIONS = {
    'Python': [
        {'question': 'What is the difference between list and tuple in Python?', 'options': ['Lists are mutable, tuples are immutable', 'Lists are immutable, tuples are mutable', 'Both are mutable', 'Both are immutable'], 'correct': 0, 'difficulty': 'easy'},
        {'question': 'What is a lambda function in Python?', 'options': ['Anonymous function', 'Named function', 'Class method', 'Generator function'], 'correct': 0, 'difficulty': 'medium'},
        {'question': 'What does the with statement do in Python?', 'options': ['Context management', 'Loop iteration', 'Function definition', 'Class inheritance'], 'correct': 0, 'difficulty': 'medium'},
        {'question': 'What is a decorator in Python?', 'options': ['Function that modifies another function', 'Loop structure', 'Data type', 'Exception handler'], 'correct': 0, 'difficulty': 'hard'}
    ],
    'JavaScript': [
        {'question': 'What is closure in JavaScript?', 'options': ['Function with access to outer scope', 'Loop structure', 'Class inheritance', 'Event handler'], 'correct': 0, 'difficulty': 'medium'},
        {'question': 'What does === operator do?', 'options': ['Strict equality check', 'Assignment', 'Loose equality check', 'Comparison'], 'correct': 0, 'difficulty': 'easy'},
        {'question': 'What is Promise in JavaScript?', 'options': ['Object for async operations', 'Loop type', 'Variable type', 'Function type'], 'correct': 0, 'difficulty': 'medium'}
    ],
    'React': [
        {'question': 'What is a React Hook?', 'options': ['Function to use state in functional components', 'Class component method', 'Routing mechanism', 'CSS framework'], 'correct': 0, 'difficulty': 'medium'},
        {'question': 'What is useState?', 'options': ['Hook to manage state', 'Lifecycle method', 'Context API', 'Router'], 'correct': 0, 'difficulty': 'easy'},
        {'question': 'What is virtual DOM?', 'options': ['In-memory representation of DOM', 'Browser DOM', 'Database', 'API'], 'correct': 0, 'difficulty': 'medium'}
    ],
    'Machine Learning': [
        {'question': 'What is overfitting?', 'options': ['Model performs well on training but poor on test data', 'Model performs poorly on all data', 'Model is too simple', 'Data is insufficient'], 'correct': 0, 'difficulty': 'medium'},
        {'question': 'What is supervised learning?', 'options': ['Learning with labeled data', 'Learning without labels', 'Reinforcement learning', 'Unsupervised clustering'], 'correct': 0, 'difficulty': 'easy'},
        {'question': 'What is gradient descent?', 'options': ['Optimization algorithm', 'Classification algorithm', 'Clustering method', 'Data preprocessing'], 'correct': 0, 'difficulty': 'hard'}
    ]
}

DEFAULT_INTERVIEW_QUESTIONS = {
    'technical': [
        {'question': 'Explain your approach to solving complex technical problems', 'expected_keywords': ['analysis', 'solution', 'approach', 'methodology', 'testing']},
        {'question': 'Describe a challenging project you worked on and how you overcame obstacles', 'expected_keywords': ['project', 'challenge', 'solution', 'outcome', 'learn']},
        {'question': 'How do you stay updated with new technologies?', 'expected_keywords': ['learning', 'reading', 'courses', 'community', 'practice']}
    ],
    'behavioral': [
        {'question': 'Tell me about a time you worked in a team to achieve a goal', 'expected_keywords': ['team', 'collaboration', 'goal', 'achievement', 'contribution']},
        {'question': 'Describe a situation where you had to learn something new quickly', 'expected_keywords': ['learn', 'quick', 'adapt', 'challenge', 'success']},
        {'question': 'How do you handle tight deadlines and pressure?', 'expected_keywords': ['deadline', 'pressure', 'prioritize', 'manage', 'deliver']}
    ],
    'situational': [
        {'question': 'How would you handle a disagreement with a team member?', 'expected_keywords': ['communication', 'respect', 'resolution', 'understanding', 'compromise']},
        {'question': 'What would you do if you discovered a critical bug just before deployment?', 'expected_keywords': ['priority', 'fix', 'communicate', 'test', 'decision']},
        {'question': 'How would you prioritize multiple urgent tasks?', 'expected_keywords': ['prioritize', 'importance', 'deadline', 'impact', 'manage']}
    ]
}

for skill, questions in DEFAULT_QUESTIONS.items():
    for q in questions:
        q_id = str(uuid.uuid4())
        question_bank_db[q_id] = {
            'id': q_id,
            'skill': skill,
            'question': q['question'],
            'options': q['options'],
            'correct': q['correct'],
            'difficulty': q['difficulty'],
            'created_at': datetime.utcnow().isoformat(),
            'created_by': 'system'
        }

for category, questions in DEFAULT_INTERVIEW_QUESTIONS.items():
    for q in questions:
        q_id = str(uuid.uuid4())
        interview_questions_db[q_id] = {
            'id': q_id,
            'question': q['question'],
            'category': category,
            'expected_keywords': q['expected_keywords'],
            'created_by': 'system',
            'created_at': datetime.utcnow().isoformat()
        }

def validate_email(email):
    return re.match(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$', email) is not None

def validate_password(password):
    if len(password) < 8:
        return False, "Password must be at least 8 characters"
    if not re.search(r'[A-Z]', password):
        return False, "Password must contain uppercase letter"
    if not re.search(r'[a-z]', password):
        return False, "Password must contain lowercase letter"
    if not re.search(r'\d', password):
        return False, "Password must contain number"
    return True, "Valid"

def token_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        token = request.headers.get('Authorization')
        if not token:
            return jsonify({'error': 'Token missing'}), 401
        try:
            if token.startswith('Bearer '):
                token = token.replace('Bearer ', '')
            data = jwt.decode(token, app.config['SECRET_KEY'], algorithms=['HS256'])
            current_user = users_db.get(data.get('user_id'))
            if not current_user:
                return jsonify({'error': 'Invalid token'}), 401
        except jwt.ExpiredSignatureError:
            return jsonify({'error': 'Token expired'}), 401
        except:
            return jsonify({'error': 'Invalid token'}), 401
        return f(current_user, *args, **kwargs)
    return decorated

def add_to_blockchain(action, data):
    try:
        prev_hash = blockchain_db[-1]['hash'] if blockchain_db else '0' * 64
        block = {
            'index': len(blockchain_db),
            'timestamp': datetime.utcnow().isoformat(),
            'action': action,
            'data': data,
            'prev_hash': prev_hash
        }
        block_string = f"{block['index']}{block['timestamp']}{block['action']}{str(block['data'])}{block['prev_hash']}"
        block['hash'] = hashlib.sha256(block_string.encode()).hexdigest()
        blockchain_db.append(block)
    except Exception as e:
        logger.error(f"Blockchain error: {e}")

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

def calculate_keyword_match(text, expected_keywords):
    try:
        if not text or not expected_keywords:
            return 0
        text_lower = text.lower()
        matched = sum(1 for keyword in expected_keywords if keyword.lower() in text_lower)
        return (matched / len(expected_keywords)) * 100
    except:
        return 0

def calculate_candidate_score(application, profile, assessment_score=None, interview_score=None):
    try:
        score = 0
        weights = {'skill_match': 40, 'assessment': 30, 'interview': 20, 'experience': 10}
        if application.get('skill_match_score'):
            score += (application['skill_match_score'] / 100) * weights['skill_match']
        if assessment_score:
            score += (assessment_score / 100) * weights['assessment']
        if interview_score:
            score += (interview_score / 100) * weights['interview']
        experience_score = min(len(profile.get('experience', '').split()) / 50 * 20, 100)
        score += (experience_score / 100) * weights['experience']
        return round(score, 2)
    except:
        return 0

def update_candidate_rankings(job_id):
    try:
        job = jobs_db.get(job_id)
        if not job:
            return []
        job_applications = [app for app in applications_db.values() if app['job_id'] == job_id]
        ranked_candidates = []
        for app in job_applications:
            profile = profiles_db.get(app['user_id'])
            user = users_db.get(app['user_id'])
            assessment = assessments_db.get(f"{app['user_id']}_{job_id}")
            interview = interviews_db.get(f"{app['user_id']}_{job_id}")
            assessment_score = assessment.get('score') if assessment else None
            interview_score = interview.get('overall_score') if interview else None
            total_score = calculate_candidate_score(app, profile, assessment_score, interview_score)
            qualified = True
            if assessment_score is not None and assessment_score < job.get('assessment_cutoff', 60):
                qualified = False
            if interview_score is not None and interview_score < job.get('interview_cutoff', 60):
                qualified = False
            if total_score < job.get('overall_cutoff', 60):
                qualified = False
            ranked_candidates.append({
                'application_id': app['id'],
                'user_id': app['user_id'],
                'name': profile.get('name', 'Unknown') if profile else 'Unknown',
                'email': user.get('email') if user else None,
                'phone': profile.get('phone') if profile else None,
                'total_score': total_score,
                'skill_match': app.get('skill_match_score', 0),
                'assessment_score': assessment_score,
                'interview_score': interview_score,
                'applied_at': app['applied_at'],
                'qualified': qualified
            })
        ranked_candidates.sort(key=lambda x: x['total_score'], reverse=True)
        for idx, candidate in enumerate(ranked_candidates):
            candidate['rank'] = idx + 1
        rankings_db[job_id] = ranked_candidates
        return ranked_candidates
    except Exception as e:
        logger.error(f"Ranking error: {e}")
        return []

def send_notification(user_id, message, notification_type='info'):
    try:
        notification_id = str(uuid.uuid4())
        notifications_db[notification_id] = {
            'id': notification_id,
            'user_id': user_id,
            'message': message,
            'type': notification_type,
            'read': False,
            'created_at': datetime.utcnow().isoformat()
        }
    except:
        pass

def find_free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('', 0))
        s.listen(1)
        return s.getsockname()[1]

@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Endpoint not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Internal server error'}), 500

@app.route('/api/auth/register', methods=['POST'])
def register():
    try:
        data = request.json
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        email = data.get('email', '').lower().strip()
        password = data.get('password', '')
        name = data.get('name', '').strip()
        role = data.get('role', 'student')
        if not email or not password or not name:
            return jsonify({'error': 'Email, password, and name required'}), 400
        if not validate_email(email):
            return jsonify({'error': 'Invalid email format'}), 400
        is_valid, message = validate_password(password)
        if not is_valid:
            return jsonify({'error': message}), 400
        if email in [u['email'] for u in users_db.values()]:
            return jsonify({'error': 'Email already registered'}), 400
        if role not in ['student', 'recruiter']:
            return jsonify({'error': 'Invalid role'}), 400
        user_id = str(uuid.uuid4())
        users_db[user_id] = {
            'id': user_id,
            'email': email,
            'password': bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8'),
            'name': name,
            'role': role,
            'created_at': datetime.utcnow().isoformat(),
            'verified': False,
            'premium': False,
            'profile_completed': False,
            'last_login': None,
            'active': True
        }
        token = jwt.encode({'user_id': user_id, 'exp': datetime.utcnow() + timedelta(days=30)}, app.config['SECRET_KEY'], algorithm='HS256')
        add_to_blockchain('USER_REGISTERED', {'user_id': user_id, 'email': email, 'role': role})
        user_response = {k: v for k, v in users_db[user_id].items() if k != 'password'}
        return jsonify({'token': token, 'user': user_response}), 201
    except Exception as e:
        logger.error(f"Registration error: {e}")
        return jsonify({'error': 'Registration failed'}), 500

@app.route('/api/auth/login', methods=['POST'])
def login():
    try:
        data = request.json
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        email = data.get('email', '').lower().strip()
        password = data.get('password', '')
        if not email or not password:
            return jsonify({'error': 'Email and password required'}), 400
        user = next((u for u in users_db.values() if u['email'] == email), None)
        if not user:
            return jsonify({'error': 'Invalid credentials'}), 401
        if not user.get('active', True):
            return jsonify({'error': 'Account deactivated'}), 403
        if not bcrypt.checkpw(password.encode('utf-8'), user['password'].encode('utf-8')):
            return jsonify({'error': 'Invalid credentials'}), 401
        user['last_login'] = datetime.utcnow().isoformat()
        token = jwt.encode({'user_id': user['id'], 'exp': datetime.utcnow() + timedelta(days=30)}, app.config['SECRET_KEY'], algorithm='HS256')
        user_response = {k: v for k, v in user.items() if k != 'password'}
        return jsonify({'token': token, 'user': user_response}), 200
    except Exception as e:
        logger.error(f"Login error: {e}")
        return jsonify({'error': 'Login failed'}), 500

@app.route('/api/job-roles', methods=['GET'])
def get_job_roles():
    return jsonify({'job_roles': JOB_ROLES}), 200

@app.route('/api/skills/categories', methods=['GET'])
def get_skill_categories():
    return jsonify({'categories': SKILL_CATEGORIES}), 200

@app.route('/api/profile', methods=['GET', 'POST', 'PUT'])
@token_required
def profile(current_user):
    try:
        if request.method in ['POST', 'PUT']:
            data = request.json
            if not data:
                return jsonify({'error': 'No data provided'}), 400
            name = data.get('name', '').strip()
            if not name:
                return jsonify({'error': 'Name required'}), 400
            profiles_db[current_user['id']] = {
                'user_id': current_user['id'],
                'name': name,
                'skills': data.get('skills', []),
                'education': data.get('education', '').strip(),
                'experience': data.get('experience', '').strip(),
                'location': data.get('location', '').strip(),
                'bio': data.get('bio', '').strip(),
                'phone': data.get('phone', '').strip(),
                'linkedin': data.get('linkedin', '').strip(),
                'github': data.get('github', '').strip(),
                'portfolio': data.get('portfolio', '').strip(),
                'updated_at': datetime.utcnow().isoformat()
            }
            users_db[current_user['id']]['profile_completed'] = True
            users_db[current_user['id']]['name'] = name
            add_to_blockchain('PROFILE_UPDATED', {'user_id': current_user['id']})
            return jsonify({'message': 'Profile saved', 'profile': profiles_db[current_user['id']]}), 200
        profile_data = profiles_db.get(current_user['id'])
        if not profile_data:
            return jsonify({'error': 'Profile not found', 'profile': None}), 404
        return jsonify({'profile': profile_data}), 200
    except Exception as e:
        logger.error(f"Profile error: {e}")
        return jsonify({'error': 'Profile operation failed'}), 500

@app.route('/api/jobs', methods=['GET', 'POST'])
@token_required
def jobs(current_user):
    try:
        if request.method == 'POST':
            if current_user['role'] != 'recruiter':
                return jsonify({'error': 'Only recruiters can post jobs'}), 403
            data = request.json
            if not data:
                return jsonify({'error': 'No data provided'}), 400
            required = ['title', 'description', 'skills']
            for field in required:
                if not data.get(field):
                    return jsonify({'error': f'{field} required'}), 400
            job_id = str(uuid.uuid4())
            jobs_db[job_id] = {
                'id': job_id,
                'posted_by': current_user['id'],
                'company_name': data.get('company_name', '').strip(),
                'title': data.get('title', '').strip(),
                'job_role': data.get('job_role', '').strip(),
                'description': data.get('description', '').strip(),
                'skills': data.get('skills', []),
                'experience_level': data.get('experience_level', '').strip(),
                'job_type': data.get('job_type', 'full-time'),
                'job_mode': data.get('job_mode', 'job'),
                'location': data.get('location', '').strip(),
                'salary_range': data.get('salary_range', '').strip(),
                'assessment_cutoff': data.get('assessment_cutoff', 60),
                'interview_cutoff': data.get('interview_cutoff', 60),
                'overall_cutoff': data.get('overall_cutoff', 60),
                'status': 'active',
                'posted_at': datetime.utcnow().isoformat(),
                'applications_count': 0
            }
            add_to_blockchain('JOB_POSTED', {'job_id': job_id, 'posted_by': current_user['id']})
            send_notification(current_user['id'], f"Job '{data.get('title')}' posted successfully", 'success')
            return jsonify({'message': 'Job posted', 'job': jobs_db[job_id]}), 201
        filtered_jobs = [j for j in jobs_db.values() if j.get('status') == 'active']
        if current_user['role'] == 'student':
            user_profile = profiles_db.get(current_user['id'])
            if user_profile and user_profile.get('skills'):
                for job in filtered_jobs:
                    job['skill_match_score'] = calculate_skill_match(job.get('skills', []), user_profile.get('skills', []))
        return jsonify({'jobs': filtered_jobs}), 200
    except Exception as e:
        logger.error(f"Jobs error: {e}")
        return jsonify({'error': 'Jobs operation failed'}), 500

@app.route('/api/jobs/<job_id>', methods=['GET', 'PUT', 'DELETE'])
@token_required
def job_detail(current_user, job_id):
    try:
        job = jobs_db.get(job_id)
        if not job:
            return jsonify({'error': 'Job not found'}), 404
        if request.method == 'GET':
            user_profile = profiles_db.get(current_user['id'])
            if user_profile and user_profile.get('skills'):
                job['skill_match_score'] = calculate_skill_match(job.get('skills', []), user_profile.get('skills', []))
            return jsonify({'job': job}), 200
        if request.method == 'PUT':
            if job['posted_by'] != current_user['id']:
                return jsonify({'error': 'Unauthorized'}), 403
            data = request.json
            if not data:
                return jsonify({'error': 'No data'}), 400
            for key in ['title', 'description', 'skills', 'location', 'status']:
                if key in data:
                    job[key] = data[key]
            job['updated_at'] = datetime.utcnow().isoformat()
            return jsonify({'message': 'Job updated', 'job': job}), 200
        if request.method == 'DELETE':
            if job['posted_by'] != current_user['id']:
                return jsonify({'error': 'Unauthorized'}), 403
            job['status'] = 'deleted'
            return jsonify({'message': 'Job deleted'}), 200
    except Exception as e:
        return jsonify({'error': 'Operation failed'}), 500

@app.route('/api/jobs/<job_id>/apply', methods=['POST'])
@token_required
def apply_job(current_user, job_id):
    try:
        if current_user['role'] != 'student':
            return jsonify({'error': 'Only students can apply'}), 403
        job = jobs_db.get(job_id)
        if not job or job.get('status') != 'active':
            return jsonify({'error': 'Job not found or inactive'}), 404
        existing = next((app for app in applications_db.values() if app['user_id'] == current_user['id'] and app['job_id'] == job_id), None)
        if existing:
            return jsonify({'error': 'Already applied'}), 400
        user_profile = profiles_db.get(current_user['id'])
        if not user_profile:
            return jsonify({'error': 'Complete profile first'}), 400
        skill_match = calculate_skill_match(job.get('skills', []), user_profile.get('skills', []))
        application_id = str(uuid.uuid4())
        applications_db[application_id] = {
            'id': application_id,
            'user_id': current_user['id'],
            'job_id': job_id,
            'skill_match_score': skill_match,
            'status': 'applied',
            'applied_at': datetime.utcnow().isoformat(),
            'assessment_completed': False,
            'interview_completed': False
        }
        job['applications_count'] = job.get('applications_count', 0) + 1
        add_to_blockchain('APPLICATION_SUBMITTED', {'application_id': application_id})
        send_notification(current_user['id'], f"Application submitted for {job['title']}", 'success')
        update_candidate_rankings(job_id)
        return jsonify({'message': 'Application submitted', 'application': applications_db[application_id]}), 201
    except Exception as e:
        return jsonify({'error': 'Application failed'}), 500

@app.route('/api/applications', methods=['GET'])
@token_required
def applications(current_user):
    try:
        if current_user['role'] == 'student':
            user_apps = [app for app in applications_db.values() if app['user_id'] == current_user['id']]
            for app in user_apps:
                job = jobs_db.get(app['job_id'])
                if job:
                    app['job_details'] = job
        else:
            job_ids = [j['id'] for j in jobs_db.values() if j['posted_by'] == current_user['id']]
            user_apps = [app for app in applications_db.values() if app['job_id'] in job_ids]
            for app in user_apps:
                profile = profiles_db.get(app['user_id'])
                if profile:
                    app['candidate_details'] = profile
        return jsonify({'applications': user_apps}), 200
    except Exception as e:
        return jsonify({'error': 'Fetch failed'}), 500

@app.route('/api/jobs/<job_id>/assessment/generate', methods=['POST'])
@token_required
def generate_assessment(current_user, job_id):
    try:
        job = jobs_db.get(job_id)
        if not job:
            return jsonify({'error': 'Job not found'}), 404
        data = request.json or {}
        num_questions = data.get('num_questions', 10)
        job_skills = job.get('skills', [])
        available_questions = [q for q in question_bank_db.values() if q['skill'] in job_skills]
        if len(available_questions) < num_questions:
            num_questions = len(available_questions) if available_questions else 0
        if num_questions == 0:
            return jsonify({'error': 'No questions available'}), 400
        selected_questions = random.sample(available_questions, num_questions)
        return jsonify({'questions': [{'id': q['id'], 'question': q['question'], 'options': q['options'], 'skill': q['skill'], 'difficulty': q['difficulty']} for q in selected_questions]}), 200
    except Exception as e:
        return jsonify({'error': 'Generation failed'}), 500

@app.route('/api/assessments/<job_id>', methods=['GET', 'POST'])
@token_required
def assessments(current_user, job_id):
    try:
        if request.method == 'POST':
            data = request.json
            if not data or not data.get('answers'):
                return jsonify({'error': 'No answers'}), 400
            answers = data.get('answers', {})
            questions = data.get('questions', [])
            correct_count = sum(1 for q in questions if answers.get(str(q['id'])) == question_bank_db.get(q['id'], {}).get('correct'))
            score = (correct_count / len(questions) * 100) if questions else 0
            assessment_id = f"{current_user['id']}_{job_id}"
            assessments_db[assessment_id] = {
                'id': assessment_id,
                'user_id': current_user['id'],
                'job_id': job_id,
                'score': round(score, 2),
                'correct_answers': correct_count,
                'total_questions': len(questions),
                'completed_at': datetime.utcnow().isoformat()
            }
            application = next((app for app in applications_db.values() if app['user_id'] == current_user['id'] and app['job_id'] == job_id), None)
            if application:
                application['assessment_completed'] = True
            add_to_blockchain('ASSESSMENT_COMPLETED', {'user_id': current_user['id'], 'job_id': job_id})
            send_notification(current_user['id'], f"Assessment completed: {score:.1f}%", 'success')
            update_candidate_rankings(job_id)
            return jsonify({'message': 'Assessment submitted', 'assessment': assessments_db[assessment_id]}), 201
        assessment = assessments_db.get(f"{current_user['id']}_{job_id}")
        return jsonify({'assessment': assessment}), 200
    except Exception as e:
        return jsonify({'error': 'Operation failed'}), 500

@app.route('/api/jobs/<job_id>/interview/generate', methods=['POST'])
@token_required
def generate_interview(current_user, job_id):
    try:
        data = request.json or {}
        num_questions = data.get('num_questions', 5)
        categories = data.get('categories', ['technical', 'behavioral'])
        selected = []
        for category in categories:
            qs = [q for q in interview_questions_db.values() if q['category'] == category]
            if qs:
                selected.extend(random.sample(qs, min(len(qs), max(1, num_questions // len(categories)))))
        return jsonify({'questions': selected[:num_questions]}), 200
    except Exception as e:
        return jsonify({'error': 'Generation failed'}), 500

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

@app.route('/api/jobs/<job_id>/rankings', methods=['GET'])
@token_required
def job_rankings(current_user, job_id):
    try:
        job = jobs_db.get(job_id)
        if not job or job['posted_by'] != current_user['id']:
            return jsonify({'error': 'Unauthorized'}), 403
        rankings = rankings_db.get(job_id, [])
        if not rankings:
            rankings = update_candidate_rankings(job_id)
        return jsonify({'rankings': rankings, 'total': len(rankings)}), 200
    except Exception as e:
        return jsonify({'error': 'Fetch failed'}), 500

@app.route('/api/saved-jobs', methods=['GET', 'POST'])
@token_required
def saved_jobs(current_user):
    try:
        if request.method == 'POST':
            data = request.json
            job_id = data.get('job_id')
            if not job_id or not jobs_db.get(job_id):
                return jsonify({'error': 'Invalid job'}), 400
            if any(s['user_id'] == current_user['id'] and s['job_id'] == job_id for s in saved_jobs_db.values()):
                return jsonify({'error': 'Already saved'}), 400
            saved_id = str(uuid.uuid4())
            saved_jobs_db[saved_id] = {'id': saved_id, 'user_id': current_user['id'], 'job_id': job_id, 'saved_at': datetime.utcnow().isoformat()}
            return jsonify({'message': 'Job saved'}), 201
        user_saved = [s for s in saved_jobs_db.values() if s['user_id'] == current_user['id']]
        for s in user_saved:
            job = jobs_db.get(s['job_id'])
            if job:
                s['job_details'] = job
        return jsonify({'saved_jobs': user_saved}), 200
    except Exception as e:
        return jsonify({'error': 'Operation failed'}), 500

@app.route('/api/saved-jobs/<job_id>', methods=['DELETE'])
@token_required
def unsave_job(current_user, job_id):
    try:
        saved = next((s for s in saved_jobs_db.values() if s['user_id'] == current_user['id'] and s['job_id'] == job_id), None)
        if saved:
            del saved_jobs_db[saved['id']]
            return jsonify({'message': 'Job unsaved'}), 200
        return jsonify({'error': 'Not found'}), 404
    except Exception as e:
        return jsonify({'error': 'Operation failed'}), 500

@app.route('/api/notifications', methods=['GET'])
@token_required
def get_notifications(current_user):
    user_notifs = sorted([n for n in notifications_db.values() if n['user_id'] == current_user['id']], key=lambda x: x['created_at'], reverse=True)
    return jsonify({'notifications': user_notifs}), 200

@app.route('/api/dashboard/recruiter', methods=['GET'])
@token_required
def recruiter_dashboard(current_user):
    try:
        if current_user['role'] != 'recruiter':
            return jsonify({'error': 'Unauthorized'}), 403
        my_jobs = [j for j in jobs_db.values() if j['posted_by'] == current_user['id']]
        job_ids = [j['id'] for j in my_jobs]
        return jsonify({
            'total_jobs': len(my_jobs),
            'active_jobs': len([j for j in my_jobs if j.get('status') == 'active']),
            'total_applications': len([a for a in applications_db.values() if a['job_id'] in job_ids]),
            'jobs': [j for j in my_jobs if j.get('status') == 'active'][:5]
        }), 200
    except Exception as e:
        return jsonify({'error': 'Fetch failed'}), 500

@app.route('/api/dashboard/student', methods=['GET'])
@token_required
def student_dashboard(current_user):
    try:
        if current_user['role'] != 'student':
            return jsonify({'error': 'Unauthorized'}), 403
        my_apps = [a for a in applications_db.values() if a['user_id'] == current_user['id']]
        return jsonify({
            'total_applications': len(my_apps),
            'profile_completed': current_user.get('profile_completed', False)
        }), 200
    except Exception as e:
        return jsonify({'error': 'Fetch failed'}), 500

@app.route('/api/question-bank', methods=['GET', 'POST'])
@token_required
def question_bank(current_user):
    try:
        if request.method == 'POST':
            if current_user['role'] != 'recruiter':
                return jsonify({'error': 'Unauthorized'}), 403
            data = request.json
            if not all(k in data for k in ['skill', 'question', 'options', 'correct', 'difficulty']):
                return jsonify({'error': 'Missing fields'}), 400
            q_id = str(uuid.uuid4())
            question_bank_db[q_id] = {**data, 'id': q_id, 'created_at': datetime.utcnow().isoformat(), 'created_by': current_user['id']}
            return jsonify({'message': 'Question added', 'question': question_bank_db[q_id]}), 201
        return jsonify({'questions': list(question_bank_db.values())}), 200
    except Exception as e:
        return jsonify({'error': 'Operation failed'}), 500

@app.route('/api/interview-questions', methods=['GET', 'POST'])
@token_required
def interview_questions_route(current_user):
    try:
        if request.method == 'POST':
            if current_user['role'] != 'recruiter':
                return jsonify({'error': 'Unauthorized'}), 403
            data = request.json
            if not data.get('question') or not data.get('category'):
                return jsonify({'error': 'Missing fields'}), 400
            q_id = str(uuid.uuid4())
            interview_questions_db[q_id] = {**data, 'id': q_id, 'created_at': datetime.utcnow().isoformat(), 'created_by': current_user['id']}
            return jsonify({'message': 'Question added', 'question': interview_questions_db[q_id]}), 201
        return jsonify({'questions': list(interview_questions_db.values())}), 200
    except Exception as e:
        return jsonify({'error': 'Operation failed'}), 500

@app.route('/api/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'healthy', 'timestamp': datetime.utcnow().isoformat(), 'users': len(users_db), 'jobs': len(jobs_db), 'applications': len(applications_db)}), 200

@app.route('/', methods=['GET'])
def index():
    return jsonify({'message': 'SOULCRUIT AI Backend', 'version': '5.1', 'status': 'running'}), 200

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
