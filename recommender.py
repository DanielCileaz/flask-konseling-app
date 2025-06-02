from datasets import load_dataset
from sentence_transformers import SentenceTransformer, util
from transformers import pipeline
import re
from langdetect import detect, DetectorFactory

# Agar hasil deteksi konsisten
DetectorFactory.seed = 0

# Load dataset dari HuggingFace
dataset = load_dataset("Amod/mental_health_counseling_conversations")["train"]
questions = [item["Context"] for item in dataset if "Context" in item and "Response" in item]
answers = [item["Response"] for item in dataset if "Context" in item and "Response" in item]

# Load Sentence-BERT model
sbert_model = SentenceTransformer('all-MiniLM-L6-v2')
question_embeddings = sbert_model.encode(questions, convert_to_tensor=True)

# Emotion classifier
emotion_model = pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base", top_k=None)

# Trigger kata untuk emosi kuat
HIGH_EMOTION_KEYWORDS = ["anxious", "angry", "lonely", "scared", "afraid", "panic", "hopeless", "depressed", "cry"]

# Pemetaan label emosi
EMOTION_MAP = {
    "joy": "positive",
    "love": "positive",
    "surprise": "neutral",
    "neutral": "neutral",
    "fear": "negative",
    "anger": "negative",
    "sadness": "negative",
    "disgust": "negative"
}

# Threshold dan label penting
MIN_SIMILARITY_THRESHOLD = 0.6
GENERAL_ADVICE = "Try journaling or talking to a friend about how you're feeling. It’s okay to not have all the answers right now."

# Saran default berdasarkan sentimen
DEFAULT_ADVICE_BY_SENTIMENT = {
    "positive": "✨ Keep up the positive energy! 😊 Maybe share your good mood with someone else or write down what made you feel this way. 📝",
    "neutral": "🚶‍♂️ Take a walk, enjoy a hobby 🎨, or try something creative to keep your mind engaged in a positive way. 🌿",
    "negative": "😔 It’s okay to feel down sometimes. Try some deep breathing 🌬️, write your thoughts in a journal 📓, or talk to someone you trust 🤝."
}

# Fungsi untuk memfilter hanya kalimat bahasa Inggris
def filter_english_only(text: str) -> str:
    sentences = re.split(r'(?<=[.!?]) +', text)
    english_sentences = []
    for sent in sentences:
        try:
            lang = detect(sent)
            if lang == 'en':  # hanya kalimat bahasa Inggris
                english_sentences.append(sent)
        except:
            pass
    return ' '.join(english_sentences).strip()

# Fungsi baru untuk memilih tipe saran berdasarkan emosi dan similarity
def choose_advice_type(emotion: str, similarity: float) -> str:
    emotion = emotion.lower()
    threshold = 0.45
    if emotion in ['joy', 'neutral']:
        threshold = 0.5
    elif emotion in ['sadness', 'fear', 'anger', 'disgust']:
        threshold = 0.4
    if similarity >= threshold:
        return 'dataset advice'
    else:
        return 'default advice'

def analyze_emotion(emotions, text):
    highest = max(emotions, key=lambda x: x["score"])
    emotion_label = highest["label"].lower()
    emotion_score = highest["score"]
    main_sentiment = EMOTION_MAP.get(emotion_label, "neutral")

    if main_sentiment == "neutral" and any(re.search(rf"\b{kw}\b", text.lower()) for kw in HIGH_EMOTION_KEYWORDS):
        main_sentiment = "negative"

    intensity = "high emotional intensity" if any(kw in text.lower() for kw in HIGH_EMOTION_KEYWORDS) else "normal"

    return {
        "dominant_emotion": emotion_label,
        "main_sentiment": main_sentiment,
        "score": round(emotion_score, 3),
        "intensity": intensity
    }

def get_recommendation_with_emotion(input_text_en):
    try:
        # Emosi
        emotion_results = emotion_model(input_text_en)[0]
        emotion_info = analyze_emotion(emotion_results, input_text_en)

        # SBERT cosine similarity
        input_embedding = sbert_model.encode(input_text_en, convert_to_tensor=True)
        cosine_scores = util.pytorch_cos_sim(input_embedding, question_embeddings)[0]
        best_idx = cosine_scores.argmax().item()

        matched_q = questions[best_idx]
        matched_a = answers[best_idx]
        similarity_score = round(cosine_scores[best_idx].item(), 2)

        # Pilih tipe advice berdasarkan fungsi baru
        advice_type = choose_advice_type(emotion_info["dominant_emotion"], similarity_score)

        if advice_type == 'default advice':
            matched_q = "You seem to be experiencing something that's difficult to explain."
            sentiment = emotion_info["main_sentiment"]
            matched_a = DEFAULT_ADVICE_BY_SENTIMENT.get(sentiment, GENERAL_ADVICE)

        # Filter kalimat bahasa Inggris saja
        matched_a = filter_english_only(matched_a)

        return emotion_info, [
            {
                "type": advice_type,
                "activity": f"Similar question: {matched_q}",
                "score": similarity_score
            },
            {
                "type": advice_type,
                "activity": f"Advice: {matched_a}",
                "score": emotion_info["score"]
            }
        ]
    except Exception as e:
        return {"error": str(e)}, [{
            "type": "error",
            "activity": f"Error occurred: {str(e)}",
            "score": 0.0
        }]
