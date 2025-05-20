import sqlite3
import warnings
warnings.filterwarnings("ignore", category=UserWarning, message="TypedStorage is deprecated")
from werkzeug.security import generate_password_hash, check_password_hash
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from collections import defaultdict, Counter
from datetime import datetime, timedelta
import pytz
import time
from recommender import get_recommendation_with_emotion

# Inisialisasi Flask
app = Flask(__name__, static_folder='frontend')
CORS(app, resources={r"/*": {"origins": "http://127.0.0.1:5500"}}, supports_credentials=True)

# ------------------ TAMBAHAN UNTUK HALAMAN LOGIN ------------------
@app.route('/')
def serve_login_page():
    return send_from_directory('frontend', 'Login.html')

@app.route('/<path:path>')
def serve_static_files(path):
    return send_from_directory('frontend', path)
# ------------------------------------------------------------------

# Fungsi koneksi database
def get_db_connection():
    conn = sqlite3.connect('database.db')
    conn.row_factory = sqlite3.Row
    return conn

# Buat tabel jika belum ada
with app.app_context():
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            email TEXT UNIQUE NOT NULL,
            password TEXT NOT NULL
        )
    ''')
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS reports (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_email TEXT NOT NULL,
            text TEXT NOT NULL,
            sentiment TEXT NOT NULL,
            suggestion TEXT NOT NULL,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    conn.commit()
    conn.close()

@app.route('/send_message', methods=['POST'])
def send_message():
    data = request.json
    text = data.get('text')
    user_email = data.get('user_email')

    if not text or not user_email:
        return jsonify({"error": "Teks dan email tidak boleh kosong"}), 400

    try:
        emotion_info, suggestions = get_recommendation_with_emotion(text)

        # Ambil saran dari indeks 1
        best_suggestion = suggestions[1]["activity"]

        # Gunakan label sentimen utama dari emosi
        sentiment = emotion_info["main_sentiment"]

    except Exception as e:
        return jsonify({"error": f"Gagal memproses rekomendasi: {str(e)}"}), 500

    # Timestamp WIB
    wib = pytz.timezone('Asia/Jakarta')
    timestamp_wib = datetime.now(wib).strftime('%Y-%m-%d %H:%M:%S')

    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute(''' 
            INSERT INTO reports (user_email, text, sentiment, suggestion, timestamp)
            VALUES (?, ?, ?, ?, ?)
        ''', (user_email, text, sentiment, best_suggestion, timestamp_wib))
        conn.commit()
        conn.close()
    except sqlite3.Error:
        return jsonify({"error": "Gagal menyimpan data"}), 500

    return jsonify({
        "sentiment": sentiment,
        "suggestions": suggestions,
        "suggestion": best_suggestion
    }), 200


@app.route('/get_history')
def get_history():
    user_email = request.args.get('user_email')
    month = request.args.get('month')
    year = request.args.get('year')
    if not user_email:
        return jsonify({"error": "User tidak ditemukan."}), 400

    conn = get_db_connection()
    cursor = conn.cursor()
    
    if month and year:
        start_date = f"{year}-{int(month):02d}-01"
        if int(month) == 12:
            end_date = f"{int(year)+1}-01-01"
        else:
            end_date = f"{year}-{int(month)+1:02d}-01"
            
    cursor.execute(''' 
        SELECT timestamp, sentiment 
        FROM reports 
        WHERE user_email = ? 
        ORDER BY timestamp ASC
    ''', (user_email,))
    rows = cursor.fetchall()
    conn.close()

    if not rows:
        return jsonify({
            "penggunaan": 0,
            "dominan": [],
            "waktu": "0 menit",
            "dates": [],
            "usage_data": [],
            "heatmap": []
        })

    usage_per_day = {}
    sentiment_per_day = defaultdict(list)
    daily_durations = defaultdict(int)  # Tambahan untuk total durasi per hari
    heatmap = []

    wib_tz = pytz.timezone('Asia/Jakarta')

    prev_time = None

    for row in rows:
        timestamp = row["timestamp"]
        sentiment = row["sentiment"]

        utc_time = datetime.strptime(timestamp, "%Y-%m-%d %H:%M:%S")
        utc_time = pytz.utc.localize(utc_time)
        local_time = utc_time.astimezone(wib_tz)

        local_date_str = local_time.strftime("%Y-%m-%d")

        usage_per_day[local_date_str] = usage_per_day.get(local_date_str, 0) + 1
        sentiment_per_day[local_date_str].append(sentiment)

        if prev_time:
            delta = (local_time - prev_time).total_seconds() / 60  # beda dalam menit
            if delta < 30:  # misal: kalau jaraknya <30 menit dianggap masih dalam satu sesi
                daily_durations[local_date_str] += delta
            else:
                daily_durations[local_date_str] += 5  # anggap interaksi kecil 5 menit
        else:
            daily_durations[local_date_str] += 5  # laporan pertama dianggap 5 menit
        
        prev_time = local_time

    # Bikin heatmap
    for date_str, count in usage_per_day.items():
        dt = datetime.strptime(date_str, "%Y-%m-%d")
        unix_ts = int(time.mktime(dt.timetuple())) * 1000

        heatmap.append({
            "title": f"Jumlah laporan: {count}",
            "start": unix_ts,
            "description": f"Total {count} laporan"
        })

    sorted_dates = sorted(usage_per_day.keys())
    date_labels = sorted_dates
    usage_data = [usage_per_day[date] for date in sorted_dates]
    dominant_data = []

    for date in sorted_dates:
        sentiments = sentiment_per_day[date]
        if sentiments:
            counter = Counter(sentiments)
            most_common = counter.most_common(1)[0][0]
            dominant_data.append(most_common)
        else:
            dominant_data.append("Netral")

    # Total waktu real
    total_minutes = sum(daily_durations.values())

    return jsonify({
        "penggunaan": sum(usage_per_day.values()),
        "dominan": dominant_data,
        "waktu": f"{int(total_minutes)} menit",
        "dates": date_labels,
        "usage_data": usage_data,
        "heatmap": heatmap
    })

@app.route('/get_suggestion', methods=['GET'])
def get_suggestion():
    user_email = request.args.get('user_email')

    if not user_email:
        return jsonify({"error": "User tidak ditemukan."}), 400

    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT suggestion FROM reports WHERE user_email = ? ORDER BY id DESC LIMIT 1", (user_email,))
    row = cursor.fetchone()
    conn.close()

    return jsonify({"suggestion": row["suggestion"] if row else "Belum ada laporan yang dikirimkan."})

@app.route('/get_chart_data', methods=['GET'])
def get_chart_data():
    user_email = request.args.get('user_email')

    if not user_email:
        return jsonify({"error": "User tidak ditemukan."}), 400

    # Ambil tanggal sekarang di zona waktu WIB
    wib = pytz.timezone('Asia/Jakarta')
    today = datetime.now(wib).strftime('%Y-%m-%d')

    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT sentiment, COUNT(sentiment) 
        FROM reports 
        WHERE user_email = ? AND DATE(timestamp) = ? 
        GROUP BY sentiment
    """, (user_email, today))
    rows = cursor.fetchall()
    conn.close()

    sentiment_map = {
        "positive": "positif",
        "neutral": "netral",
        "negative": "negatif"
    }

    label_counts = {"positif": 0, "netral": 0, "negatif": 0}
    for row in rows:
        label = sentiment_map.get(row[0], "netral")
        label_counts[label] += row[1]

    labels = list(label_counts.keys())
    values = list(label_counts.values())

    return jsonify({"labels": labels, "values": values})

@app.route('/register', methods=['POST'])
def register():
    conn = None
    try:
        data = request.json
        email = data.get('email')
        password = data.get('password')

        if not email or not password:
            return jsonify({"error": "Email dan Password wajib diisi"}), 400

        hashed_password = generate_password_hash(password)

        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("INSERT INTO users (email, password) VALUES (?, ?)", (email, hashed_password))
        conn.commit()
        return jsonify({"message": "Registrasi berhasil!"}), 201
    except sqlite3.IntegrityError:
        return jsonify({"error": "Email sudah digunakan"}), 400
    finally:
        if conn:
            conn.close()

@app.route('/login', methods=['POST'])
def login():
    data = request.json
    email = data.get('email')
    password = data.get('password')

    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT password FROM users WHERE email=?", (email,))
    user = cursor.fetchone()
    conn.close()

    if user and check_password_hash(user[0], password):
        return jsonify({"message": "Login berhasil!", "user_email": email}), 200
    else:
        return jsonify({"error": "Email atau password salah"}), 401

if __name__ == "__main__":
    app.run(debug=True)