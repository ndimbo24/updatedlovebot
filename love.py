import os
import sys
import logging
from datetime import datetime
from flask import Flask, request, jsonify, session
from flask_session import Session
from flask_cors import CORS
from dotenv import load_dotenv
from openai import OpenAI

# ----------------------------
# Load environment variables
# ----------------------------
load_dotenv()
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
FLASK_SECRET_KEY = os.environ.get("FLASK_SECRET_KEY", "supersecretkey")
PORT = int(os.environ.get("PORT", 5000))  # Render sets PORT automatically

if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY not set in .env")

# ----------------------------
# Logging setup
# ----------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)

# ----------------------------
# OpenAI client
# ----------------------------
client = OpenAI(api_key=OPENAI_API_KEY)

# ----------------------------
# Flask setup
# ----------------------------
app = Flask(__name__)
app.secret_key = FLASK_SECRET_KEY
app.config["SESSION_TYPE"] = "filesystem"
app.config["SESSION_FILE_DIR"] = "./flask_session"
os.makedirs(app.config["SESSION_FILE_DIR"], exist_ok=True)
Session(app)

# Allow Vite frontend (can restrict later)
CORS(app, resources={r"/api/*": {"origins": "*"}})

# ----------------------------
# Helpers
# ----------------------------
LOG_FILE = "chat_log.txt"

def log_message(user_message: str, bot_reply: str):
    """Append messages to chat log."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(f"[{timestamp}] USER: {user_message}\n[{timestamp}] BOT: {bot_reply}\n\n")

def safe_extract_reply(completion) -> str:
    try:
        choice = completion.choices[0]
        msg = getattr(choice, "message", choice)
        content = msg.get("content") if isinstance(msg, dict) else getattr(msg, "content", None)
        return content or "Sorry, I could not generate a response."
    except Exception:
        return "Sorry, I could not generate a response."

# ----------------------------
# API Endpoints
# ----------------------------
@app.route("/api/chat", methods=["POST"])
def chat():
    try:
        data = request.get_json() or {}
        user_message = (data.get("message") or "").strip()
        if not user_message:
            return jsonify({"error": "message required"}), 400

        # --- Keyword filtering for love/relationship topics ---
        love_keywords = ["love", "relationship", "heartbreak", "trust", "emotions", "dating"]
        if not any(word in user_message.lower() for word in love_keywords):
            return jsonify({"reply": "I'm sorry, I only provide advice about love and relationships."})

        # --- Initialize session ---
        if "messages" not in session:
            session["messages"] = [
                {
                    "role": "system",
                    "content": (
                        "You are an AI chatbot specialized ONLY in love, relationships, "
                        "breakups, trust, and human connection. "
                        "If the user asks about anything outside these topics, "
                        "politely reply: 'I'm sorry, I only provide advice about love and relationships.' "
                        "Always reply in the same language the user uses."
                    )
                }
            ]
        if "history" not in session:
            session["history"] = []

        # --- Append user message ---
        session["messages"].append({"role": "user", "content": user_message})
        session["history"].append({"sender": "user", "text": user_message})
        session.modified = True

        # --- Call OpenAI API ---
        completion = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=session["messages"],
            max_tokens=400,
            temperature=0.8
        )

        bot_reply = safe_extract_reply(completion)

        # --- Save assistant reply ---
        session["messages"].append({"role": "assistant", "content": bot_reply})
        session["history"].append({"sender": "bot", "text": bot_reply})
        session.modified = True

        # --- Log conversation ---
        log_message(user_message, bot_reply)
        logging.info(f"Handled /api/chat: {user_message[:50]}...")

        return jsonify({"reply": bot_reply})

    except Exception as e:
        logging.exception("Error in /api/chat")
        return jsonify({"error": "Server error: connection to model failed"}), 500

@app.route("/api/clear", methods=["POST"])
def clear_chat():
    session.pop("messages", None)
    session.pop("history", None)
    session.modified = True
    logging.info("Chat session cleared")
    return jsonify({"status": "cleared"})

@app.route("/api/history", methods=["GET"])
def history():
    return jsonify(session.get("history", []))

# ----------------------------
# Serve frontend (optional)
# ----------------------------
@app.route("/", defaults={"path": ""})
@app.route("/<path:path>")
def serve_frontend(path):
    dist_dir = os.path.join(os.path.dirname(__file__), "frontend", "dist")
    if path != "" and os.path.exists(os.path.join(dist_dir, path)):
        from flask import send_from_directory
        return send_from_directory(dist_dir, path)
    else:
        from flask import send_from_directory
        return send_from_directory(dist_dir, "index.html")

# ----------------------------
# Run Flask server
# ----------------------------
if __name__ == "__main__":
    logging.info(f"Starting Flask server on 0.0.0.0:{PORT}")
    app.run(host="0.0.0.0", port=PORT, debug=False)
