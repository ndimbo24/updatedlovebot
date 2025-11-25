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

# Render-friendly session config
app.config["SESSION_TYPE"] = "filesystem"
app.config["SESSION_FILE_DIR"] = "./flask_session"
os.makedirs(app.config["SESSION_FILE_DIR"], exist_ok=True)

Session(app)

# Allow Vite frontend
CORS(app, resources={r"/api/*": {"origins": "*"}})

# ----------------------------
# Helpers
# ----------------------------
LOG_FILE = "chat_log.txt"

def log_message(user_message: str, bot_reply: str):
    """Append messages to log file."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(f"[{timestamp}] USER: {user_message}\n[{timestamp}] BOT: {bot_reply}\n\n")

def safe_extract_reply(completion) -> str:
    """Extract assistant reply safely."""
    try:
        choice = completion.choices[0]
        msg = getattr(choice, "message", choice)
        content = msg.get("content") if isinstance(msg, dict) else getattr(msg, "content", None)
        return content or "Sorry, I could not generate a response."
    except Exception:
        return "Sorry, I could not generate a response."

# ----------------------------
# Chat Endpoint
# ----------------------------
@app.route("/api/chat", methods=["POST"])
def chat():
    try:
        data = request.get_json() or {}
        user_message = (data.get("message") or "").strip()

        if not user_message:
            return jsonify({"error": "message required"}), 400

        # Swahili + English love-topic keywords
        love_keywords = [
            "love", "relationship", "heartbreak", "trust", "emotions", "dating",
            "mapenzi", "penzi", "moyo", "hisia", "uhusiano", "kuachana", "kuvumiliana"
        ]

        msg_lower = user_message.lower()

        if not any(word in msg_lower for word in love_keywords):
            return jsonify({
                "reply": "Samahani, naweza kujibu maswali kuhusu mapenzi, mahusiano, hisia na mambo ya moyo tu ❤️."
            })

        # Initialize session memory
        if "messages" not in session:
            session["messages"] = [
                {
                    "role": "system",
                    "content": (
                        "You are a chatbot SPECIALIZED ONLY in love, relationships, heartbreak, "
                        "trust, emotions, and dating. If asked anything else, reply: "
                        "'Samahani, naweza kujibu maswali ya mapenzi tu.' "
                        "Always reply in the same language the user uses."
                    )
                }
            ]

        if "history" not in session:
            session["history"] = []

        # Save user message
        session["messages"].append({"role": "user", "content": user_message})
        session["history"].append({"sender": "user", "text": user_message})
        session.modified = True

        # --- OpenAI Completion ---
        completion = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=session["messages"],
            max_tokens=250,
            temperature=0.8
        )

        bot_reply = safe_extract_reply(completion)

        # Save reply
        session["messages"].append({"role": "assistant", "content": bot_reply})
        session["history"].append({"sender": "bot", "text": bot_reply})
        session.modified = True

        # Log conversation
        log_message(user_message, bot_reply)

        return jsonify({"reply": bot_reply})

    except Exception as e:
        logging.exception("Error in /api/chat")
        return jsonify({"error": "Server error: connection to model failed"}), 500

# ----------------------------
# Clear Session
# ----------------------------
@app.route("/api/clear", methods=["POST"])
def clear_chat():
    session.pop("messages", None)
    session.pop("history", None)
    session.modified = True
    return jsonify({"status": "cleared"})

# ----------------------------
# History
# ----------------------------
@app.route("/api/history", methods=["GET"])
def history():
    return jsonify(session.get("history", []))

# ----------------------------
# Run
# ----------------------------
if __name__ == "__main__":
    logging.info(f"Starting Flask server on 0.0.0.0:{PORT}")
    app.run(host="0.0.0.0", port=PORT, debug=False)
