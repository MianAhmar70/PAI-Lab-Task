from flask import Flask, request, jsonify, render_template
from chatbot import HealthChatbotModel
import os

app = Flask(__name__)

chatbot = HealthChatbotModel()

# Load CSV data at startup
try:
    chatbot.load_data('data/combined_health_qa.csv')
except Exception as e:
    print(f"[ERROR] Failed to load chatbot data: {e}")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    data = request.get_json()
    user_input = data.get('message')
    if not user_input:
        return jsonify({'response': "Please enter a valid query."})
    response = chatbot.get_answer(user_input)
    return jsonify({'response': response})

if __name__ == '__main__':
    print("[INFO] Starting Flask server...")
    app.run(debug=True, use_reloader=False)
    app.run(debug=True)

