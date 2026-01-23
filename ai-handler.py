from flask import Flask, request, jsonify
from flask_cors import CORS
import anthropic
import os
from typing import Optional

app = Flask(__name__)
CORS(app)

# Store data summaries in memory
data_store = {}

def summarize_data(data: str, api_key: str) -> str:
    """Summarize input data using Claude"""
    client = anthropic.Anthropic(api_key=api_key)
    
    message = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=1000,
        messages=[{
            "role": "user",
            "content": f"Summarize the following data concisely, highlighting key points and structure:\n\n{data}"
        }]
    )
    
    return message.content[0].text

def answer_question(question: str, context: str, api_key: str) -> str:
    """Answer question based on data context"""
    client = anthropic.Anthropic(api_key=api_key)
    
    message = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=1000,
        messages=[{
            "role": "user",
            "content": f"Based on this data:\n\n{context}\n\nAnswer this question: {question}"
        }]
    )
    
    return message.content[0].text

@app.route('/upload', methods=['POST'])
def upload_data():
    """Upload and summarize data"""
    try:
        data = request.json.get('data')
        api_key = request.json.get('api_key')
        session_id = request.json.get('session_id', 'default')
        
        if not data or not api_key:
            return jsonify({'error': 'Missing data or api_key'}), 400
        
        # Summarize the data
        summary = summarize_data(data, api_key)
        
        # Store both original data and summary
        data_store[session_id] = {
            'original': data,
            'summary': summary
        }
        
        return jsonify({
            'success': True,
            'summary': summary,
            'session_id': session_id
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/ask', methods=['POST'])
def ask_question():
    """Ask a question about uploaded data"""
    try:
        question = request.json.get('question')
        api_key = request.json.get('api_key')
        session_id = request.json.get('session_id', 'default')
        
        if not question or not api_key:
            return jsonify({'error': 'Missing question or api_key'}), 400
        
        if session_id not in data_store:
            return jsonify({'error': 'No data uploaded for this session'}), 400
        
        # Use original data for context
        context = data_store[session_id]['original']
        answer = answer_question(question, context, api_key)
        
        return jsonify({
            'success': True,
            'answer': answer
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/sessions', methods=['GET'])
def list_sessions():
    """List all active sessions"""
    return jsonify({
        'sessions': list(data_store.keys())
    })

@app.route('/clear/<session_id>', methods=['DELETE'])
def clear_session(session_id):
    """Clear a specific session"""
    if session_id in data_store:
        del data_store[session_id]
        return jsonify({'success': True})
    return jsonify({'error': 'Session not found'}), 404

if __name__ == '__main__':
    app.run(debug=True, port=5000)
