from flask import Flask, request, render_template, session, jsonify
from conversation_manager import ConversationManager  # Import the ConversationManager class

app = Flask(__name__)
app.secret_key = 'your_secret_key'  # Required for using sessions

# Initialize the ConversationManager
conv_manager = ConversationManager()

@app.route("/", methods=["GET"])
def index():
    # Initialize the conversation history in the session
    if 'history' not in session:
        session['history'] = conv_manager.get_history()
    return render_template("index.html", history=session['history'])

@app.route("/get_response", methods=["POST"])
def get_response():
    user_input = request.form["user_input"]
    
    # Generate a response using the ConversationManager
    response_text = conv_manager.generate_response(user_input)
    
    # Update the session with the new conversation history
    session['history'] = conv_manager.get_history()
    session.modified = True
    
    return jsonify({"response": response_text, "history": session['history']})

if __name__ == "__main__":
    app.run(debug=True)