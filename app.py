from flask import Flask, render_template, request

#We import the processor.py to access to chatbot_response function
import processor

app = Flask(__name__)
app.static_folder = 'static'

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/get")
def get_bot_response():
    userText = request.args.get('msg')
    #We call chatbot_response from processor
    return str(processor.chatbot_response(userText))

if __name__ == "__main__":
    app.run() 
    
    
    