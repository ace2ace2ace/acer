from flask import Flask, request, jsonify
import cohere
import os
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__, static_folder='static', static_url_path='')
co = cohere.Client(os.getenv('COHERE_API_KEY'))


@app.route('/')
def home():

    return app.send_static_file('index.html')


@app.route('/generate-story', methods=['POST'])
def generate_story():
    data = request.json
    format = data.get('format') 
    topic = data.get('topic')
    length = data.get('length')

    prompt = f"""Write a {length} story with the following elements:
    Format: {format}
    Topic: {topic}

    
At the end of the essay, please Explain how to write a good essay for high school students"""

    response = co.generate(
        model='command',
        prompt=prompt,
        max_tokens=2000,
        temperature=0.8,
        k=0,
        stop_sequences=[],
        return_likelihoods='NONE'
    )

    return jsonify({'story': response.generations[0].text})


if __name__ == '__main__':
    app.run(debug=True)
