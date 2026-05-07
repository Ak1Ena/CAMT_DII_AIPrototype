from flask import Flask,  flash, redirect, request, render_template, make_response, url_for
import json
import sys 
#import pandas as pd

app = Flask(__name__)

@app.route("/") 
def helloworld():
    return "Hello, World!"

@app.route("/name") 
def name():
    return "Siraphop Khatchamat ID672110163"


@app.route('/receive_get',methods=['GET']) 
def web_service_API_GET():

    msg = request.args.get('msg')
    name = request.args.get('name')
    
    print(f'the input message from GET is {msg} from {name}.')
    
    return f'{msg} from {name} received!'


@app.route('/request_POST',methods=['POST']) 
def web_service_API_POST():
        payload = request.data.decode("utf-8")
        inmessage = json.loads(payload) # ทำ json
        print(inmessage)
        json_data = json.dumps({'y': 'POST received!'}) # ส่งกลับไปว่าได้รับเเล้ววว
        return json_data
    

@app.route("/resume")
def resume():
    return """
    <!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Resume - [Your Full Name]</title>
    <style>
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            line-height: 1.6;
            color: #333;
            max-width: 800px;
            margin: 0 auto;
            padding: 40px 20px;
            background-color: #fff;
        }
        header {
            text-align: center;
            margin-bottom: 30px;
        }
        h1 {
            margin: 0 0 10px 0;
            font-size: 2.5em;
            color: #2c3e50;
        }
        .contact-info {
            font-size: 0.95em;
            color: #555;
        }
        .contact-info a {
            color: #2980b9;
            text-decoration: none;
        }
        .contact-info a:hover {
            text-decoration: underline;
        }
        section {
            margin-bottom: 25px;
        }
        h2 {
            font-size: 1.4em;
            color: #2c3e50;
            border-bottom: 2px solid #ecf0f1;
            padding-bottom: 5px;
            margin-bottom: 15px;
            text-transform: uppercase;
            letter-spacing: 1px;
        }
        .item-header {
            display: flex;
            justify-content: space-between;
            align-items: baseline;
            margin-bottom: 5px;
        }
        .item-title {
            font-weight: bold;
            font-size: 1.1em;
        }
        .item-subtitle {
            font-style: italic;
            color: #555;
        }
        .item-date {
            font-size: 0.9em;
            color: #666;
            text-align: right;
        }
        ul {
            margin-top: 0;
            padding-left: 20px;
        }
        li {
            margin-bottom: 8px;
        }
        .skills-category {
            font-weight: bold;
        }
    </style>
</head>
<body>

    <header>
        <h1>Siraphop Khatchamat</h1>
        <div class="contact-info">
            Chiang Mai, Thailand &nbsp;|&nbsp; 
            [Phone Number] &nbsp;|&nbsp; 
            <a href="mailto:your.email@example.com">[Email Address]</a> &nbsp;|&nbsp; 
            <a href="https://github.com/yourusername" target="_blank">GitHub</a> &nbsp;|&nbsp; 
            <a href="https://linkedin.com/in/yourusername" target="_blank">LinkedIn</a>
        </div>
    </header>

    <section>
        <h2>Professional Summary</h2>
        <p>[Optional: Detail-oriented software developer with hands-on experience in full-stack web development, workflow automation, and local AI deployment. Passionate about building efficient tools and optimizing systems in Arch Linux environments.]</p>
    </section>

    <section>
        <h2>Education</h2>
        <div class="item-header">
            <div>
                <span class="item-title">Chiang Mai University (CMU)</span> | Chiang Mai, Thailand
                <div class="item-subtitle">CAMT, CMU</div>
            </div>
        </div>
    </section>

    <section>
        <h2>Professional Experience</h2>
        <div class="item-header">
            <div>
                <span class="item-title">Appman</span> 
                <div class="item-subtitle">Software Developer Intern</div>
            </div>
            <div class="item-date">1 Dec 2025 – Present</div>
        </div>
    </section>

    <section>
        <h2>Technical Projects</h2>
        
        <div class="item-header">
            <span class="item-title">Workflow Automation & Logic Design</span>
        </div>
        <ul>
            <li>Designed and maintained complex automation workflows using <strong>n8n</strong>, streamlining data processing and internal tasks.</li>
            <li>Developed custom TypeScript clustering algorithms and integrated React-based web SDKs to enhance application functionality.</li>
        </ul>

        <div class="item-header">
            <span class="item-title">Local AI & Audio Deployment</span>
        </div>
        <ul>
            <li>Successfully deployed and ran local Large Language Models (LLMs) using <strong>Ollama</strong> on CPU-limited hardware.</li>
            <li>Integrated Text-to-Speech (TTS) models, specifically <strong>Kokoro-82M (ONNX)</strong>, with Python scripts to enable real-time voice cloning and interaction.</li>
        </ul>
    </section>

    <section>
        <h2>Technical Skills</h2>
        <ul>
            <li><span class="skills-category">Programming Languages:</span> TypeScript, JavaScript, Python</li>
            <li><span class="skills-category">Frontend & Backend:</span> React, Node.js</li>
            <li><span class="skills-category">Tools & Infrastructure:</span> n8n, Git / GitHub, Arch Linux (pacman/yay), Google Chrome dev tools</li>
            <li><span class="skills-category">AI & Machine Learning:</span> Local LLM deployment (Ollama), ONNX models, TTS generation</li>
        </ul>
    </section>

</body>
</html>
    """
if __name__ == "__main__":   # run code 
    app.run(host='0.0.0.0',debug=True,port=5002)#host='0.0.0.0' = run on internet ,port=5002 (port บน server เหมือนประตู) / localhost รันบนเครื่องเรายังไม่ใช่ internet

