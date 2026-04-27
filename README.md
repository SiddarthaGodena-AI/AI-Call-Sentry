🎙️ AI Call Analysis & Conversation Intelligence System
An end-to-end AI-powered call analysis platform built with FastAPI + AWS + Vertex AI, designed to:

🎧 Process audio recordings
📝 Generate speaker-wise transcripts
😊 Perform sentiment analysis
📊 Rate conversations
🤖 Generate summaries, Q&A, and chatbot responses using LLMs

🚀 Key Features
🎧 1. Audio Processing Pipeline
Noise reduction (noisereduce)
Audio normalization (pydub)
Silence-based segmentation
Supports multi-speaker conversations

📝 2. Speech-to-Text (AWS Transcribe)
Converts audio → text
Detects speaker labels (Speaker 1, Speaker 2)
Produces structured transcript

😊 3. Sentiment Analysis (AWS Comprehend)
Chunk-based sentiment detection
Speaker-level sentiment analysis
Aggregated sentiment scoring:
Positive
Negative
Neutral
Mixed

⭐ 4. Conversation Rating Engine
Automatically scores conversations:
⭐⭐⭐⭐⭐ → Positive
⭐⭐⭐ → Mixed
⭐ → Negative

📄 5. Transcript Management
Formats conversation into readable structure
Saves transcripts as .txt
Enables further AI analysis

🤖 6. Multi-LLM AI Analysis

Supports dual AI models:
🔹 AWS Bedrock (Jamba)
Evaluation summaries
Q&A generation
Chat responses

🔹 Google Vertex AI (Gemini)
Advanced conversation analysis
Chat with transcript
Context-aware responses

💬 7. Chatbot on Transcripts
Ask questions based on conversation
Context-aware answers
Works on stored transcripts

📊 8. AI Evaluation & Insights
Generate:
📄 Summary reports
❓ Q&A from conversations
💡 Insights for customer support

☁️ 9. Cloud Integration
AWS S3 → File storage
AWS Transcribe → Speech-to-text
AWS Comprehend → Sentiment
AWS Bedrock → LLM inference
Vertex AI (Gemini) → Advanced AI

📜 10. Logging & Monitoring
Rotating logs (RotatingFileHandler)
Scheduled upload to S3 (every 5 mins)
Tracks user activity

🛠️ Tech Stack
Backend: FastAPI
Audio Processing: pydub, librosa, noisereduce
Speech-to-Text: AWS Transcribe
NLP: AWS Comprehend
LLMs:
AWS Bedrock (Jamba)
Google Vertex AI (Gemini)
Cloud Storage: AWS S3
Frontend: Jinja2 Templates

📌 API Endpoints
🔹 Upload & Analyze Audio
POST /transcribe_and_analyze
Upload audio file
Returns:
Transcript
Conversation rating
Speaker sentiment

🔹 Generate Summary
POST /generate

🔹 Generate Q&A
POST /generate_qna

🔹 Chat with Transcript
POST /chat_with_transcript

🔹 Chatbot UI
GET /chatbot

⚙️ Setup Instructions
1️⃣ Clone Repository
git clone https://github.com/your-username/ai-call-analysis.git
cd ai-call-analysis

2️⃣ Install Dependencies
pip install -r requirements.txt

3️⃣ Setup AWS Credentials
export AWS_ACCESS_KEY_ID=your_key
export AWS_SECRET_ACCESS_KEY=your_secret
export AWS_DEFAULT_REGION=us-east-1

4️⃣ Setup Google Vertex AI
export GOOGLE_APPLICATION_CREDENTIALS="path/to/service-account.json"

5️⃣ Run Server
uvicorn app:app --reload

6️⃣ Open UI
http://localhost:8000

🧪 How It Works
Upload call audio
Audio preprocessing:
Noise reduction
Segmentation
Transcription via AWS
Sentiment analysis
Transcript formatting & storage
AI models generate:
Summary
Q&A
Chat responses
