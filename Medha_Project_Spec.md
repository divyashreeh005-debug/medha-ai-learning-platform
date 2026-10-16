# Medhā — AI-Powered Video-to-Interactive Learning Platform

## 1. Project Overview

### Name

**Medhā (मेधा)** refers to intellect, intelligence, and understanding. The name reflects the project's goal of helping students turn educational content into clearer, structured understanding.


**Medhā** is an AI-powered learning platform that converts educational video content into an interactive learning experience.

A student can either:

1. Upload a video file, or
2. Paste a YouTube video URL.

The system obtains the video's transcript, processes the educational content using NLP, Deep Learning and Generative AI, and transforms it into:

- Smart Summary Cards
- Topic-wise explanations
- Simple explanations for difficult concepts
- Real-world analogies
- Key takeaways
- Concept relationships / concept map
- Timestamp-based topic navigation
- Context-aware AI chatbot for student doubts

### Core idea

> **Instead of making students passively watch a long educational video, Medhā converts the video into an interactive knowledge and learning environment.**

---

# 2. Problem Statement

Educational videos are widely used for learning, but students often face difficulties when learning from long-form video content.

A student may need to:

- Watch an entire video to find one specific concept.
- Rewind repeatedly when a concept is difficult.
- Create notes manually.
- Search elsewhere for a simpler explanation.
- Identify the important concepts from a long lecture.
- Connect related concepts presented at different points in the video.
- Remember the key points after completing the lecture.
- Ask questions and obtain answers specifically based on the lecture.

Conventional video platforms mainly provide playback, captions/transcripts and basic search. They do not automatically transform the educational content into a structured, personalized learning representation.

### Proposed problem

Develop an AI-based system that can understand the textual content of an educational video and automatically transform it into structured learning resources and an interactive question-answering environment.

---

# 3. Proposed Solution

Medhā accepts either a video file or a YouTube URL.

### Input

```text
Video File
     OR
YouTube URL
```

### Processing

```text
Transcript Acquisition
        ↓
Text Cleaning
        ↓
Semantic Chunking
        ↓
Topic / Concept Extraction
        ↓
Embeddings
        ↓
Vector Database
        ↓
LLM + RAG
```

### Output

```text
Smart Summary Cards
Concept Map
Simple Explanations
Real-World Analogies
Key Takeaways
Timestamp-based Search
AI Doubt Chatbot
```

The chatbot uses retrieval from the processed lecture rather than relying only on the general knowledge of an LLM.

---

# 4. Main Objectives

1. Convert educational videos into structured learning material.
2. Support both YouTube URLs and locally uploaded videos.
3. Automatically identify major topics and concepts.
4. Generate concise, topic-wise summary cards.
5. Provide simplified explanations of difficult concepts.
6. Generate real-world analogies to improve understanding.
7. Build relationships between concepts.
8. Allow students to ask questions about the video.
9. Retrieve relevant portions of the lecture for chatbot responses.
10. Provide timestamps so students can return to the original explanation.
11. Reduce the time required to revise long educational videos.
12. Provide a single interactive interface for learning from video content.

---

# 5. Target Users

### Primary users

- College students
- MCA / BCA / B.Tech students
- Online learners
- Students using YouTube educational content

### Secondary users

- Teachers
- Trainers
- Online course creators

---

# 6. Key Features

## 6.1 Dual Input

### Option A — YouTube URL

```text
Paste YouTube URL
        ↓
Retrieve available transcript
        ↓
Process transcript
```

### Option B — Video Upload

```text
Upload MP4/MKV/etc.
        ↓
Extract audio
        ↓
Speech-to-text model
        ↓
Transcript
```

The rest of the processing pipeline is common to both inputs.

---

# 7. Smart Summary Cards

Instead of generating one large summary, Medhā divides the lecture into meaningful concepts.

Example:

### Topic: Inheritance

**What it means**

A class can reuse properties and methods from another class.

**Example**

```text
Vehicle
   ↓
  Car
```

**Key idea**

Inheritance promotes reuse of existing functionality.

**Video section**

`12:40 – 15:20`

This allows students to understand the topic without reading a long generated paragraph.

---

# 8. Simple Explanation

For difficult concepts, the system can generate a beginner-friendly explanation.

Example:

### Technical explanation

> Polymorphism allows objects of different types to respond to the same interface in different ways.

### Simple explanation

> Different objects can respond to the same instruction in their own way.

This provides multiple levels of understanding without requiring the student to search elsewhere.

---

# 9. Real-World Analogy

Students can request an analogy for a difficult concept.

Example:

### Concept: Inheritance

**Analogy:**

A child can inherit characteristics from their parents. Similarly, a child class can inherit properties and methods from a parent class.

The analogy is generated from the identified concept and its context.

---

# 10. Key Takeaways

For every topic, Medhā can generate a short revision section.

Example:

### Inheritance — Remember

- A child class can reuse parent functionality.
- It reduces repeated code.
- It represents a parent-child relationship.
- The child can also add its own functionality.

This makes the system useful for quick revision.

---

# 11. Concept Map

Medhā can identify relationships between concepts.

Example:

```text
                    OOP
                     |
       +-------------+-------------+
       |             |             |
     Class         Object      Inheritance
                                   |
                                   ↓
                             Polymorphism
```

The concept map helps students understand how different topics are connected.

A student can click a concept to view:

- Its explanation
- Related concepts
- Key takeaways
- Relevant transcript
- Video timestamp

---

# 12. AI Doubt Chatbot

The chatbot is a major interaction feature, but it is not simply a generic chatbot.

The chatbot is grounded in the processed lecture.

### Example

Student:

> "Why is inheritance useful?"

Medhā:

> "In the lecture, inheritance is introduced as a way to reuse functionality from an existing class. It helps reduce repeated code and establishes a parent-child relationship."

The response can include:

```text
Relevant section: 12:40
[Jump to video]
```

### RAG-based approach

```text
Student Question
       ↓
Question Embedding
       ↓
Vector Search
       ↓
Relevant Transcript Chunks
       ↓
LLM
       ↓
Context-grounded Answer
```

This reduces the chance of the chatbot answering unrelated to the uploaded lecture.

---

# 13. Timestamp-Based Learning

Each transcript chunk retains its timestamp.

Example:

```text
00:00 – Introduction
03:25 – Classes
07:40 – Objects
12:40 – Inheritance
18:15 – Polymorphism
```

If a student searches:

> "Where is inheritance explained?"

Medhā can return:

> **12:40 – 18:00**

and provide a link/button to return to that point in the video.

---

# 14. Methodology

## Step 1 — Input Acquisition

The user selects one of two options:

```text
             INPUT
               |
       +-------+-------+
       |               |
 YouTube URL       Video File
```

### YouTube path

The system retrieves the available transcript.

### Local video path

Audio is extracted from the video and passed through a speech-to-text model such as Whisper.

---

## Step 2 — Transcript Processing

Raw transcript may contain:

- Repeated words
- Unnecessary filler words
- Broken sentences
- Timestamp fragments
- Recognition errors

The preprocessing stage cleans and normalizes the transcript.

```text
Raw Transcript
      ↓
Cleaning
      ↓
Sentence Segmentation
      ↓
Timestamp Preservation
```

---

## Step 3 — Semantic Chunking

The transcript is divided into meaningful chunks.

Instead of blindly splitting every N characters, the system attempts to preserve complete concepts.

Example:

```text
Transcript
   ↓
Introduction
   ↓
Classes
   ↓
Objects
   ↓
Inheritance
   ↓
Polymorphism
```

Each chunk stores:

```text
Chunk ID
Topic
Text
Start Timestamp
End Timestamp
```

---

## Step 4 — Concept and Topic Extraction

The system identifies important concepts using NLP/LLM-based processing.

Example:

```text
Transcript
    ↓
Concept Extraction
    ↓
Class
Object
Inheritance
Polymorphism
Encapsulation
```

Relationships between concepts are also identified.

Example:

```text
Class → Object
Inheritance → Polymorphism
```

---

## Step 5 — Embedding Generation

Each transcript chunk is converted into a numerical vector representation using a sentence-transformer embedding model.

```text
Text Chunk
    ↓
Embedding Model
    ↓
Vector
```

These vectors represent the semantic meaning of the text.

---

## Step 6 — Vector Storage

Embeddings and metadata are stored in a vector database such as FAISS or Qdrant.

Example:

```text
Vector
+
Transcript
+
Topic
+
Timestamp
```

This allows semantic retrieval.

A query such as:

> "Where does the teacher explain reuse of code?"

can retrieve the chunk discussing inheritance even if the exact phrase "reuse of code" does not appear in the transcript.

---

# 15. Generative AI Layer

The LLM uses the retrieved context to generate:

- Summary cards
- Simple explanations
- Analogies
- Key takeaways
- Chatbot answers
- Concept descriptions

### Example pipeline

```text
Retrieved Context
       +
Student Request
       ↓
      LLM
       ↓
Structured Response
```

Structured output can be requested in JSON format so that the frontend can reliably render cards.

---

# 16. RAG Architecture

Medhā uses Retrieval-Augmented Generation for lecture-based questions.

```text
                 Student
                   |
                Question
                   |
                   ↓
             Query Embedding
                   |
                   ↓
             Vector Search
                   |
             Top-K Chunks
                   |
                   ↓
        +----------------------+
        | Transcript Context   |
        +----------------------+
                   |
                   ↓
                  LLM
                   |
                   ↓
          Grounded Answer
                   |
                   ↓
          Relevant Timestamp
```

---

# 17. System Architecture

```text
                         MEDHĀ
                            |
             +--------------+--------------+
             |                             |
       YouTube URL                    Video Upload
             |                             |
             ↓                             ↓
   Transcript Retrieval              Audio Extraction
             |                             |
             |                          Whisper
             |                             |
             +-------------+---------------+
                           |
                           ↓
                    Transcript Store
                           |
                           ↓
                  Text Preprocessing
                           |
                           ↓
                   Semantic Chunking
                           |
              +------------+------------+
              |                         |
              ↓                         ↓
       Topic / Concept             Embeddings
        Extraction                    |
              |                       ↓
              |                  Vector DB
              |                       |
              +-----------+-----------+
                          |
                          ↓
                     RAG Engine
                          |
                          ↓
                         LLM
                          |
          +---------------+----------------+
          |               |                |
          ↓               ↓                ↓
   Summary Cards    Concept Map       AI Chatbot
          |               |                |
          ↓               ↓                ↓
 Simple Explanation   Relationships   Student Doubts
 Analogies            Key Points      Timestamp
```

---

# 18. Technology Stack

## Frontend

- React / Next.js
- Tailwind CSS
- JavaScript / TypeScript

## Backend

- Python
- FastAPI

## Speech Processing

- OpenAI Whisper / Whisper-compatible model
- FFmpeg for audio extraction

## NLP / Deep Learning

- Hugging Face Transformers
- Sentence Transformers
- Transformer-based embedding model

## Generative AI

### Primary model

**Qwen3-8B-Instruct**

For local deployment on the project development machine, use a **4-bit GGUF quantized version of Qwen3-8B-Instruct**.

Possible local runtimes:

- LM Studio
- Ollama

The LLM is responsible for:

- Generating summary cards
- Simplifying concepts
- Generating real-world analogies
- Producing key takeaways
- Answering student doubts
- Generating structured output for the frontend

The model is used **after retrieval**, so it receives relevant transcript context rather than the entire lecture whenever possible.

### Why Qwen3-8B-Instruct?

The project requires an instruction-following model for summarization, explanation, structured generation and RAG-based question answering. An 8B instruction model provides a practical balance between capability and local deployment requirements for the POC.

**Note:** `Qwen3-8B-Base` should not be used as the primary application model; the POC should use the instruction/chat-oriented variant.

## RAG

- FAISS or Qdrant
- Sentence Transformers
- Custom retrieval pipeline or LangChain

## Database

- PostgreSQL / MongoDB

Use the normal database for:

- Users
- Videos
- Sessions
- Generated content
- Learning history

Use the vector database for:

- Transcript embeddings
- Semantic retrieval

## Video

- FFmpeg
- YouTube transcript retrieval
- HTML5 video player

## Deployment

Possible options:

- Local development
- Docker
- Render / Railway / similar cloud deployment
- Local server

---

# 19. Suggested Technology Choice for the POC

To keep the POC manageable:

```text
Frontend       → Streamlit
Backend        → Python / FastAPI
Speech         → Whisper
Embeddings     → all-MiniLM-L6-v2
Vector DB      → FAISS
GenAI          → Qwen3-8B-Instruct
Local Model    → Qwen3-8B-Instruct GGUF (4-bit)
Runtime        → LM Studio / Ollama
Video          → FFmpeg
```

Once the pipeline works, the interface can be migrated to:

```text
React / Next.js + FastAPI
```

---

# 20. Data Flow

```text
USER
 |
 |-- YouTube URL
 |        OR
 |-- Video Upload
 |
 ↓
TRANSCRIPT ACQUISITION
 |
 ↓
TEXT PREPROCESSING
 |
 ↓
SEMANTIC CHUNKING
 |
 ↓
TOPIC & CONCEPT EXTRACTION
 |
 +----------------------+
 |                      |
 ↓                      ↓
Embeddings           LLM Processing
 |                      |
 ↓                      |
Vector DB                |
 |                      |
 +----------+-----------+
            |
            ↓
        RAG ENGINE
            |
            ↓
     LEARNING CONTENT
            |
   +--------+--------+----------+
   |        |        |          |
 Summary  Simple   Analogy   Takeaways
 Cards   Explain
            |
            ↓
       AI CHATBOT
            |
            ↓
       Student Doubt
            |
            ↓
    Semantic Retrieval
            |
            ↓
      Context + LLM
            |
            ↓
       Answer + Timestamp
```

---

# 21. Example End-to-End Scenario

### Input

Student pastes:

```text
YouTube URL
"Python OOP Tutorial"
```

### Step 1

Medhā obtains the transcript.

### Step 2

It identifies:

```text
1. Classes
2. Objects
3. Inheritance
4. Polymorphism
5. Encapsulation
```

### Step 3

It generates:

```text
Summary Cards
    ↓
Classes
Objects
Inheritance
Polymorphism
Encapsulation
```

### Step 4

Student opens **Inheritance**.

The system shows:

```text
Inheritance

Simple explanation:
A child class can reuse functionality
from a parent class.

Analogy:
A child inherits characteristics from
their parents.

Key points:
• Reuse functionality
• Parent-child relationship
• Reduces duplication

Video:
12:40 – 18:00
```

### Step 5

Student asks:

> "I still don't understand inheritance."

The chatbot retrieves the relevant lecture section and explains it using the lecture context.

### Step 6

Student clicks:

> **Watch relevant section**

The video starts around the relevant timestamp.

---

# 22. What Makes Medhā Different

The project should **not** be presented as simply:

> "An AI video summarizer."

The differentiation is the transformation of a lecture into multiple interconnected learning representations.

```text
Traditional approach:

Video → Transcript → Summary


Medhā:

Video
  ↓
Transcript
  ↓
Concepts
  ↓
Relationships
  ↓
Summary Cards
  ↓
Simple Explanation
  ↓
Real-world Analogy
  ↓
Key Takeaways
  ↓
Context-aware AI Tutor
  ↓
Relevant Video Timestamp
```

### Main innovation

> **Medhā converts passive video content into an interactive, structured learning environment rather than producing only a conventional summary.**

---

# 23. Deep Learning + Generative AI Components

## Deep Learning

The project can use transformer/deep-learning models for:

- Speech recognition through Whisper
- Semantic representation through sentence-transformer embeddings
- Natural language understanding

## Generative AI

The LLM generates:

- Summaries
- Explanations
- Analogies
- Key takeaways
- Concept descriptions
- Chatbot responses

## RAG

RAG connects the LLM to the actual lecture content.

This allows the chatbot to answer questions based on the selected video rather than treating the conversation as a completely generic question-answering task.

---

# 24. Selected AI Model Architecture

The POC uses separate models/components for separate responsibilities instead of asking one LLM to perform the entire pipeline.

```text
Educational Video / YouTube
            ↓
          Whisper
            ↓
        Transcript
            ↓
   Sentence Transformer
     (all-MiniLM-L6-v2)
            ↓
          FAISS
            ↓
    Relevant Transcript
          Chunks
            ↓
     Qwen3-8B-Instruct
     (4-bit GGUF locally)
            ↓
   ┌────────┼───────────┐
   ↓        ↓           ↓
Summary  Explanation  Chatbot
Cards    + Analogy    Answers
   ↓        ↓           ↓
         Key Takeaways
```

### Component responsibilities

| Component | Responsibility |
|---|---|
| Whisper | Speech-to-text for uploaded videos |
| YouTube transcript retrieval | Obtain transcript when a YouTube URL is provided |
| all-MiniLM-L6-v2 | Convert transcript chunks and queries into embeddings |
| FAISS | Semantic retrieval of relevant transcript chunks |
| Qwen3-8B-Instruct | Generate explanations, summaries, analogies, takeaways and chatbot responses |
| LM Studio / Ollama | Local inference runtime for Qwen |
| FastAPI | Backend API layer |
| React/Next.js or Streamlit | User interface |

### Important POC design principle

The LLM should not be treated as the database or retrieval mechanism.

```text
Question
   ↓
Embedding
   ↓
FAISS retrieval
   ↓
Relevant lecture context
   ↓
Qwen3-8B-Instruct
   ↓
Answer
```

This makes the chatbot **lecture-grounded** and gives the project a clear RAG architecture.

---

# 31. POC Scope

The first working POC should implement:

### Must-have

- [x] YouTube URL input
- [x] Transcript acquisition
- [x] Transcript cleaning
- [x] Semantic chunking
- [x] Embeddings
- [x] FAISS vector search
- [x] AI summary cards
- [x] Simple explanations
- [x] Real-world analogies
- [x] Key takeaways
- [x] RAG chatbot
- [x] Relevant timestamp retrieval

### Second stage

- [ ] Local video upload
- [ ] Whisper transcription
- [ ] Interactive concept map
- [ ] User accounts
- [ ] Saved learning sessions
- [ ] Learning history
- [ ] Better video player integration

---

# 25. Development Plan

## Phase 1 — Core Pipeline

```text
YouTube URL
     ↓
Transcript
     ↓
Clean Transcript
     ↓
Chunking
```

## Phase 2 — AI Processing

```text
Chunks
  ↓
Embeddings
  ↓
FAISS
  ↓
RAG
```

## Phase 3 — Learning Content

Implement:

- Summary cards
- Simple explanations
- Analogies
- Key takeaways

## Phase 4 — Chatbot

Implement:

```text
Question
 ↓
Retrieve relevant chunks
 ↓
LLM
 ↓
Answer
```

## Phase 5 — Timestamp Navigation

Connect retrieved chunks to their original timestamps.

## Phase 6 — Video Upload

Add:

```text
Video
 ↓
FFmpeg
 ↓
Whisper
 ↓
Transcript
```

## Phase 7 — UI

Build a clean dashboard:

```text
┌──────────────────────────────────────┐
│              MEDHĀ                 │
├──────────────────────────────────────┤
│ YouTube URL:                         │
│ [________________________] [Process] │
│                                      │
│ Topics                               │
│ ┌─────────┐ ┌────────────┐           │
│ │ Classes │ │ Inheritance│           │
│ └─────────┘ └────────────┘           │
│                                      │
│ Summary Cards                        │
│                                      │
│ [Ask Medhā...]                     │
└──────────────────────────────────────┘
```

---

# 26. Expected Outcome

Medhā should allow a student to take a long educational video and quickly obtain:

- A structured understanding of the lecture
- Important concepts
- Short summary cards
- Beginner-friendly explanations
- Real-world analogies
- Key revision points
- Relationships between concepts
- Answers to doubts
- Relevant locations in the original video

The system therefore acts as an **AI learning companion built around educational video content**.

---

# 27. Future Enhancements

Possible future work:

1. Multimodal understanding of video slides and diagrams.
2. Automatic extraction of code shown on screen.
3. Personalized learning paths.
4. Difficulty-adaptive explanations.
5. Voice-based AI tutor.
6. Support for multiple languages.
7. Offline/local LLM deployment.
8. Knowledge graph generation.
9. Teacher analytics dashboard.
10. Integration with learning management systems.
11. Automatic lecture comparison.
12. Cross-video knowledge search.

---

# 28. Proposed Project Title

### Primary title

**Medhā: AI-Powered Video-to-Interactive Learning Platform**

### More academic title

**Medhā: A Generative AI and Retrieval-Augmented Framework for Transforming Educational Videos into Interactive Learning Resources**

### Short title

**Medhā — AI Video Learning Companion**

---

# 29. One-Line Project Definition

> **Medhā transforms educational videos into structured, searchable and interactive learning experiences using Deep Learning, NLP, RAG and Generative AI.**

---

# 30. Project POC Success Criteria

The POC is considered successful if a user can:

```text
1. Paste a YouTube URL
          ↓
2. Obtain the transcript
          ↓
3. Process the lecture
          ↓
4. View topic-wise summary cards
          ↓
5. Open simple explanations
          ↓
6. Generate real-world analogies
          ↓
7. View key takeaways
          ↓
8. Ask a question
          ↓
9. Receive an answer grounded in the lecture
          ↓
10. Jump to the relevant video timestamp
```

This is the **minimum end-to-end demonstration** of Medhā.
