# Adaptive English Learning Assistant

An NLP based adaptive English learning system that analyzes learner writing, identifies grammar errors, estimates CEFR proficiency, and generates personalized grammar practice based on the learner's weaknesses.

## Overview

The Adaptive English Learning Assistant is designed to provide more personalized English grammar practice than a traditional fixed exercise system.

Instead of giving every learner the same exercises, the application analyzes writing samples, identifies recurring grammar problems, records learner performance, and recommends practice based on areas that need improvement.

The system uses a local language model through Ollama for language analysis and combines it with structured grammar exercises and learner performance tracking.

## Features

* English writing analysis
* Grammar error detection and correction
* CEFR level estimation from A1 to C2
* Personalized grammar practice
* Adaptive exercise difficulty
* Tracking of learner strengths and weaknesses
* Practice history and accuracy tracking
* Detection of repeated or similar practice questions
* Anonymous learner profiles
* Local learner data storage
* FastAPI based backend
* Local LLM integration through Ollama

## How It Works

The application follows a basic adaptive learning cycle:

1. The learner submits an English writing sample.
2. The system analyzes the text for grammar problems.
3. Grammar errors are categorized by area, such as articles, verb tense, prepositions, word order, or subject verb agreement.
4. The learner's performance profile is updated.
5. The system identifies weaker grammar areas.
6. Practice exercises are selected or generated for those areas.
7. The learner's answers are recorded.
8. Future exercise difficulty and recommendations are adjusted according to performance.

This allows practice to become increasingly personalized as the learner uses the system.

## Grammar Areas

The system currently supports grammar categories including:

* Past tense
* Present tense
* Future tense
* Present perfect
* Articles
* Prepositions
* Subject verb agreement
* Verb forms
* Adjectives
* Adverbs
* Pronouns
* Conjunctions
* Conditionals
* Modal verbs
* Sentence structure
* Word order
* Countable and uncountable nouns
* Plurals
* Gerunds and infinitives
* Comparatives and superlatives

## Technologies

* Python
* FastAPI
* Pydantic
* Ollama
* Large Language Models
* JSON based local learner storage

## Project Structure

```text
adaptive-english-learning-assistant/
│
├── main.py
├── learner_profile.py
├── practice_engine.py
├── requirements.txt
├── README.md
├── .gitignore
└── .env.example
```

### `main.py`

Contains the FastAPI application, API endpoints, learner sessions, writing analysis logic, and Ollama integration.

### `learner_profile.py`

Manages learner profiles, grammar error history, CEFR history, practice performance, weakness scores, and personalized recommendations.

### `practice_engine.py`

Contains structured grammar exercises, exercise selection logic, difficulty levels, and duplicate question detection.

## Installation

Clone the repository:

```bash
git clone https://github.com/YOUR-USERNAME/YOUR-REPOSITORY.git
cd YOUR-REPOSITORY
```

Create a virtual environment:

```bash
python -m venv .venv
```

Activate it.

On macOS or Linux:

```bash
source .venv/bin/activate
```

On Windows:

```bash
.venv\Scripts\activate
```

Install the dependencies:

```bash
pip install -r requirements.txt
```

## Ollama Setup

This project uses Ollama to run the language model locally.

Install Ollama and download the model used by the application:

```bash
ollama pull llama3.2:3b
```

The default model is:

```text
llama3.2:3b
```

A different Ollama model can be selected using the `OLLAMA_MODEL` environment variable.

Example:

```env
OLLAMA_MODEL=llama3.2:3b
COOKIE_SECURE=0
```

## Running the Application

Start Ollama if it is not already running.

Then run the FastAPI application:

```bash
uvicorn main:app --reload
```

The application will normally be available at:

```text
http://127.0.0.1:8000
```

## Learner Data

Learner profiles are stored locally in:

```text
data/learners/
```

These files contain locally generated learner information and are excluded from the Git repository through `.gitignore`.

## Privacy

The application uses anonymous learner identifiers rather than requiring users to create accounts.

Learner records are stored locally and are not intended to be committed to the public GitHub repository.

## Current Status

This project is an experimental educational NLP application and is under active development.

Future improvements may include better learner modeling, expanded grammar coverage, improved exercise generation, more detailed progress visualization, and further evaluation of the accuracy of language model based grammar analysis.

## Purpose

This project explores how Natural Language Processing and adaptive learning techniques can be combined to provide more individualized English language learning support.

In particular, it focuses on connecting automatic analysis of learner writing with personalized grammar practice rather than treating grammar correction and practice as separate tasks.

## Author

Chih-Han Hsu
