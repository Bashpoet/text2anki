# text2anki: AI-Powered Anki Flashcard Generator for Language Learners

**text2anki** is a Python script that transforms any text into personalized Anki flashcards using Natural Language Processing (NLP) and AI. By leveraging spaCy for text analysis and a Large Language Model (LLM) for content generation, it creates detailed, customizable flashcards tailored to your language learning needs—making the process efficient and effective.

## 📚 Available Versions

This repository contains multiple versions of the text2anki script:

1. **`main.py`** - The core implementation with support for multiple languages, domain detection, and frequency filtering
2. **`main_tts_update.py`** - Adds text-to-speech functionality to generate audio for cards
3. **`text2anki_german_tts.py`** - German-focused version with TTS, compound noun handling and specialized German grammar support

## ✨ Features

- **Multi-Language Support:** Works with any language supported by spaCy, adapting to features like noun genders
- **Smart Text Processing:** Uses spaCy for tokenization, part-of-speech tagging, and dependency parsing
- **AI-Powered Insights:** Employs LLMs to generate definitions, conjugations, and example sentences
- **Custom Anki Decks:** Creates tailored flashcards with specialized models for nouns and verbs
- **Personalized Learning:** Excludes user-specified known words to focus on new vocabulary
- **Data Tracking:** Stores flashcard metadata in an SQLite database for progress tracking
- **Progress Resumption:** Can resume interrupted processing jobs
- **Domain Detection:** Identifies text domains (e.g., weather, travel) for contextually aware definitions
- **Frequency Filtering:** Prioritizes common words based on frequency data
- **Audio Pronunciation:** Generates audio files for words, sentences, and examples (in TTS versions)
- **German Language Support:** Special handling for German compound nouns and verb detection

## 🔍 Language-Specific Features

### German (text2anki_german_tts.py)
- Automatic recognition of German compound nouns
- Special handling for German verbs often incorrectly tagged as nouns
- German-specific domain terminology
- German text-to-speech pronunciation

### Spanish (main.py)
- Detection of Spanish verb forms with complex conjugations
- Spanish domain-specific compound expressions
- Gender recognition for Spanish nouns

## 📋 Requirements

- **Python 3.7+**
- **spaCy:** `pip install spacy`
- **genanki:** `pip install genanki`
- **OpenAI:** `pip install openai`
- **gTTS:** `pip install gtts` (for TTS versions)
- **SQLite3:** Included in Python standard library
- **spaCy Language Models:**
  - German: `python -m spacy download de_core_news_sm`
  - Spanish: `python -m spacy download es_core_news_sm`
  - English: `python -m spacy download en_core_web_sm`

## 🚀 Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Bashpoet/text2anki/
   cd text2anki
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Download spaCy language models:**
   ```bash
   # For German
   python -m spacy download de_core_news_sm
   
   # For Spanish
   python -m spacy download es_core_news_sm
   
   # For English
   python -m spacy download en_core_web_sm
   ```

4. **Set your OpenAI API key:**
   ```bash
   export OPENAI_API_KEY='your-api-key-here'
   ```

## 🎓 Usage

### Basic Usage
```bash
python main.py --lang es_core_news_sm --input sample_text.txt --output Spanish_Deck.apkg
```

### With Audio Support (TTS version)
```bash
python text2anki_german_tts.py --lang de_core_news_sm --input german_text.txt --output German_Deck.apkg
```

### Full Options
```bash
python text2anki_german_tts.py \
  --lang de_core_news_sm \
  --input german_text.txt \
  --output German_Deck.apkg \
  --known-words known_german.txt \
  --frequency-list german_freq.txt \
  --frequency-threshold 3000 \
  --deck-name "My German Learning" \
  --audio \
  --verbose
```

### Arguments
- `--lang`: spaCy language model to use
- `--input`: Path to input text file
- `--output`: Output Anki deck name
- `--known-words`: Path to file containing words to exclude
- `--frequency-list`: Path to file containing word frequencies
- `--frequency-threshold`: Only process words in the top N most frequent words
- `--deck-name`: Custom name for the Anki deck
- `--audio`: Generate audio for words and sentences (default in TTS version)
- `--no-audio`: Disable audio generation
- `--verbose`: Enable verbose logging
- `--debug`: Enable debug logging
- `--resume`: Resume from last run if interrupted

## 🔧 How It Works

1. **Text Parsing**: The script loads and parses text using spaCy's NLP pipeline
2. **Word Selection**: It identifies nouns, verbs, and other learnable elements
3. **Frequency Filtering**: Words are filtered based on frequency data
4. **LLM Processing**: The script sends prompts to an LLM to generate definitions, examples, etc.
5. **Audio Generation** (TTS versions): Text-to-speech creates audio for pronunciation
6. **Deck Creation**: An Anki deck is assembled with all the data and media
7. **Progress Tracking**: All processed words are stored in a database for future reference

## 📊 Technical Architecture

### Core Components
- **NLP Processing Pipeline**: spaCy-based processing with custom components
- **LLM Service**: Handles API communication with exponential backoff and caching
- **Database Manager**: SQLite storage for tracking processed items and enabling resumption
- **Card Models**: Specialized templates for different word types
- **Audio Manager** (TTS versions): Manages audio file generation and caching

### Parallel Processing
The script implements parallel processing with ThreadPoolExecutor to efficiently handle LLM API calls, significantly improving processing speed.

## 📜 License

This project is licensed under the MIT License.

## 🙏 Acknowledgements

- [spaCy](https://spacy.io/) for NLP capabilities
- [genanki](https://github.com/kerrickstaley/genanki) for Anki deck generation
- [OpenAI](https://openai.com/) for LLM functionality
- [gTTS](https://gtts.readthedocs.io/) for text-to-speech support
