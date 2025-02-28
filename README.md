# text2anki: AI-Powered Anki Flashcard Generator for Language Learners

**text2anki** is a Python script that transforms any text into personalized Anki flashcards using Natural Language Processing (NLP) and AI. By leveraging spaCy for text analysis and a Large Language Model (LLM) for content generation, it creates detailed, customizable flashcards tailored to your language learning needs—making the process efficient and effective.

---

## ✨ Features

- **Multi-Language Support:** Works with any language supported by spaCy (e.g., Spanish, French, English), adapting to features like noun genders.
- **Smart Text Processing:** Uses spaCy for tokenization, part-of-speech tagging, and dependency parsing to identify key linguistic elements.
- **AI-Powered Insights:** Employs an LLM (e.g., OpenAI's GPT) to generate definitions, conjugations, and example sentences.
- **Custom Anki Decks:** Creates tailored flashcards with `genanki`, using specialized models for nouns and verbs.
- **Personalized Learning:** Excludes user-specified known words to focus on new vocabulary.
- **Data Tracking:** Stores flashcard metadata (e.g., word, definition, sentence) in an SQLite database for progress tracking.

---

## 📋 Requirements

- **Python 3.7+**
- **spaCy:** `pip install spacy`
- **genanki:** `pip install genanki`
- **OpenAI:** `pip install openai` (optional, for LLM functionality)
- **SQLite3:** Included in Python standard library
- **spaCy Language Model:** e.g., `python -m spacy download es_core_news_sm` for Spanish

---

## 🚀 Installation

1. **Clone the repository:**
   ```bash
   git clone <https://github.com/Bashpoet/text2anki/>
   cd <repository_directory>
   ```

2. **Install dependencies:**
   ```bash
   pip install spacy genanki openai
   ```

3. **Download a spaCy language model:**
   ```bash
   python -m spacy download es_core_news_sm  # For Spanish
   ```
   *Note: Replace `es_core_news_sm` with the model for your language (e.g., `en_core_web_sm` for English).*

4. **Set your OpenAI API key (optional):**
   ```bash
   export OPENAI_API_KEY='your-api-key-here'
   ```
   *Note: Required for LLM functionality.*

---

## 🎓 Usage

### 1. Prepare Your Text
Save the text you want to learn from in a `.txt` file (e.g., `sample_text.txt`).

### 2. (Optional) List Known Words
Create a `.txt` file with one word per line (e.g., `known_words.txt`) to exclude familiar vocabulary.

### 3. Run the Script
```bash
python main.py --lang <spacy_model> --input <input_file> --output <output_file> --known-words <known_words_file>
```

**Arguments:**
- `--lang`: spaCy model (e.g., `es_core_news_sm` for Spanish)
- `--input`: Path to input text file
- `--output`: Output Anki deck name (default: `Language_Learning.apkg`)
- `--known-words`: Path to known words file (optional)

### 4. Import into Anki
Open the generated `.apkg` file in Anki to start learning!

### Examples

**Spanish:**
```bash
python main.py --lang es_core_news_sm --input sample_text.txt --output Spanish_Deck.apkg --known-words known_words.txt
```

**English:**
```bash
python main.py --lang en_core_web_sm --input english_text.txt --output English_Deck.apkg
```

---

## ⚙️ Configuration

- **Language-Specific Fields:** Includes fields like "Gender" for languages with gendered nouns (e.g., Spanish) and omits them for others (e.g., English).
- **Exclude Known Words:** Use `--known-words` to skip familiar terms.
- **Custom Anki Models:** Uses specialized models for nouns and verbs with relevant fields.

---

## 🔮 Potential Extensions

- Support for additional languages (e.g., Italian, Chinese).
- Enhanced LLM prompts for grammar insights.
- Text-to-speech for pronunciation.
- Image generation for visual aids.
- Word frequency analysis for prioritization.

---

## 📜 License

This project is licensed under the MIT License. 

---

## 🙏 Acknowledgements

- [spaCy](https://spacy.io/) for NLP capabilities.
- [genanki](https://github.com/kerrickstaley/genanki) for Anki deck generation.
- [OpenAI](https://openai.com/) for LLM functionality.

---
