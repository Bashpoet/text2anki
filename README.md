# text2anki

**text2anki:** This script intelligently extracts key linguistic elements from text and uses a Large Language Model (LLM) to create comprehensive Anki flashcards for efficient language learning.

---

## Overview

This project leverages Natural Language Processing (NLP) and Large Language Models (LLMs) to automatically generate Anki flashcards from text input, streamlining language acquisition. It focuses on vocabulary, grammar, and usage examples extracted from user-provided texts, supporting multiple languages and excluding known words for personalized learning.

---

## Features

- **Multi-Language Support:** Dynamically loads spaCy models and adjusts prompts based on the target language (e.g., includes gender for nouns in languages like Spanish or French).
- **Text Processing:** Uses spaCy for tokenization, part-of-speech tagging, and dependency parsing to identify key linguistic elements.
- **LLM Integration:** Generates definitions, conjugations, and example sentences using LLMs (e.g., OpenAI models).
- **Anki Deck Creation:** Creates customizable Anki decks with `genanki`, including tailored models for nouns and verbs.
- **Exclude Known Words:** Optionally skips user-specified known words to focus on new vocabulary.
- **Database Storage:** Saves metadata (e.g., sentence, word, definition) in an SQLite database for tracking.
- **Prompt Engineering:** Crafts structured, language-specific prompts for consistent LLM responses.
- **Parallel Processing:** Speeds up generation with parallel LLM API calls.

---

## Requirements

- **Python 3.7+**
- **spaCy**: `pip install spacy`
- **genanki**: `pip install genanki`
- **OpenAI**: `pip install openai`
- **SQLite3**: Included in Python standard library
- **spaCy Language Model**: Install for your target language (e.g., `python -m spacy download es_core_news_sm` for Spanish)

---
## Installation

1. **Clone the repository:**
   ```bash
   git clone <repository_url>
   cd <repository_directory>
   ```

2. **Install dependencies:**
   ```bash
   pip install spacy genanki openai
   ```

3. **Download a spaCy language model:**
   ```bash
   python -m spacy download es_core_news_sm  # Spanish
   python -m spacy download en_core_web_sm  # English
   ```

4. **Set your OpenAI API key:**
   ```bash
   export OPENAI_API_KEY='your-api-key-here'
   ```

## Usage

### Prepare input text
Save your text in a `.txt` file (e.g., `sample_text.txt`).

### (Optional) List known words
Create a `.txt` file with one word per line (e.g., `known_words.txt`) to exclude them.

### Run the script
```bash
python main.py --lang <spacy_model> --input <input_file> --output <output_file> --known-words <known_words_file>
```

**Arguments:**
- `--lang`: spaCy model (e.g., `es_core_news_sm`).
- `--input`: Path to input text file.
- `--output`: Output deck name (default: `Language_Learning.apkg`).
- `--known-words`: Path to known words file (optional).

### Examples

**Spanish:**
```bash
python main.py --lang es_core_news_sm --input sample_text.txt --output Spanish_Deck.apkg --known-words known_words.txt
```

**English:**
```bash
python main.py --lang en_core_web_sm --input english_text.txt --output English_Deck.apkg
```

### Import into Anki
Open the generated `.apkg` file in Anki.

## Configuration

- **Language Adjustments:** Automatically includes fields like "Gender" for gendered languages (e.g., Spanish) and omits them for others (e.g., English).
- **Known Words:** Use `--known-words` to skip familiar vocabulary.
- **Anki Models:** Custom models for nouns and verbs with relevant fields.

## Potential Extensions

- Add support for more languages (e.g., Italian, Chinese).
- Enhance prompts for advanced grammar insights.
- Integrate text-to-speech or image generation.
- Incorporate word frequency analysis or user feedback.

## License

Licensed under the MIT License. See the `LICENSE` file for details.

## Acknowledgements

- [spaCy](https://spacy.io/)
- [genanki](https://github.com/kerrickstaley/genanki)
- [OpenAI](https://openai.com/)
