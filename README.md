# text2anki
Intelligent Flashcard Generator: This script intelligently extracts key linguistic elements from text and uses an LLM to create comprehensive Anki flashcards for efficient language learning.


# Anki Deck Generator for Language Learning

This project leverages Natural Language Processing (NLP) and Large Language Models (LLMs) to automatically generate Anki flashcards from text input, aiding in language acquisition. It's designed to streamline the process of creating personalized Anki decks, focusing on vocabulary, grammar, and usage examples extracted from user-provided texts.

## Features

*   **Text Processing:** Utilizes spaCy for tokenization, part-of-speech tagging, and dependency parsing to analyze input text and identify key linguistic elements.
*   **LLM Integration:** Employs LLMs (such as OpenAI's models or Hugging Face's transformers) to generate explanations, definitions, and example sentences for selected words and phrases.
*   **Anki Deck Creation:** Uses genanki to programmatically create Anki decks and notes, with customizable models for different card types (e.g., nouns, verbs).
*   **Database Storage (Optional):** Optionally stores processed sentences, focus words, and LLM responses in an SQLite database for tracking and analysis.
*   **Prompt Engineering:** Includes a flexible prompt generation system to tailor LLM queries for specific grammatical or contextual information.

## Requirements

*   Python 3.7+
*   spaCy (`pip install spacy`)
*   genanki (`pip install genanki`)
*   OpenAI Python library (if using OpenAI's API) (`pip install openai`) or Transformers (`pip install transformers`)
*   spaCy language model (e.g., `python -m spacy download es_core_news_sm` for Spanish)

## Installation

1.  Clone this repository:
    ```bash
    git clone <repository_url>
    cd <repository_directory>
    ```
2.  Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```
3.  Download the necessary spaCy language model:
    ```bash
    python -m spacy download es_core_news_sm # Or your desired language model
    ```

## Usage

1.  **Prepare your input text:** Place the text you want to process in a `.txt` file (e.g., `sample_text.txt`).
2.  **Set up your LLM API key:** If using OpenAI, set the `OPENAI_API_KEY` environment variable.
3.  **Run the script:**
    ```bash
    python main.py
    ```
    This will generate an Anki deck file (`.apkg`) in the project directory.

## Configuration

*   **`NLP_MODEL`:** Specify the spaCy language model to use (e.g., `"en_core_web_sm"` for English).
*   **`DB_PATH`:** Set the path for the SQLite database (if used).
*   **Anki Model and Deck IDs:** Modify the `MyBaseModel` class and `anki_deck` object in `main.py` to customize card formats and deck information.

## Potential Extensions

*   **Constituency Parsing:** Integrate libraries like Stanza for more detailed syntactic analysis.
*   **TTS and Image Generation:** Add text-to-speech and image generation capabilities to enhance cards.
*   **Advanced Prompt Engineering:** Refine LLM prompts for more nuanced grammatical insights.
*   **Frequency Lists and User Feedback:** Incorporate word frequency analysis and user feedback mechanisms.
*   **Error Handling and Rate Limiting:** Implement robust error handling and API rate limiting.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details. (You should create a LICENSE file)

## Acknowledgements

*   spaCy
*   genanki
*   OpenAI / Hugging Face
