import os
import spacy
import genanki
import openai
import sqlite3
from typing import List, Dict, Any
import argparse
from concurrent.futures import ThreadPoolExecutor

# ---------------------------------------------
# I. SETUP & INITIALIZATION
# ---------------------------------------------

# 1. Parse command-line arguments
parser = argparse.ArgumentParser(description="Generate Anki flashcards from text using spaCy and LLM.")
parser.add_argument("--lang", required=True, help="spaCy language model (e.g., es_core_news_sm for Spanish)")
parser.add_argument("--input", required=True, help="Input text file")
parser.add_argument("--output", default="Language_Learning.apkg", help="Output Anki deck file name")
parser.add_argument("--known-words", help="File containing known words to exclude")
args = parser.parse_args()

# 2. Load spaCy model
try:
    nlp = spacy.load(args.lang)
except OSError:
    print(f"spaCy model '{args.lang}' not found. Please install it first.")
    exit(1)

# 3. Set up OpenAI API
openai.api_key = os.getenv("OPENAI_API_KEY")
if not openai.api_key:
    print("OpenAI API key not found. Set the OPENAI_API_KEY environment variable.")
    exit(1)

# 4. Initialize Database
DB_PATH = "anki_sentences.db"
conn = sqlite3.connect(DB_PATH)
cursor = conn.cursor()
cursor.execute("""
CREATE TABLE IF NOT EXISTS processed_sentences (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    sentence TEXT,
    focus_word TEXT,
    card_type TEXT,
    prompt TEXT,
    response TEXT,
    definition TEXT,
    example TEXT
);
""")
conn.commit()

# 5. Determine if the language has gender
LANG_HAS_GENDER = {
    "es": True,  # Spanish
    "fr": True,  # French
    "de": True,  # German
    "en": False,  # English
    # Add more languages as needed
}
lang_code = args.lang.split("_")[0]  # e.g., "es" from "es_core_news_sm"
has_gender = LANG_HAS_GENDER.get(lang_code, False)

# 6. Define Anki Models
class MyModel(genanki.Model):
    def __init__(self, model_id, name, fields, templates):
        super().__init__(model_id=model_id, name=name, fields=fields, templates=templates)

# Noun Model
noun_fields = [
    {"name": "Word"},
    {"name": "Sentence"},
    {"name": "Definition"},
    {"name": "Example"}
]
if has_gender:
    noun_fields.insert(2, {"name": "Gender"})

noun_templates = [
    {
        "Name": "Noun Card",
        "Front": "{{Word}}\n{{Sentence}}",
        "Back": ("{{Gender}}\n" if has_gender else "") + "{{Definition}}\n{{Example}}"
    }
]
NounModel = MyModel(
    model_id=1607392311,
    name="NounModel",
    fields=noun_fields,
    templates=noun_templates
)

# Verb Model
verb_fields = [
    {"name": "Word"},
    {"name": "Sentence"},
    {"name": "Conjugation"},
    {"name": "Definition"},
    {"name": "Example"}
]
verb_templates = [
    {
        "Name": "Verb Card",
        "Front": "{{Word}}\n{{Sentence}}",
        "Back": "{{Conjugation}}\n{{Definition}}\n{{Example}}"
    }
]
VerbModel = MyModel(
    model_id=1607392312,
    name="VerbModel",
    fields=verb_fields,
    templates=verb_templates
)

# 7. Create Anki Deck
anki_deck = genanki.Deck(
    deck_id=1234567890,  # Arbitrary unique ID
    name="Language Learning Deck"
)

# 8. Load known words if provided
known_words = set()
if args.known_words:
    try:
        with open(args.known_words, 'r', encoding='utf-8') as f:
            known_words = set(line.strip().lower() for line in f)
    except FileNotFoundError:
        print(f"Known words file '{args.known_words}' not found.")
        exit(1)

# ---------------------------------------------
# II. MAIN PROCESSING LOGIC
# ---------------------------------------------

def read_text_file(file_path: str) -> str:
    """
    Loads text from a file.
    
    Args:
        file_path (str): Path to the input text file.
    
    Returns:
        str: The content of the file.
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    except FileNotFoundError:
        print(f"Input file '{file_path}' not found.")
        exit(1)

def split_into_sentences(text: str) -> List[str]:
    """
    Uses spaCy's sentence segmentation to split the text into sentences.
    
    Args:
        text (str): The input text.
    
    Returns:
        List[str]: A list of sentences.
    """
    doc = nlp(text)
    return [sent.text.strip() for sent in doc.sents]

def identify_potential_focus_elements(doc: spacy.tokens.Doc) -> List[Dict[str, Any]]:
    """
    Identifies potential elements to focus on (nouns and verbs), excluding known words.
    
    Args:
        doc (spacy.tokens.Doc): The processed spaCy document.
    
    Returns:
        List[Dict[str, Any]]: List of focus elements with metadata.
    """
    elements = []
    for token in doc:
        if token.pos_ in ["NOUN", "PROPN"] and token.lemma_.lower() not in known_words:
            elements.append({
                "text": token.text,
                "pos": token.pos_,
                "lemma": token.lemma_,
                "card_type": "Noun Card"
            })
        elif token.pos_ == "VERB" and token.lemma_.lower() not in known_words:
            elements.append({
                "text": token.text,
                "pos": token.pos_,
                "lemma": token.lemma_,
                "card_type": "Verb Card"
            })
    return elements

def generate_llm_prompts(element: Dict[str, Any], sentence: str) -> List[str]:
    """
    Creates a structured prompt for the LLM based on the card type and language.
    
    Args:
        element (Dict[str, Any]): The focus element metadata.
        sentence (str): The sentence containing the focus word.
    
    Returns:
        List[str]: A list of prompts for the LLM.
    """
    prompts = []
    card_type = element["card_type"]
    focus_word = element["text"]
    if card_type == "Noun Card":
        if has_gender:
            prompts.append(
                f"For the noun '{focus_word}' in the sentence '{sentence}', provide: Gender: <masculine/feminine/neuter>. Definition: <definition>. Example: <example sentence>."
            )
        else:
            prompts.append(
                f"For the noun '{focus_word}' in the sentence '{sentence}', provide: Definition: <definition>. Example: <example sentence>."
            )
    elif card_type == "Verb Card":
        prompts.append(
            f"For the verb '{focus_word}' in the sentence '{sentence}', provide: Conjugation in present tense (first-person singular): <conjugation>. Definition: <definition>. Example: <example sentence>."
        )
    return prompts

def call_llm_api(prompts: List[str]) -> List[str]:
    """
    Calls the LLM API with the given prompts in parallel.
    
    Args:
        prompts (List[str]): List of prompts to send to the LLM.
    
    Returns:
        List[str]: List of responses from the LLM.
    """
    responses = []
    with ThreadPoolExecutor() as executor:
        future_to_prompt = {executor.submit(openai.Completion.create, engine="text-davinci-003", prompt=p, max_tokens=100, temperature=0.7): p for p in prompts}
        for future in future_to_prompt:
            try:
                response = future.result()
                responses.append(response.choices[0].text.strip())
            except Exception as e:
                print(f"LLM API error for prompt '{future_to_prompt[future]}': {e}")
                responses.append("Error: Unable to generate response")
    return responses

def process_llm_responses(responses: List[str], card_type: str) -> Dict[str, Any]:
    """
    Extracts structured data from LLM responses using spaCy.
    
    Args:
        responses (List[str]): List of responses from the LLM.
        card_type (str): Type of card ("Noun Card" or "Verb Card").
    
    Returns:
        Dict[str, Any]: Extracted data (e.g., gender, definition, example).
    """
    extracted_data = {}
    for res in responses:
        doc = nlp(res)
        for sent in doc.sents:
            text = sent.text
            if card_type == "Noun Card":
                if has_gender and "Gender:" in text:
                    extracted_data["gender"] = text.split("Gender:")[1].split(".")[0].strip()
                if "Definition:" in text:
                    extracted_data["definition"] = text.split("Definition:")[1].split(".")[0].strip()
                if "Example:" in text:
                    extracted_data["example"] = text.split("Example:")[1].strip()
            elif card_type == "Verb Card":
                if "Conjugation:" in text:
                    extracted_data["conjugation"] = text.split("Conjugation:")[1].split(".")[0].strip()
                if "Definition:" in text:
                    extracted_data["definition"] = text.split("Definition:")[1].split(".")[0].strip()
                if "Example:" in text:
                    extracted_data["example"] = text.split("Example:")[1].strip()
    return extracted_data

def create_anki_card(extracted_data: Dict[str, Any], sentence: str, focus_word: str, card_type: str) -> genanki.Note:
    """
    Creates an Anki note based on the card type and extracted data.
    
    Args:
        extracted_data (Dict[str, Any]): Data extracted from LLM response.
        sentence (str): The original sentence.
        focus_word (str): The word to focus on.
        card_type (str): Type of card ("Noun Card" or "Verb Card").
    
    Returns:
        genanki.Note: The Anki note object.
    """
    if card_type == "Noun Card":
        fields = [focus_word, sentence]
        if has_gender:
            fields.append(extracted_data.get("gender", ""))
        fields.extend([extracted_data.get("definition", ""), extracted_data.get("example", "")])
        return genanki.Note(model=NounModel, fields=fields)
    elif card_type == "Verb Card":
        fields = [focus_word, sentence, extracted_data.get("conjugation", ""), extracted_data.get("definition", ""), extracted_data.get("example", "")]
        return genanki.Note(model=VerbModel, fields=fields)

def store_data_in_db(sentence: str, element_text: str, card_type: str, prompt: str, response: str, extracted_data: Dict[str, Any]):
    """
    Stores processed data in the SQLite database.
    
    Args:
        sentence (str): The original sentence.
        element_text (str): The focus word.
        card_type (str): Type of card.
        prompt (str): The prompt sent to the LLM.
        response (str): The LLM's response.
        extracted_data (Dict[str, Any]): Extracted data from the response.
    """
    definition = extracted_data.get("definition", "")
    example = extracted_data.get("example", "")
    cursor.execute("""
    INSERT INTO processed_sentences (sentence, focus_word, card_type, prompt, response, definition, example)
    VALUES (?, ?, ?, ?, ?, ?, ?)
    """, (sentence, element_text, card_type, prompt, response, definition, example))
    conn.commit()

def process_text(text: str):
    """
    Main processing pipeline: splits text into sentences, identifies focus elements,
    generates prompts, queries LLM, processes responses, creates Anki cards, and stores data.
    
    Args:
        text (str): The input text to process.
    """
    sentences = split_into_sentences(text)
    for sentence in sentences:
        doc = nlp(sentence)
        potential_focus_elements = identify_potential_focus_elements(doc)
        for element in potential_focus_elements:
            focus_word = element["text"]
            card_type = element["card_type"]
            prompts = generate_llm_prompts(element, sentence)
            responses = call_llm_api(prompts)
            for prompt, response in zip(prompts, responses):
                extracted_data = process_llm_responses([response], card_type)
                card_note = create_anki_card(extracted_data, sentence, focus_word, card_type)
                anki_deck.add_note(card_note)
                store_data_in_db(sentence, focus_word, card_type, prompt, response, extracted_data)

def export_anki_deck(output_filename: str):
    """
    Exports the Anki deck to a .apkg file.
    
    Args:
        output_filename (str): The name of the output Anki file.
    """
    genanki.Package(anki_deck).write_to_file(output_filename)
    print(f"Anki deck exported to {output_filename}")

# ---------------------------------------------
# MAIN SCRIPT EXECUTION
# ---------------------------------------------
if __name__ == "__main__":
    text_input = read_text_file(args.input)
    process_text(text_input)
    export_anki_deck(args.output)
    conn.close()
