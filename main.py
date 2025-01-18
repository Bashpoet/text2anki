import os
import spacy
import genanki
import openai  # or from transformers import pipeline, etc., depending on your LLM usage
import sqlite3
from typing import List, Dict, Any

# ---------------------------------------------
# I. SETUP & INITIALIZATION
# ---------------------------------------------

# 1. Load spaCy and LLM
#    Adjust model name for target language, e.g., "es_core_news_sm" for Spanish.
NLP_MODEL = "es_core_news_sm"  # Change to the appropriate model for your target language
nlp = spacy.load(NLP_MODEL)

# For demonstration, we'll set up OpenAI, but you could adapt for Hugging Face or others.
openai.api_key = os.getenv("OPENAI_API_KEY")  # Make sure your API key is available as an env variable


# 2. Optional: Initialize Database
DB_PATH = "anki_sentences.db"
conn = sqlite3.connect(DB_PATH)
cursor = conn.cursor()

# Example table creation (if storing data about sentences, tokens, etc.)
cursor.execute("""
CREATE TABLE IF NOT EXISTS processed_sentences (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    sentence TEXT,
    focus_word TEXT,
    card_type TEXT,
    response TEXT
);
""")
conn.commit()

# 3. Define Anki Model(s) and Deck
#    We'll define a single sample model here. You can define more specialized ones for nouns, verbs, etc.
class MyBaseModel(genanki.Model):
    def __init__(self):
        super().__init__(
            model_id=1607392310,
            name="CustomModel",
            fields=[
                {"name": "Front"},
                {"name": "Back"},
            ],
            templates=[
                {
                    "Name": "Card 1",
                    "Front": "{{Front}}",
                    "Back": "{{Back}}",
                }
            ],
        )

anki_deck = genanki.Deck(
    deck_id=1234567890,  # Arbitrary unique ID
    name="Language Learning Deck"
)

# (Optional) Additional specialized models for nouns, verbs, etc. can be defined similarly


# ---------------------------------------------
# II. MAIN PROCESSING LOGIC
# ---------------------------------------------

def read_text_file(file_path: str) -> str:
    """
    Loads text from a file (txt, pdf if you handle PDF reading, etc.)
    For simplicity, we'll assume it's a plain text file.
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    return content


def split_into_sentences(text: str) -> List[str]:
    """
    Uses spaCy's sentence segmentation to split the text into sentences.
    """
    doc = nlp(text)
    sentences = [sent.text.strip() for sent in doc.sents]
    return sentences


def identify_potential_focus_elements(doc: spacy.tokens.Doc) -> List[Dict[str, Any]]:
    """
    Identifies potential elements to focus on (nouns, verbs, phrases, etc.)
    Here, as a minimal example, we pick out nouns and verbs.
    In practice, you'll want frequency checks, user-known words, grammar rules, etc.
    """
    elements = []
    for token in doc:
        if token.pos_ in ["NOUN", "PROPN"]:
            elements.append({
                "text": token.text,
                "pos": token.pos_,
                "lemma": token.lemma_,
                "card_type": "Noun Card"  # Or more sophisticated logic
            })
        elif token.pos_ == "VERB":
            elements.append({
                "text": token.text,
                "pos": token.pos_,
                "lemma": token.lemma_,
                "card_type": "Verb Card"
            })
        # Add as many categories as you like here
    return elements


def generate_llm_prompts(element: Dict[str, Any], sentence: str) -> List[str]:
    """
    Creates prompts for the LLM depending on the card type and context.
    """
    prompts = []
    card_type = element["card_type"]
    focus_word = element["text"]

    # Example prompts for a noun card
    if card_type == "Noun Card":
        # Your refined logic can be more elaborate
        prompts.append(
            f"Explain the gender and number of the noun '{focus_word}' in this sentence: '{sentence}'."
        )
        prompts.append(
            f"Provide a short definition and one additional example sentence of the noun '{focus_word}'."
        )

    elif card_type == "Verb Card":
        prompts.append(
            f"Conjugate the verb '{focus_word}' in present tense (first-person singular) based on its usage in this sentence: '{sentence}'."
        )
        prompts.append(
            f"Give a brief definition and an example sentence for the verb '{focus_word}'."
        )

    # Additional prompts for phrases, clauses, etc. can follow similarly
    return prompts


def call_llm_api(prompts: List[str]) -> List[str]:
    """
    Calls the LLM with the given prompts.
    Returns a list of response strings.
    """
    responses = []
    for prompt in prompts:
        # Example: Using OpenAI's API
        try:
            response = openai.Completion.create(
                engine="text-davinci-003",  # or another model
                prompt=prompt,
                max_tokens=100,
                temperature=0.7
            )
            text = response.choices[0].text.strip()
            responses.append(text)
        except Exception as e:
            print(f"LLM API error: {e}")
            responses.append("")
    return responses


def process_llm_responses(responses: List[str]) -> Dict[str, Any]:
    """
    Extract relevant info from the LLM's text responses.
    In a real scenario, you'd parse for definitions, morphological data, etc.
    We'll just store them verbatim for this demo.
    """
    extracted_data = {
        "definitions": [],
        "examples": []
    }
    # A naive approach: treat each response as either a definition or example.
    # You'd tailor this to your prompt design.
    for res in responses:
        # For now, we just throw each entire response into definitions or examples in turn
        if "example" in res.lower():
            extracted_data["examples"].append(res)
        else:
            extracted_data["definitions"].append(res)
    return extracted_data


def create_anki_card(extracted_data: Dict[str, Any],
                     sentence: str,
                     focus_word: str,
                     card_type: str) -> genanki.Note:
    """
    Build an Anki note using the 'CustomModel'.
    Some fields can combine the extracted data for front/back.
    """
    front_text = f"Focus Word: {focus_word}\n\nSentence: {sentence}"
    back_text = ""

    # Combine definitions in some coherent manner
    if extracted_data["definitions"]:
        defs = "\n".join(extracted_data["definitions"])
        back_text += f"Definitions:\n{defs}\n\n"

    if extracted_data["examples"]:
        exs = "\n".join(extracted_data["examples"])
        back_text += f"Examples:\n{exs}\n"

    note = genanki.Note(
        model=MyBaseModel(),
        fields=[front_text, back_text]
    )
    return note


def store_data_in_db(sentence: str,
                     element_text: str,
                     card_type: str,
                     response: str):
    """
    Saves minimal info to the SQLite DB. Expand as needed.
    """
    cursor.execute("""
    INSERT INTO processed_sentences (sentence, focus_word, card_type, response)
    VALUES (?, ?, ?, ?)
    """, (sentence, element_text, card_type, response))
    conn.commit()


def process_text(text: str):
    """
    Main pipeline hooking all steps together:
      - Sentence segmentation
      - Identify potential focus elements
      - Generate prompts
      - Query LLM
      - Process responses
      - Create and add Anki cards
      - Store in DB
    """
    sentences = split_into_sentences(text)
    for sentence in sentences:
        # spaCy doc object for the current sentence
        doc = nlp(sentence)
        # Identify potential focus elements (nouns, verbs, etc.)
        potential_focus_elements = identify_potential_focus_elements(doc)

        for element in potential_focus_elements:
            focus_word = element["text"]
            card_type = element["card_type"]

            # Generate LLM prompts
            prompts = generate_llm_prompts(element, sentence)

            # Call LLM API
            responses = call_llm_api(prompts)

            # Process the responses
            extracted_data = process_llm_responses(responses)

            # Create an Anki card
            card_note = create_anki_card(extracted_data, sentence, focus_word, card_type)
            anki_deck.add_note(card_note)

            # Store minimal data in DB
            store_data_in_db(sentence, focus_word, card_type, str(responses))


def export_anki_deck(output_filename: str = "Language_Learning.apkg"):
    """
    Final step: Package up the deck as a .apkg file.
    """
    # Build a package and write to file
    genanki.Package(anki_deck).write_to_file(output_filename)
    print(f"Anki deck exported to {output_filename}")


# ---------------------------------------------
# MAIN SCRIPT EXECUTION
# ---------------------------------------------
if __name__ == "__main__":
    # Example usage:
    # 1) Read text
    text_input = read_text_file("sample_text.txt")  # Provide your own text file
    # 2) Process the text
    process_text(text_input)
    # 3) Export the deck
    export_anki_deck("MyLanguageDeck.apkg")

    print("All done! Your deck should now be ready for import into Anki.")
