import os
import time
import spacy
import genanki
import openai
import sqlite3
import logging
import argparse
from typing import List, Dict, Any, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache

# ---------------------------------------------
# Configure Logging
# ---------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("anki_generator.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("anki_generator")


# ---------------------------------------------
# Anki Card Models
# ---------------------------------------------
class AnkiCardModels:
    """Class to manage Anki card models."""
    
    def __init__(self, has_gender: bool):
        self.has_gender = has_gender
        self.models = self._create_models()
    
    def _create_models(self) -> Dict[str, genanki.Model]:
        """Create and return Anki card models."""
        class CustomModel(genanki.Model):
            def __init__(self, model_id, name, fields, templates):
                super().__init__(model_id=model_id, name=name, fields=fields, templates=templates)
        
        # Noun Model
        noun_fields = [
            {"name": "Word"},
            {"name": "Sentence"},
            {"name": "Definition"},
            {"name": "Example"}
        ]
        if self.has_gender:
            noun_fields.insert(2, {"name": "Gender"})
        
        noun_templates = [
            {
                "name": "Noun Card",
                "qfmt": "{{Word}}<br>{{Sentence}}",
                "afmt": "{{FrontSide}}<hr id='answer'>" + 
                        ("{{Gender}}<br>" if self.has_gender else "") + 
                        "{{Definition}}<br>{{Example}}"
            }
        ]
        
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
                "name": "Verb Card",
                "qfmt": "{{Word}}<br>{{Sentence}}",
                "afmt": "{{FrontSide}}<hr id='answer'>" + 
                        "{{Conjugation}}<br>{{Definition}}<br>{{Example}}"
            }
        ]
        
        return {
            "Noun Card": CustomModel(
                model_id=1607392311,
                name="NounModel",
                fields=noun_fields,
                templates=noun_templates
            ),
            "Verb Card": CustomModel(
                model_id=1607392312,
                name="VerbModel",
                fields=verb_fields,
                templates=verb_templates
            )
        }
    
    def get_model(self, card_type: str) -> genanki.Model:
        """Get the appropriate model for a card type."""
        return self.models.get(card_type)


# ---------------------------------------------
# Database Manager
# ---------------------------------------------
class DatabaseManager:
    """Class to manage database operations."""
    
    def __init__(self, db_path: str = "anki_sentences.db"):
        """Initialize the database manager."""
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path)
        self.cursor = self.conn.cursor()
        self._setup_db()
    
    def _setup_db(self):
        """Set up the database schema if it doesn't exist."""
        self.cursor.execute("""
        CREATE TABLE IF NOT EXISTS processed_sentences (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            sentence TEXT,
            focus_word TEXT,
            card_type TEXT,
            prompt TEXT,
            response TEXT,
            definition TEXT,
            example TEXT,
            gender TEXT,
            conjugation TEXT,
            processed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(sentence, focus_word, card_type)
        );
        """)
        
        self.cursor.execute("""
        CREATE TABLE IF NOT EXISTS processing_status (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            input_file TEXT,
            total_sentences INTEGER,
            processed_sentences INTEGER,
            start_time TIMESTAMP,
            end_time TIMESTAMP,
            status TEXT
        );
        """)
        
        self.conn.commit()
    
    def store_card_data(self, sentence: str, focus_word: str, card_type: str, 
                      prompt: str, response: str, extracted_data: Dict[str, Any]):
        """Store processed card data in the database."""
        try:
            self.cursor.execute("""
            INSERT OR REPLACE INTO processed_sentences 
            (sentence, focus_word, card_type, prompt, response, definition, example, gender, conjugation)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                sentence, 
                focus_word, 
                card_type, 
                prompt, 
                response, 
                extracted_data.get("definition", ""),
                extracted_data.get("example", ""),
                extracted_data.get("gender", ""),
                extracted_data.get("conjugation", "")
            ))
            self.conn.commit()
        except sqlite3.Error as e:
            logger.error(f"Database error: {e}")
            self.conn.rollback()
    
    def is_sentence_word_processed(self, sentence: str, focus_word: str, card_type: str) -> bool:
        """Check if a sentence-word pair has already been processed."""
        try:
            self.cursor.execute("""
            SELECT id FROM processed_sentences 
            WHERE sentence = ? AND focus_word = ? AND card_type = ?
            """, (sentence, focus_word, card_type))
            return self.cursor.fetchone() is not None
        except sqlite3.Error as e:
            logger.error(f"Database error: {e}")
            return False
    
    def start_processing_job(self, input_file: str, total_sentences: int) -> int:
        """Record the start of a processing job and return its ID."""
        try:
            self.cursor.execute("""
            INSERT INTO processing_status 
            (input_file, total_sentences, processed_sentences, start_time, status)
            VALUES (?, ?, 0, CURRENT_TIMESTAMP, 'STARTED')
            """, (input_file, total_sentences))
            self.conn.commit()
            return self.cursor.lastrowid
        except sqlite3.Error as e:
            logger.error(f"Database error: {e}")
            return -1
    
    def update_processing_status(self, job_id: int, processed_count: int, status: str = None):
        """Update the status of a processing job."""
        try:
            if status:
                self.cursor.execute("""
                UPDATE processing_status 
                SET processed_sentences = ?, status = ?, 
                end_time = CASE WHEN ? IN ('COMPLETED', 'FAILED') THEN CURRENT_TIMESTAMP ELSE end_time END
                WHERE id = ?
                """, (processed_count, status, status, job_id))
            else:
                self.cursor.execute("""
                UPDATE processing_status 
                SET processed_sentences = ?
                WHERE id = ?
                """, (processed_count, job_id))
            self.conn.commit()
        except sqlite3.Error as e:
            logger.error(f"Database error: {e}")
    
    def get_processing_status(self, job_id: int) -> Optional[Dict[str, Any]]:
        """Get the status of a processing job."""
        try:
            self.cursor.execute("""
            SELECT * FROM processing_status WHERE id = ?
            """, (job_id,))
            row = self.cursor.fetchone()
            if row:
                columns = [description[0] for description in self.cursor.description]
                return dict(zip(columns, row))
            return None
        except sqlite3.Error as e:
            logger.error(f"Database error: {e}")
            return None
    
    def close(self):
        """Close the database connection."""
        self.conn.close()


# ---------------------------------------------
# LLM Service
# ---------------------------------------------
class LLMService:
    """Class to handle LLM API interactions."""
    
    def __init__(self, api_key: str, engine: str = "gpt-3.5-turbo"):
        """Initialize the LLM service."""
        openai.api_key = api_key
        self.engine = engine
        # Simple cache for responses to avoid duplicate calls
        self.response_cache = {}
    
    def call_with_retry(self, prompt: str, max_retries: int = 3, base_delay: float = 1.0) -> Dict[str, Any]:
        """Call the OpenAI API with exponential backoff retry."""
        cache_key = hash(prompt)
        if cache_key in self.response_cache:
            logger.info("Using cached response")
            return self.response_cache[cache_key]
        
        for attempt in range(max_retries):
            try:
                if self.engine.startswith("gpt"):  # ChatCompletion API
                    response = openai.ChatCompletion.create(
                        model=self.engine,
                        messages=[
                            {"role": "system", "content": "You are a helpful language learning assistant."},
                            {"role": "user", "content": prompt}
                        ],
                        max_tokens=150,
                        temperature=0.7
                    )
                    result = response.choices[0].message.content.strip()
                else:  # Completion API
                    response = openai.Completion.create(
                        engine=self.engine,
                        prompt=prompt,
                        max_tokens=150,
                        temperature=0.7
                    )
                    result = response.choices[0].text.strip()
                
                # Cache the result
                self.response_cache[cache_key] = result
                return result
            except (openai.error.RateLimitError, openai.error.ServiceUnavailableError) as e:
                if attempt == max_retries - 1:
                    logger.error(f"LLM API error after {max_retries} attempts: {e}")
                    raise
                delay = base_delay * (2 ** attempt)
                logger.warning(f"LLM API error: {e}. Retrying in {delay}s...")
                time.sleep(delay)
            except Exception as e:
                logger.error(f"Unexpected LLM API error: {e}")
                raise
    
    def batch_process(self, prompts: List[str], max_workers: int = 3) -> List[str]:
        """Process multiple prompts in parallel with a limited number of workers."""
        responses = []
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_prompt = {
                executor.submit(self.call_with_retry, prompt): prompt 
                for prompt in prompts
            }
            for future in future_to_prompt:
                try:
                    response = future.result()
                    responses.append(response)
                except Exception as e:
                    logger.error(f"Error processing prompt: {e}")
                    responses.append(f"Error: {str(e)}")
        
        return responses


# ---------------------------------------------
# Anki Card Generator
# ---------------------------------------------
class AnkiCardGenerator:
    """Main class for generating Anki cards from text."""
    
    def __init__(self, options: Dict[str, Any]):
        """Initialize the generator with options."""
        self.options = options
        
        # Set up language processing
        lang_code = options["lang"].split("_")[0]  # e.g., "es" from "es_core_news_sm"
        self.has_gender = self._language_has_gender(lang_code)
        
        try:
            self.nlp = spacy.load(options["lang"])
        except OSError:
            logger.error(f"spaCy model '{options['lang']}' not found. Please install it first.")
            raise
        
        # Set up OpenAI
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            logger.error("OpenAI API key not found. Set the OPENAI_API_KEY environment variable.")
            raise ValueError("OpenAI API key not found")
        
        # Initialize components
        self.llm = LLMService(api_key)
        self.db = DatabaseManager()
        self.card_models = AnkiCardModels(self.has_gender)
        
        # Create Anki deck
        self.anki_deck = genanki.Deck(
            deck_id=1234567890,  # Arbitrary unique ID
            name=options.get("deck_name", "Language Learning Deck")
        )
        
        # Load known words if provided
        self.known_words = self._load_known_words(options.get("known_words"))
    
    def _language_has_gender(self, lang_code: str) -> bool:
        """Determine if the language has grammatical gender."""
        LANG_HAS_GENDER = {
            "es": True,  # Spanish
            "fr": True,  # French
            "de": True,  # German
            "it": True,  # Italian
            "pt": True,  # Portuguese
            "ru": True,  # Russian
            "en": False,  # English
            "ja": False,  # Japanese
            "zh": False,  # Chinese
            "ko": False,  # Korean
        }
        return LANG_HAS_GENDER.get(lang_code, False)
    
    def _load_known_words(self, known_words_file: Optional[str]) -> set:
        """Load known words from file if provided."""
        known_words = set()
        if known_words_file:
            try:
                with open(known_words_file, 'r', encoding='utf-8') as f:
                    known_words = set(line.strip().lower() for line in f)
                logger.info(f"Loaded {len(known_words)} known words from {known_words_file}")
            except FileNotFoundError:
                logger.warning(f"Known words file '{known_words_file}' not found.")
        return known_words
    
    def read_text_file(self, file_path: str) -> str:
        """Read text from a file."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
        except FileNotFoundError:
            logger.error(f"Input file '{file_path}' not found.")
            raise
    
    def split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences using spaCy."""
        doc = self.nlp(text)
        return [sent.text.strip() for sent in doc.sents]
    
    def identify_focus_elements(self, doc: spacy.tokens.Doc) -> List[Dict[str, Any]]:
        """Identify potential elements to focus on (nouns and verbs)."""
        elements = []
        for token in doc:
            if token.pos_ in ["NOUN", "PROPN"] and token.lemma_.lower() not in self.known_words:
                elements.append({
                    "text": token.text,
                    "pos": token.pos_,
                    "lemma": token.lemma_,
                    "card_type": "Noun Card"
                })
            elif token.pos_ == "VERB" and token.lemma_.lower() not in self.known_words:
                elements.append({
                    "text": token.text,
                    "pos": token.pos_,
                    "lemma": token.lemma_,
                    "card_type": "Verb Card"
                })
        return elements
    
    def generate_llm_prompt(self, element: Dict[str, Any], sentence: str) -> str:
        """Create a structured prompt for the LLM."""
        card_type = element["card_type"]
        focus_word = element["text"]
        
        if card_type == "Noun Card":
            if self.has_gender:
                return (
                    f"I'm learning a language and need help with the noun '{focus_word}' from this sentence: '{sentence}'. "
                    f"Please provide me with the following information in JSON format:\n"
                    f"{{\n"
                    f"  \"gender\": \"[masculine/feminine/neuter]\",\n"
                    f"  \"definition\": \"[simple definition in English]\",\n"
                    f"  \"example\": \"[example sentence using this word in a different context]\"\n"
                    f"}}"
                )
            else:
                return (
                    f"I'm learning a language and need help with the noun '{focus_word}' from this sentence: '{sentence}'. "
                    f"Please provide me with the following information in JSON format:\n"
                    f"{{\n"
                    f"  \"definition\": \"[simple definition in English]\",\n"
                    f"  \"example\": \"[example sentence using this word in a different context]\"\n"
                    f"}}"
                )
        elif card_type == "Verb Card":
            return (
                f"I'm learning a language and need help with the verb '{focus_word}' from this sentence: '{sentence}'. "
                f"Please provide me with the following information in JSON format:\n"
                f"{{\n"
                f"  \"conjugation\": \"[conjugation in present tense, first-person singular]\",\n"
                f"  \"definition\": \"[simple definition in English]\",\n"
                f"  \"example\": \"[example sentence using this word in a different context]\"\n"
                f"}}"
            )
    
    def extract_json_from_response(self, response: str) -> Dict[str, Any]:
        """Extract JSON data from LLM response."""
        import json
        import re
        
        # Try to find JSON object in the response
        match = re.search(r'{.*?}', response, re.DOTALL)
        if match:
            try:
                # Parse the JSON object
                json_str = match.group(0)
                data = json.loads(json_str)
                return data
            except json.JSONDecodeError:
                logger.warning(f"Failed to parse JSON from response: {response}")
        
        # Fallback extraction if JSON parsing fails
        extracted = {}
        
        # Extract fields based on patterns
        if '"gender"' in response.lower() or 'gender:' in response.lower():
            gender_match = re.search(r'"gender"\s*:\s*"([^"]+)"', response) or re.search(r'gender:\s*(\w+)', response, re.IGNORECASE)
            if gender_match:
                extracted["gender"] = gender_match.group(1).strip()
        
        if '"definition"' in response.lower() or 'definition:' in response.lower():
            def_match = re.search(r'"definition"\s*:\s*"([^"]+)"', response) or re.search(r'definition:\s*(.+?)(?:\n|$)', response, re.IGNORECASE)
            if def_match:
                extracted["definition"] = def_match.group(1).strip()
        
        if '"example"' in response.lower() or 'example:' in response.lower():
            example_match = re.search(r'"example"\s*:\s*"([^"]+)"', response) or re.search(r'example:\s*(.+?)(?:\n|$)', response, re.IGNORECASE)
            if example_match:
                extracted["example"] = example_match.group(1).strip()
        
        if '"conjugation"' in response.lower() or 'conjugation:' in response.lower():
            conj_match = re.search(r'"conjugation"\s*:\s*"([^"]+)"', response) or re.search(r'conjugation:\s*(.+?)(?:\n|$)', response, re.IGNORECASE)
            if conj_match:
                extracted["conjugation"] = conj_match.group(1).strip()
        
        return extracted
    
    def create_anki_note(self, extracted_data: Dict[str, Any], sentence: str, 
                        focus_word: str, card_type: str) -> genanki.Note:
        """Create an Anki note based on the card type and extracted data."""
        model = self.card_models.get_model(card_type)
        
        if card_type == "Noun Card":
            fields = [focus_word, sentence]
            if self.has_gender:
                fields.append(extracted_data.get("gender", ""))
            fields.extend([extracted_data.get("definition", ""), extracted_data.get("example", "")])
        elif card_type == "Verb Card":
            fields = [
                focus_word, 
                sentence, 
                extracted_data.get("conjugation", ""), 
                extracted_data.get("definition", ""), 
                extracted_data.get("example", "")
            ]
        else:
            logger.error(f"Unknown card type: {card_type}")
            raise ValueError(f"Unknown card type: {card_type}")
        
        return genanki.Note(model=model, fields=fields)
    
    def process_sentence(self, sentence: str) -> List[genanki.Note]:
        """Process a single sentence and return generated notes."""
        doc = self.nlp(sentence)
        focus_elements = self.identify_focus_elements(doc)
        notes = []
        
        for element in focus_elements:
            focus_word = element["text"]
            card_type = element["card_type"]
            
            # Skip if already processed
            if self.db.is_sentence_word_processed(sentence, focus_word, card_type):
                logger.debug(f"Skipping already processed: {focus_word} in '{sentence}'")
                continue
            
            prompt = self.generate_llm_prompt(element, sentence)
            
            try:
                response = self.llm.call_with_retry(prompt)
                extracted_data = self.extract_json_from_response(response)
                
                if not extracted_data:
                    logger.warning(f"Failed to extract data for '{focus_word}' in sentence: '{sentence}'")
                    continue
                
                note = self.create_anki_note(extracted_data, sentence, focus_word, card_type)
                notes.append(note)
                
                # Store in database
                self.db.store_card_data(sentence, focus_word, card_type, prompt, response, extracted_data)
                
            except Exception as e:
                logger.error(f"Error processing '{focus_word}' in '{sentence}': {e}")
        
        return notes
    
    def process_text(self, text: str, progress_callback=None) -> None:
        """Process the entire text and generate Anki cards."""
        sentences = self.split_into_sentences(text)
        total_sentences = len(sentences)
        logger.info(f"Processing {total_sentences} sentences")
        
        # Start a processing job
        job_id = self.db.start_processing_job(self.options.get("input", "unknown"), total_sentences)
        
        for i, sentence in enumerate(sentences):
            try:
                notes = self.process_sentence(sentence)
                for note in notes:
                    self.anki_deck.add_note(note)
                
                # Update progress
                if progress_callback:
                    progress_callback(i + 1, total_sentences)
                
                # Update database status
                if job_id > 0:
                    self.db.update_processing_status(job_id, i + 1)
                
                # Log progress every 10 sentences
                if (i + 1) % 10 == 0 or i + 1 == total_sentences:
                    logger.info(f"Processed {i + 1}/{total_sentences} sentences")
            
            except Exception as e:
                logger.error(f"Error processing sentence: '{sentence}': {e}")
                if job_id > 0:
                    self.db.update_processing_status(job_id, i + 1, "FAILED")
        
        if job_id > 0:
            self.db.update_processing_status(job_id, total_sentences, "COMPLETED")
    
    def export_anki_deck(self) -> None:
        """Export the Anki deck to a .apkg file."""
        output_file = self.options.get("output", "Language_Learning.apkg")
        genanki.Package(self.anki_deck).write_to_file(output_file)
        logger.info(f"Anki deck exported to {output_file}")
    
    def cleanup(self) -> None:
        """Clean up resources."""
        self.db.close()


# ---------------------------------------------
# CLI Application
# ---------------------------------------------
def create_parser() -> argparse.ArgumentParser:
    """Create command line argument parser."""
    parser = argparse.ArgumentParser(description="Generate Anki flashcards from text using spaCy and LLM.")
    parser.add_argument("--lang", required=True, help="spaCy language model (e.g., es_core_news_sm for Spanish)")
    parser.add_argument("--input", required=True, help="Input text file")
    parser.add_argument("--output", default="Language_Learning.apkg", help="Output Anki deck file name")
    parser.add_argument("--known-words", help="File containing known words to exclude")
    parser.add_argument("--deck-name", default="Language Learning Deck", help="Name for the Anki deck")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose logging")
    parser.add_argument("--resume", action="store_true", help="Resume from last run if interrupted")
    return parser


def display_progress(current: int, total: int) -> None:
    """Display a progress bar."""
    percent = int(current * 100 / total)
    bar_length = 50
    filled_length = int(bar_length * current / total)
    bar = '█' * filled_length + '-' * (bar_length - filled_length)
    print(f'\rProgress: |{bar}| {percent}% ({current}/{total})', end='')
    if current == total:
        print()


def main() -> None:
    """Main function to run the Anki card generator."""
    parser = create_parser()
    args = parser.parse_args()
    
    # Set log level
    if args.verbose:
        logger.setLevel(logging.DEBUG)
    
    try:
        # Convert args to dict
        options = vars(args)
        
        # Initialize generator
        generator = AnkiCardGenerator(options)
        
        # Read input file
        input_text = generator.read_text_file(args.input)
        
        # Process text
        generator.process_text(input_text, progress_callback=display_progress)
        
        # Export deck
        generator.export_anki_deck()
        
        # Cleanup
        generator.cleanup()
        
    except Exception as e:
        logger.error(f"Error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    main()
