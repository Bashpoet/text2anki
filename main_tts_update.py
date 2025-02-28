import os
import time
import spacy
import genanki
import openai
import sqlite3
import logging
import argparse
import tempfile
import requests
import hashlib
from gtts import gTTS
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
# Audio Manager for TTS
# ---------------------------------------------
class AudioManager:
    """Class to handle text-to-speech and audio file management."""
    
    def __init__(self, lang_code: str, media_dir: str = "./media"):
        """Initialize the audio manager.
        
        Args:
            lang_code: Language code for TTS (e.g., 'es' for Spanish)
            media_dir: Directory to store audio files
        """
        self.lang_code = lang_code
        self.media_dir = media_dir
        self.audio_files = []
        
        # Create media directory if it doesn't exist
        os.makedirs(media_dir, exist_ok=True)
    
    def generate_audio(self, text: str, word: str = None) -> str:
        """Generate audio file for the given text.
        
        Args:
            text: Text to convert to speech
            word: Optional word to include in the filename for identification
            
        Returns:
            str: Path to the generated audio file
        """
        try:
            # Create a unique filename based on the content
            if word:
                # Use the word as part of the filename for better organization
                sanitized_word = ''.join(c for c in word if c.isalnum())
                content_hash = hashlib.md5(text.encode()).hexdigest()[:8]
                filename = f"{sanitized_word}_{content_hash}.mp3"
            else:
                content_hash = hashlib.md5(text.encode()).hexdigest()
                filename = f"audio_{content_hash}.mp3"
            
            filepath = os.path.join(self.media_dir, filename)
            
            # Skip if file already exists
            if os.path.exists(filepath):
                logger.debug(f"Audio file already exists: {filepath}")
                self.audio_files.append(filepath)
                return filename
            
            # Generate audio using gTTS
            tts = gTTS(text=text, lang=self.lang_code, slow=False)
            tts.save(filepath)
            
            logger.debug(f"Generated audio file: {filepath}")
            self.audio_files.append(filepath)
            return filename
            
        except Exception as e:
            logger.error(f"Error generating audio for '{text}': {e}")
            return None
    
    def get_audio_files(self) -> List[str]:
        """Get list of all generated audio files."""
        return self.audio_files


# ---------------------------------------------
# Anki Card Models
# ---------------------------------------------
class AnkiCardModels:
    """Class to manage Anki card models."""
    
    def __init__(self, has_gender: bool, has_audio: bool):
        self.has_gender = has_gender
        self.has_audio = has_audio
        self.models = self._create_models()
    
    def _create_models(self) -> Dict[str, genanki.Model]:
        """Create and return Anki card models."""
        class CustomModel(genanki.Model):
            def __init__(self, model_id, name, fields, templates, css=None):
                super().__init__(model_id=model_id, name=name, fields=fields, templates=templates, css=css)
        
        # Shared CSS for both models
        css = """
        .card {
            font-family: Arial, sans-serif;
            font-size: 20px;
            text-align: center;
            color: black;
            background-color: white;
            padding: 20px;
        }
        .word {
            font-weight: bold;
            font-size: 24px;
            color: #2c3e50;
        }
        .sentence {
            font-style: italic;
            color: #7f8c8d;
            margin-top: 10px;
        }
        .definition {
            margin-top: 15px;
            color: #2980b9;
        }
        .example {
            margin-top: 10px;
            color: #27ae60;
        }
        .gender {
            color: #e74c3c;
            font-weight: bold;
        }
        .conjugation {
            color: #8e44ad;
        }
        .audio-btn {
            margin-top: 10px;
        }
        """
        
        # Noun Model
        noun_fields = [
            {"name": "Word"},
            {"name": "Sentence"},
            {"name": "Definition"},
            {"name": "Example"}
        ]
        
        if self.has_gender:
            noun_fields.insert(2, {"name": "Gender"})
            
        if self.has_audio:
            noun_fields.append({"name": "WordAudio"})
            noun_fields.append({"name": "SentenceAudio"})
            noun_fields.append({"name": "ExampleAudio"})
        
        # Front template with audio buttons
        noun_front = """
        <div class="word">{{Word}}</div>
        {{#WordAudio}}<div class="audio-btn">[sound:{{WordAudio}}]</div>{{/WordAudio}}
        <div class="sentence">{{Sentence}}</div>
        {{#SentenceAudio}}<div class="audio-btn">[sound:{{SentenceAudio}}]</div>{{/SentenceAudio}}
        """
        
        # Back template with audio buttons
        noun_back = """
        {{FrontSide}}
        <hr id='answer'>
        """
        
        if self.has_gender:
            noun_back += """<div class="gender">{{Gender}}</div>"""
        
        noun_back += """
        <div class="definition">{{Definition}}</div>
        <div class="example">{{Example}}</div>
        {{#ExampleAudio}}<div class="audio-btn">[sound:{{ExampleAudio}}]</div>{{/ExampleAudio}}
        """
        
        noun_templates = [
            {
                "name": "Noun Card",
                "qfmt": noun_front,
                "afmt": noun_back
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
        
        if self.has_audio:
            verb_fields.append({"name": "WordAudio"})
            verb_fields.append({"name": "SentenceAudio"})
            verb_fields.append({"name": "ExampleAudio"})
        
        # Front template with audio buttons
        verb_front = """
        <div class="word">{{Word}}</div>
        {{#WordAudio}}<div class="audio-btn">[sound:{{WordAudio}}]</div>{{/WordAudio}}
        <div class="sentence">{{Sentence}}</div>
        {{#SentenceAudio}}<div class="audio-btn">[sound:{{SentenceAudio}}]</div>{{/SentenceAudio}}
        """
        
        # Back template with audio buttons
        verb_back = """
        {{FrontSide}}
        <hr id='answer'>
        <div class="conjugation">{{Conjugation}}</div>
        <div class="definition">{{Definition}}</div>
        <div class="example">{{Example}}</div>
        {{#ExampleAudio}}<div class="audio-btn">[sound:{{ExampleAudio}}]</div>{{/ExampleAudio}}
        """
        
        verb_templates = [
            {
                "name": "Verb Card",
                "qfmt": verb_front,
                "afmt": verb_back
            }
        ]
        
        return {
            "Noun Card": CustomModel(
                model_id=1607392311,
                name="NounModel",
                fields=noun_fields,
                templates=noun_templates,
                css=css
            ),
            "Verb Card": CustomModel(
                model_id=1607392312,
                name="VerbModel",
                fields=verb_fields,
                templates=verb_templates,
                css=css
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
            word_audio TEXT,
            sentence_audio TEXT,
            example_audio TEXT,
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
                      prompt: str, response: str, extracted_data: Dict[str, Any],
                      audio_files: Dict[str, str] = None):
        """Store processed card data in the database."""
        try:
            self.cursor.execute("""
            INSERT OR REPLACE INTO processed_sentences 
            (sentence, focus_word, card_type, prompt, response, definition, example, gender, conjugation,
             word_audio, sentence_audio, example_audio)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                sentence, 
                focus_word, 
                card_type, 
                prompt, 
                response, 
                extracted_data.get("definition", ""),
                extracted_data.get("example", ""),
                extracted_data.get("gender", ""),
                extracted_data.get("conjugation", ""),
                audio_files.get("word", "") if audio_files else "",
                audio_files.get("sentence", "") if audio_files else "",
                audio_files.get("example", "") if audio_files else ""
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
        self.lang_code = lang_code
        self.has_gender = self._language_has_gender(lang_code)
        self.has_audio = options.get("audio", True)
        
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
        self.card_models = AnkiCardModels(self.has_gender, self.has_audio)
        
        # Initialize audio manager if needed
        if self.has_audio:
            self.audio_manager = AudioManager(lang_code)
        else:
            self.audio_manager = None
        
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
    
    def generate_audio_files(self, word: str, sentence: str, example: str) -> Dict[str, str]:
        """Generate audio files for word, sentence, and example."""
        if not self.has_audio or not self.audio_manager:
            return {}
        
        audio_files = {}
        
        # Generate audio for the word
        word_audio = self.audio_manager.generate_audio(word, word)
        if word_audio:
            audio_files["word"] = word_audio
        
        # Generate audio for the sentence
        sentence_audio = self.audio_manager.generate_audio(sentence)
        if sentence_audio:
            audio_files["sentence"] = sentence_audio
        
        # Generate audio for the example
        if example:
            example_audio = self.audio_manager.generate_audio(example)
            if example_audio:
                audio_files["example"] = example_audio
        
        return audio_files
    
    def create_anki_note(self, extracted_data: Dict[str, Any], sentence: str, 
                        focus_word: str, card_type: str, audio_files: Dict[str, str] = None) -> genanki.Note:
        """Create an Anki note based on the card type and extracted data."""
        model = self.card_models.get_model(card_type)
        
        if card_type == "Noun Card":
            fields = [focus_word, sentence]
            if self.has_gender:
                fields.append(extracted_data.get("gender", ""))
            fields.extend([extracted_data.get("definition", ""), extracted_data.get("example", "")])
            
            # Add audio fields if available
            if self.has_audio and audio_files:
                fields.append(audio_files.get("word", ""))
                fields.append(audio_files.get("sentence", ""))
                fields.append(audio_files.get("example", ""))
        
        elif card_type == "Verb Card":
            fields = [
                focus_word, 
                sentence, 
                extracted_data.get("conjugation", ""), 
                extracted_data.get("definition", ""), 
                extracted_data.get("example", "")
            ]
            
            # Add audio fields if available
            if self.has_audio and audio_files:
                fields.append(audio_files.get("word", ""))
                fields.append(audio_files.get("sentence", ""))
                fields.append(audio_files.get("example", ""))
        
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
                
                # Generate audio files if needed
                audio_files = {}
                if self.has_audio:
                    example = extracted_data.get("example", "")
                    audio_files = self.generate_audio_files(focus_word, sentence, example)
                
                # Create Anki note
                note = self.create_anki_note(extracted_data, sentence, focus_word, card_type, audio_files)
                notes.append(note)
                
                # Store in database
                self.db.store_card_data(sentence, focus_word, card_type, prompt, response, extracted_data, audio_files)
                
            except Exception as e:
                logger.error(f"Error processing '{focus_word}' in '{sentence}': {e}")
        
        return notes
    
def process_text(self, text: str, progress_callback=None) -> None:
    """Process the entire text and generate Anki cards."""
    sentences = self.split_into_sentences(text)
    total_sentences = len(sentences)
    logger.info(f"Processing {total_sentences} sentences")
    
    # Start a processing job
    job_id = self.db.start_processing_job(self.options['input_file'], total_sentences)
    
    processed_count = 0
    for sentence in sentences:
        try:
            notes = self.process_sentence(sentence)
            for note in notes:
                self.anki_deck.add_note(note)
        except Exception as e:
            logger.error(f"Error processing sentence: {sentence}\n{e}")
            # Continue to next sentence
        processed_count += 1
        if progress_callback:
            progress_callback(processed_count, total_sentences)
        self.db.update_processing_status(job_id, processed_count)
    
    # Update final status
    self.db.update_processing_status(job_id, processed_count, 'COMPLETED')
    
    # Generate Anki package
    if self.has_audio:
        media_files = self.audio_manager.get_audio_files()
    else:
        media_files = []
    
    package = genanki.Package(self.anki_deck, media_files)
    package.write_to_file(self.options['output_file'])
    logger.info(f"Anki deck generated: {self.options['output_file']}")
