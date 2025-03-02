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
            lang_code: Language code for TTS (e.g., 'de' for German)
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
        .domain {
            font-size: 14px;
            color: #95a5a6;
            margin-top: 5px;
        }
        .frequency {
            font-size: 12px;
            color: #7f8c8d;
            margin-top: 2px;
        }
        """
        
        # Noun Model
        noun_fields = [
            {"name": "Word"},
            {"name": "Sentence"},
            {"name": "Definition"},
            {"name": "Example"},
            {"name": "Domain"},
            {"name": "Frequency"}
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
        <div class="domain">{{Domain}}</div>
        <div class="frequency">Frequency rank: {{Frequency}}</div>
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
            {"name": "Example"},
            {"name": "Domain"},
            {"name": "Frequency"}
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
        <div class="domain">{{Domain}}</div>
        <div class="frequency">Frequency rank: {{Frequency}}</div>
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
            domain TEXT,
            frequency INTEGER,
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
                      domain: str = "", frequency: int = 0,
                      audio_files: Dict[str, str] = None):
        """Store processed card data in the database."""
        try:
            self.cursor.execute("""
            INSERT OR REPLACE INTO processed_sentences 
            (sentence, focus_word, card_type, prompt, response, definition, example, gender, conjugation, 
             domain, frequency, word_audio, sentence_audio, example_audio)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
                domain,
                frequency,
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
    
    def call_with_retry(self, prompt: str, max_retries: int = 3, base_delay: float = 1.0) -> str:
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
        lang_code = options["lang"].split("_")[0]  # e.g., "de" from "de_core_news_sm"
        self.lang_code = lang_code
        self.has_gender = self._language_has_gender(lang_code)
        self.has_audio = options.get("audio", True)
        
        try:
            self.nlp = spacy.load(options["lang"])
            # Add custom processing for German
            if self.lang_code == "de":
                self.add_custom_pos_rules()
            # Add multi-word expression recognition
            self.add_compound_recognition()
        except OSError:
            logger.error(f"spaCy model '{options['lang']}' not found. Please install it first.")
            raise
        
        # Set up OpenAI
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            logger.error("OpenAI API key not found. Set the OPENAI_API_KEY environment variable.")
            raise ValueError("OpenAI API key not found")
        
        # Configure frequency filtering
        self.frequency_threshold = options.get("frequency_threshold", 5000)  # Default: top 5000 words
        self.frequency_lists = self._load_frequency_lists(options.get("frequency_list"))
        
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
        
        # Domain dictionaries
        self.domains = {
            "weather": {
                "terms": {"wetter", "temperatur", "regen", "wind", "sturm", "niederschlag", 
                         "meteorologisch", "feuchtigkeit", "bewölkt", "sonnig", "kalt", "hitze", 
                         "prognose", "grad", "celsius", "druck", "atmosphärisch"},
                "compounds": ["klimawandel", "kaltfront", "hochdruck", "luftmasse",
                             "niederschlagsmenge", "wettervorhersage", "wettersystem",
                             "hitzewelle", "klarer himmel", "starkregen"]
            },
            "travel": {
                "terms": {"reise", "hotel", "tourismus", "buchung", "flug", "zug", "ticket", "fahrschein",
                         "reisepass", "koffer", "gepäck", "urlaub", "ziel", "strand", "berg"},
                "compounds": ["reisebüro", "touristenvisum", "direktflug", "erste klasse",
                             "halbpension", "all-inclusive", "historisches zentrum"]
            }
            # Add more domains as needed
        }
    
    def add_custom_pos_rules(self):
        """Add custom POS tagging rules to handle specific German cases."""
        @spacy.Language.component("german_verb_corrections")
        def german_verb_fixes(doc):
            # Common German weather verbs often mistaken
            weather_verbs = {
                "regnet": "VERB",
                "schneit": "VERB",
                "dämmert": "VERB",
                "dunkelt": "VERB",
                "donnert": "VERB",
                "blitzt": "VERB"
            }
            
            # German verb endings
            verb_endings = [
                "en", "eln", "ern",  # Infinitive
                "e", "st", "t", "en", "et", "en",  # Present tense
                "te", "test", "te", "ten", "tet", "ten"  # Simple past
            ]
            
            for token in doc:
                # Check specific weather verbs
                if token.text.lower() in weather_verbs:
                    token.pos_ = weather_verbs[token.text.lower()]
                
                # Check verb endings for potential verbs tagged as nouns
                elif token.pos_ == "NOUN":
                    for ending in verb_endings:
                        if token.text.lower().endswith(ending) and len(token.text) > len(ending) + 1:
                            # Basic check to avoid false positives
                            if len(token.text) > 3:  # Avoid very short words
                                token.pos_ = "VERB"
                                break
            
            return doc
        
        # Add to pipeline
        if "german_verb_corrections" not in self.nlp.pipe_names:
            self.nlp.add_pipe("german_verb_corrections", before="parser")
    
    def add_compound_recognition(self):
        """Add compound noun recognition to spaCy pipeline."""
        @spacy.Language.component("compound_recognizer")
        def recognize_compounds(doc):
            # Start with common compounds for the language
            compounds = []
            
            # Add language-specific common compounds
            if self.lang_code == "de":  # German
                compounds = [
                    "bahnhof", "rathaus", "hauptstadt", "buchhaltung", "kindergarten", 
                    "tageszeitung", "lebensmittel", "arbeitsplatz", "krankenhaus",
                    "geburtsurkunde", "jahreszeit", "staatsangehörigkeit"
                ]
            elif self.lang_code == "fr":  # French
                compounds = [
                    "pomme de terre", "chemin de fer", "carte de crédit",
                    "salle de bain", "coup d'état", "arc-en-ciel"
                ]
            
            # Add domain-specific compounds from all domains
            for domain_name, domain_data in self.domains.items():
                compounds.extend(domain_data.get("compounds", []))
            
            # Find compounds in the text
            for compound in compounds:
                compound_tokens = self.nlp(compound)
                compound_length = len(compound_tokens)
                
                for i in range(len(doc) - compound_length + 1):
                    if doc[i:i+compound_length].text.lower() == compound.lower():
                        # Mark tokens as part of compound
                        with doc.retokenize() as retokenizer:
                            retokenizer.merge(doc[i:i+compound_length])
            
            return doc
        
        # Add to pipeline
        if "compound_recognizer" not in self.nlp.pipe_names:
            self.nlp.add_pipe("compound_recognizer", last=True)
    
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
    
    def _load_frequency_lists(self, frequency_file: Optional[str]) -> Dict[str, int]:
        """Load word frequency information."""
        frequencies = {}
        
        # Default frequency lists by language
        default_lists = {
            "de": "german_frequency.txt",
            "es": "spanish_frequency.txt",
            "fr": "french_frequency.txt",
            "en": "english_frequency.txt",
            # Add more languages as needed
        }
        
        # If no file provided, use default for the language
        if not frequency_file and self.lang_code in default_lists:
            frequency_file = os.path.join(os.path.dirname(__file__), "data", default_lists[self.lang_code])
        
        if frequency_file and os.path.exists(frequency_file):
            try:
                with open(frequency_file, 'r', encoding='utf-8') as f:
                    # Expected format: word,rank per line
                    for i, line in enumerate(f):
                        parts = line.strip().split(',')
                        if len(parts) >= 2:
                            word, rank = parts[0], int(parts[1])
                            frequencies[word.lower()] = rank
                        else:
                            # If just a list of words, assign rank by line number
                            frequencies[line.strip().lower()] = i + 1
                logger.info(f"Loaded {len(frequencies)} word frequencies")
            except Exception as e:
                logger.warning(f"Error loading frequency file: {e}")
        
        return frequencies
    
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
    
    def detect_domain(self, text: str) -> str:
        """Detect the domain of the text based on keyword presence."""
        text_lower = text.lower()
        
        # Count terms from each domain
        domain_scores = {}
        for domain, domain_data in self.domains.items():
            terms = domain_data.get("terms", set())
            score = sum(1 for term in terms if term in text_lower)
            if score > 0:
                domain_scores[domain] = score
        
        # Return the domain with the highest score
        if domain_scores:
            max_domain = max(domain_scores.items(), key=lambda x: x[1])
            if max_domain[1] >= 2:  # Require at least 2 terms to determine domain
                return max_domain[0]
        
        return "general"  # Default domain
    
    def identify_focus_elements(self, doc: spacy.tokens.Doc, domain: str) -> List[Dict[str, Any]]:
        """Identify potential elements to focus on with domain awareness and frequency filtering."""
        elements = []
        domain_data = self.domains.get(domain, {})
        domain_terms = domain_data.get("terms", set())
        
        for token in doc:
            # Skip common words, stop words, and known words
            if token.is_stop or token.lemma_.lower() in self.known_words:
                continue
            
            # Get frequency information
            word_frequency = self.frequency_lists.get(token.lemma_.lower(), float('inf'))
            is_common_enough = word_frequency <= self.frequency_threshold
            
            # Check if it's a domain-specific term
            is_domain_term = token.lemma_.lower() in domain_terms
            
            # Determine if we should process this token
            should_process = (is_common_enough or is_domain_term)
            
            if should_process and token.pos_ in ["NOUN", "PROPN"]:
                elements.append({
                    "text": token.text,
                    "pos": token.pos_,
                    "lemma": token.lemma_,
                    "card_type": "Noun Card",
                    "domain": domain,
                    "frequency": word_frequency,
                    "is_domain_term": is_domain_term
                })
            elif should_process and token.pos_ == "VERB":
                elements.append({
                    "text": token.text,
                    "pos": token.pos_,
                    "lemma": token.lemma_,
                    "card_type": "Verb Card",
                    "domain": domain,
                    "frequency": word_frequency,
                    "is_domain_term": is_domain_term
                })
        
        # Sort by frequency to prioritize common words
        elements.sort(key=lambda x: x.get("frequency", float('inf')))
        
        return elements
    
    def generate_llm_prompt(self, element: Dict[str, Any], sentence: str) -> str:
        """Create a structured prompt for the LLM with domain awareness."""
        card_type = element["card_type"]
        focus_word = element["text"]
        domain = element.get("domain", "general")
        is_domain_term = element.get("is_domain_term", False)
        
        # Create base prompt
        prompt = f"I'm learning German and need help with the {element['pos'].lower()} '{focus_word}' from this sentence: '{sentence}'."
        
        # Add domain context if it's a domain term
        if domain != "general" and is_domain_term:
            prompt += f" This is a term related to {domain}."
        
        # Structure based on card type
        if card_type == "Noun Card":
            if self.has_gender:
                prompt += f" Please provide the following information in JSON format:\n{{\n  \"gender\": \"[masculine/feminine/neuter]\",\n  \"definition\": \"[simple definition in English]\",\n  \"example\": \"[example sentence using this word in a different context]\"\n}}"
            else:
                prompt += f" Please provide the following information in JSON format:\n{{\n  \"definition\": \"[simple definition in English]\",\n  \"example\": \"[example sentence using this word in a different context]\"\n}}"
        elif card_type == "Verb Card":
            prompt += f" Please provide the following information in JSON format:\n{{\n  \"conjugation\": \"[conjugation in present tense, first-person singular]\",\n  \"definition\": \"[simple definition in English]\",\n  \"example\": \"[example sentence using this word in a different context]\"\n}}"
        
        # Add instruction for proper response format
        prompt += "\n\nPlease respond ONLY with the JSON object containing the requested information. Do not include any additional text before or after the JSON."
        
        # Emphasize completeness
        prompt += "\n\nIMPORTANT: Please include ALL the requested fields in your response, even if you're not sure about some values."
        
        return prompt
    
    def extract_json_from_response(self, response: str) -> Dict[str, Any]:
        """Extract JSON data from LLM response with robust fallback."""
        import json
        import re
        
        # Try to find JSON object in the response
        match = re.search(r'{.*?}', response, re.DOTALL)
        if match:
            try:
                # Parse the JSON object
                json_str = match.group(0)
                data = json.loads(json_str)
                
                # Log if it's missing any expected fields
                logger.debug(f"Extracted data: {data}")
                
                return data
            except json.JSONDecodeError:
                logger.warning(f"Failed to parse JSON from response: {response}")
        
        # More robust fallback extraction for non-JSON responses
        extracted = {}
        
        # Extract fields using more flexible pattern matching
        patterns = {
            "gender": [r'"gender":\s*"([^"]+)"', r'gender:\s*([^\n]+)', r'Gender:\s*([^\n]+)'],
            "definition": [r'"definition":\s*"([^"]+)"', r'definition:\s*([^\n]+)', r'Definition:\s*([^\n]+)'],
            "example": [r'"example":\s*"([^"]+)"', r'example:\s*([^\n]+)', r'Example:\s*([^\n]+)'],
            "conjugation": [r'"conjugation":\s*"([^"]+)"', r'conjugation:\s*([^\n]+)', r'Conjugation:\s*([^\n]+)']
        }
        
        for field, field_patterns in patterns.items():
            for pattern in field_patterns:
                match = re.search(pattern, response, re.IGNORECASE | re.DOTALL)
                if match:
                    extracted[field] = match.group(1).strip()
                    break
        
        # Ensure all fields have defaults
        for field in ["definition", "example", "gender", "conjugation"]:
            if field not in extracted:
                extracted[field] = ""
        
        if not extracted.get("definition") and not extracted.get("example"):
            logger.warning(f"Failed to extract meaningful data from response: {response}")
        
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
                        focus_word: str, card_type: str, domain: str = "general",
                        frequency: int = 0, audio_files: Dict[str, str] = None) -> genanki.Note:
        """Create an Anki note based on the card type and extracted data."""
        model = self.card_models.get_model(card_type)
        
        if card_type == "Noun Card":
            fields = [focus_word, sentence]
            
            if self.has_gender:
                fields.append(extracted_data.get("gender", ""))
                
            fields.extend([
                extracted_data.get("definition", ""),
                extracted_data.get("example", ""),
                domain,
                str(frequency) if frequency < float('inf') else "Unknown"
            ])
            
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
                extracted_data.get("example", ""),
                domain,
                str(frequency) if frequency < float('inf') else "Unknown"
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
        # Parse sentence
        doc = self.nlp(sentence)
        
        # Detect domain
        domain = self.detect_domain(sentence)
        logger.debug(f"Detected domain for sentence: {domain}")
        
        # Find focus elements
        focus_elements = self.identify_focus_elements(doc, domain)
        notes = []
        
        for element in focus_elements:
            focus_word = element["text"]
            card_type = element["card_type"]
            element_domain = element.get("domain", "general")
            frequency = element.get("frequency", float('inf'))
            
            # Skip if already processed
            if self.db.is_sentence_word_processed(sentence, focus_word, card_type):
                logger.debug(f"Skipping already processed: {focus_word} in '{sentence}'")
                continue
            
            try:
                # Generate prompt
                prompt = self.generate_llm_prompt(element, sentence)
                
                # Get LLM response with retry
                response = self.llm.call_with_retry(prompt)
                logger.debug(f"LLM response for '{focus_word}': {response}")
                
                # Extract data from response
                extracted_data = self.extract_json_from_response(response)
                
                # Validate extracted data
                if not extracted_data or (not extracted_data.get("definition") and not extracted_data.get("example")):
                    logger.warning(f"No useful data extracted for '{focus_word}' in sentence: '{sentence}'")
                    continue
                
                # Generate audio files if needed
                audio_files = {}
                if self.has_audio:
                    example = extracted_data.get("example", "")
                    audio_files = self.generate_audio_files(focus_word, sentence, example)
                
                # Create note
                note = self.create_anki_note(
                    extracted_data, sentence, focus_word, card_type, 
                    domain=element_domain, frequency=frequency, audio_files=audio_files
                )
                notes.append(note)
                
                # Store in database
                self.db.store_card_data(
                    sentence, focus_word, card_type, prompt, response, 
                    extracted_data, domain=element_domain, frequency=frequency,
                    audio_files=audio_files
                )
                
            except Exception as e:
                logger.error(f"Error processing '{focus_word}' in '{sentence}': {e}")
                # Continue with next element
        
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
                # Skip very short sentences
                if len(sentence.split()) < 3:
                    logger.debug(f"Skipping short sentence: '{sentence}'")
                    continue
                
                logger.info(f"Processing sentence {i+1}/{total_sentences}: '{sentence}'")
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
        output_file = self.options.get("output", "German_Learning.apkg")
        
        # Prepare media files if audio is enabled
        if self.has_audio and self.audio_manager:
            media_files = self.audio_manager.get_audio_files()
            package = genanki.Package(self.anki_deck, media_files)
        else:
            package = genanki.Package(self.anki_deck)
            
        package.write_to_file(output_file)
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
    parser.add_argument("--lang", required=True, help="spaCy language model (e.g., de_core_news_sm for German)")
    parser.add_argument("--input", required=True, help="Input text file")
    parser.add_argument("--output", default="German_Learning.apkg", help="Output Anki deck file name")
    parser.add_argument("--known-words", help="File containing known words to exclude")
    parser.add_argument("--frequency-list", help="File containing word frequencies")
    parser.add_argument("--frequency-threshold", type=int, default=5000, 
                       help="Only process words in the top N most frequent words (default: 5000)")
    parser.add_argument("--deck-name", default="German Learning Deck", help="Name for the Anki deck")
    parser.add_argument("--audio", action="store_true", default=True, help="Generate audio for words and sentences")
    parser.add_argument("--no-audio", dest="audio", action="store_false", help="Disable audio generation")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose logging")
    parser.add_argument("--debug", "-d", action="store_true", help="Enable debug logging")
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
    if args.debug:
        logger.setLevel(logging.DEBUG)
    elif args.verbose:
        logger.setLevel(logging.INFO)
    else:
        logger.setLevel(logging.WARNING)
    
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
        
        print(f"\nSuccess! Anki deck created: {args.output}")
        
    except KeyboardInterrupt:
        print("\nProcess interrupted by user. Run with --resume to continue later.")
        return 1
    except Exception as e:
        logger.error(f"Error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    main()
