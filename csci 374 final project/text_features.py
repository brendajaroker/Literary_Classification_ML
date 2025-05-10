import pandas as pd
import requests
import re
import logging
import spacy
import nltk
from nltk.corpus import wordnet, stopwords
from nltk.tokenize import sent_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.feature_selection import SelectKBest, chi2, mutual_info_classif, f_classif
from gensim import corpora
from gensim.models import LdaModel, Word2Vec, FastText
from sklearn.decomposition import NMF
from gensim.models.phrases import Phrases, Phraser
import numpy as np
from collections import Counter
import time
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
import pickle

# Download NLTK data
nltk.download('wordnet')
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger()

# Load SpaCy model - use medium model for better language understanding
try:
    nlp = spacy.load("en_core_web_md", disable=["ner"])
except OSError:
    print("Downloading en_core_web_md...")
    os.system("python -m spacy download en_core_web_md")
    nlp = spacy.load("en_core_web_md", disable=["ner"])

# Load stop words with custom literary additions
stop_words = set(stopwords.words('english'))
# Remove some stopwords that could be stylistically important
for word in ['no', 'not', 'very', 'few', 'more', 'most', 'against', 'own']:
    if word in stop_words:
        stop_words.remove(word)

lemmatizer = WordNetLemmatizer()

# Enhanced movement-specific features with more comprehensive term lists
# Enhanced movement-specific features with more comprehensive term lists
movement_specific_features = {
    "Romanticism": [
        # Original terms
        "forlorn", "miserly", "passion", "heart", "sublime", "nature", "soul", "emotion", 
        "beauty", "imagination", "dream", "vision", "melancholy", "solitude", "wild", 
        "supernatural", "gothic", "quest", "pastoral", "medieval", "individualism",
        "spontaneous", "liberty", "innocence", "nostalgic", "mysterious", "wanderer",
        # Additional terms
        "spiritual", "transcendent", "awe", "rapture", "reverie", "moonlight", "divine",
        "eternal", "infinite", "sublime", "idyllic", "myth", "legend", "fantasy", "fairy",
        "sentiment", "feeling", "passion", "yearning", "longing", "exotic", "faraway",
        "primal", "authentic", "genuine", "untamed", "primitive", "folk", "ballad",
        "freedom", "revolution", "visionary", "dreamer", "madness", "despair", "ecstasy",
        "genius", "prodigy", "creativity", "inspiration", "muse"
    ],
    
    "Realism": [
        # Original terms
        "palpable", "manifest", "society", "class", "work", "money", "family", "everyday", 
        "objective", "accurate", "ordinary", "common", "realistic", "detailed", "precise",
        "practical", "factual", "empirical", "mundane", "middle-class", "city", "urban",
        "industrial", "scientific", "documentary", "literal", "observable", "concrete",
        # Additional terms
        "actual", "real", "authentic", "genuine", "natural", "truthful", "unembellished",
        "unvarnished", "straightforward", "direct", "plain", "unadorned", "unpretentious",
        "unromantic", "unsentimental", "prosaic", "matter-of-fact", "workday", "humdrum",
        "commonplace", "conventional", "normal", "regular", "standard", "average", "typical",
        "customary", "habitual", "routine", "hourly", "daily", "weekly", "monthly", "yearly",
        "labor", "trade", "profession", "job", "employment", "industry", "commerce", "business",
        "transaction", "exchange", "banking", "economics", "market", "domestic", "household"
    ],
    
    "Naturalism": [
        # Original terms
        "fate", "instinct", "survival", "poverty", "force", "determinism", "environment", 
        "heredity", "struggle", "brutal", "harsh", "primitive", "animal", "darwinian", 
        "savage", "scientific", "experiment", "observation", "lower-class", "squalor", 
        "disease", "violence", "pessimism", "inevitable", "biological", "evolutionary",
        # Additional terms
        "amoral", "indifferent", "mechanical", "physiological", "genetic", "psychological",
        "depravity", "degeneration", "corruption", "vice", "alcoholism", "addiction", "slum",
        "tenement", "filth", "dirt", "grime", "sewage", "stench", "sweat", "labor", "toil",
        "drudgery", "beast", "brute", "ape", "primate", "organism", "species", "specimen",
        "predator", "prey", "victim", "natural selection", "adaptation", "degeneration",
        "entropy", "devolution", "regression", "atavism", "clinical", "pathological",
        "antisocial", "criminal", "delinquent", "vagrant", "proletariat", "underclass"
    ],
    
    "Gothicism": [
        # Original terms
        "shadow", "grave", "fear", "dark", "ghost", "horror", "supernatural", "mysterious", 
        "castle", "ruin", "ancient", "terror", "death", "decay", "eerie", "monstrous", 
        "macabre", "grotesque", "nocturnal", "haunted", "spectral", "melancholy", "doom",
        "gloomy", "curse", "omen", "diabolic", "dread", "foreboding", "sinister",
        # Additional terms
        "abbey", "cathedral", "monastery", "crypt", "dungeon", "chamber", "tower", "turret",
        "labyrinth", "passageway", "corridor", "staircase", "cellar", "attic", "mansion",
        "ancestral", "inheritance", "bloodline", "descendant", "nobility", "aristocracy",
        "villain", "tyrant", "madman", "lunatic", "maniac", "fiend", "demon", "devil",
        "specter", "apparition", "phantom", "wraith", "revenant", "banshee", "ghoul",
        "vampire", "werewolf", "monster", "creature", "beast", "abomination", "undead",
        "coffin", "tomb", "catacombs", "graveyard", "cemetery", "mausoleum", "sepulcher",
        "wail", "scream", "shriek", "howl", "moan", "groan", "whisper", "hiss"
    ],
    
    "Transcendentalism": [
        # Original terms
        "spirit", "divine", "truth", "individual", "universe", "self-reliance", "intuition", 
        "consciousness", "oversoul", "nature", "harmony", "moral", "simplicity", "spiritual", 
        "inherent", "goodness", "wilderness", "inherent", "transcend", "enlightenment", "intellect", 
        "meditation", "ideal", "reform", "higher", "eternal", "transparent", "reverence",
        # Additional terms
        "emerson", "thoreau", "walden", "concord", "brook farm", "dial", "pantheism",
        "universal", "cosmic", "unity", "oneness", "whole", "integrity", "authentic", "genuine",
        "transparent", "sincerity", "candor", "contemplation", "reflection", "solitude",
        "hermit", "pond", "lake", "forest", "woods", "clearing", "path", "journey", "quest",
        "seeking", "introspection", "self-examination", "self-knowledge", "insight", "wisdom",
        "perception", "comprehension", "understanding", "realization", "awakening", "epiphany",
        "revelation", "enlightenment", "illumination", "clarity", "veil", "beyond", "above"
    ],
    
    "Modernism": [
        # Original terms
        "fragment", "chaos", "memory", "time", "consciousness", "stream", "alienation", 
        "interior", "subjective", "psychological", "disillusion", "existential", "cynical",
        "abstract", "innovation", "experiment", "epiphany", "urban", "flux", "relativity",
        "discontinuity", "skepticism", "absurd", "irony", "juxtaposition", "montage",
        # Additional terms
        "joyce", "woolf", "eliot", "pound", "faulkner", "hemingway", "kafka", "proust",
        "metropolis", "city", "apartment", "skyscraper", "automobile", "machine", "industry",
        "technology", "progress", "science", "freud", "psychoanalysis", "unconscious", "dream",
        "symbol", "metaphor", "allusion", "intertextuality", "pastiche", "collage", "palimpsest",
        "world war", "trench", "soldier", "veteran", "shellshock", "trauma", "wound", "scar",
        "disillusionment", "despair", "emptiness", "void", "nothingness", "meaninglessness",
        "isolation", "loneliness", "estrangement", "outsider", "exile", "rootless", "displaced",
        "cosmopolitan", "international", "expatriate", "bohemian", "avant-garde", "manifesto",
        "movement", "rupture", "break", "tradition", "convention", "orthodoxy", "establishment"
    ],
    
    "Renaissance": [
        # Original terms
        "classical", "humanism", "virtue", "rhetoric", "reason", "renaissance", "pagan",
        "greco-roman", "courtly", "sonnet", "wit", "allegory", "platonic", "pastoral",
        "renaissance", "civilization", "worldly", "skeptical", "renaissance", "knowledge",
        "patronage", "learning", "discovery", "rebirth", "harmony", "proportion", "counsel",
        # Additional terms
        "italy", "florence", "venice", "medici", "pope", "machiavelli", "petrarch", "boccaccio",
        "aristotle", "plato", "cicero", "virgil", "horace", "ovid", "seneca", "plutarch",
        "monarchy", "court", "prince", "duke", "nobility", "aristocracy", "merchant", "guild",
        "education", "liberal arts", "trivium", "quadrivium", "grammar", "logic", "rhetoric",
        "arithmetic", "geometry", "astronomy", "music", "university", "academy", "scholar",
        "manuscript", "codex", "printing", "press", "vernacular", "latin", "greek", "translation",
        "oration", "discourse", "dialogue", "debate", "eloquence", "persuasion", "statecraft",
        "governance", "republic", "politics", "diplomacy", "civic", "commonwealth", "citizen"
    ]
}

# Temporal markers by era (approximate publication dates)
temporal_markers = {
    "Renaissance": [
        "thee", "thou", "thy", "hath", "doth", "art", "wert", "hast", "forsooth", "methinks",
        "ye", "thine", "tis", "twas", "whence", "thither", "hither", "wherefore", "verily",
        "maketh", "giveth", "sayeth", "deem", "perchance", "mayhap", "yea", "nay", "prithee"
    ],
    "Romanticism/Gothic": [
        "shall", "upon", "ere", "whilst", "'tis", "thence", "whence", "aught", "naught",
        "sublime", "o'er", "amidst", "betwixt", "betimes", "anon", "oft", "ofttimes", "oft-times",
        "withal", "fain", "athwart", "語言", "thither", "yon", "yonder", "alack", "alas"
    ],
    "Victorian": [
        "should", "would", "ought", "must", "quite", "indeed", "perhaps", "rather",
        "pray", "truly", "certainly", "scarcely", "presently", "particularly", "exceedingly",
        "remarkably", "decidedly", "tolerably", "uncommonly", "excessively", "prodigiously"
    ],
    "Modern": [
        "isn't", "don't", "can't", "won't", "couldn't", "shouldn't", "wouldn't",
        "gonna", "gotta", "wanna", "yeah", "nope", "okay", "ok", "stuff", "thing", "like",
        "whatever", "anyway", "anyhow", "sorta", "kinda", "hell", "damn", "goddamn"
    ]
}

# Function to count syllables in a word with improved accuracy
def count_syllables(word):
    try:
        word = word.lower().strip()
        if not word:
            return 0
        
        special_cases = {
            'the': 1, 'every': 2, 'different': 3, 'difficult': 3, 'beautiful': 3,
            'absolutely': 4, 'area': 2, 'business': 2, 'camera': 3, 'poem': 2,
            'poems': 1, 'through': 1, 'thoroughly': 3, 'thought': 1, 'variable': 4,
            'society': 3, 'science': 2
        }
        
        if word in special_cases:
            return special_cases[word]
        
        if word.endswith('es') and not word.endswith(('ies', 'ves', 'oes')):
            word = word[:-2]
        elif word.endswith('ed') and not word.endswith('ied'):
            word = word[:-2]
        
        vowels = 'aeiouy'
        syllable_count = 0
        prev_char_was_vowel = False
        
        for i, char in enumerate(word):
            is_vowel = char in vowels
            
            if is_vowel and prev_char_was_vowel and i > 0:
                if word[i-1:i+1] not in ['io', 'ia', 'eo', 'ua', 'uo', 'ie']:
                    syllable_count += 1
            elif is_vowel and not prev_char_was_vowel:
                syllable_count += 1
                
            prev_char_was_vowel = is_vowel
        
        if word.endswith('e') and not word.endswith(('le', 'ie', 'ee', 'oe')):
            syllable_count = max(1, syllable_count - 1)
        
        if word.endswith('le') and len(word) > 2 and word[-3] not in vowels:
            syllable_count += 1
            
        if word.endswith('y') and len(word) > 1 and word[-2] not in vowels:
            syllable_count += 1
        
        return max(1, syllable_count)
    except Exception as e:
        logger.error(f"Error counting syllables for word '{word}': {e}")
        return 1

# Enhanced function to calculate readability metrics
def calculate_readability_metrics(text):
    try:
        sentences = sent_tokenize(text[:100000])
        if not sentences:
            return {
                "flesch_reading_ease": 0,
                "gunning_fog": 0,
                "smog_index": 0,
                "automated_readability_index": 0,
                "coleman_liau_index": 0
            }
        
        words = []
        for sentence in sentences:
            words.extend([word for word in sentence.split() if word.strip()])
        
        if not words:
            return {
                "flesch_reading_ease": 0,
                "gunning_fog": 0,
                "smog_index": 0,
                "automated_readability_index": 0,
                "coleman_liau_index": 0
            }
        
        word_count = len(words)
        syllable_counts = [count_syllables(word) for word in words]
        complex_words = sum(1 for count in syllable_counts if count > 2)
        
        total_chars = sum(len(word) for word in words)
        
        avg_sentence_length = word_count / len(sentences)
        avg_syllables_per_word = sum(syllable_counts) / word_count
        
        flesch = 206.835 - (1.015 * avg_sentence_length) - (84.6 * avg_syllables_per_word)
        gunning_fog = 0.4 * (avg_sentence_length + (100 * complex_words / word_count))
        smog = 1.043 * np.sqrt(30 * complex_words / len(sentences)) + 3.1291
        ari = 4.71 * (total_chars / word_count) + 0.5 * (word_count / len(sentences)) - 21.43
        L = (total_chars / word_count) * 100
        S = (len(sentences) / word_count) * 100
        coleman_liau = 0.0588 * L - 0.296 * S - 15.8
        
        return {
            "flesch_reading_ease": flesch,
            "gunning_fog": gunning_fog,
            "smog_index": smog,
            "automated_readability_index": ari,
            "coleman_liau_index": coleman_liau
        }
    except Exception as e:
        logger.error(f"Error in readability calculation: {e}")
        return {
            "flesch_reading_ease": 0,
            "gunning_fog": 0,
            "smog_index": 0,
            "automated_readability_index": 0,
            "coleman_liau_index": 0
        }

# Function to download and clean Gutenberg text
def download_gutenberg_text(gid):
    urls = [
        f"https://www.gutenberg.org/files/{gid}/{gid}-0.txt",
        f"https://www.gutenberg.org/files/{gid}/{gid}.txt",
        f"https://www.gutenberg.org/cache/epub/{gid}/pg{gid}.txt"
    ]
    
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    
    for attempt in range(5):
        for url in urls:
            try:
                response = requests.get(url, timeout=20, headers=headers)
                response.raise_for_status()
                text = response.text
                
                start_markers = [
                    "*** START OF THIS PROJECT GUTENBERG EBOOK",
                    "***START OF THE PROJECT GUTENBERG EBOOK",
                    "*** START OF THE PROJECT GUTENBERG EBOOK",
                    "***START OF THIS PROJECT GUTENBERG EBOOK",
                    "START OF THE PROJECT GUTENBERG EBOOK",
                    "START OF THIS PROJECT GUTENBERG EBOOK",
                    "*** START OF THE PROJECT GUTENBERG",
                    "*** START OF THIS PROJECT GUTENBERG",
                    "Produced by",
                    "Transcribed from",
                    "This eBook was produced by"
                ]
                
                end_markers = [
                    "*** END OF THIS PROJECT GUTENBERG EBOOK",
                    "***END OF THE PROJECT GUTENBERG EBOOK",
                    "*** END OF THE PROJECT GUTENBERG EBOOK",
                    "***END OF THIS PROJECT GUTENBERG EBOOK",
                    "END OF THE PROJECT GUTENBERG EBOOK",
                    "END OF THIS PROJECT GUTENBERG EBOOK",
                    "*** END OF THE PROJECT GUTENBERG",
                    "*** END OF THIS PROJECT GUTENBERG",
                    "End of Project Gutenberg",
                    "End of the Project Gutenberg"
                ]
                
                start_idx = -1
                for marker in start_markers:
                    pos = text.find(marker)
                    if pos != -1:
                        line_end = text.find('\n', pos)
                        if line_end != -1:
                            start_idx = line_end + 1
                        else:
                            start_idx = pos + len(marker)
                        break
                
                end_idx = -1
                for marker in end_markers:
                    pos = text.find(marker)
                    if pos != -1:
                        end_idx = pos
                        break
                
                if start_idx != -1 and end_idx != -1:
                    text = text[start_idx:end_idx].strip()
                elif start_idx != -1:
                    text = text[start_idx:].strip()
                elif end_idx != -1:
                    text = text[:end_idx].strip()
                
                if len(text) > 500000:
                    chapter_markers = [
                        "\nCHAPTER I", "\nCHAPTER 1", "\nI.", "\n1.",
                        "\nPREFACE", "\nINTRODUCTION", "\nFOREWORD"
                    ]
                    for marker in chapter_markers:
                        pos = text.find(marker)
                        if pos > 1000:
                            text = text[pos:]
                            break
                
                text = re.sub(r'\r\n', '\n', text)
                text = re.sub(r'\n{3,}', '\n\n', text)
                text = re.sub(r'_+', '', text)
                text = re.sub(r'\[.*?\]', '', text)
                text = re.sub(r'[\x00-\x1F\x7F-\x9F]', '', text)
                
                if len(text.split()) > 1000:
                    return text
                
            except requests.RequestException as e:
                logger.warning(f"Failed to download {url}: {e}")
        
        sleep_time = 2 * (2 ** attempt)
        logger.warning(f"Retry {attempt+1}/5, waiting {sleep_time} seconds...")
        time.sleep(sleep_time)
    
    logger.error(f"No valid text found for GutenbergID {gid} after 5 attempts")
    return None

# Complete extract_syntactic_features
def extract_syntactic_features(text):
    sentences = sent_tokenize(text[:200000])
    if not sentences:
        return {}
    
    sentence_lengths = [len(sent.split()) for sent in sentences]
    
    avg_sentence_length = np.mean(sentence_lengths) if sentence_lengths else 0
    var_sentence_length = np.var(sentence_lengths) if sentence_lengths else 0
    max_sentence_length = max(sentence_lengths) if sentence_lengths else 0
    median_sentence_length = np.median(sentence_lengths) if sentence_lengths else 0
    
    sample_size = min(len(text), 100000)
    doc = nlp(text[:sample_size])
    
    text_length = max(len(text), 1)
    punctuation_counts = {
        "exclamation_marks": text.count("!") / (text_length / 10000),
        "question_marks": text.count("?") / (text_length / 10000),
        "semicolons": text.count(";") / (text_length / 10000),
        "colons": text.count(":") / (text_length / 10000),
        "ellipses": (text.count("...") + text.count(". . .")) / (text_length / 10000),
        "dashes": (text.count("—") + text.count("--") + text.count("–")) / (text_length / 10000),
        "parentheses": (text.count("(") + text.count(")")) / (text_length / 10000),
        "quotation_marks": (text.count('"') + text.count('"') + text.count('"')) / (text_length / 10000),
        "apostrophes": (text.count("'") + text.count("'") + text.count("'")) / (text_length / 10000)
    }
    
    pos_counts = Counter(token.pos_ for token in doc)
    total_tokens = sum(pos_counts.values()) or 1
    
    pos_features = {
        "noun_freq": pos_counts.get("NOUN", 0) / total_tokens * 100,
        "proper_noun_freq": pos_counts.get("PROPN", 0) / total_tokens * 100,
        "adj_freq": pos_counts.get("ADJ", 0) / total_tokens * 100,
        "adv_freq": pos_counts.get("ADV", 0) / total_tokens * 100,
        "verb_freq": pos_counts.get("VERB", 0) / total_tokens * 100,
        "det_freq": pos_counts.get("DET", 0) / total_tokens * 100,
        "conj_freq": (pos_counts.get("CONJ", 0) + pos_counts.get("CCONJ", 0) + pos_counts.get("SCONJ", 0)) / total_tokens * 100,
        "intj_freq": pos_counts.get("INTJ", 0) / total_tokens * 100,
        "prep_freq": pos_counts.get("ADP", 0) / total_tokens * 100,
        "pron_freq": pos_counts.get("PRON", 0) / total_tokens * 100
    }
    
    verb_tokens = [token for token in doc if token.pos_ == "VERB"]
    verb_features = {}
    if verb_tokens:
        past_verbs = sum(1 for t in verb_tokens if t.tag_ in ["VBD", "VBN"])
        present_verbs = sum(1 for t in verb_tokens if t.tag_ in ["VBZ", "VBP", "VBG"])
        future_hints = sum(1 for t in doc if t.lower_ in ["will", "shall", "going"])
        
        verb_features = {
            "past_tense_ratio": past_verbs / len(verb_tokens) * 100,
            "present_tense_ratio": present_verbs / len(verb_tokens) * 100,
            "future_hints_ratio": future_hints / total_tokens * 100
        }
    else:
        verb_features = {
            "past_tense_ratio": 0,
            "present_tense_ratio": 0,
            "future_hints_ratio": 0
        }
    
    words = [token.text for token in doc if token.is_alpha]
    syllable_counts = [count_syllables(word) for word in words]
    avg_syllables_per_word = np.mean(syllable_counts) if syllable_counts else 0
    polysyllabic_ratio = sum(1 for count in syllable_counts if count > 2) / (len(syllable_counts) or 1) * 100
    
    narrative_markers = {
        "temporal_adverbs": sum(1 for token in doc if token.lower_ in [
            "when", "then", "whilst", "after", "before", "now", "soon", "later",
            "early", "previously", "subsequently", "immediately", "presently",
            "yesterday", "today", "tomorrow", "once", "again", "already", "yet",
            "still", "always", "never", "ever", "often", "seldom", "rarely"
        ]) / total_tokens * 100,
        "locative_preps": sum(1 for token in doc if token.lower_ in [
            "in", "at", "on", "from", "to", "toward", "between", "among",
            "within", "throughout", "around", "across", "behind", "beyond",
            "above", "below", "beside", "under", "over", "near", "far",
            "inside", "outside", "along", "against"
        ]) / total_tokens * 100,
        "causal_connectives": sum(1 for token in doc if token.lower_ in [
            "because", "since", "therefore", "thus", "hence", "consequently",
            "as", "due", "owing", "resulting"
        ]) / total_tokens * 100,
        "dialogue_markers": sum(1 for token in doc if token.lower_ in [
            "said", "say", "says", "replied", "asked", "exclaimed", "whispered",
            "shouted", "cried", "murmured", "muttered", "answered"
        ]) / total_tokens * 100
    }
    
    return {
        "avg_sentence_length": avg_sentence_length,
        "var_sentence_length": var_sentence_length,
        "max_sentence_length": max_sentence_length,
        "median_sentence_length": median_sentence_length,
        **punctuation_counts,
        **pos_features,
        **verb_features,
        "avg_syllables_per_word": avg_syllables_per_word,
        "polysyllabic_ratio": polysyllabic_ratio,
        **narrative_markers
    }

# Text preprocessing for feature extraction
def preprocess_text(text, for_vectorizer=True):
    """
    Preprocess text for TF-IDF, topic modeling, or embeddings.
    
    Args:
        text: Input text
        for_vectorizer: If True, return a single string for TF-IDF; else, return list of tokens
    
    Returns:
        Preprocessed text or tokens
    """
    try:
        doc = nlp(text[:100000])  # Limit for performance
        tokens = []
        for token in doc:
            if token.is_alpha and not token.is_stop and token.text.lower() not in stop_words:
                lemma = lemmatizer.lemmatize(token.text.lower())
                if len(lemma) > 2:  # Ignore very short tokens
                    tokens.append(lemma)
        
        if for_vectorizer:
            return " ".join(tokens)
        return tokens
    except Exception as e:
        logger.error(f"Error preprocessing text: {e}")
        return "" if for_vectorizer else []

# Extract TF-IDF features
def extract_tfidf_features(texts, max_features=1000):
    """
    Extract TF-IDF features from a list of texts.
    
    Args:
        texts: List of preprocessed texts
        max_features: Maximum number of features to keep
    
    Returns:
        Feature matrix, feature names, vectorizer
    """
    try:
        vectorizer = TfidfVectorizer(max_features=max_features, ngram_range=(1, 2), min_df=2)
        X = vectorizer.fit_transform(texts)
        return X.toarray(), vectorizer.get_feature_names_out(), vectorizer
    except Exception as e:
        logger.error(f"Error in TF-IDF extraction: {e}")
        return np.zeros((len(texts), max_features)), [], None

# Extract LDA topic features
def extract_lda_features(texts, num_topics=10):
    """
    Extract LDA topic probabilities as features.
    
    Args:
        texts: List of tokenized texts
        num_topics: Number of topics
    
    Returns:
        Topic probability matrix
    """
    try:
        dictionary = corpora.Dictionary(texts)
        dictionary.filter_extremes(no_below=2, no_above=0.5)
        corpus = [dictionary.doc2bow(text) for text in texts]
        lda = LdaModel(corpus, num_topics=num_topics, id2word=dictionary, passes=10, random_state=42)
        topic_probs = np.zeros((len(texts), num_topics))
        for i, doc in enumerate(corpus):
            topics = lda[doc]
            for topic_id, prob in topics:
                topic_probs[i, topic_id] = prob
        return topic_probs
    except Exception as e:
        logger.error(f"Error in LDA extraction: {e}")
        return np.zeros((len(texts), num_topics))

# Extract NMF topic features
def extract_nmf_features(texts, num_topics=10):
    """
    Extract NMF topic features.
    
    Args:
        texts: List of preprocessed texts
        num_topics: Number of topics
    
    Returns:
        Topic feature matrix
    """
    try:
        vectorizer = TfidfVectorizer(max_features=1000, min_df=2)
        X = vectorizer.fit_transform(texts)
        nmf = NMF(n_components=num_topics, random_state=42)
        W = nmf.fit_transform(X)
        return W
    except Exception as e:
        logger.error(f"Error in NMF extraction: {e}")
        return np.zeros((len(texts), num_topics))

# Extract word embeddings (Word2Vec)
def extract_word2vec_features(texts, vector_size=100):
    """
    Extract mean Word2Vec vectors for each text.
    
    Args:
        texts: List of tokenized texts
        vector_size: Size of embedding vectors
    
    Returns:
        Mean embedding matrix
    """
    try:
        model = Word2Vec(sentences=texts, vector_size=vector_size, window=5, min_count=2, workers=4, seed=42)
        embeddings = np.zeros((len(texts), vector_size))
        for i, tokens in enumerate(texts):
            valid_tokens = [t for t in tokens if t in model.wv]
            if valid_tokens:
                embeddings[i] = np.mean([model.wv[t] for t in valid_tokens], axis=0)
        return embeddings
    except Exception as e:
        logger.error(f"Error in Word2Vec extraction: {e}")
        return np.zeros((len(texts), vector_size))

# Extract movement-specific term frequencies
def extract_movement_term_features(text):
    """
    Count frequencies of movement-specific terms.
    
    Args:
        text: Input text
    
    Returns:
        Dictionary of term frequencies
    """
    features = {}
    doc = nlp(text[:100000])
    tokens = [token.text.lower() for token in doc if token.is_alpha]
    total_tokens = len(tokens) or 1
    
    for movement, terms in movement_specific_features.items():
        term_count = sum(tokens.count(term) for term in terms)
        features[f"{movement}_term_freq"] = term_count / total_tokens * 100
    
    return features

# Extract temporal marker frequencies
def extract_temporal_features(text):
    """
    Count frequencies of temporal markers.
    
    Args:
        text: Input text
    
    Returns:
        Dictionary of marker frequencies
    """
    features = {}
    doc = nlp(text[:100000])
    tokens = [token.text.lower() for token in doc if token.is_alpha]
    total_tokens = len(tokens) or 1
    
    for era, markers in temporal_markers.items():
        marker_count = sum(tokens.count(marker) for marker in markers)
        features[f"{era}_marker_freq"] = marker_count / total_tokens * 100
    
    return features

# Feature selection
def select_features(X, y, feature_names, k=500):
    """
    Select top k features using multiple methods.
    
    Args:
        X: Feature matrix
        y: Labels
        feature_names: List of feature names
        k: Number of features to select
    
    Returns:
        Selected feature matrix, selected feature names
    """
    try:
        selector_chi2 = SelectKBest(chi2, k=k//3)
        selector_mi = SelectKBest(mutual_info_classif, k=k//3)
        selector_f = SelectKBest(f_classif, k=k//3)
        
        X_chi2 = selector_chi2.fit_transform(X, y)
        X_mi = selector_mi.fit_transform(X, y)
        X_f = selector_f.fit_transform(X, y)
        
        chi2_mask = selector_chi2.get_support()
        mi_mask = selector_mi.get_support()
        f_mask = selector_f.get_support()
        
        selected_mask = np.logical_or(np.logical_or(chi2_mask, mi_mask), f_mask)
        selected_features = X[:, selected_mask]
        selected_names = feature_names[selected_mask]
        
        return selected_features, selected_names
    except Exception as e:
        logger.error(f"Error in feature selection: {e}")
        return X, feature_names

# Main function to process texts and extract features
def process_texts(metadata_file, output_file="text_features.csv", max_texts=None):
    """
    Process texts and extract features, saving to CSV.
    
    Args:
        metadata_file: CSV file with GutenbergID, Author, Movement
        output_file: Output CSV file
        max_texts: Maximum number of texts to process (for testing)
    """
    try:
        # Load metadata
        df = pd.read_csv(metadata_file)
        if max_texts:
            df = df[:max_texts]
        
        logger.info(f"Processing {len(df)} texts")
        
        # Download texts in parallel
        texts = []
        with ThreadPoolExecutor(max_workers=10) as executor:
            future_to_gid = {executor.submit(download_gutenberg_text, row['GutenbergID']): row['GutenbergID'] for _, row in df.iterrows()}
            for future in as_completed(future_to_gid):
                gid = future_to_gid[future]
                try:
                    text = future.result()
                    if text:
                        texts.append(text)
                        logger.info(f"Downloaded text for GutenbergID {gid}")
                    else:
                        texts.append("")
                        logger.warning(f"Failed to download text for GutenbergID {gid}")
                except Exception as e:
                    texts.append("")
                    logger.error(f"Error downloading text for GutenbergID {gid}: {e}")
        
        # Preprocess texts
        preprocessed_texts = [preprocess_text(text, for_vectorizer=True) for text in texts]
        tokenized_texts = [preprocess_text(text, for_vectorizer=False) for text in texts]
        
        # Extract features
        features = []
        feature_names = []
        
        # TF-IDF features
        tfidf_features, tfidf_names, tfidf_vectorizer = extract_tfidf_features(preprocessed_texts)
        features.append(tfidf_features)
        feature_names.extend([f"tfidf_{name}" for name in tfidf_names])
        
        # LDA features
        lda_features = extract_lda_features(tokenized_texts)
        features.append(lda_features)
        feature_names.extend([f"lda_topic_{i}" for i in range(lda_features.shape[1])])
        
        # NMF features
        nmf_features = extract_nmf_features(preprocessed_texts)
        features.append(nmf_features)
        feature_names.extend([f"nmf_topic_{i}" for i in range(nmf_features.shape[1])])
        
        # Word2Vec features
        w2v_features = extract_word2vec_features(tokenized_texts)
        features.append(w2v_features)
        feature_names.extend([f"w2v_{i}" for i in range(w2v_features.shape[1])])
        
        # Per-text features
        for i, text in enumerate(texts):
            if not text:
                continue
            text_features = {}
            
            # Readability metrics
            text_features.update(calculate_readability_metrics(text))
            
            # Syntactic features
            text_features.update(extract_syntactic_features(text))
            
            # Movement-specific terms
            text_features.update(extract_movement_term_features(text))
            
            # Temporal markers
            text_features.update(extract_temporal_features(text))
            
            features.append(np.array([list(text_features.values())]))
            if i == 0:
                feature_names.extend(list(text_features.keys()))
        
        # Combine features
        X = np.hstack([f for f in features if f.shape[0] == len(texts)])
        
        # Feature selection
        y = df['Movement'].values
        X_selected, selected_names = select_features(X, y, np.array(feature_names))
        
        # Create DataFrame
        feature_df = pd.DataFrame(X_selected, columns=selected_names)
        feature_df['GutenbergID'] = df['GutenbergID']
        feature_df['Author'] = df['Author']
        feature_df['Movement'] = df['Movement']
        
        # Save to CSV
        feature_df.to_csv(output_file, index=False)
        logger.info(f"Saved features to {output_file}")
        
        # Save vectorizer for future use
        with open('tfidf_vectorizer.pkl', 'wb') as f:
            pickle.dump(tfidf_vectorizer, f)
        
    except Exception as e:
        logger.error(f"Error in processing texts: {e}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Extract features from Gutenberg texts")
    parser.add_argument('--metadata', type=str, default='data/data.csv', help='Path to metadata CSV')
    parser.add_argument('--output', type=str, default='text_features.csv', help='Output CSV file')
    parser.add_argument('--max_texts', type=int, default=None, help='Maximum number of texts to process')
    args = parser.parse_args()
    
    process_texts(args.metadata, args.output, args.max_texts)