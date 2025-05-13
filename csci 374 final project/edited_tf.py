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
from sklearn.feature_selection import SelectKBest, chi2, mutual_info_classif
from gensim import corpora
from gensim.models import LdaModel
import numpy as np
from collections import Counter
import time
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

# Download NLTK data
nltk.download('wordnet')
nltk.download('stopwords')
nltk.download('punkt')

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger()

# Load SpaCy model
nlp = spacy.load("en_core_web_sm", disable=["ner"])  # Keep lemmatizer but disable NER for speed

# Load stop words
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

# Enhanced movement-specific features
# These are expanded with more period-appropriate and stylistically relevant terms
movement_specific_features = {
    "Romanticism": [
        "forlorn", "miserly", "passion", "heart", "sublime", "nature", "soul", "emotion", 
        "beauty", "imagination", "dream", "vision", "melancholy", "solitude", "wild", 
        "supernatural", "gothic", "quest", "pastoral", "medieval", "individualism",
        "spontaneous", "liberty", "innocence", "nostalgic", "mysterious", "wanderer"
    ],
    "Realism": [
        "palpable", "manifest", "society", "class", "work", "money", "family", "everyday", 
        "objective", "accurate", "ordinary", "common", "realistic", "detailed", "precise",
        "practical", "factual", "empirical", "mundane", "middle-class", "city", "urban",
        "industrial", "scientific", "documentary", "literal", "observable", "concrete"
    ],
    "Naturalism": [
        "fate", "instinct", "survival", "poverty", "force", "determinism", "environment", 
        "heredity", "struggle", "brutal", "harsh", "primitive", "animal", "darwinian", 
        "savage", "scientific", "experiment", "observation", "lower-class", "squalor", 
        "disease", "violence", "pessimism", "inevitable", "biological", "evolutionary"
    ],
    "Gothicism": [
        "shadow", "grave", "fear", "dark", "ghost", "horror", "supernatural", "mysterious", 
        "castle", "ruin", "ancient", "terror", "death", "decay", "eerie", "monstrous", 
        "macabre", "grotesque", "nocturnal", "haunted", "spectral", "melancholy", "doom",
        "gloomy", "curse", "omen", "diabolic", "dread", "foreboding", "sinister"
    ],
    "Transcendentalism": [
        "spirit", "divine", "truth", "individual", "universe", "self-reliance", "intuition", 
        "consciousness", "oversoul", "nature", "harmony", "moral", "simplicity", "spiritual", 
        "inherent", "goodness", "wilderness", "transcend", "enlightenment", "intellect", 
        "meditation", "ideal", "reform", "higher", "eternal", "transparent", "reverence"
    ],
    "Modernism": [
        "fragment", "chaos", "memory", "time", "consciousness", "stream", "alienation", 
        "interior", "subjective", "psychological", "disillusion", "existential", "cynical",
        "abstract", "innovation", "experiment", "epiphany", "urban", "flux", "relativity",
        "discontinuity", "skepticism", "absurd", "irony", "juxtaposition", "montage"
    ],
    "Renaissance": [
        "classical", "humanism", "virtue", "rhetoric", "reason", "renaissance", "pagan",
        "greco-roman", "courtly", "sonnet", "wit", "allegory", "platonic", "pastoral",
        "renaissance", "civilization", "worldly", "skeptical", "renaissance", "knowledge",
        "patronage", "learning", "discovery", "rebirth", "harmony", "proportion", "counsel"
    ]
}

# Temporal markers by era (approximate publication dates)
temporal_markers = {
    "Renaissance": ["thee", "thou", "thy", "hath", "doth", "art", "wert", "hast", "forsooth", "methinks"],
    "Romanticism/Gothic": ["shall", "upon", "ere", "whilst", "'tis", "thence", "whence", "aught", "naught"],
    "Victorian": ["should", "would", "ought", "must", "quite", "indeed", "perhaps", "rather"],
    "Modern": ["isn't", "don't", "can't", "won't", "couldn't", "shouldn't", "wouldn't"]
}

# Function to count syllables in a word
def count_syllables(word):
    try:
        word = word.lower().strip()
        if not word:
            return 0
        vowels = 'aeiouy'
        syllable_count = 0
        prev_char_was_vowel = False
        for char in word:
            is_vowel = char in vowels
            if is_vowel and not prev_char_was_vowel:
                syllable_count += 1
            prev_char_was_vowel = is_vowel
        if word.endswith('e') and not word.endswith('le'):
            syllable_count = max(1, syllable_count - 1)  # Silent 'e'
        if word.endswith('le') and len(word) > 2 and word[-3] not in vowels:
            syllable_count += 1  # Handle 'le' as syllable
        for diphthong in ['ia', 'io', 'ua', 'uo']:
            if diphthong in word:
                syllable_count = max(1, syllable_count - 1)
        return max(1, syllable_count)
    except Exception as e:
        logger.error(f"Error counting syllables for word '{word}': {e}")
        return 1

# Function to calculate readability metrics
def calculate_readability_metrics(text):
    try:
        sentences = sent_tokenize(text[:100000])  # Limit for performance
        if not sentences:
            return {
                "flesch_reading_ease": 0,
                "gunning_fog": 0,
                "smog_index": 0
            }
        
        # Get word counts and syllable counts
        words = []
        for sentence in sentences:
            words.extend([word for word in sentence.split() if word.strip()])
        
        if not words:
            return {
                "flesch_reading_ease": 0,
                "gunning_fog": 0,
                "smog_index": 0
            }
        
        word_count = len(words)
        syllable_counts = [count_syllables(word) for word in words]
        complex_words = sum(1 for count in syllable_counts if count > 2)
        
        # Calculate metrics
        avg_sentence_length = word_count / len(sentences)
        avg_syllables_per_word = sum(syllable_counts) / word_count
        
        # Flesch Reading Ease
        flesch = 206.835 - (1.015 * avg_sentence_length) - (84.6 * avg_syllables_per_word)
        
        # Gunning Fog
        gunning_fog = 0.4 * (avg_sentence_length + (100 * complex_words / word_count))
        
        # SMOG Index
        smog = 1.043 * np.sqrt(30 * complex_words / len(sentences)) + 3.1291
        
        return {
            "flesch_reading_ease": flesch,
            "gunning_fog": gunning_fog,
            "smog_index": smog
        }
    except Exception as e:
        logger.error(f"Error in readability calculation: {e}")
        return {
            "flesch_reading_ease": 0,
            "gunning_fog": 0,
            "smog_index": 0
        }

# Function to download and clean Gutenberg text with retry logic
def download_gutenberg_text(gid):
    urls = [
        f"https://www.gutenberg.org/files/{gid}/{gid}-0.txt",
        f"https://www.gutenberg.org/files/{gid}/{gid}.txt",
        f"https://www.gutenberg.org/cache/epub/{gid}/pg{gid}.txt"
    ]
    
    for attempt in range(3):  # Retry up to 3 times
        for url in urls:
            try:
                response = requests.get(url, timeout=15)
                response.raise_for_status()
                text = response.text
                
                # Find and extract the main content
                start_markers = [
                    "*** START OF THIS PROJECT GUTENBERG EBOOK",
                    "***START OF THE PROJECT GUTENBERG EBOOK",
                    "*** START OF THE PROJECT GUTENBERG EBOOK",
                    "***START OF THIS PROJECT GUTENBERG EBOOK",
                    "START OF THE PROJECT GUTENBERG EBOOK",
                    "START OF THIS PROJECT GUTENBERG EBOOK"
                ]
                
                end_markers = [
                    "*** END OF THIS PROJECT GUTENBERG EBOOK",
                    "***END OF THE PROJECT GUTENBERG EBOOK",
                    "*** END OF THE PROJECT GUTENBERG EBOOK",
                    "***END OF THIS PROJECT GUTENBERG EBOOK",
                    "END OF THE PROJECT GUTENBERG EBOOK",
                    "END OF THIS PROJECT GUTENBERG EBOOK"
                ]
                
                # Find start and end positions
                start_idx = -1
                for marker in start_markers:
                    pos = text.find(marker)
                    if pos != -1:
                        start_idx = pos + len(marker)
                        break
                
                end_idx = -1
                for marker in end_markers:
                    pos = text.find(marker)
                    if pos != -1:
                        end_idx = pos
                        break
                
                # Extract and clean the main content
                if start_idx != -1 and end_idx != -1:
                    text = text[start_idx:end_idx].strip()
                
                # Clean up the text
                text = re.sub(r'\r\n', '\n', text)  # Normalize line endings
                text = re.sub(r'\n{2,}', '\n\n', text)  # Normalize paragraph breaks
                text = re.sub(r'_+', '', text)  # Remove underscores used for emphasis
                text = re.sub(r'\[.*?\]', '', text)  # Remove footnote references
                
                return text
            except requests.RequestException as e:
                logger.warning(f"Failed to download {url}: {e}")
        
        # Wait before retrying
        time.sleep(2)
    
    logger.error(f"No valid text found for GutenbergID {gid} after 3 attempts")
    return None

# Function to extract syntactic and narrative features
def extract_syntactic_features(text):
    # Tokenize text for basic statistics
    sentences = sent_tokenize(text[:200000])  # Limit for performance
    if not sentences:
        return {}
    
    # Get sentence lengths
    sentence_lengths = [len(sent.split()) for sent in sentences]
    
    # Basic statistics
    avg_sentence_length = np.mean(sentence_lengths) if sentence_lengths else 0
    var_sentence_length = np.var(sentence_lengths) if sentence_lengths else 0
    max_sentence_length = max(sentence_lengths) if sentence_lengths else 0
    median_sentence_length = np.median(sentence_lengths) if sentence_lengths else 0
    
    # Process a sample of the text with SpaCy for detailed analysis
    sample_size = min(len(text), 100000)  # Process at most 100K characters
    doc = nlp(text[:sample_size])
    
    # Count punctuation
    punctuation_counts = {
        "exclamation_marks": text.count("!") / max(len(text) / 10000, 1),  # Normalize by 10K chars
        "question_marks": text.count("?") / max(len(text) / 10000, 1),
        "semicolons": text.count(";") / max(len(text) / 10000, 1),
        "colons": text.count(":") / max(len(text) / 10000, 1),
        "ellipses": text.count("...") / max(len(text) / 10000, 1),
        "dashes": (text.count("—") + text.count("--")) / max(len(text) / 10000, 1),
        "parentheses": (text.count("(") + text.count(")")) / max(len(text) / 10000, 1)
    }
    
    # Count parts of speech
    pos_counts = Counter(token.pos_ for token in doc)
    total_tokens = sum(pos_counts.values()) or 1
    
    pos_features = {
        "noun_freq": pos_counts.get("NOUN", 0) / total_tokens * 100,
        "proper_noun_freq": pos_counts.get("PROPN", 0) / total_tokens * 100,
        "adj_freq": pos_counts.get("ADJ", 0) / total_tokens * 100,
        "adv_freq": pos_counts.get("ADV", 0) / total_tokens * 100,
        "verb_freq": pos_counts.get("VERB", 0) / total_tokens * 100,
        "det_freq": pos_counts.get("DET", 0) / total_tokens * 100,
        "conj_freq": pos_counts.get("CONJ", 0) / total_tokens * 100,
        "intj_freq": pos_counts.get("INTJ", 0) / total_tokens * 100
    }
    
    # Calculate verb tense distribution
    verb_tokens = [token for token in doc if token.pos_ == "VERB"]
    verb_features = {}
    if verb_tokens:
        # Approximating tense - not perfect but gives a signal
        past_verbs = sum(1 for t in verb_tokens if t.tag_ in ["VBD", "VBN"])
        present_verbs = sum(1 for t in verb_tokens if t.tag_ in ["VBZ", "VBP", "VBG"])
        verb_features = {
            "past_tense_ratio": past_verbs / len(verb_tokens) * 100,
            "present_tense_ratio": present_verbs / len(verb_tokens) * 100
        }
    else:
        verb_features = {
            "past_tense_ratio": 0,
            "present_tense_ratio": 0
        }
    
    # Calculate syllables per word
    words = [token.text for token in doc if token.is_alpha]
    syllable_counts = [count_syllables(word) for word in words]
    avg_syllables_per_word = np.mean(syllable_counts) if syllable_counts else 0
    
    # Narrative features
    narrative_markers = {
        "temporal_adverbs": sum(1 for token in doc if token.lower_ in [
            "when", "then", "whilst", "after", "before", "now", "soon", "later",
            "early", "previously", "subsequently", "immediately", "presently"
        ]) / total_tokens * 100,
        
        "locative_preps": sum(1 for token in doc if token.lower_ in [
            "in", "at", "on", "from", "to", "toward", "between", "among",
            "within", "throughout", "around", "across", "over", "under"
        ]) / total_tokens * 100,
        
        "dialogue_markers": (
            text.count('"') + text.count("'") + 
            sum(1 for token in doc if token.lower_ in [
                "said", "asked", "replied", "answered", "exclaimed", "whispered",
                "shouted", "murmured", "spoke", "uttered", "cried", "responded"
            ])
        ) / total_tokens * 100,
        
        "conditional_verbs": sum(1 for token in doc if token.lower_ in [
            "would", "could", "should", "might", "may", "must", "ought"
        ]) / total_tokens * 100,
        
        "first_person_pronouns": sum(1 for token in doc if token.lower_ in [
            "i", "me", "my", "mine", "myself", "we", "us", "our", "ours", "ourselves"
        ]) / total_tokens * 100,
        
        "third_person_pronouns": sum(1 for token in doc if token.lower_ in [
            "he", "him", "his", "himself", "she", "her", "hers", "herself",
            "they", "them", "their", "theirs", "themselves"
        ]) / total_tokens * 100
    }
    
    # Calculate readability metrics
    readability = calculate_readability_metrics(text[:200000])
    
    # Calculate sentence structure complexity
    sentence_fragments = sum(1 for sent in sentences if len(sent.split()) < 5) / len(sentences) * 100
    complex_sentences = sum(1 for sent in sentences if len(sent.split()) > 30) / len(sentences) * 100
    
    # Temporal marker frequencies
    temporal_markers_features = {}
    for era, markers in temporal_markers.items():
        era_key = f"{era.lower()}_markers"
        text_lower = text.lower()
        temporal_markers_features[era_key] = sum(text_lower.count(marker) for marker in markers) / max(len(text.split()) / 1000, 1)
    
    # Combine all features
    return {
        # Sentence statistics
        "avg_sentence_length": avg_sentence_length,
        "var_sentence_length": var_sentence_length,
        "max_sentence_length": max_sentence_length,
        "median_sentence_length": median_sentence_length,
        
        # Word statistics
        "avg_syllables_per_word": avg_syllables_per_word,
        
        # Punctuation
        **punctuation_counts,
        
        # Parts of speech
        **pos_features,
        
        # Verb tense
        **verb_features,
        
        # Narrative style
        **narrative_markers,
        
        # Readability
        **readability,
        
        # Sentence complexity
        "sentence_fragments": sentence_fragments,
        "complex_sentences": complex_sentences,
        
        # Temporal markers
        **temporal_markers_features
    }

# Function to extract movement-specific vocabulary features
def extract_movement_features(text, movement):
    text_lower = text.lower()
    movement_features = {}
    
    # Count occurrences of movement-specific terms
    word_count = len(text.split())
    for mov, words in movement_specific_features.items():
        prefix = f"{mov.lower()}_words"
        # Normalize by word count and scale up for readability
        movement_features[prefix] = sum(text_lower.count(word) for word in words) / (word_count or 1) * 1000
    
    # Additional stylistic features
    doc = nlp(text[:100000])  # Process first 100K characters for efficiency
    total_tokens = len(doc) or 1
    
    # Count capitalized words (not at sentence start)
    sentences = list(doc.sents)
    capitalized_words = 0
    for sent in sentences:
        sent_tokens = [token for token in sent if token.is_alpha]
        if len(sent_tokens) > 1:  # Skip single-word sentences
            capitalized_words += sum(1 for token in sent_tokens[1:] if token.text[0].isupper())
    
    movement_features["capitalized_words"] = capitalized_words / total_tokens * 100
    
    # Analyze paragraph structure
    paragraphs = [p for p in text.split('\n\n') if p.strip()]
    if paragraphs:
        movement_features["avg_paragraph_length"] = np.mean([len(p.split()) for p in paragraphs])
        movement_features["var_paragraph_length"] = np.var([len(p.split()) for p in paragraphs])
    else:
        movement_features["avg_paragraph_length"] = 0
        movement_features["var_paragraph_length"] = 0
    
    # Measure vocabulary richness
    words = [token.text.lower() for token in doc if token.is_alpha]
    unique_words = len(set(words))
    total_words = len(words)
    movement_features["type_token_ratio"] = unique_words / (total_words or 1)
    
    # Lexical diversity (moving window)
    if total_words > 1000:
        window_size = 1000
        windows = [words[i:i+window_size] for i in range(0, len(words)-window_size, window_size//2)]
        ttr_windows = [len(set(window)) / len(window) for window in windows]
        movement_features["mattr"] = np.mean(ttr_windows)  # Moving-Average Type-Token Ratio
    else:
        movement_features["mattr"] = movement_features["type_token_ratio"]
    
    return movement_features

# Function to get expanded synonyms using WordNet
def get_expanded_terms(seed_terms, depth=1):
    expanded_terms = set(seed_terms)
    
    for term in seed_terms:
        # Get WordNet synsets
        synsets = wordnet.synsets(term)
        for synset in synsets[:3]:  # Limit to first 3 synsets per term
            # Add synonyms
            for lemma in synset.lemmas():
                expanded_terms.add(lemma.name().lower().replace('_', ' '))
            
            # Add hypernyms (one level up)
            if depth > 0:
                for hypernym in synset.hypernyms():
                    for lemma in hypernym.lemmas():
                        expanded_terms.add(lemma.name().lower().replace('_', ' '))
    
    return list(expanded_terms)

# Function to extract semantic features using improved TF-IDF and LDA
def extract_semantic_features(texts, movements, gids):
    # Prepare text for vectorization
    processed_texts = []
    for text in texts:
        # Take a sample to speed up processing
        sample = text[:200000]
        # Basic cleaning
        sample = re.sub(r'[^\w\s]', ' ', sample)  # Replace punctuation with space
        sample = re.sub(r'\s+', ' ', sample).strip()  # Normalize whitespace
        processed_texts.append(sample)
    
    # Expanded movement-specific terms for each movement
    expanded_terms = {}
    for movement, terms in movement_specific_features.items():
        expanded_terms[movement] = get_expanded_terms(terms)
    
    # Create movement-specific dictionaries
    movement_dictionaries = {}
    for movement, terms in expanded_terms.items():
        movement_dictionaries[movement] = terms
    
    # Create custom TF-IDF with expanded movement-specific vocabulary
    vocab = set()
    for terms in movement_dictionaries.values():
        vocab.update(terms)
    
    # Use CountVectorizer for count-based features with custom vocabulary
    count_vectorizer = CountVectorizer(
        max_features=300,
        vocabulary=list(vocab),
        ngram_range=(1, 2)  # Include bigrams
    )
    
    # Generate count matrix
    try:
        count_matrix = count_vectorizer.fit_transform(processed_texts)
        count_feature_names = count_vectorizer.get_feature_names_out()
        
        # Normalize counts by document length
        doc_lengths = count_matrix.sum(axis=1).A1
        normalized_counts = count_matrix.copy()
        for i in range(len(processed_texts)):
            if doc_lengths[i] > 0:
                normalized_counts[i] = normalized_counts[i] / doc_lengths[i]
        
        # TF-IDF for general vocabulary capture
        tfidf_vectorizer = TfidfVectorizer(
            max_features=200,
            stop_words='english',
            ngram_range=(1, 2),
            max_df=0.8,
            min_df=5
        )
        
        tfidf_matrix = tfidf_vectorizer.fit_transform(processed_texts)
        tfidf_feature_names = tfidf_vectorizer.get_feature_names_out()
        
        # LDA topic modeling
        processed_tokens = []
        for text in processed_texts:
            tokens = [word for word in text.lower().split() if word not in stop_words and len(word) > 2]
            processed_tokens.append(tokens)
        
        dictionary = corpora.Dictionary(processed_tokens)
        corpus = [dictionary.doc2bow(text) for text in processed_tokens]
        
        n_topics = min(15, len(processed_texts) // 5)  # Scale topics with dataset size
        lda = LdaModel(
            corpus=corpus,
            id2word=dictionary,
            num_topics=n_topics,
            passes=10,
            alpha='auto',
            random_state=42
        )
        
        # Extract topic distributions
        topic_distributions = []
        for bow in corpus:
            topics = lda[bow]
            topic_vec = [0] * n_topics
            for topic_id, weight in topics:
                topic_vec[topic_id] = weight
            topic_distributions.append(topic_vec)
        
        # Calculate movement-specific term frequencies
        movement_term_freqs = []
        for text in processed_texts:
            text_lower = text.lower()
            movement_freqs = {}
            total_words = len(text_lower.split())
            for movement, terms in movement_dictionaries.items():
                term_count = sum(text_lower.count(term) for term in terms)
                movement_freqs[f"{movement.lower()}_term_freq"] = term_count / (total_words or 1) * 1000
            movement_term_freqs.append(movement_freqs)
        
        # Combine features
        feature_matrices = [
            normalized_counts.toarray(),  # Movement-specific vocabulary counts
            tfidf_matrix.toarray(),       # General TF-IDF features
            np.array(topic_distributions)  # Topic distributions
        ]
        
        feature_matrix = np.hstack(feature_matrices)
        
        # Generate feature names
        feature_names = (
            [f"count_{name}" for name in count_feature_names] +
            [f"tfidf_{name}" for name in tfidf_feature_names] +
            [f"topic_{i}" for i in range(n_topics)]
        )
        
        # Movement-term frequencies
        movement_term_matrix = []
        for freqs in movement_term_freqs:
            movement_term_matrix.append([freqs.get(f"{movement.lower()}_term_freq", 0) for movement in movement_specific_features])
        
        movement_term_matrix = np.array(movement_term_matrix)
        movement_term_names = [f"{movement.lower()}_term_freq" for movement in movement_specific_features]
        
        # Add movement term frequencies to feature matrix
        full_feature_matrix = np.hstack([feature_matrix, movement_term_matrix])
        full_feature_names = feature_names + movement_term_names
        
        return full_feature_matrix, full_feature_names
    
    except Exception as e:
        logger.error(f"Error in semantic feature extraction: {e}")
        return np.zeros((len(processed_texts), 1)), ["error"]

# Main processing function
def main():
    # Create output directory for intermediate files
    os.makedirs("data", exist_ok=True)
    
    # Load dataset
    try:
        data_df = pd.read_csv("data/data.csv")
    except FileNotFoundError:
        logger.error("data/data.csv not found. Please ensure the file exists.")
        exit(1)
    
    # Filter out non-prose works if needed
    if "IsProse" in data_df.columns:
        data_df = data_df[data_df["IsProse"] == True]
    
    # Check if any cached texts exist
    cache_dir = os.path.join("data", "text_cache")
    os.makedirs(cache_dir, exist_ok=True)
    
    # Process texts and extract features
    logger.info("Starting feature extraction process...")
    features_list = []
    texts = []
    gids = []
    movements = []
    authors = []
    titles = []
    
    # Process texts in parallel
    def process_text(row):
        gid = row['GutenbergID']
        title = row['Title']
        movement = row['Movement']
        author = row['Author']
        
        cache_file = os.path.join(cache_dir, f"{gid}.txt")
        
        # Check if text is cached
        if os.path.exists(cache_file):
            with open(cache_file, 'r', encoding='utf-8', errors='replace') as f:
                text = f.read()
            logger.info(f"Loaded cached text for {title} (GutenbergID: {gid})")
        else:
            # Download text
            logger.info(f"Downloading {title} (GutenbergID: {gid}) by {author}...")
            text = download_gutenberg_text(gid)
            if text:
                # Cache the text
                with open(cache_file, 'w', encoding='utf-8') as f:
                    f.write(text)
            else:
                return None
        
        # Extract features
        syntactic_features = extract_syntactic_features(text)
        movement_features = extract_movement_features(text, movement)
        
        return {
            'gid': gid,
            'title': title,
            'movement': movement,
            'author': author,
            'text': text,
            'features': {**syntactic_features, **movement_features}
        }
    
    # Process in parallel with thread pool
    with ThreadPoolExecutor(max_workers=5) as executor:
        future_to_row = {executor.submit(process_text, row): row for _, row in data_df.iterrows()}
        for future in as_completed(future_to_row):
            result = future.result()
            if result:
                texts.append(result['text'])
                gids.append(result['gid'])
                movements.append(result['movement'])
                authors.append(result['author'])
                titles.append(result['title'])
                features_list.append(result['features'])
    
    # Extract semantic features
    logger.info("Extracting semantic features...")
    semantic_matrix, semantic_feature_names = extract_semantic_features(texts, movements, gids)
    
    # Reduce semantic features to ~82 to total 120 features
    target_semantic_features = 82
    if semantic_matrix.shape[1] > target_semantic_features:
        logger.info(f"Reducing {semantic_matrix.shape[1]} semantic features to {target_semantic_features}...")
        selector = SelectKBest(score_func=mutual_info_classif, k=target_semantic_features)
        semantic_matrix = selector.fit_transform(semantic_matrix, movements)
        selected_indices = selector.get_support()
        semantic_feature_names = [semantic_feature_names[i] for i in range(len(semantic_feature_names)) if selected_indices[i]]
    
    # Combine all features
    logger.info("Combining features...")
    feature_df = pd.DataFrame(features_list)
    semantic_df = pd.DataFrame(semantic_matrix, columns=semantic_feature_names)

    # Create final DataFrame
    final_df = pd.concat([
        pd.DataFrame({
            'GutenbergID': gids,
            'Author': authors,
            'Movement': movements
        }),
        feature_df,
        semantic_df
    ], axis=1)
    
    # Remove any duplicate GutenbergIDs
    final_df = final_df.drop_duplicates(subset=['GutenbergID'])
    
    # Save to CSV
    output_path = "text_features.csv"
    final_df.to_csv(output_path, index=False)
    logger.info(f"Feature extraction complete. Saved {len(final_df)} samples to {output_path}")
    logger.info(f"Total features: {len(final_df.columns) - 3}")  # Subtract GutenbergID, Author, Movement

if __name__ == "__main__":
    main()