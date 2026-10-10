from flask import Flask, request, render_template_string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem import WordNetLemmatizer, PorterStemmer, LancasterStemmer
import math
import string
import re

# --- NLTK Downloads (Runs once) ---
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('tokenizers/punkt_tab')
    nltk.data.find('corpora/stopwords')
    nltk.data.find('corpora/wordnet')
except LookupError:
    print("Downloading required NLTK data...")
    nltk.download('punkt')
    nltk.download('punkt_tab')  # Required for newer NLTK versions
    nltk.download('stopwords')
    nltk.download('wordnet')
    nltk.download('omw-1.4')    # Required for wordnet lemmatizer in some environments

app = Flask(__name__)

# ==========================================
# 1. Text Preprocessing Pipeline
# ==========================================
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()
porter = PorterStemmer()
lancaster = LancasterStemmer()

def analyze_sentence(sentence):
    """Tokenize, remove stop-words, lemmatize, Porter-stem and Lancaster-stem."""
    # Clean and tokenize
    text = sentence.lower()
    text = re.sub(f"[{re.escape(string.punctuation)}]", "", text)
    tokens = word_tokenize(text)

    # 1. Raw tokens
    token_str = " ".join(tokens)

    # 2. Stop-word Removal
    filtered_words = [w for w in tokens if w not in stop_words]

    # 3. Lemmatization (dictionary form)
    lemmatized_words = [lemmatizer.lemmatize(w) for w in filtered_words]

    # 4. Porter Stemming
    porter_words = [porter.stem(w) for w in filtered_words]

    # 5. Lancaster Stemming (more aggressive)
    lancaster_words = [lancaster.stem(w) for w in filtered_words]

    return {
        'original': sentence,
        'tokens': token_str,
        'filtered': " ".join(filtered_words),
        'lemmatized': " ".join(lemmatized_words),
        'porter': " ".join(porter_words),
        'lancaster': " ".join(lancaster_words),
        'final_tokens': porter_words  # classifier input: stop-free Porter stems
    }

# ==========================================
# 2. Naive Bayes Text Classifier (From Scratch)
# ==========================================
class TextNaiveBayes:
    def __init__(self):
        self.vocab = set()
        self.class_word_counts = {}
        self.class_counts = {}
        self.total_docs = 0

    def fit(self, X_train, y_train):
        for text_tokens, label in zip(X_train, y_train):
            self.total_docs += 1
            if label not in self.class_counts:
                self.class_counts[label] = 0
                self.class_word_counts[label] = {}

            self.class_counts[label] += 1

            for word in text_tokens:
                self.vocab.add(word)
                self.class_word_counts[label][word] = self.class_word_counts[label].get(word, 0) + 1

    def predict(self, text_tokens):
        best_label = None
        max_prob = -float('inf')

        for label in self.class_counts:
            # P(Class) - Prior
            prob = math.log(self.class_counts[label] / self.total_docs)

            total_words_in_class = sum(self.class_word_counts[label].values())
            vocab_size = len(self.vocab)

            # P(Word | Class) - Likelihood with Laplace Smoothing
            for word in text_tokens:
                count = self.class_word_counts[label].get(word, 0)
                prob += math.log((count + 1) / (total_words_in_class + vocab_size))

            if prob > max_prob:
                max_prob = prob
                best_label = label

        return best_label

# ==========================================
# 3. Training the Model with Dummy Data
# ==========================================
training_data = [
    ("The new smartphone has a great camera and battery life.", "Technology"),
    ("Artificial intelligence is transforming software development.", "Technology"),
    ("The quarterback threw a touchdown pass in the final quarter.", "Sports"),
    ("The team won the championship after a grueling match.", "Sports"),
    ("Eating vegetables and exercising daily improves heart health.", "Health"),
    ("The vaccine has significantly reduced the spread of the virus.", "Health")
]

classifier = TextNaiveBayes()
X_train_processed = [analyze_sentence(text)['final_tokens'] for text, label in training_data]
y_train = [label for text, label in training_data]
classifier.fit(X_train_processed, y_train)

# ==========================================
# 4. Flask HTML Template & Routing
# ==========================================
HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>Pattern Analysis - NLP</title>
    <style>
        body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; background: #f4f7f6; margin: 0; padding: 30px; }
        .container { max-width: 1150px; margin: auto; }
        .card { background: white; padding: 25px; border-radius: 10px; box-shadow: 0 5px 15px rgba(0,0,0,0.08); margin-bottom: 30px; border-left: 5px solid #34495e; }
        h1, h2 { color: #2c3e50; }
        textarea, input[type="text"] { width: 100%; padding: 10px; border: 1px solid #ccc; border-radius: 4px; box-sizing: border-box; margin-bottom: 15px; font-family: inherit;}
        button { background: #34495e; color: white; border: none; padding: 10px 20px; cursor: pointer; border-radius: 5px; font-weight: bold; }
        button:hover { background: #2c3e50; }
        table { width: 100%; border-collapse: collapse; margin-top: 15px; font-size: 0.88em; }
        th, td { border: 1px solid #ddd; padding: 10px; text-align: left; vertical-align: top; }
        th { background: #f9f9f9; color: #333;}
        .category-badge { background: #e74c3c; color: white; padding: 4px 8px; border-radius: 4px; font-weight: bold; white-space: nowrap; }
    </style>
</head>
<body>
    <div class="container">
        <h1>Experiment 5: Text-Based Pattern Analysis</h1>

        <!-- PART 1: Paragraph Analysis -->
        <div class="card">
            <h2>Part 1: Paragraph Analysis & Classification</h2>
            <p>Enter a paragraph. The system extracts sentences, applies stop-word removal, lemmatization,
            Porter + Lancaster stemming, then classifies each sentence with Naive Bayes.</p>
            <form method="POST">
                <input type="hidden" name="form_type" value="paragraph">
                <textarea name="paragraph_text" rows="4" required>{{ saved_para if saved_para else "The software update improved the phone's performance. Meanwhile, the striker scored a beautiful goal for the club. Doctors recommend drinking plenty of water." }}</textarea>
                <button type="submit">Analyze Paragraph</button>
            </form>

            {% if para_results %}
            <table>
                <tr>
                    <th>Original Sentence</th>
                    <th>Tokens</th>
                    <th>Stop-words Removed</th>
                    <th>Lemmatized</th>
                    <th>Porter Stem</th>
                    <th>Lancaster Stem</th>
                    <th>Predicted Category</th>
                </tr>
                {% for res in para_results %}
                <tr>
                    <td>{{ res.original }}</td>
                    <td style="color: #7f8c8d;">{{ res.tokens }}</td>
                    <td style="color: #7f8c8d;">{{ res.filtered }}</td>
                    <td style="color: #2980b9;">{{ res.lemmatized }}</td>
                    <td style="color: #27ae60;">{{ res.porter }}</td>
                    <td style="color: #8e44ad;">{{ res.lancaster }}</td>
                    <td><span class="category-badge">{{ res.category }}</span></td>
                </tr>
                {% endfor %}
            </table>
            {% endif %}
        </div>

        <!-- PART 2: Single Sentence Prediction -->
        <div class="card" style="border-left-color: #e67e22;">
            <h2>Part 2: Single Sentence Classification</h2>
            <form method="POST">
                <input type="hidden" name="form_type" value="sentence">
                <input type="text" name="single_sentence" value="{{ saved_sent if saved_sent else 'The basketball player injured his ankle.' }}" required>
                <button type="submit" style="background: #e67e22;">Predict Category</button>
            </form>

            {% if single_result %}
            <div style="background: #fef5e7; padding: 15px; border-radius: 8px; margin-top: 15px; border: 1px solid #fad7a1;">
                <p><strong>Input:</strong> {{ single_result.original }}</p>
                <p><strong>Tokens:</strong> {{ single_result.tokens }}</p>
                <p><strong>Stop-words removed:</strong> {{ single_result.filtered }}</p>
                <p><strong>Lemmatized:</strong> {{ single_result.lemmatized }}</p>
                <p><strong>Porter stem:</strong> {{ single_result.porter }}</p>
                <p><strong>Lancaster stem:</strong> {{ single_result.lancaster }}</p>
                <p><strong>Predicted Category:</strong> <span class="category-badge" style="background:#d35400;">{{ single_result.category }}</span></p>
            </div>
            {% endif %}
        </div>
    </div>
</body>
</html>
"""

@app.route('/', methods=['GET', 'POST'])
def index():
    para_results = None
    single_result = None
    saved_para = ""
    saved_sent = ""

    if request.method == 'POST':
        form_type = request.form.get('form_type')

        if form_type == 'paragraph':
            text = request.form.get('paragraph_text')
            saved_para = text
            # Split paragraph into individual sentences
            sentences = sent_tokenize(text)

            para_results = []
            for sentence in sentences:
                analyzed = analyze_sentence(sentence)
                category = classifier.predict(analyzed['final_tokens'])
                analyzed['category'] = category
                para_results.append(analyzed)

        elif form_type == 'sentence':
            sentence = request.form.get('single_sentence')
            saved_sent = sentence
            analyzed = analyze_sentence(sentence)
            category = classifier.predict(analyzed['final_tokens'])
            analyzed['category'] = category
            single_result = analyzed

    return render_template_string(HTML_TEMPLATE,
                                  para_results=para_results,
                                  single_result=single_result,
                                  saved_para=saved_para,
                                  saved_sent=saved_sent)

if __name__ == '__main__':
    print("Starting Text Analysis App...")
    print("Access at: http://127.0.0.1:5000")
    app.run(debug=True)
