import sys
import os
import time
import spacy
from icecream import ic
from flask import Flask, render_template_string, request, jsonify
from markupsafe import Markup
sys.path.append('..')
from sentence_transformers import SentenceTransformer


import main.answer_generation as answer_generation
from domynclaimalign.utils.model_utils import extract_atomic_facts_with_mappings
from domynclaimalign.main.compute_traces import compute_traces
from domynclaimalign.main.hallucination_claim_support import HallucinationClaimSupport_advanced_withMapping

os.environ["TOKENIZERS_PARALLELISM"] = "false"
app = Flask(__name__)


# Initialize models and components with timing


init_start = time.time()

MODEL_ID_sentence_transformer = 'all-mpnet-base-v2'
WIKI_BASE_DIR = "../data/wiki"
WIKI_INDEX_DIR = "../data/wiki_index"
UNIGRAM_PATH = "../data/wiki_unigram_probs/olmo-7b_wiki_unigram_probs.json"
MODEL_ID = "allenai/OLMo-7B"
MODEL_CACHE = "../data/model_cache/"
SPACY_MODEL = "en_core_web_sm"

# Load models (cache if needed)
sbert_model = SentenceTransformer(MODEL_ID_sentence_transformer)
HallucinationClaimSupport_obj = HallucinationClaimSupport_advanced_withMapping(threshold = 0.6)

import spacy
try:
    nlp = spacy.load(SPACY_MODEL)
except OSError:
    nlp = spacy.load(SPACY_MODEL)

chat_history = []


# --- HTML Template ---
TEMPLATE = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Agent Traceability Chatbot</title>
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
    <style>
        body { background: #f8fafc; }
        .chat-container { max-width: 900px; margin: 40px auto; background: #fff; border-radius: 16px; box-shadow: 0 2px 16px #0001; padding: 32px; }
        .chat-bubble { padding: 16px; border-radius: 12px; margin-bottom: 12px; }
        .user-bubble { background: #e3f2fd; text-align: right; }
        .agent-bubble { background: #f1f8e9; }
        .highlight { cursor: pointer; padding: 2px 4px; border-radius: 4px; transition: box-shadow 0.2s; }
        .highlight-strong { background: #d1e7dd; border-bottom: 3px solid #198754; }
        .highlight-medium { background: #fff3cd; border-bottom: 3px solid #ffc107; }
        .highlight-weak { background: #f8d7da; border-bottom: 3px solid #dc3545; }
        .highlight:hover { box-shadow: 0 0 0 2px #0d6efd33; }
        .matched-doc-expander { margin-bottom: 10px; }
        .matched-doc-title { font-weight: bold; color: #198754; }
        .matched-doc-low { color: #dc3545; }
        .fact-badge { background: #0d6efd; color: #fff; border-radius: 4px; padding: 2px 6px; font-size: 0.9em; margin-right: 4px; }
    </style>
</head>
<body>
<div class="container-fluid">
    <div class="row">
        <div class="col-md-4">
            <h5 class="mt-3">Matched Documents</h5>
                        {% if matched_docs %}
                                {% for doc_score in matched_docs %}
                                {% set idx = loop.index0 %}
                                {% set doc = doc_score[0] %}
                                {% set score = doc_score[1] %}
                                <div class="accordion matched-doc-expander" id="docAccordion{{idx}}">
                                    <div class="accordion-item">
                                        <h2 class="accordion-header" id="heading{{idx}}">
                                            <button class="accordion-button collapsed matched-doc-title {% if idx >= top_half %}matched-doc-low{% endif %}" type="button" data-bs-toggle="collapse" data-bs-target="#collapse{{idx}}" aria-expanded="false" aria-controls="collapse{{idx}}">
                                                {{ 'High' if idx < top_half else 'Low' }} relevance (Score: {{ '%.3f' % score }})
                                            </button>
                                        </h2>
                                        <div id="collapse{{idx}}" class="accordion-collapse collapse" aria-labelledby="heading{{idx}}" data-bs-parent="#docAccordion{{idx}}">
                                            <div class="accordion-body" style="white-space: pre-wrap;">{{ doc|e }}</div>
                                        </div>
                                    </div>
                                </div>
                                {% endfor %}
            {% else %}
                <div class="alert alert-info">No matched documents found.</div>
            {% endif %}
        </div>
        <div class="col-md-8">
            <div class="chat-container">
                <h2 class="mb-4">💬 Agent Traceability Chatbot</h2>
                <div id="chat-box">
                    {% for turn in chat_history %}
                        <div class="chat-bubble user-bubble">You: {{ turn['question'] }}</div>
                        <div class="chat-bubble agent-bubble">
                            Agent: <span id="answer-{{ loop.index0 }}">{{ turn['answer_html']|safe }}</span>
                        </div>
                    {% endfor %}
                </div>
                <form id="chat-form" class="mt-4" autocomplete="off">
                    <div class="input-group">
                        <input type="text" class="form-control" id="user-input" name="user_input" placeholder="Ask a question..." required autofocus>
                        <button class="btn btn-primary" type="submit">Send</button>
                    </div>
                </form>
                <div id="progress-area" class="mt-3" style="display:none;">
                    <div class="d-flex align-items-center">
                        <div class="spinner-border text-primary me-2" role="status" style="width:2rem;height:2rem;">
                          <span class="visually-hidden">Loading...</span>
                        </div>
                        <span id="progress-msg">Processing your question...</span>
                    </div>
                </div>
            </div>
        </div>
    </div>
</div>

<!-- Modal for sentence details -->
<div class="modal fade" id="sentenceModal" tabindex="-1" aria-labelledby="sentenceModalLabel" aria-hidden="true">
  <div class="modal-dialog modal-lg">
    <div class="modal-content">
      <div class="modal-header">
        <h5 class="modal-title" id="sentenceModalLabel">Sentence Details</h5>
        <button type="button" class="btn-close" data-bs-dismiss="modal" aria-label="Close"></button>
      </div>
      <div class="modal-body" id="modal-body-content">
        <!-- Details will be loaded here -->
      </div>
    </div>
  </div>
</div>

<script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/js/bootstrap.bundle.min.js"></script>
<script src="https://code.jquery.com/jquery-3.6.0.min.js"></script>
<script>
$(function() {
    $('#chat-form').on('submit', function(e) {
        e.preventDefault();
        var user_input = $('#user-input').val();
        if (!user_input) return;
        // Show spinner and disable input
        $('#progress-area').show();
        $('#user-input').prop('disabled', true);
        $('#chat-form button[type=submit]').prop('disabled', true);
        $('#progress-msg').text('Processing your question...');
        $.post('/chat', {user_input: user_input}, function(data) {
            location.reload();
        }).fail(function() {
            $('#progress-msg').text('An error occurred.');
            setTimeout(function(){
                $('#progress-area').hide();
                $('#user-input').prop('disabled', false);
                $('#chat-form button[type=submit]').prop('disabled', false);
            }, 2000);
        });
    });

    // Initialize modal once
    var sentenceModal = new bootstrap.Modal(document.getElementById('sentenceModal'), {
        backdrop: true,
        keyboard: true,
        focus: true
    });

    // Delegate click for highlights
    $(document).on('click', '.highlight', function(e) {
        e.preventDefault();
        e.stopPropagation();
        var idx = $(this).data('idx');
        var turn = $(this).data('turn');
        $.get('/sentence_details', {turn: turn, idx: idx}, function(data) {
            $('#modal-body-content').html(data.html);
            sentenceModal.show();
        }).fail(function() {
            console.error('Failed to load sentence details');
        });
    });

    // Handle modal close events
    $('#sentenceModal').on('hidden.bs.modal', function () {
        $('#modal-body-content').empty();
    });
});
</script>
</body>
</html>
'''

# --- Helper functions ---
def highlight_answer(answer, hallucination_results, facts_by_sentence, turn_idx):
    start_time = time.time()

    # Pre-index hallucination results for O(1) lookup instead of O(n) search
    halluc_index_start = time.time()
    halluc_lookup = {}
    for hr in hallucination_results:
        claim = hr.get('claim', '').strip()
        if claim:
            halluc_lookup[claim] = hr
    halluc_index_end = time.time()

    # Use same spaCy segmentation as extract_atomic_facts_with_mappings
    spacy_start = time.time()
    doc = nlp(answer)
    text_sents = [sent.text.strip() for sent in doc.sents if len(sent.text.split()) > 4]
    spacy_end = time.time()

    if not text_sents:
        return Markup.escape(answer)
    
    html_parts = []
    
    processing_start = time.time()
    for idx, sent in enumerate(text_sents):
        # Check if this sentence has any facts
        sentence_facts = facts_by_sentence[idx] if idx < len(facts_by_sentence) else []
        
        if sentence_facts:
            # This sentence has facts, analyze each fact's support status
            supported_count = 0
            unsupported_count = 0
            
            for fact_mapping in sentence_facts:
                fact_text = fact_mapping.get('fact', '').strip()
                
                # Fast O(1) lookup instead of O(n) search
                hallucination_result = halluc_lookup.get(fact_text)
                
                if hallucination_result:
                    status = hallucination_result.get('status', '')
                    if status.lower().startswith('support'):
                        supported_count += 1
                    else:
                        unsupported_count += 1
                else:
                    # If no hallucination result found, consider it unsupported
                    unsupported_count += 1
            
            # Determine highlighting based on fact support status
            if unsupported_count == 0:
                # All facts are supported - no color highlight, but still clickable
                cls = 'highlight'
            elif supported_count == 0:
                # All facts are unsupported - red highlight
                cls = 'highlight highlight-weak'
            else:
                # Mixed support - yellow highlight
                cls = 'highlight highlight-medium'
            
            html_parts.append(Markup(f'<span class="{cls}" data-idx="{idx}" data-turn="{turn_idx}">{Markup.escape(sent)}</span>'))
        else:
            # No facts for this sentence, just add as plain text
            html_parts.append(Markup.escape(sent))
        
        # Add space between sentences
        html_parts.append(Markup(' '))
    
    processing_end = time.time()

    html_generation_start = time.time()
    result = Markup(''.join(str(part) for part in html_parts))
    html_generation_end = time.time()

    total_time = time.time() - start_time
    ic("highlight_answer", total_time)
    return result

@app.route('/', methods=['GET'])
def index():
    start_time = time.time()

    # Show chat and matched docs for last turn
    matched_docs = []
    top_half = 0
    if chat_history:
        last = chat_history[-1]
        matched_docs = last.get('matched_docs', [])
        n = len(matched_docs)
        top_half = max(1, n//2)
    
    render_start = time.time()
    result = render_template_string(TEMPLATE, chat_history=chat_history, matched_docs=matched_docs, top_half=top_half)
    render_end = time.time()

    total_time = time.time() - start_time
    ic("index_render", total_time)
    return result


@app.route('/chat', methods=['POST'])
def chat():
    start_time = time.time()
    user_input = request.form.get('user_input', '')
    if not user_input.strip():
        return jsonify({'error': 'Empty input'}), 400

    # 1. Generate answer using generate_answer (like Streamlit)
    answer = answer_generation.generate_answer(user_input, MODEL_ID, MODEL_CACHE)
    ic(answer)
    # 2. Compute traces (like Streamlit)
    restricted_docs_query, restricted_docs_answer, span_docs, trace_error, _, _ = compute_traces(
        user_input, answer, MODEL_ID, MODEL_CACHE, UNIGRAM_PATH, SPACY_MODEL, MODEL_ID_sentence_transformer, WIKI_BASE_DIR, WIKI_INDEX_DIR
    )
    ic(len(restricted_docs_query), len(restricted_docs_answer), len(span_docs), trace_error)
    # 3. Prepare matched_docs for display (combine top docs from both traces)
    # For simplicity, combine top 10 from each, as in Streamlit
    matched_docs = []
    for d in sorted(restricted_docs_query, key=lambda d: -d.get('alignment_score', 0))[:10]:
        matched_docs.append((d['doc']['text'], d.get('alignment_score', 0)))
    for d in sorted(restricted_docs_answer, key=lambda d: -d.get('alignment_score', 0))[:10]:
        matched_docs.append((d['doc']['text'], d.get('alignment_score', 0)))
    for d in sorted(span_docs, key=lambda d: -d.get('bm25_score', 0))[:10]:
        matched_docs.append((d['full_doc_text'], d.get('bm25_score', 0)))

    facts_result = extract_atomic_facts_with_mappings(answer)
    all_facts = facts_result.get('all_facts', [])
    fact_mappings = facts_result.get('fact_mappings', [])
    n = len(matched_docs)
    highly_relevant_docs = matched_docs[:max(1, n//4)] if n > 0 else []      
    # 4. Hallucination claim support (keep using HallucinationClaimSupport_advanced_withMapping)
    # Use restricted_docs_query and restricted_docs_answer as in Streamlit
    hallucination_results = HallucinationClaimSupport_obj.check_claims(
        fact_mappings, highly_relevant_docs, sbert_model, use_numerical=False, use_only_entailment=False
    )
    ic(hallucination_results)
    # 5. Map facts to sentences (for highlighting)
    doc = nlp(answer)
    text_sents = [sent.text.strip() for sent in doc.sents if len(sent.text.split()) > 4]
    facts_by_sentence = [[] for _ in text_sents]
    # If hallucination_results have 'sent_idx', map facts to sentences
    for hr in hallucination_results:
        sent_idx = hr.get('sent_idx', 0)
        if 0 <= sent_idx < len(facts_by_sentence):
            facts_by_sentence[sent_idx].append({'fact': hr.get('claim', '')})

    # 6. Highlight answer
    answer_html = highlight_answer(answer, hallucination_results, facts_by_sentence, len(chat_history))

    chat_history.append({
        'question': user_input,
        'answer': answer,
        'answer_html': answer_html,
        'hallucination_results': hallucination_results,
        'facts_by_sentence': facts_by_sentence,
        'matched_docs': matched_docs,
    })

    total_time = time.time() - start_time
    ic("chat_total", total_time)
    return jsonify({'ok': True})

@app.route('/sentence_details', methods=['GET'])
def sentence_details():
    start_time = time.time()

    turn = int(request.args.get('turn', 0))
    idx = int(request.args.get('idx', 0))

    if turn >= len(chat_history):
        return jsonify({'error': 'Invalid turn'}), 400
    turn_data = chat_history[turn]
    hallucination_results = turn_data.get('hallucination_results', [])
    facts_by_sentence = turn_data.get('facts_by_sentence', [])
    if idx >= len(facts_by_sentence):
        return jsonify({'error': 'Invalid sentence index'}), 400
    
    # Get the sentence text using same spaCy segmentation
    spacy_start = time.time()
    answer = turn_data.get('answer', '')
    doc = nlp(answer)
    text_sents = [sent.text.strip() for sent in doc.sents if len(sent.text.split()) > 4]
    spacy_end = time.time()

    sentence_text = text_sents[idx] if idx < len(text_sents) else "Unknown sentence"
    facts = facts_by_sentence[idx] if idx < len(facts_by_sentence) else []
    
    if not facts:
        html = f'<div><b>Sentence:</b> <span style="background:#f8f9fa">{Markup.escape(sentence_text)}</span></div><div>No facts found for this sentence.</div>'
        total_time = time.time() - start_time
        return jsonify({'html': html})
    
    # Build HTML for multiple facts
    html_start = time.time()
    
    # Pre-index hallucination results for faster lookup
    halluc_lookup = {}
    for hr in hallucination_results:
        claim = hr.get('claim', '').strip()
        if claim:
            halluc_lookup[claim] = hr
    
    html = f'<div><b>Sentence:</b> <span style="background:#f8f9fa">{Markup.escape(sentence_text)}</span></div><hr>'
    
    for fact_idx, fact_mapping in enumerate(facts):
        fact_text = fact_mapping.get('fact', '').strip()
        
        # Fast O(1) lookup instead of O(n) search
        hallucination_result = halluc_lookup.get(fact_text)
        
        if hallucination_result:
            status = hallucination_result.get('status', '')
            max_sim = hallucination_result.get('max_sim', 0)
            avg_topk_sim = hallucination_result.get('avg_topk_sim', 0)
            entailment_score = hallucination_result.get('entailment_score', 0)
            topk_sents = hallucination_result.get('topk_sents', [])
            
            color = 'green' if status.lower().startswith('support') else 'red'
            
            html += f'''
            <div class="mb-3">
                <div><b>Fact {fact_idx + 1}:</b> <span class="fact-badge">{Markup.escape(fact_text)}</span></div>
                <div><b>Status:</b> <span style="color:{color};font-weight:bold">{status}</span></div>
                <div><b>Max Similarity:</b> {max_sim:.3f}</div>
                <div><b>Avg Top-k Similarity:</b> {avg_topk_sim:.3f}</div>
                <div><b>Entailment Score:</b> {entailment_score:.3f}</div>
                <div><b>Supporting Sentences:</b></div>
                <ul>
            '''
            
            for sent, score in topk_sents:
                html += f'<li><span style="color:#1f77b4">{Markup.escape(sent)}</span> <span style="color:#888">(sim: {score:.2f})</span></li>'
            
            html += '</ul></div>'
            
            if fact_idx < len(facts) - 1:
                html += '<hr>'
        else:
            html += f'''
            <div class="mb-3">
                <div><b>Fact {fact_idx + 1}:</b> <span class="fact-badge">{Markup.escape(fact_text)}</span></div>
                <div><i>No hallucination analysis available for this fact.</i></div>
            </div>
            '''
    
    html_end = time.time()
    
    total_time = time.time() - start_time
    ic("sentence_details", total_time)
    return jsonify({'html': html})

if __name__ == '__main__':
    app_start = time.time()
    app.run(debug=True, port=8504)
