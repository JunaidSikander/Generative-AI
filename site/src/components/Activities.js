import React, { useMemo, useState } from 'react';

const GITHUB_LESSON = 'https://github.com/JunaidSikander/Generative-AI/blob/main/modules/';

function LessonLink({ file, children = 'Read the original lesson' }) {
  return (
    <a className="lesson-link" href={`${GITHUB_LESSON}${file}`} target="_blank" rel="noreferrer">
      {children} ↗
    </a>
  );
}

const quizItems = [
  { example: 'A rule-based thermostat switches heating on below a set temperature.', answer: 'AI', why: 'It performs a goal-directed task using hand-written rules, without learning from examples.' },
  { example: 'A spam filter learns from labeled examples of spam and legitimate email.', answer: 'Machine learning', why: 'It learns a decision pattern from data rather than relying only on hand-written rules.' },
  { example: 'A neural network with many layers identifies objects in photos.', answer: 'Deep learning', why: 'Deep learning is machine learning built with multi-layer neural networks.' },
  { example: 'A language model writes a new product description from a short brief.', answer: 'Generative AI', why: 'The system generates new content; current generative models commonly use deep learning.' },
  { example: 'An expert system applies a clinician-authored decision tree to symptoms.', answer: 'AI', why: 'This is a symbolic, rule-based AI system; it need not learn from data.' },
  { example: 'A random forest predicts whether a loan application will default.', answer: 'Machine learning', why: 'The model combines decision trees trained on historical examples.' },
  { example: 'A large neural network produces a new image from a text description.', answer: 'Generative AI', why: 'It creates content, typically using a deep-learning model.' },
  { example: 'A convolutional neural network labels a medical scan as containing a finding or not.', answer: 'Deep learning', why: 'It is a multi-layer neural network that learns visual features from data.' },
];
const classOptions = ['AI', 'Machine learning', 'Deep learning', 'Generative AI'];

export function FoundationsActivity() {
  const [index, setIndex] = useState(0);
  const [choice, setChoice] = useState('');
  const [checked, setChecked] = useState(false);
  const [score, setScore] = useState(0);
  const [answered, setAnswered] = useState(0);
  const item = quizItems[index];

  function checkAnswer(event) {
    event.preventDefault();
    if (!choice || checked) return;
    setChecked(true);
    setAnswered((value) => value + 1);
    if (choice === item.answer) setScore((value) => value + 1);
  }
  function nextQuestion() {
    setIndex((value) => (value + 1) % quizItems.length);
    setChoice('');
    setChecked(false);
  }
  function reset() {
    setIndex(0);
    setChoice('');
    setChecked(false);
    setScore(0);
    setAnswered(0);
  }

  return (
    <section className="activity-card" id="module-1" aria-labelledby="module-1-title">
      <div className="activity-heading"><span className="module-number">01</span><div><p className="eyebrow">Module 1 · Foundations</p><h2 id="module-1-title">Classify the AI</h2></div></div>
      <p>Choose the most specific category for each example. These categories overlap: deep learning is a type of machine learning, and many generative AI systems use deep learning.</p>
      <p className="score-line" aria-live="polite">Score: {score} correct out of {answered} answered</p>
      <form onSubmit={checkAnswer}>
        <fieldset className="question-box">
          <legend>Example {index + 1} of {quizItems.length}</legend>
          <p className="question-text">{item.example}</p>
          {classOptions.map((option) => (
            <label className="choice-row" key={option}>
              <input type="radio" name="classification" value={option} checked={choice === option} onChange={(event) => setChoice(event.target.value)} disabled={checked} />
              <span>{option}</span>
            </label>
          ))}
        </fieldset>
        {checked && <p className={`feedback ${choice === item.answer ? 'correct' : 'incorrect'}`} role="status">{choice === item.answer ? 'Correct.' : `Not quite — the best answer is ${item.answer}.`} {item.why}</p>}
        <div className="button-row">
          {!checked ? <button className="button button--primary" type="submit" disabled={!choice}>Check answer</button> : <button className="button button--primary" type="button" onClick={nextQuestion}>Next example</button>}
          {answered > 0 && <button className="button button--secondary" type="button" onClick={reset}>Reset quiz</button>}
        </div>
      </form>
      <LessonLink file="01-foundations.md" />
    </section>
  );
}

const phraseVectors = {
  'A dog plays in the park': [0.9, 0.2, 0.1, 0.1],
  'A puppy runs outside': [0.85, 0.25, 0.08, 0.12],
  'A train arrives at the station': [0.1, 0.9, 0.15, 0.05],
  'A cat naps on a sofa': [0.65, 0.35, 0.1, 0.1],
};
function cosine(a, b) {
  const dot = a.reduce((sum, value, i) => sum + value * b[i], 0);
  const normA = Math.sqrt(a.reduce((sum, value) => sum + value * value, 0));
  const normB = Math.sqrt(b.reduce((sum, value) => sum + value * value, 0));
  return dot / (normA * normB);
}

export function TokensActivity() {
  const [text, setText] = useState('Tokenization turns language into pieces.');
  const [first, setFirst] = useState('A dog plays in the park');
  const [second, setSecond] = useState('A puppy runs outside');
  const tokens = useMemo(() => text.match(/[\p{L}\p{N}]+|[^\s\p{L}\p{N}]/gu) || [], [text]);
  const similarity = cosine(phraseVectors[first], phraseVectors[second]);
  return (
    <section className="activity-card" id="module-3" aria-labelledby="module-3-title">
      <div className="activity-heading"><span className="module-number">03</span><div><p className="eyebrow">Module 3 · Tokens and embeddings</p><h2 id="module-3-title">Explore tokens and similarity</h2></div></div>
      <p>Try how text can be split into pieces, then compare a tiny set of illustrative vectors.</p>
      <label className="field-label" htmlFor="token-text">Text to split</label>
      <textarea id="token-text" rows="3" value={text} onChange={(event) => setText(event.target.value)} />
      <p className="score-line" aria-live="polite">{tokens.length} illustrative pieces (not an exact model token count)</p>
      <ol className="token-list" aria-label="Illustrative token pieces">{tokens.map((token, i) => <li key={`${i}-${token}`}><span className="token-index">{i + 1}</span>{token}</li>)}</ol>
      <div className="notice"><strong>Simulation:</strong> this simple word-and-punctuation splitter is not BPE or a model tokenizer. The vectors below are hand-authored teaching examples, not model embeddings.</div>
      <div className="compare-grid">
        <label className="field-label" htmlFor="phrase-one">First sample phrase<select id="phrase-one" value={first} onChange={(event) => setFirst(event.target.value)}>{Object.keys(phraseVectors).map((phrase) => <option key={phrase}>{phrase}</option>)}</select></label>
        <label className="field-label" htmlFor="phrase-two">Second sample phrase<select id="phrase-two" value={second} onChange={(event) => setSecond(event.target.value)}>{Object.keys(phraseVectors).map((phrase) => <option key={phrase}>{phrase}</option>)}</select></label>
      </div>
      <div className="similarity-result" aria-live="polite"><span>Illustrative cosine similarity</span><strong>{similarity.toFixed(3)}</strong><small>Higher means these sample vectors point in more similar directions.</small></div>
      <LessonLink file="03-tokens-embeddings-similarity.md" />
    </section>
  );
}

const startingPrompt = 'Summarize this customer feedback: “The setup was easy, but the app crashed twice.”';
const improvedPrompt = 'You are a support analyst. Summarize the customer feedback in one sentence, then return JSON with keys sentiment and issue. Feedback: “The setup was easy, but the app crashed twice.”';
function simulatedPromptOutput(prompt) {
  const hasJson = /json|structured|schema/i.test(prompt);
  const hasRole = /you are|act as|role:/i.test(prompt);
  const hasLimit = /one sentence|\b\d+ words?\b/i.test(prompt);
  if (hasJson) return '{\n  "sentiment": "mixed",\n  "issue": "The app crashed twice after an easy setup."\n}';
  const qualifier = hasLimit ? 'In one sentence: ' : '';
  return `${qualifier}${hasRole ? 'As a support analyst, ' : ''}the customer found setup easy but experienced two app crashes.`;
}

export function PromptActivity() {
  const [before, setBefore] = useState(startingPrompt);
  const [after, setAfter] = useState(improvedPrompt);
  const [compared, setCompared] = useState(false);
  return (
    <section className="activity-card" id="module-5" aria-labelledby="module-5-title">
      <div className="activity-heading"><span className="module-number">05</span><div><p className="eyebrow">Module 5 · Prompt engineering</p><h2 id="module-5-title">Edit a prompt, compare outputs</h2></div></div>
      <p>Compare a minimal instruction with a more specific one. Edit either prompt and run the local preview.</p>
      <div className="compare-grid">
        <label className="field-label" htmlFor="prompt-before">Starting prompt<textarea id="prompt-before" rows="5" value={before} onChange={(event) => { setBefore(event.target.value); setCompared(false); }} /></label>
        <label className="field-label" htmlFor="prompt-after">Edited prompt<textarea id="prompt-after" rows="5" value={after} onChange={(event) => { setAfter(event.target.value); setCompared(false); }} /></label>
      </div>
      <button className="button button--primary" type="button" onClick={() => setCompared(true)}>Compare illustrative outputs</button>
      <div className="notice"><strong>Simulated output — no AI model is called.</strong> The preview uses a few local rules to demonstrate how role, format, and length instructions can change an answer. It is not generated text.</div>
      <div className="compare-grid output-grid" aria-live="polite">
        <article className="output-box"><h3>Starting prompt preview</h3><pre>{simulatedPromptOutput(before)}</pre></article>
        <article className="output-box"><h3>Edited prompt preview</h3><pre>{compared ? simulatedPromptOutput(after) : 'Select “Compare illustrative outputs” to run the local preview.'}</pre></article>
      </div>
      <LessonLink file="05-prompt-engineering.md" />
    </section>
  );
}

const collection = [
  { id: 'D1', title: 'A guide to growing tomatoes', text: 'Tomato plants need full sun, steady watering, and support as they grow.', vector: [0.1, 0.95, 0.05, 0.1] },
  { id: 'D2', title: 'Python functions for beginners', text: 'A Python function groups reusable instructions and can return a value.', vector: [0.05, 0.05, 0.95, 0.1] },
  { id: 'D3', title: 'Getting enough restful sleep', text: 'A consistent bedtime routine and a cool, dark room can support better sleep.', vector: [0.05, 0.1, 0.1, 0.95] },
  { id: 'D4', title: 'How planets orbit stars', text: 'Gravity keeps planets moving in orbit around stars such as our Sun.', vector: [0.95, 0.1, 0.05, 0.05] },
  { id: 'D5', title: 'Preparing a simple vegetable soup', text: 'Simmer chopped vegetables in broth until tender, then season to taste.', vector: [0.1, 0.85, 0.1, 0.1] },
  { id: 'D6', title: 'Writing tests for code', text: 'Automated tests check that a program behaves as expected after changes.', vector: [0.05, 0.05, 0.9, 0.15] },
];
function vectorForQuery(query) {
  const categories = [
    /space|planet|mars|orbit|star|astronom/i,
    /recipe|cook|tomato|soup|bread|food|kitchen|vegetable/i,
    /python|code|function|program|software|test/i,
    /health|sleep|exercise|rest|wellness/i,
  ];
  const vector = categories.map((pattern) => pattern.test(query) ? 1 : 0);
  if (vector.every((value) => value === 0)) return { vector: [0.5, 0.5, 0.5, 0.5], fallback: true };
  return { vector, fallback: false };
}

export function VectorActivity() {
  const [query, setQuery] = useState('How do planets move around a star?');
  const [submitted, setSubmitted] = useState(query);
  const search = vectorForQuery(submitted);
  const results = collection.map((document) => ({ ...document, score: cosine(search.vector, document.vector) })).sort((a, b) => b.score - a.score).slice(0, 3);
  function submit(event) { event.preventDefault(); setSubmitted(query); }
  return (
    <section className="activity-card" id="module-7" aria-labelledby="module-7-title">
      <div className="activity-heading"><span className="module-number">07</span><div><p className="eyebrow">Module 7 · Vector databases</p><h2 id="module-7-title">Search a sample vector collection</h2></div></div>
      <p>Enter a topic to retrieve the closest sample records by cosine similarity.</p>
      <form className="search-form" onSubmit={submit}><label className="field-label" htmlFor="vector-query">Search query<input id="vector-query" value={query} onChange={(event) => setQuery(event.target.value)} /><span className="field-hint">Try astronomy, cooking, Python, or sleep terms.</span></label><button className="button button--primary" type="submit">Search collection</button></form>
      <div className="notice"><strong>Simulation:</strong> this six-record collection uses hand-authored topic vectors and local keyword routing. It does not create embeddings or contact a vector database.</div>
      <h3 className="results-heading">Top 3 sample results for “{submitted}”</h3>
      {search.fallback && <p className="field-hint" role="status">No known topic keyword matched; a neutral sample query vector is used.</p>}
      <ol className="result-list">{results.map((result) => <li key={result.id}><div className="result-topline"><strong>{result.title}</strong><span>similarity {result.score.toFixed(3)}</span></div><p>{result.text}</p><small>Record {result.id} · sample vector</small></li>)}</ol>
      <LessonLink file="07-vector-databases.md" />
    </section>
  );
}

const passages = [
  { id: 'guide-sun', title: 'Home gardening guide', page: 4, text: 'Tomatoes grow best with at least six hours of direct sunlight each day. Water the soil consistently and avoid leaving roots in standing water.' },
  { id: 'guide-water', title: 'Home gardening guide', page: 5, text: 'Support tomato vines with a stake or cage as they grow. Check soil moisture regularly; containers may dry out faster than garden beds.' },
  { id: 'guide-pests', title: 'Home gardening guide', page: 8, text: 'Inspect leaves weekly for pests. Remove damaged leaves and use a garden-safe control method appropriate for the pest.' },
  { id: 'city-herbs', title: 'City balcony planting notes', page: 2, text: 'Choose a container with drainage holes and place sun-loving plants where they receive several hours of direct light.' },
];
const stopWords = new Set(['what', 'how', 'when', 'where', 'which', 'does', 'do', 'the', 'a', 'an', 'is', 'are', 'can', 'i', 'my', 'to', 'for', 'with', 'and', 'of', 'in', 'on', 'it']);
function passageMatches(question) {
  const terms = question.toLowerCase().match(/[a-z]+/g) || [];
  const usefulTerms = terms.filter((term) => !stopWords.has(term));
  return passages.map((passage) => {
    const text = `${passage.title} ${passage.text}`.toLowerCase();
    const matched = usefulTerms.filter((term) => text.includes(term));
    return { ...passage, matched, score: usefulTerms.length ? matched.length / usefulTerms.length : 0 };
  }).filter((passage) => passage.score > 0).sort((a, b) => b.score - a.score).slice(0, 2);
}

export function RagActivity() {
  const [question, setQuestion] = useState('How much sunlight do tomato plants need?');
  const [submitted, setSubmitted] = useState(question);
  const hits = useMemo(() => passageMatches(submitted), [submitted]);
  function submit(event) { event.preventDefault(); setSubmitted(question); }
  return (
    <section className="activity-card" id="module-8" aria-labelledby="module-8-title">
      <div className="activity-heading"><span className="module-number">08</span><div><p className="eyebrow">Module 8 · Retrieval-augmented generation</p><h2 id="module-8-title">Guided RAG demo</h2></div></div>
      <p>Ask a question to retrieve matching passages from a tiny local collection and inspect the citations used in the answer.</p>
      <form className="search-form" onSubmit={submit}><label className="field-label" htmlFor="rag-question">Question<input id="rag-question" value={question} onChange={(event) => setQuestion(event.target.value)} /></label><button className="button button--primary" type="submit">Retrieve and answer</button></form>
      <div className="notice"><strong>Simulated RAG — no model is called.</strong> Retrieval uses local keyword overlap; the answer is a deterministic excerpt assembled from the retrieved sample passages. This is not an AI-generated response.</div>
      <div className="rag-layout" aria-live="polite">
        <article className="answer-box"><h3>Answer preview</h3>{hits.length ? <p>The retrieved material says: “{hits[0].text}” <a href={`#passage-${hits[0].id}`}>[1]</a>{hits[1] && <> Another relevant passage notes: “{hits[1].text}” <a href={`#passage-${hits[1].id}`}>[2]</a></>}</p> : <p>No matching passage was found in the local sample collection, so this demo cannot answer from its sources.</p>}</article>
        <div><h3 className="results-heading">Retrieved passages ({hits.length})</h3>{hits.length ? <ol className="passage-list">{hits.map((passage, index) => <li id={`passage-${passage.id}`} key={passage.id}><span className="citation">[{index + 1}]</span><strong>{passage.title}, p. {passage.page}</strong><p>{passage.text}</p><small>Keyword overlap score: {passage.score.toFixed(2)}</small></li>)}</ol> : <p className="empty-state">Try asking about tomato sunlight, watering, supports, pests, or containers.</p>}</div>
      </div>
      <LessonLink file="08-rag.md" />
    </section>
  );
}
