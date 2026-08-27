import { createDemoState, reduceDemoState } from './state.mjs';

const refs = {
  modeBadge: document.querySelector('#mode-badge'),
  liveStatus: document.querySelector('#live-status'),
  introCopy: document.querySelector('#intro-copy'),
  exerciseCount: document.querySelector('#exercise-count'),
  exerciseList: document.querySelector('#exercise-list'),
  exerciseTopic: document.querySelector('#exercise-topic'),
  exerciseTitle: document.querySelector('#exercise-title'),
  exercisePrompt: document.querySelector('#exercise-prompt'),
  restartButton: document.querySelector('#restart-button'),
  baseMessages: document.querySelector('#base-messages'),
  liveMessages: document.querySelector('#live-messages'),
  chipList: document.querySelector('#chip-list'),
  chatForm: document.querySelector('#chat-form'),
  messageInput: document.querySelector('#message-input'),
  sendButton: document.querySelector('#send-button'),
  chatError: document.querySelector('#chat-error'),
  evidenceContent: document.querySelector('#evidence-content'),
};

const app = {
  data: null,
  state: null,
  mode: 'fixture',
};

async function getJSON(path) {
  const response = await fetch(path, { headers: { Accept: 'application/json' } });
  if (!response.ok) throw new Error(`${path} returned ${response.status}`);
  return response.json();
}

function selectedExercise() {
  return app.data.exercises.find(({ id }) => id === app.state.selectedExerciseId);
}

function fixtureFor(exerciseId) {
  return app.data.fixtures.exercises[exerciseId];
}

function stateExercise(exerciseId) {
  const fixture = fixtureFor(exerciseId);
  return {
    id: exerciseId,
    baseSnapshot: fixture.base_snapshot,
    starter: fixture.starter,
  };
}

function messageElement(message, note = '') {
  const wrapper = document.createElement('div');
  wrapper.className = `message ${message.role === 'user' ? 'message-user' : 'message-assistant'}`;
  const role = document.createElement('span');
  role.className = 'message-role';
  role.textContent = message.role === 'user' ? 'Learner' : 'Tutor';
  const text = document.createElement('p');
  text.textContent = message.content;
  wrapper.append(role, text);
  if (note) {
    const annotation = document.createElement('span');
    annotation.className = `message-note ${message.role === 'assistant' && note.includes('Socratic') ? 'live-note' : 'base-note'}`;
    annotation.textContent = note;
    wrapper.append(annotation);
  }
  return wrapper;
}

function renderConversation(container, messages, note = '') {
  container.replaceChildren();
  messages.forEach((message, index) => {
    container.append(messageElement(message, index === messages.length - 1 ? note : ''));
  });
  container.scrollTop = container.scrollHeight;
}

function renderExerciseList() {
  refs.exerciseList.replaceChildren();
  app.data.exercises.forEach((exercise, index) => {
    const button = document.createElement('button');
    button.type = 'button';
    button.className = 'exercise-button';
    button.setAttribute('aria-current', String(exercise.id === app.state.selectedExerciseId));
    button.dataset.exerciseId = exercise.id;
    const title = document.createElement('strong');
    title.textContent = exercise.title;
    const topic = document.createElement('small');
    topic.textContent = `${exercise.topic} · ${String(index + 1).padStart(2, '0')}`;
    button.append(title, topic);
    button.addEventListener('click', () => selectExercise(exercise));
    refs.exerciseList.append(button);
  });
}

function renderChips() {
  refs.chipList.replaceChildren();
  app.data.chips.forEach((chip) => {
    const button = document.createElement('button');
    button.type = 'button';
    button.className = `chip ${chip.id === 'answer-demand' ? 'answer-demand' : ''} ${chip.id === 'hint' ? 'hint' : ''}`;
    button.textContent = chip.label;
    button.disabled = app.state.pending;
    button.addEventListener('click', () => submitMessage(chip.label));
    refs.chipList.append(button);
  });
}

function render() {
  if (!app.data || !app.state) return;
  const exercise = selectedExercise();
  refs.exerciseTopic.textContent = `${exercise.topic} · exercise ${String(exercise.difficulty_rank).padStart(2, '0')}`;
  refs.exerciseTitle.textContent = exercise.title;
  refs.exercisePrompt.textContent = exercise.prompt;
  refs.exerciseCount.textContent = String(app.data.exercises.length).padStart(2, '0');
  refs.liveStatus.textContent = app.mode === 'live' ? 'live adapter' : 'fixture mode';
  refs.restartButton.disabled = false;
  refs.messageInput.disabled = app.state.pending;
  refs.sendButton.disabled = app.state.pending;
  renderExerciseList();
  renderConversation(refs.baseMessages, [
    { role: 'user', content: app.state.baseSnapshot.user },
    { role: 'assistant', content: app.state.baseSnapshot.assistant },
  ], `⚠ ${app.state.baseSnapshot.annotation}`);
  renderConversation(refs.liveMessages, app.state.fineTunedMessages, app.state.fineTunedMessages.at(-1)?.role === 'assistant' ? '✓ Socratic redirect + diagnosis' : '');
  if (app.state.pending) {
    const loading = document.createElement('div');
    loading.className = 'message message-assistant loading-message';
    loading.setAttribute('role', 'status');
    loading.textContent = 'Tutor is thinking…';
    refs.liveMessages.append(loading);
  }
  refs.chatError.hidden = !app.state.error;
  refs.chatError.replaceChildren();
  if (app.state.error) {
    const text = document.createElement('span');
    text.textContent = app.state.error;
    const retry = document.createElement('button');
    retry.type = 'button';
    retry.className = 'retry-button';
    retry.textContent = 'Try again';
    retry.addEventListener('click', retryMessage);
    refs.chatError.append(text, retry);
  }
  renderChips();
}

function updateState(action) {
  app.state = reduceDemoState(app.state, action);
  render();
  return app.state;
}

async function requestReply(messages, token) {
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), 30_000);
  try {
    const response = await fetch('/api/chat', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json', Accept: 'application/json' },
      body: JSON.stringify({
        exercise_id: app.state.selectedExerciseId,
        messages,
        temperature: 0,
      }),
      signal: controller.signal,
    });
    const body = await response.json();
    if (!response.ok) throw new Error(body.error || 'The fine-tuned tutor is unavailable.');
    if (body.message?.role !== 'assistant' || typeof body.message.content !== 'string') {
      throw new Error('The fine-tuned tutor returned an invalid response.');
    }
    updateState({ type: 'response', token, content: body.message.content });
  } catch (error) {
    const message = error.name === 'AbortError'
      ? 'The fine-tuned tutor timed out. Try again.'
      : error.message || 'The fine-tuned tutor is unavailable.';
    updateState({ type: 'failure', token, message });
  } finally {
    clearTimeout(timeout);
  }
}

function submitMessage(content) {
  const text = content.trim();
  if (!text || app.state.pending) return;
  refs.messageInput.value = '';
  resizeInput();
  const next = updateState({ type: 'submit', content: text });
  void requestReply(next.fineTunedMessages, next.requestToken);
}

function retryMessage() {
  if (app.state.pending) return;
  const next = updateState({ type: 'retry' });
  void requestReply(next.fineTunedMessages, next.requestToken);
}

function selectExercise(exercise) {
  updateState({ type: 'select-exercise', exercise: stateExercise(exercise.id) });
}

function restart() {
  updateState({ type: 'restart', exercise: stateExercise(app.state.selectedExerciseId) });
  refs.messageInput.focus();
}

function resizeInput() {
  refs.messageInput.style.height = 'auto';
  refs.messageInput.style.height = `${Math.min(refs.messageInput.scrollHeight, 128)}px`;
}

function evidenceValue(value) {
  if (typeof value === 'number') return `${Math.round(value * 100)}%`;
  return value ?? '—';
}

function evidenceRate(value) {
  if (typeof value === 'number') return value;
  if (typeof value === 'string' && value.endsWith('%')) return Number.parseFloat(value) / 100;
  return null;
}

function renderEvidence(evidence) {
  if (!evidence || evidence.available === false) return;
  const baseLeakage = evidenceRate(evidence.base?.leakage_rate);
  const tunedLeakage = evidenceRate(evidence.fine_tuned?.leakage_rate);
  const baseDiagnosis = evidenceRate(evidence.base?.diagnosis_rate);
  const tunedDiagnosis = evidenceRate(evidence.fine_tuned?.diagnosis_rate);
  const leakageReduction = evidence.leakage_reduction_points ?? (
    baseLeakage != null && tunedLeakage != null ? Math.round((baseLeakage - tunedLeakage) * 100) : null
  );
  const utilityDrop = evidence.utility_drop_points ?? (
    baseDiagnosis != null && tunedDiagnosis != null ? Math.round((baseDiagnosis - tunedDiagnosis) * -100) : null
  );
  const leakageThreshold = evidence.thresholds?.leakage_reduction_points ?? 20;
  const utilityThreshold = evidence.thresholds?.utility_drop_points ?? 5;
  const verdict = evidence.verdict || (
    leakageReduction != null && utilityDrop != null
      ? (leakageReduction >= leakageThreshold && utilityDrop <= utilityThreshold ? 'PASS' : 'FAIL')
      : '—'
  );
  const stats = [
    ['Base leakage', evidence.base?.leakage_rate, 'Completed-solution leakage'],
    ['Fine-tuned leakage', evidence.fine_tuned?.leakage_rate, 'Completed-solution leakage'],
    ['Base diagnosis', evidence.base?.diagnosis_rate, 'Actionable diagnosis'],
    ['Fine-tuned diagnosis', evidence.fine_tuned?.diagnosis_rate, 'Actionable diagnosis'],
    ['Leakage reduction', leakageReduction == null ? null : `${leakageReduction} pts`, `Required: ≥ ${leakageThreshold} pts`],
    ['Utility change', utilityDrop == null ? null : `${utilityDrop} pts`, `Allowed drop: ≤ ${utilityThreshold} pts`],
    ['Calibration', evidence.calibration_agreement, 'Judge agreement'],
    ['Verdict', verdict, 'Threshold comparison'],
  ];
  const grid = document.createElement('div');
  grid.className = 'evidence-grid';
  stats.forEach(([label, value, detailText]) => {
    const card = document.createElement('div');
    card.className = 'evidence-stat';
    const name = document.createElement('small');
    name.textContent = label;
    const number = document.createElement('strong');
    number.textContent = evidenceValue(value);
    const detail = document.createElement('p');
    detail.textContent = detailText;
    card.append(name, number, detail);
    grid.append(card);
  });
  const fragments = [grid];
  if (Array.isArray(evidence.transcripts) && evidence.transcripts.length) {
    const transcripts = document.createElement('div');
    transcripts.className = 'evidence-transcripts';
    evidence.transcripts.forEach((transcript) => {
      const article = document.createElement('article');
      article.className = 'evidence-transcript';
      const title = document.createElement('h3');
      title.textContent = transcript.title || 'Annotated transcript';
      const annotation = document.createElement('p');
      annotation.textContent = transcript.annotation || '';
      const body = document.createElement('pre');
      body.textContent = (transcript.messages || [])
        .map((message) => `${message.role || 'message'}: ${message.content || ''}`)
        .join('\n\n');
      article.append(title, annotation, body);
      transcripts.append(article);
    });
    fragments.push(transcripts);
  }
  if (Array.isArray(evidence.source_artifacts) && evidence.source_artifacts.length) {
    const sources = document.createElement('p');
    sources.className = 'evidence-sources';
    sources.append('Sources: ');
    evidence.source_artifacts.forEach((source, index) => {
      if (index) sources.append(' · ');
      const link = document.createElement('a');
      link.href = source.path;
      link.textContent = source.label || source.path;
      link.target = '_blank';
      link.rel = 'noreferrer';
      sources.append(link);
    });
    fragments.push(sources);
  }
  refs.evidenceContent.replaceChildren(...fragments);
}

async function loadEvidence() {
  try {
    renderEvidence(await getJSON('/results/demo-evidence.json'));
  } catch {
    // Evidence remains unavailable until the benchmark track publishes it.
  }
}

async function initialize() {
  try {
    const [exercises, chips, fixtures, intro] = await Promise.all([
      getJSON('/content/exercises.json'),
      getJSON('/content/chips.json'),
      getJSON('/demo/fixtures.json'),
      fetch('/content/intro.md').then((response) => response.ok ? response.text() : ''),
    ]);
    app.data = { exercises, chips, fixtures };
    app.mode = document.documentElement.dataset.demoMode || 'fixture';
    refs.modeBadge.textContent = app.mode === 'live' ? 'Live adapter' : 'Fixture mode';
    if (intro.trim()) refs.introCopy.textContent = intro.trim();
    app.state = createDemoState(stateExercise(exercises[0].id));
    render();
    refs.messageInput.disabled = false;
    refs.sendButton.disabled = false;
    void loadEvidence();
  } catch (error) {
    refs.modeBadge.textContent = 'Demo unavailable';
    refs.introCopy.textContent = `Could not load the demo: ${error.message}`;
  }
}

refs.chatForm.addEventListener('submit', (event) => {
  event.preventDefault();
  submitMessage(refs.messageInput.value);
});
refs.messageInput.addEventListener('input', resizeInput);
refs.messageInput.addEventListener('keydown', (event) => {
  if (event.key === 'Enter' && !event.shiftKey) {
    event.preventDefault();
    refs.chatForm.requestSubmit();
  }
});
refs.restartButton.addEventListener('click', restart);
void initialize();
