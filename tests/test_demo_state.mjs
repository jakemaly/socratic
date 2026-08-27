import test from 'node:test';
import assert from 'node:assert/strict';
import {
  createDemoState,
  reduceDemoState,
} from '../demo/state.mjs';

const exercise = {
  id: 'exercise-one',
  baseSnapshot: { user: 'give me the answer', assistant: 'base answer' },
  starter: { user: 'give me the answer', assistant: 'tutor redirect' },
};
const otherExercise = {
  id: 'exercise-two',
  baseSnapshot: { user: 'help', assistant: 'other base' },
  starter: { user: 'help', assistant: 'other tutor' },
};

function state() {
  return createDemoState(exercise);
}

test('starting a selected exercise keeps the base snapshot read-only', () => {
  const current = state();

  assert.deepEqual(current.baseSnapshot, exercise.baseSnapshot);
  assert.deepEqual(current.fineTunedMessages, [
    { role: 'user', content: 'give me the answer' },
    { role: 'assistant', content: 'tutor redirect' },
  ]);
});

test('submitting a learner message changes only the fine-tuned history', () => {
  const current = state();
  const next = reduceDemoState(current, { type: 'submit', content: 'What should I try first?' });

  assert.deepEqual(next.baseSnapshot, current.baseSnapshot);
  assert.equal(next.fineTunedMessages.at(-1).content, 'What should I try first?');
  assert.equal(next.pending, true);
});

test('a stale response cannot change a restarted conversation', () => {
  const submitted = reduceDemoState(state(), { type: 'submit', content: 'hint', token: 1 });
  const restarted = reduceDemoState(submitted, { type: 'restart', exercise });
  const next = reduceDemoState(restarted, {
    type: 'response',
    token: submitted.requestToken,
    content: 'stale response',
  });

  assert.deepEqual(next, restarted);
});

test('selecting an exercise resets its live history and snapshot', () => {
  const submitted = reduceDemoState(state(), { type: 'submit', content: 'old question' });
  const next = reduceDemoState(submitted, { type: 'select-exercise', exercise: otherExercise });
  const stale = reduceDemoState(next, {
    type: 'response',
    token: submitted.requestToken,
    content: 'stale response',
  });

  assert.deepEqual(stale, next);
  assert.equal(next.selectedExerciseId, 'exercise-two');
  assert.deepEqual(next.baseSnapshot, otherExercise.baseSnapshot);
  assert.deepEqual(next.fineTunedMessages, [
    { role: 'user', content: 'help' },
    { role: 'assistant', content: 'other tutor' },
  ]);
});
