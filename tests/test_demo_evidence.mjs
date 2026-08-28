import test from 'node:test';
import assert from 'node:assert/strict';
import { isCompleteEvidence, summarizeEvidence } from '../demo/evidence.mjs';

function completeEvidence() {
  return {
    available: true,
    base: { leakage_rate: '60%', diagnosis_rate: 0.7 },
    fine_tuned: { leakage_rate: 0.3, diagnosis_rate: '70%' },
    leakage_reduction_points: 30,
    utility_drop_points: 0,
    thresholds: { leakage_reduction_points: 20, utility_drop_points: 5 },
    calibration_agreement: '95%',
    verdict: 'PASS',
    transcripts: [
      {
        title: 'Actionable diagnosis',
        annotation: 'The tutor identifies the learner block.',
        messages: [
          { role: 'user', content: 'I am stuck.' },
          { role: 'assistant', content: 'Which step behaves differently than expected?' },
        ],
      },
      {
        title: 'Answer redirect',
        annotation: 'The tutor redirects a request for the finished answer.',
        messages: [
          { role: 'user', content: 'Give me the answer.' },
          { role: 'assistant', content: 'What is the first value the program needs?' },
        ],
      },
    ],
    source_artifacts: [
      { label: 'Full benchmark', path: '/results/benchmark.json' },
      { label: 'Summary', path: '/results/summary.md' },
      { label: 'Annotated transcripts', path: '/results/evidence-transcripts.md' },
    ],
  };
}

test('evidence fallback treats lower diagnosis as utility loss', () => {
  const summary = summarizeEvidence({
    base: { leakage_rate: '60%', diagnosis_rate: '70%' },
    fine_tuned: { leakage_rate: '30%', diagnosis_rate: '60%' },
  });

  assert.equal(summary.leakageReduction, 30);
  assert.equal(summary.utilityDrop, 10);
  assert.equal(summary.verdict, 'FAIL');
});

test('evidence fallback allows diagnosis improvements', () => {
  const summary = summarizeEvidence({
    base: { leakage_rate: 0.6, diagnosis_rate: 0.6 },
    fine_tuned: { leakage_rate: 0.3, diagnosis_rate: 0.7 },
  });

  assert.equal(summary.utilityDrop, -10);
  assert.equal(summary.verdict, 'PASS');
});

test('complete evidence requires the full presentation record', () => {
  assert.equal(isCompleteEvidence(completeEvidence()), true);

  const incompleteRecords = [
    { ...completeEvidence(), thresholds: {} },
    { ...completeEvidence(), calibration_agreement: 'unknown' },
    { ...completeEvidence(), transcripts: completeEvidence().transcripts.slice(0, 1) },
    {
      ...completeEvidence(),
      transcripts: completeEvidence().transcripts.map((transcript, index) => (
        index === 0 ? { ...transcript, annotation: '   ' } : transcript
      )),
    },
    { ...completeEvidence(), source_artifacts: completeEvidence().source_artifacts.slice(0, 2) },
  ];

  incompleteRecords.forEach((record) => assert.equal(isCompleteEvidence(record), false));
});
