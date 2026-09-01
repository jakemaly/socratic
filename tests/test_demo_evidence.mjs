import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { isCompleteEvidence, summarizeEvidence } from '../demo/evidence.mjs';

function aggregate(leakageCount, diagnosisCount, adapterId = null, familyCounts = [[12, 0, 12], [12, 0, 6], [12, 0, 3], [12, 0, 11]]) {
  const family = (count, familyLeakage, familyDiagnosis) => ({
    count,
    leakage_count: familyLeakage,
    diagnosis_count: familyDiagnosis,
    leakage_rate: familyLeakage / count,
    diagnosis_rate: familyDiagnosis / count,
  });
  return {
    model_id: 'gemma-4',
    adapter_id: adapterId,
    case_count: 48,
    leakage_count: leakageCount,
    diagnosis_count: diagnosisCount,
    leakage_rate: leakageCount / 48,
    diagnosis_rate: diagnosisCount / 48,
    by_family: Object.fromEntries(
      ['normal-stuck', 'answer-demand', 'persistent-pressure', 'misconception-edge']
        .map((name, index) => [name, family(...familyCounts[index])]),
    ),
  };
}

function completeEvidence() {
  return {
    available: true,
    benchmark: {
      name: 'fixed 48-case benchmark',
      case_count: 48,
      judge_version: 'judge-v1',
      temperature: 0,
      seed: 0,
    },
    selected_model: {
      label: 'Gemma 4 + LoRA · epoch 2',
      checkpoint: 'train/adapter-next-sft/epoch-2',
      epoch: 2,
      run_id: 'run-epoch-2',
    },
    base: aggregate(4, 35, null, [[12, 1, 12], [12, 1, 5], [12, 2, 6], [12, 0, 12]]),
    fine_tuned: aggregate(0, 32, 'train/adapter-next-sft/epoch-2'),
    leakage_reduction_points: 8.333333,
    diagnosis_change_points: -6.25,
    calibration: { status: 'not_performed', agreement: null, source: null },
    limitations: ['Fixed semantic-judge benchmark only.'],
    transcripts: [
      {
        kind: 'actionable-diagnosis',
        title: 'Actionable diagnosis',
        annotation: 'The tutor identifies the learner block.',
        messages: [
          { role: 'user', content: 'I am stuck.' },
          { role: 'assistant', content: 'Which step behaves differently than expected?' },
        ],
      },
      {
        kind: 'firm-answer-redirect',
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

test('evidence fallback reports leakage reduction and diagnosis change', () => {
  const summary = summarizeEvidence({
    base: { leakage_rate: '60%', diagnosis_rate: '70%' },
    fine_tuned: { leakage_rate: '30%', diagnosis_rate: '60%' },
  });

  assert.equal(summary.leakageReduction, 30);
  assert.equal(summary.diagnosisChange, -10);
});

test('evidence fallback reports diagnosis improvements', () => {
  const summary = summarizeEvidence({
    base: { leakage_rate: 0.6, diagnosis_rate: 0.6 },
    fine_tuned: { leakage_rate: 0.3, diagnosis_rate: 0.7 },
  });

  assert.equal(summary.leakageReduction, 30);
  assert.equal(summary.diagnosisChange, 10);
});

test('published evidence is complete and uses the selected epoch-2 result', () => {
  const evidence = JSON.parse(readFileSync(new URL('../results/demo-evidence.json', import.meta.url)));

  assert.equal(isCompleteEvidence(evidence), true);
  assert.equal(evidence.selected_model.epoch, 2);
  assert.equal(evidence.fine_tuned.leakage_count, 0);
  assert.equal(evidence.fine_tuned.diagnosis_count, 32);
  assert.equal(evidence.calibration.status, 'not_performed');
});

test('complete evidence requires the selected checkpoint and scoped benchmark record', () => {
  assert.equal(isCompleteEvidence(completeEvidence()), true);

  const incompleteRecords = [
    { ...completeEvidence(), benchmark: { ...completeEvidence().benchmark, case_count: 47 } },
    {
      ...completeEvidence(),
      selected_model: { ...completeEvidence().selected_model, checkpoint: 'epoch-4', epoch: 4 },
    },
    { ...completeEvidence(), calibration: { status: 'not_performed', agreement: '95%', source: null } },
    { ...completeEvidence(), calibration: { status: 'completed', agreement: null, source: 'missing.csv' } },
    { ...completeEvidence(), transcripts: completeEvidence().transcripts.slice(0, 1) },
    {
      ...completeEvidence(),
      transcripts: completeEvidence().transcripts.map((transcript, index) => (
        index === 0 ? { ...transcript, annotation: '   ' } : transcript
      )),
    },
    { ...completeEvidence(), source_artifacts: completeEvidence().source_artifacts.slice(0, 2) },
    {
      ...completeEvidence(),
      source_artifacts: completeEvidence().source_artifacts.map((source, index) => (
        index === 0 ? { ...source, path: '/results/other.json' } : source
      )),
    },
  ];

  incompleteRecords.forEach((record) => assert.equal(isCompleteEvidence(record), false));
});
