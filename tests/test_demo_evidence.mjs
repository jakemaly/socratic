import test from 'node:test';
import assert from 'node:assert/strict';
import { summarizeEvidence } from '../demo/evidence.mjs';

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
