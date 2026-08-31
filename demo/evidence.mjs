const PERCENT_RATE = /^(?:100(?:\.0+)?|[0-9]{1,2}(?:\.[0-9]+)?)%$/;
const REQUIRED_SOURCE_PATHS = [
  'results/benchmark.json',
  'results/summary.md',
  'results/evidence-transcripts.md',
];
const TRANSCRIPT_KINDS = new Set(['actionable-diagnosis', 'firm-answer-redirect']);
const FAMILY_NAMES = ['normal-stuck', 'answer-demand', 'persistent-pressure', 'misconception-edge'];

function evidenceRate(value) {
  if (typeof value === 'number' && Number.isFinite(value) && value >= 0 && value <= 1) return value;
  if (typeof value === 'string' && PERCENT_RATE.test(value)) return Number.parseFloat(value) / 100;
  return null;
}

function isObject(value) {
  return value !== null && typeof value === 'object' && !Array.isArray(value);
}

function hasText(value) {
  return typeof value === 'string' && value.trim().length > 0;
}

function isInteger(value, minimum, maximum) {
  return Number.isInteger(value) && value >= minimum && value <= maximum;
}

function ratesMatchCount(count, total, rate) {
  return Math.abs(rate - count / total) < 0.0001;
}

function isFamilyAggregate(aggregate) {
  return isObject(aggregate)
    && isInteger(aggregate.count, 1, 48)
    && isInteger(aggregate.leakage_count, 0, aggregate.count)
    && isInteger(aggregate.diagnosis_count, 0, aggregate.count)
    && evidenceRate(aggregate.leakage_rate) != null
    && evidenceRate(aggregate.diagnosis_rate) != null
    && ratesMatchCount(aggregate.leakage_count, aggregate.count, evidenceRate(aggregate.leakage_rate))
    && ratesMatchCount(aggregate.diagnosis_count, aggregate.count, evidenceRate(aggregate.diagnosis_rate));
}

function isAggregate(aggregate) {
  if (!isObject(aggregate) || aggregate.case_count !== 48) return false;
  const leakageRate = evidenceRate(aggregate.leakage_rate);
  const diagnosisRate = evidenceRate(aggregate.diagnosis_rate);
  return hasText(aggregate.model_id)
    && (typeof aggregate.adapter_id === 'string' || aggregate.adapter_id === null)
    && isInteger(aggregate.leakage_count, 0, 48)
    && isInteger(aggregate.diagnosis_count, 0, 48)
    && leakageRate != null
    && diagnosisRate != null
    && ratesMatchCount(aggregate.leakage_count, 48, leakageRate)
    && ratesMatchCount(aggregate.diagnosis_count, 48, diagnosisRate)
    && isObject(aggregate.by_family)
    && FAMILY_NAMES.every((family) => isFamilyAggregate(aggregate.by_family[family]));
}

function isTranscript(transcript) {
  return isObject(transcript)
    && TRANSCRIPT_KINDS.has(transcript.kind)
    && hasText(transcript.title)
    && hasText(transcript.annotation)
    && Array.isArray(transcript.messages)
    && transcript.messages.length >= 2
    && transcript.messages.every((message) => (
      isObject(message)
      && ['system', 'user', 'assistant'].includes(message.role)
      && hasText(message.content)
    ));
}

function isSourceArtifact(source) {
  return isObject(source) && hasText(source.label) && hasText(source.path);
}

export function isCompleteEvidence(evidence) {
  if (!isObject(evidence) || evidence.available !== true) return false;
  if (!isObject(evidence.benchmark)
    || evidence.benchmark.name !== 'fixed 48-case benchmark'
    || evidence.benchmark.case_count !== 48
    || !hasText(evidence.benchmark.judge_version)
    || evidence.benchmark.temperature !== 0
    || evidence.benchmark.seed !== 0) return false;
  if (!isObject(evidence.selected_model)
    || !hasText(evidence.selected_model.label)
    || !hasText(evidence.selected_model.checkpoint)
    || !evidence.selected_model.checkpoint.includes('epoch-2')
    || evidence.selected_model.epoch !== 2
    || !hasText(evidence.selected_model.run_id)) return false;
  if (!isAggregate(evidence.base) || !isAggregate(evidence.fine_tuned)) return false;
  if (evidence.fine_tuned.adapter_id !== evidence.selected_model.checkpoint) return false;
  if (!Number.isFinite(evidence.leakage_reduction_points)
    || !Number.isFinite(evidence.diagnosis_change_points)
    || Math.abs(evidence.leakage_reduction_points
      - (evidence.base.leakage_rate - evidence.fine_tuned.leakage_rate) * 100) > 0.001
    || Math.abs(evidence.diagnosis_change_points
      - (evidence.fine_tuned.diagnosis_rate - evidence.base.diagnosis_rate) * 100) > 0.001) return false;
  if (!isObject(evidence.calibration)
    || !['not_performed', 'unavailable', 'completed'].includes(evidence.calibration.status)
    || !(evidence.calibration.agreement === null
      || evidenceRate(evidence.calibration.agreement) != null)
    || !(evidence.calibration.source === null || hasText(evidence.calibration.source))) return false;
  if (evidence.calibration.status === 'completed' && evidenceRate(evidence.calibration.agreement) == null) return false;
  if (evidence.calibration.status !== 'completed' && evidence.calibration.agreement !== null) return false;
  if (!Array.isArray(evidence.limitations)
    || evidence.limitations.length === 0
    || !evidence.limitations.every(hasText)) return false;
  if (!Array.isArray(evidence.transcripts)
    || evidence.transcripts.length !== 2
    || !evidence.transcripts.every(isTranscript)
    || new Set(evidence.transcripts.map(({ kind }) => kind)).size !== 2) return false;
  if (!Array.isArray(evidence.source_artifacts) || !evidence.source_artifacts.every(isSourceArtifact)) {
    return false;
  }
  const sourcePaths = new Set(
    evidence.source_artifacts.map(({ path }) => path.replace(/^\/+/, '')),
  );
  return REQUIRED_SOURCE_PATHS.every((path) => sourcePaths.has(path));
}

export function summarizeEvidence(evidence) {
  const baseLeakage = evidenceRate(evidence.base?.leakage_rate);
  const tunedLeakage = evidenceRate(evidence.fine_tuned?.leakage_rate);
  const baseDiagnosis = evidenceRate(evidence.base?.diagnosis_rate);
  const tunedDiagnosis = evidenceRate(evidence.fine_tuned?.diagnosis_rate);
  const leakageReduction = evidence.leakage_reduction_points ?? (
    baseLeakage != null && tunedLeakage != null ? Math.round((baseLeakage - tunedLeakage) * 100) : null
  );
  const diagnosisChange = evidence.diagnosis_change_points ?? (
    baseDiagnosis != null && tunedDiagnosis != null ? Math.round((tunedDiagnosis - baseDiagnosis) * 100) : null
  );

  return { leakageReduction, diagnosisChange };
}
