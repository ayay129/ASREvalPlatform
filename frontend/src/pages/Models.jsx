import { useCallback, useEffect, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import {
  RefreshCw, Layers, GitMerge, FlaskConical, ChevronRight,
  CheckCircle2, AlertCircle, Loader2, FolderOpen, X,
} from 'lucide-react'
import { api } from '../api'
import { COLORS } from '../theme'
import StatusBadge from '../components/StatusBadge'
import DatasetPicker from '../components/DatasetPicker'
import GpuPicker from '../components/GpuPicker'

export default function Models() {
  const [runs, setRuns] = useState([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState('')
  const [mergingIds, setMergingIds] = useState(new Set())
  const [evalRun, setEvalRun] = useState(null)      // 当前要评测的 run
  const navigate = useNavigate()

  const load = useCallback(async () => {
    setLoading(true)
    setError('')
    try {
      const data = await api.listTrainRuns({ status: 'completed' })
      setRuns(data)
    } catch (err) {
      console.error(err)
      setError('Failed to load models')
    } finally {
      setLoading(false)
    }
  }, [])

  useEffect(() => { load() }, [load])

  // 有正在 merging 的任务时自动轮询
  useEffect(() => {
    const hasMerging = runs.some(r => r.phase === 'merging') || mergingIds.size > 0
    if (!hasMerging) return
    const timer = setInterval(load, 3000)
    return () => clearInterval(timer)
  }, [runs, mergingIds, load])

  const handleMerge = async (run, e) => {
    e.stopPropagation()
    if (!confirm(`Merge LoRA adapter for "${run.name}"?`)) return
    setMergingIds(prev => new Set(prev).add(run.id))
    try {
      await api.mergeTrainRun(run.id)
      load()
    } catch (err) {
      alert(err.response?.data?.detail || 'Merge failed')
      setMergingIds(prev => {
        const next = new Set(prev)
        next.delete(run.id)
        return next
      })
    }
  }

  const handleEval = (run, e) => {
    e.stopPropagation()
    setEvalRun(run)
  }

  // 按模型分类：has merged / has adapter only / no model
  const merged = runs.filter(r => r.merged_model_path)
  const adapterOnly = runs.filter(r => r.checkpoint_path && !r.merged_model_path)

  return (
    <div>
      {/* Header */}
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 28 }}>
        <div>
          <h1 style={{ fontSize: 24, fontWeight: 700, color: COLORS.textDark }}>
            <Layers size={22} style={{ marginRight: 8, verticalAlign: -3 }} />
            Models
          </h1>
          <p style={{ color: COLORS.textLight, fontSize: 14, marginTop: 4 }}>
            Manage fine-tuned models from completed training runs
          </p>
        </div>
        <button onClick={load} style={refreshBtn}>
          <RefreshCw size={15} /> Refresh
        </button>
      </div>

      {error && <div style={errorBox}>{error}</div>}

      {/* Stats */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: 14, marginBottom: 24 }}>
        <StatCard
          icon={<CheckCircle2 size={20} />}
          label="Merged Models"
          value={merged.length}
          color={COLORS.success}
        />
        <StatCard
          icon={<GitMerge size={20} />}
          label="Adapters Only"
          value={adapterOnly.length}
          color={COLORS.warning}
          sub="Need merge before evaluation"
        />
        <StatCard
          icon={<Layers size={20} />}
          label="Total Completed"
          value={runs.length}
          color={COLORS.accent}
        />
      </div>

      {loading && runs.length === 0 ? (
        <div style={emptyState}>Loading...</div>
      ) : runs.length === 0 ? (
        <div style={emptyState}>
          <Layers size={40} color={COLORS.textLight} />
          <div style={{ marginTop: 12, fontWeight: 600, color: COLORS.textMid }}>No completed training runs</div>
          <div style={{ marginTop: 6, fontSize: 13, color: COLORS.textLight }}>
            Complete a training run to see your fine-tuned models here.
          </div>
          <button
            onClick={() => navigate('/train-runs/new')}
            style={{ ...actionBtn(COLORS.accent), marginTop: 16 }}
          >
            Start Training
          </button>
        </div>
      ) : (
        <div style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
          {/* Merged models first, then adapter-only */}
          {merged.length > 0 && (
            <>
              <SectionHeader title="Merged Models" subtitle="Ready for evaluation" count={merged.length} />
              {merged.map(run => (
                <ModelCard
                  key={run.id}
                  run={run}
                  isMerging={mergingIds.has(run.id) || run.phase === 'merging'}
                  onMerge={handleMerge}
                  onEval={handleEval}
                  onClick={() => navigate(`/train-runs/${run.id}`)}
                />
              ))}
            </>
          )}

          {adapterOnly.length > 0 && (
            <>
              <SectionHeader
                title="LoRA Adapters"
                subtitle="Merge required before evaluation"
                count={adapterOnly.length}
                style={{ marginTop: merged.length > 0 ? 20 : 0 }}
              />
              {adapterOnly.map(run => (
                <ModelCard
                  key={run.id}
                  run={run}
                  isMerging={mergingIds.has(run.id) || run.phase === 'merging'}
                  onMerge={handleMerge}
                  onEval={handleEval}
                  onClick={() => navigate(`/train-runs/${run.id}`)}
                />
              ))}
            </>
          )}
        </div>
      )}

      {/* Eval modal */}
      {evalRun && (
        <EvalModal
          run={evalRun}
          onClose={() => setEvalRun(null)}
          onSuccess={(evalId) => navigate(`/report/${evalId}`)}
        />
      )}
    </div>
  )
}


// ──────────────────── Sub-components ────────────────────

function SectionHeader({ title, subtitle, count, style = {} }) {
  return (
    <div style={{ display: 'flex', alignItems: 'baseline', gap: 8, ...style }}>
      <h2 style={{ fontSize: 15, fontWeight: 700, color: COLORS.textDark, margin: 0 }}>{title}</h2>
      <span style={{
        padding: '2px 8px', borderRadius: 10, fontSize: 11, fontWeight: 700,
        background: COLORS.secondary1, color: COLORS.secondary2,
      }}>
        {count}
      </span>
      {subtitle && (
        <span style={{ fontSize: 12, color: COLORS.textLight }}>{subtitle}</span>
      )}
    </div>
  )
}

function StatCard({ icon, label, value, color, sub }) {
  return (
    <div style={{
      background: COLORS.card, borderRadius: 12, padding: '18px 20px',
      border: `1px solid ${COLORS.border}`,
      display: 'flex', alignItems: 'center', gap: 14,
    }}>
      <div style={{
        width: 42, height: 42, borderRadius: 10,
        background: color + '18', color,
        display: 'flex', alignItems: 'center', justifyContent: 'center',
      }}>
        {icon}
      </div>
      <div>
        <div style={{ fontSize: 22, fontWeight: 700, color: COLORS.textDark }}>{value}</div>
        <div style={{ fontSize: 12, color: COLORS.textLight, marginTop: 2 }}>{label}</div>
        {sub && <div style={{ fontSize: 11, color: COLORS.textLight, marginTop: 2 }}>{sub}</div>}
      </div>
    </div>
  )
}

function ModelCard({ run, isMerging, onMerge, onEval, onClick }) {
  const hasMerged = !!run.merged_model_path
  const hasAdapter = !!run.checkpoint_path
  const isMergeFailed = run.phase === 'merge_failed'

  return (
    <div style={cardStyle} onClick={onClick}>
      <div style={{ flex: 1, minWidth: 0 }}>
        {/* Row 1: Name + badges */}
        <div style={{ display: 'flex', alignItems: 'center', gap: 10, marginBottom: 8 }}>
          <span style={{ fontWeight: 700, fontSize: 15, color: COLORS.textDark }}>
            {run.name}
          </span>
          <span style={{ fontSize: 12, color: COLORS.textLight }}>#{run.id}</span>
          {hasMerged && (
            <span style={badge(COLORS.success)}>
              <CheckCircle2 size={11} /> Merged
            </span>
          )}
          {!hasMerged && hasAdapter && !isMerging && !isMergeFailed && (
            <span style={badge(COLORS.warning)}>
              <FolderOpen size={11} /> Adapter Only
            </span>
          )}
          {isMerging && (
            <span style={badge(COLORS.secondary2)}>
              <Loader2 size={11} style={{ animation: 'spin 1s linear infinite' }} /> Merging...
            </span>
          )}
          {isMergeFailed && (
            <span style={badge(COLORS.danger)}>
              <AlertCircle size={11} /> Merge Failed
            </span>
          )}
        </div>

        {/* Row 2: Model + paths */}
        <div style={{ fontSize: 12, color: COLORS.textMid, lineHeight: 1.8 }}>
          <span style={{ fontWeight: 600 }}>Base:</span> {run.base_model}
          {hasAdapter && (
            <div style={pathRow}>
              <span style={{ fontWeight: 600, color: COLORS.textMid }}>Adapter:</span>
              <code style={codePath}>{run.checkpoint_path}</code>
            </div>
          )}
          {hasMerged && (
            <div style={pathRow}>
              <span style={{ fontWeight: 600, color: COLORS.success }}>Merged:</span>
              <code style={{ ...codePath, color: COLORS.success }}>{run.merged_model_path}</code>
            </div>
          )}
          {isMergeFailed && run.error_message && (
            <div style={{ color: COLORS.danger, fontSize: 11, marginTop: 4 }}>
              {run.error_message.length > 120
                ? run.error_message.slice(0, 120) + '...'
                : run.error_message}
            </div>
          )}
        </div>
      </div>

      {/* Actions */}
      <div style={{ display: 'flex', gap: 8, alignItems: 'center', flexShrink: 0 }}>
        {(!hasMerged && hasAdapter) && (
          <button
            onClick={e => onMerge(run, e)}
            disabled={isMerging}
            style={{
              ...actionBtn(COLORS.secondary2),
              opacity: isMerging ? 0.6 : 1,
              cursor: isMerging ? 'not-allowed' : 'pointer',
            }}
          >
            <GitMerge size={14} />
            {isMerging ? 'Merging...' : isMergeFailed ? 'Retry Merge' : 'Merge'}
          </button>
        )}
        {hasMerged && (
          <button onClick={e => onEval(run, e)} style={actionBtn(COLORS.accent)}>
            <FlaskConical size={14} /> Evaluate
          </button>
        )}
        <ChevronRight size={16} color={COLORS.textLight} />
      </div>
    </div>
  )
}


// ──────────────────── EvalModal (reused logic) ────────────────────

function EvalModal({ run, onClose, onSuccess }) {
  const [testPath, setTestPath] = useState(run.test_data_path || '')
  const [datasetName, setDatasetName] = useState('')
  const [tokenizeMode, setTokenizeMode] = useState('auto')
  const [gpuId, setGpuId] = useState('')
  const [submitting, setSubmitting] = useState(false)
  const [err, setErr] = useState('')

  const handleSubmit = async () => {
    if (!testPath.trim()) { setErr('Please select a test dataset'); return }
    setSubmitting(true)
    setErr('')
    try {
      const res = await api.evaluateTrainRun(run.id, {
        test_data_path: testPath.trim(),
        dataset_name: datasetName.trim(),
        tokenize_mode: tokenizeMode,
        gpu_id: gpuId || undefined,
      })
      onSuccess(res.id)
    } catch (e) {
      setErr(e.response?.data?.detail || e.message || 'Failed')
      setSubmitting(false)
    }
  }

  return (
    <div style={modalOverlay} onClick={onClose}>
      <div style={modalBox} onClick={e => e.stopPropagation()}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 20 }}>
          <h2 style={{ fontSize: 18, fontWeight: 700, color: COLORS.textDark, margin: 0 }}>
            <FlaskConical size={18} style={{ marginRight: 8, verticalAlign: -3 }} />
            Evaluate Model
          </h2>
          <button onClick={onClose} style={{ background: 'none', border: 'none', cursor: 'pointer', color: COLORS.textLight }}>
            <X size={18} />
          </button>
        </div>

        <div style={{ marginBottom: 12, padding: '10px 14px', background: COLORS.bg, borderRadius: 8, fontSize: 13 }}>
          <div style={{ color: COLORS.textLight, fontSize: 11, marginBottom: 4 }}>Model: {run.name}</div>
          <code style={{ color: COLORS.textDark, fontSize: 12, wordBreak: 'break-all' }}>
            {run.merged_model_path}
          </code>
        </div>

        <div style={{ marginBottom: 16 }}>
          <DatasetPicker
            kind="train_manifest"
            value={testPath}
            onChange={setTestPath}
            label="Test dataset (JSONL manifest)"
          />
          <div style={{ fontSize: 11, color: COLORS.warning, marginTop: 6, lineHeight: 1.5 }}>
            Note: The test_data used during training was for validation.
            For final evaluation, select your held-out <b>test.jsonl</b> manifest.
          </div>
        </div>

        <div style={{ marginBottom: 16 }}>
          <label style={labelStyle}>Dataset name (optional)</label>
          <input
            style={inputStyle}
            placeholder="e.g. mn-test-cv22"
            value={datasetName}
            onChange={e => setDatasetName(e.target.value)}
          />
        </div>

        <div style={{ marginBottom: 16 }}>
          <label style={labelStyle}>Tokenize mode</label>
          <select style={inputStyle} value={tokenizeMode} onChange={e => setTokenizeMode(e.target.value)}>
            <option value="auto">auto (detect language)</option>
            <option value="char">char (character-level)</option>
            <option value="space">space (space-delimited)</option>
            <option value="whisper">whisper (Whisper tokenizer)</option>
          </select>
        </div>

        <div style={{ marginBottom: 20 }}>
          <GpuPicker value={gpuId} onChange={setGpuId} label="GPU for inference" />
        </div>

        {err && <div style={{ ...errorBox, marginBottom: 16 }}>{err}</div>}

        <div style={{ display: 'flex', justifyContent: 'flex-end', gap: 10 }}>
          <button onClick={onClose} style={cancelBtn}>Cancel</button>
          <button onClick={handleSubmit} disabled={submitting} style={submitBtn}>
            {submitting ? 'Starting...' : 'Run Evaluation'}
          </button>
        </div>
      </div>
    </div>
  )
}


// ──────────────────── Styles ────────────────────

const cardStyle = {
  background: COLORS.card,
  borderRadius: 12,
  padding: '18px 22px',
  border: `1px solid ${COLORS.border}`,
  display: 'flex',
  alignItems: 'center',
  gap: 16,
  cursor: 'pointer',
  transition: 'box-shadow 0.15s',
}

const badge = (color) => ({
  display: 'inline-flex', alignItems: 'center', gap: 4,
  padding: '2px 10px', borderRadius: 10, fontSize: 11, fontWeight: 600,
  background: color + '18', color,
})

const pathRow = {
  display: 'flex', alignItems: 'center', gap: 6,
  marginTop: 2,
}

const codePath = {
  fontSize: 11, color: COLORS.textMid, wordBreak: 'break-all',
  fontFamily: 'ui-monospace, SFMono-Regular, Menlo, monospace',
}

const actionBtn = (bg) => ({
  display: 'flex', alignItems: 'center', gap: 6,
  padding: '7px 14px', borderRadius: 8, fontSize: 12, fontWeight: 600,
  border: 'none', background: bg, color: '#fff', cursor: 'pointer',
  whiteSpace: 'nowrap',
})

const refreshBtn = {
  display: 'flex', alignItems: 'center', gap: 6,
  padding: '8px 16px', borderRadius: 8,
  border: `1px solid ${COLORS.secondary2}`,
  background: 'transparent', color: COLORS.secondary2,
  cursor: 'pointer', fontSize: 13, fontWeight: 600,
}

const emptyState = {
  textAlign: 'center', padding: 60,
  color: COLORS.textLight,
  background: COLORS.card, borderRadius: 12,
  border: `1px solid ${COLORS.border}`,
}

const errorBox = {
  marginBottom: 16, padding: '12px 16px', borderRadius: 8,
  background: COLORS.danger + '18',
  border: `1px solid ${COLORS.danger}40`,
  color: COLORS.danger, fontSize: 13,
}

const modalOverlay = {
  position: 'fixed', inset: 0, zIndex: 1000,
  background: 'rgba(0,0,0,0.45)',
  display: 'flex', alignItems: 'center', justifyContent: 'center',
}

const modalBox = {
  background: '#fff', borderRadius: 14, padding: '28px 32px',
  width: 520, maxHeight: '85vh', overflow: 'auto',
  boxShadow: '0 20px 60px rgba(0,0,0,0.25)',
}

const labelStyle = {
  display: 'block', fontSize: 13, fontWeight: 600,
  color: COLORS.textMid, marginBottom: 6,
}

const inputStyle = {
  width: '100%', padding: '9px 12px', borderRadius: 8,
  border: `1px solid ${COLORS.border}`, fontSize: 13,
  color: COLORS.textDark, outline: 'none',
  boxSizing: 'border-box',
}

const cancelBtn = {
  padding: '8px 18px', borderRadius: 8, fontSize: 13, fontWeight: 600,
  border: `1px solid ${COLORS.border}`, background: '#fff',
  color: COLORS.textMid, cursor: 'pointer',
}

const submitBtn = {
  padding: '8px 22px', borderRadius: 8, fontSize: 13, fontWeight: 600,
  border: 'none', background: COLORS.accent, color: '#fff',
  cursor: 'pointer',
}
