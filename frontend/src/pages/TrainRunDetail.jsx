import { useCallback, useEffect, useRef, useState } from 'react'
import { useNavigate, useParams } from 'react-router-dom'
import { ArrowLeft, RefreshCw, Terminal, Activity, FlaskConical, X } from 'lucide-react'
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer,
} from 'recharts'
import { api } from '../api'
import { COLORS } from '../theme'
import StatusBadge from '../components/StatusBadge'
import DatasetPicker from '../components/DatasetPicker'
import GpuPicker from '../components/GpuPicker'

const fmtDate = (v) => (v ? new Date(v).toLocaleString() : '—')
const fmtNum = (v, d = 4) => (v == null || Number.isNaN(v) ? '—' : Number(v).toFixed(d))

export default function TrainRunDetail() {
  const { id } = useParams()
  const navigate = useNavigate()
  const [run, setRun] = useState(null)
  const [metrics, setMetrics] = useState({ train: [], eval: [] })
  const [log, setLog] = useState({ lines: [], total_lines: 0, path: null })
  const [error, setError] = useState('')
  const [autoScroll, setAutoScroll] = useState(true)
  const [showEvalModal, setShowEvalModal] = useState(false)
  const logBoxRef = useRef(null)

  const load = useCallback(async () => {
    try {
      const [r, m, l] = await Promise.all([
        api.getTrainRun(id),
        api.getTrainRunMetrics(id),
        api.getTrainRunLog(id, 500),
      ])
      setRun(r)
      setMetrics(m)
      setLog(l)
      setError('')
    } catch (err) {
      console.error(err)
      setError(err.response?.data?.detail || 'Failed to load train run')
    }
  }, [id])

  useEffect(() => { load() }, [load])

  // 在运行阶段每 2s 刷新一次
  useEffect(() => {
    if (!run) return
    if (run.status !== 'running' && run.status !== 'queued') return
    const t = setInterval(load, 2000)
    return () => clearInterval(t)
  }, [run, load])

  // 自动滚到底
  useEffect(() => {
    if (autoScroll && logBoxRef.current) {
      logBoxRef.current.scrollTop = logBoxRef.current.scrollHeight
    }
  }, [log, autoScroll])

  if (error && !run) {
    return (
      <div>
        <button onClick={() => navigate('/train-runs')} style={backBtn}>
          <ArrowLeft size={15} /> Back
        </button>
        <div style={errorBox}>{error}</div>
      </div>
    )
  }

  if (!run) {
    return <div style={{ color: COLORS.textLight }}>Loading…</div>
  }

  const pct = run.total_steps > 0
    ? Math.min(100, (run.current_step / run.total_steps) * 100)
    : 0

  return (
    <div>
      {/* Header */}
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 24 }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 16 }}>
          <button onClick={() => navigate('/train-runs')} style={backBtn}>
            <ArrowLeft size={15} /> Back
          </button>
          <div>
            <h1 style={{ fontSize: 22, fontWeight: 700, color: COLORS.textDark, margin: 0 }}>
              {run.name} <span style={{ color: COLORS.textLight, fontSize: 14, fontWeight: 400 }}>#{run.id}</span>
            </h1>
            <div style={{ display: 'flex', alignItems: 'center', gap: 10, marginTop: 6 }}>
              <StatusBadge status={run.status} />
              {run.phase && <span style={phaseChip}>phase: {run.phase}</span>}
              {run.pid && <span style={pidChip}>pid: {run.pid}</span>}
              <span style={{ color: COLORS.textLight, fontSize: 12 }}>
                created {fmtDate(run.created_at)}
              </span>
            </div>
          </div>
        </div>
        <div style={{ display: 'flex', gap: 10 }}>
          {run.status === 'completed' && run.merged_model_path && (
            <button onClick={() => setShowEvalModal(true)} style={evalBtn}>
              <FlaskConical size={15} /> Evaluate
            </button>
          )}
          <button onClick={load} style={refreshBtn}>
            <RefreshCw size={15} /> Refresh
          </button>
        </div>
      </div>

      {/* Progress + quick stats */}
      <div style={twoCol}>
        <div style={card}>
          <h2 style={sectionTitle}><Activity size={16} /> Progress</h2>
          <div style={{ marginTop: 8 }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 13, color: COLORS.textMid }}>
              <span>{run.current_step} / {run.total_steps || '?'}</span>
              <span>{pct.toFixed(1)}%</span>
            </div>
            <div style={progressTrack}>
              <div style={{ ...progressFill, width: `${pct}%` }} />
            </div>
          </div>

          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: 14, marginTop: 22 }}>
            <Stat label="Epoch" value={fmtNum(run.current_epoch, 3)} />
            <Stat label="Loss" value={fmtNum(run.current_loss)} />
            <Stat label="Eval Loss" value={fmtNum(run.current_eval_loss)} />
          </div>

          <div style={{ marginTop: 20, fontSize: 12, color: COLORS.textLight, lineHeight: 1.7 }}>
            <div>Started: {fmtDate(run.started_at)}</div>
            <div>Completed: {fmtDate(run.completed_at)}</div>
            {run.checkpoint_path && <div>Adapter: <code>{run.checkpoint_path}</code></div>}
            {run.merged_model_path && <div>Merged model: <code>{run.merged_model_path}</code></div>}
          </div>

          {run.error_message && (
            <div style={errorBox}>{run.error_message}</div>
          )}
        </div>

        <div style={card}>
          <h2 style={sectionTitle}>Config</h2>
          <div style={kvList}>
            <KV k="Base model" v={run.base_model} />
            <KV k="Train data" v={run.train_data_path} mono />
            <KV k="Test data" v={run.test_data_path} mono />
            <KV k="Output dir" v={run.output_dir} mono />
            <KV k="Language / Task" v={`${run.language} · ${run.task}`} />
            <KV k="Epochs" v={run.num_train_epochs} />
            <KV k="Learning rate" v={run.learning_rate} />
            <KV k="Train bs / Eval bs" v={`${run.per_device_train_batch_size} / ${run.per_device_eval_batch_size}`} />
            <KV k="Grad accum" v={run.gradient_accumulation_steps} />
            <KV k="AdaLoRA / fp16 / 8-bit" v={`${run.use_adalora} · ${run.fp16} · ${run.use_8bit}`} />
          </div>
        </div>
      </div>

      {/* Loss curve */}
      <div style={{ ...card, marginTop: 16 }}>
        <h2 style={sectionTitle}>Loss Curve</h2>
        {metrics.train.length === 0 ? (
          <div style={emptyHint}>No loss logs yet. Waiting for the first logging_steps.</div>
        ) : (
          <div style={{ width: '100%', height: 280 }}>
            <ResponsiveContainer>
              <LineChart
                data={mergeLossSeries(metrics.train, metrics.eval)}
                margin={{ top: 10, right: 20, left: 0, bottom: 0 }}
              >
                <CartesianGrid strokeDasharray="3 3" stroke={COLORS.border} />
                <XAxis dataKey="epoch" stroke={COLORS.textMid} tick={{ fontSize: 12 }}
                  label={{ value: 'epoch', position: 'insideBottomRight', offset: -4, fill: COLORS.textLight, fontSize: 11 }} />
                <YAxis stroke={COLORS.textMid} tick={{ fontSize: 12 }} />
                <Tooltip />
                <Legend />
                <Line type="monotone" dataKey="loss" stroke={COLORS.accent} dot={false} name="train loss" />
                <Line type="monotone" dataKey="eval_loss" stroke={COLORS.danger} dot={{ r: 3 }} name="eval loss" />
              </LineChart>
            </ResponsiveContainer>
          </div>
        )}
      </div>

      {/* Log tail */}
      <div style={{ ...card, marginTop: 16 }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
          <h2 style={{ ...sectionTitle, marginBottom: 0 }}><Terminal size={16} /> Log ({log.total_lines} lines)</h2>
          <label style={{ fontSize: 12, color: COLORS.textMid, display: 'flex', gap: 6, alignItems: 'center' }}>
            <input
              type="checkbox"
              checked={autoScroll}
              onChange={e => setAutoScroll(e.target.checked)}
            />
            auto-scroll
          </label>
        </div>
        <div ref={logBoxRef} style={logBox}>
          {log.lines.length === 0 ? (
            <div style={{ color: '#888' }}>No log yet.</div>
          ) : (
            log.lines.map((line, i) => (
              <div key={i} style={logLine(line)}>{line}</div>
            ))
          )}
        </div>
        {log.path && (
          <div style={{ fontSize: 11, color: COLORS.textLight, marginTop: 8 }}>
            path: <code>{log.path}</code>
          </div>
        )}
      </div>

      {/* Eval modal */}
      {showEvalModal && (
        <EvalModal
          run={run}
          onClose={() => setShowEvalModal(false)}
          onSuccess={(evalId) => navigate(`/report/${evalId}`)}
        />
      )}
    </div>
  )
}

// ──────────────────── EvalModal ────────────────────

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
          <div style={{ color: COLORS.textLight, fontSize: 11, marginBottom: 4 }}>Merged model</div>
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


// ──────────────────── helpers ────────────────────

function mergeLossSeries(train, evalPts) {
  // 把 train loss 和 eval loss 按 epoch 合并到同一条时间轴上，
  // 便于 recharts 画双线。
  const map = new Map()
  for (const p of train) {
    const key = p.epoch ?? 0
    map.set(key, { epoch: key, loss: p.loss })
  }
  for (const p of evalPts) {
    const key = p.epoch ?? 0
    const prev = map.get(key) || { epoch: key }
    map.set(key, { ...prev, eval_loss: p.eval_loss })
  }
  return Array.from(map.values()).sort((a, b) => a.epoch - b.epoch)
}

function Stat({ label, value }) {
  return (
    <div>
      <div style={{ fontSize: 11, color: COLORS.textLight, textTransform: 'uppercase', fontWeight: 600 }}>{label}</div>
      <div style={{ fontSize: 22, fontWeight: 700, color: COLORS.textDark, marginTop: 4 }}>{value}</div>
    </div>
  )
}

function KV({ k, v, mono }) {
  return (
    <div style={{ display: 'flex', gap: 10, fontSize: 13 }}>
      <div style={{ width: 140, color: COLORS.textLight, flexShrink: 0 }}>{k}</div>
      <div style={{
        color: COLORS.textDark,
        wordBreak: 'break-all',
        fontFamily: mono ? 'ui-monospace, SFMono-Regular, Menlo, monospace' : 'inherit',
        fontSize: mono ? 12 : 13,
      }}>
        {String(v ?? '—')}
      </div>
    </div>
  )
}

function logLine(line) {
  let color = COLORS.textDark
  if (line.includes('[stderr]')) color = '#b86b22'
  if (line.includes('[merge/')) color = COLORS.secondary2
  if (line.includes('Traceback') || line.includes('Error')) color = COLORS.danger
  if (line.startsWith('#')) color = COLORS.textLight
  return {
    color,
    whiteSpace: 'pre-wrap',
    wordBreak: 'break-all',
    padding: '1px 0',
  }
}

// ──────────────────── styles ────────────────────

const card = {
  background: COLORS.card,
  borderRadius: 12,
  padding: '22px 24px',
  border: `1px solid ${COLORS.border}`,
}

const twoCol = {
  display: 'grid',
  gridTemplateColumns: '1.1fr 1fr',
  gap: 16,
}

const sectionTitle = {
  fontSize: 15,
  fontWeight: 700,
  color: COLORS.textDark,
  margin: 0,
  marginBottom: 16,
  paddingBottom: 10,
  borderBottom: `1px solid ${COLORS.border}`,
  display: 'flex',
  alignItems: 'center',
  gap: 8,
}

const kvList = {
  display: 'flex',
  flexDirection: 'column',
  gap: 10,
}

const progressTrack = {
  marginTop: 8,
  height: 10,
  borderRadius: 6,
  background: COLORS.secondary1 + '80',
  overflow: 'hidden',
}

const progressFill = {
  height: '100%',
  background: `linear-gradient(90deg, ${COLORS.primary}, ${COLORS.accent})`,
  transition: 'width 0.4s ease',
}

const logBox = {
  marginTop: 12,
  background: '#1E2A3A',
  color: '#E8EEF7',
  borderRadius: 8,
  padding: 14,
  fontFamily: 'ui-monospace, SFMono-Regular, Menlo, monospace',
  fontSize: 12,
  lineHeight: 1.5,
  maxHeight: 420,
  overflow: 'auto',
}

const backBtn = {
  display: 'flex', alignItems: 'center', gap: 6,
  padding: '7px 14px', borderRadius: 8,
  border: `1px solid ${COLORS.border}`,
  background: '#fff', color: COLORS.textMid,
  cursor: 'pointer', fontSize: 13,
}

const refreshBtn = {
  display: 'flex', alignItems: 'center', gap: 6,
  padding: '8px 16px', borderRadius: 8,
  border: `1px solid ${COLORS.secondary2}`,
  background: 'transparent', color: COLORS.secondary2,
  cursor: 'pointer', fontSize: 13, fontWeight: 600,
}

const phaseChip = {
  padding: '3px 10px',
  borderRadius: 20,
  fontSize: 12,
  fontWeight: 600,
  background: COLORS.secondary1 + '80',
  color: COLORS.secondary2,
}

const pidChip = {
  padding: '3px 10px',
  borderRadius: 20,
  fontSize: 11,
  fontWeight: 600,
  background: COLORS.bg,
  color: COLORS.textLight,
  border: `1px solid ${COLORS.border}`,
}

const emptyHint = {
  padding: 24,
  textAlign: 'center',
  color: COLORS.textLight,
  fontSize: 13,
}

const errorBox = {
  marginTop: 16,
  padding: '12px 16px',
  borderRadius: 8,
  background: COLORS.danger + '18',
  border: `1px solid ${COLORS.danger}40`,
  color: COLORS.danger,
  fontSize: 13,
  whiteSpace: 'pre-wrap',
}

const evalBtn = {
  display: 'flex', alignItems: 'center', gap: 6,
  padding: '8px 16px', borderRadius: 8,
  border: 'none',
  background: COLORS.accent, color: '#fff',
  cursor: 'pointer', fontSize: 13, fontWeight: 600,
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
