import { useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { Play, ChevronDown } from 'lucide-react'
import { api } from '../api'
import { COLORS } from '../theme'
import DatasetPicker from '../components/DatasetPicker'

const TOKENIZE_OPTIONS = [
  { value: 'auto',   label: 'Auto (detect language)' },
  { value: 'space',  label: 'Space (Mongolian, Arabic, Russian…)' },
  { value: 'char',   label: 'Char (Chinese, Japanese, Thai…)' },
  { value: 'tsheg',  label: 'Tsheg (Tibetan)' },
]

export default function NewEval() {
  const [form, setForm] = useState({
    model_name: '',
    dataset_name: '',
    dataset_path: '',
    tokenize_mode: 'auto',
  })
  const [submitting, setSubmitting] = useState(false)
  const [error, setError] = useState('')
  const navigate = useNavigate()

  const set = (k, v) => setForm(f => ({ ...f, [k]: v }))

  const handleSubmit = async (e) => {
    e.preventDefault()
    if (!form.model_name.trim()) { setError('Model name is required'); return }
    if (!form.dataset_path.trim()) { setError('Dataset path is required'); return }
    setError('')
    setSubmitting(true)
    try {
      const ev = await api.createEval({
        model_name: form.model_name.trim(),
        dataset_name: form.dataset_name.trim() || form.model_name.trim(),
        dataset_path: form.dataset_path.trim(),
        tokenize_mode: form.tokenize_mode,
      })
      navigate('/dashboard')
    } catch (err) {
      setError(err.response?.data?.detail || 'Failed to create evaluation')
    } finally {
      setSubmitting(false)
    }
  }

  return (
    <div style={{ maxWidth: 700 }}>
      <div style={{ marginBottom: 28 }}>
        <h1 style={{ fontSize: 24, fontWeight: 700, color: COLORS.textDark }}>New Evaluation</h1>
        <p style={{ color: COLORS.textLight, fontSize: 14, marginTop: 4 }}>
          Configure and launch a new ASR evaluation job
        </p>
      </div>

      <form onSubmit={handleSubmit}>
        <div style={card}>
          <h2 style={sectionTitle}>Model Info</h2>

          <div style={field}>
            <label style={labelStyle}>Model Name *</label>
            <input
              style={input}
              placeholder="e.g. Ganaa0614, whisper-large-v3"
              value={form.model_name}
              onChange={e => set('model_name', e.target.value)}
            />
          </div>

          <div style={field}>
            <label style={labelStyle}>Display Name (optional)</label>
            <input
              style={input}
              placeholder="Defaults to model name"
              value={form.dataset_name}
              onChange={e => set('dataset_name', e.target.value)}
            />
          </div>
        </div>

        <div style={{ ...card, marginTop: 16 }}>
          <h2 style={sectionTitle}>Dataset</h2>

          <div style={field}>
            <DatasetPicker
              kind="eval_csv"
              label="Dataset"
              required
              value={form.dataset_path}
              onChange={v => set('dataset_path', v)}
              placeholder="/dataset/csv-results/common_test.csv"
            />
          </div>

          <div style={field}>
            <label style={labelStyle}>Tokenize Mode</label>
            <div style={{ position: 'relative' }}>
              <select
                style={{ ...input, appearance: 'none', paddingRight: 36, cursor: 'pointer' }}
                value={form.tokenize_mode}
                onChange={e => set('tokenize_mode', e.target.value)}
              >
                {TOKENIZE_OPTIONS.map(o => (
                  <option key={o.value} value={o.value}>{o.label}</option>
                ))}
              </select>
              <ChevronDown size={15} style={{ position: 'absolute', right: 12, top: '50%', transform: 'translateY(-50%)', color: COLORS.textLight, pointerEvents: 'none' }} />
            </div>
            <span style={{ fontSize: 11, color: COLORS.textLight, marginTop: 4, display: 'block' }}>
              Use <strong>space</strong> for Mongolian
            </span>
          </div>
        </div>

        {error && (
          <div style={{ marginTop: 16, padding: '12px 16px', borderRadius: 8, background: COLORS.danger + '18', border: `1px solid ${COLORS.danger}40`, color: COLORS.danger, fontSize: 13 }}>
            {error}
          </div>
        )}

        <div style={{ display: 'flex', gap: 12, marginTop: 20 }}>
          <button
            type="button"
            onClick={() => navigate('/dashboard')}
            style={{ padding: '10px 24px', borderRadius: 8, border: `1px solid ${COLORS.border}`, background: '#fff', color: COLORS.textMid, cursor: 'pointer', fontSize: 14 }}
          >
            Cancel
          </button>
          <button
            type="submit"
            disabled={submitting}
            style={{
              display: 'flex', alignItems: 'center', gap: 8,
              padding: '10px 28px', borderRadius: 8, border: 'none',
              background: submitting ? COLORS.textLight : COLORS.accent,
              color: '#fff', cursor: submitting ? 'not-allowed' : 'pointer',
              fontSize: 14, fontWeight: 600,
            }}
          >
            <Play size={15} />
            {submitting ? 'Submitting…' : 'Run Evaluation'}
          </button>
        </div>
      </form>
    </div>
  )
}

const card = {
  background: '#fff', borderRadius: 12, padding: '24px 28px',
  border: `1px solid ${COLORS.border}`,
  boxShadow: '0 2px 8px rgba(100,130,173,0.07)',
}
const sectionTitle = {
  fontSize: 15, fontWeight: 700, color: COLORS.textDark, marginBottom: 20,
  paddingBottom: 12, borderBottom: `1px solid ${COLORS.border}`,
}
const field = { marginBottom: 20 }
const labelStyle = {
  display: 'block', fontSize: 13, fontWeight: 600,
  color: COLORS.textMid, marginBottom: 6,
}
const input = {
  width: '100%', padding: '9px 12px', borderRadius: 8,
  border: `1px solid ${COLORS.border}`, fontSize: 14,
  color: COLORS.textDark, outline: 'none', background: '#fff',
  fontFamily: 'inherit',
}
