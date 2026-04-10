import { useState, useEffect } from 'react'
import { useNavigate } from 'react-router-dom'
import { Play, FolderOpen, ChevronDown } from 'lucide-react'
import { api } from '../api'
import { COLORS } from '../theme'

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
  const [datasets, setDatasets] = useState([])
  const [submitting, setSubmitting] = useState(false)
  const [error, setError] = useState('')
  const navigate = useNavigate()

  useEffect(() => {
    api.listDatasets().then(d => setDatasets(d.datasets || [])).catch(() => {})
  }, [])

  const set = (k, v) => setForm(f => ({ ...f, [k]: v }))

  const pickDataset = (ds) => {
    set('dataset_path', ds.path)
    if (!form.dataset_name) set('dataset_name', ds.name)
  }

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

          {datasets.length > 0 && (
            <div style={field}>
              <label style={labelStyle}>Quick-pick from scanned datasets</label>
              <div style={{ display: 'flex', flexWrap: 'wrap', gap: 8 }}>
                {datasets.map(ds => (
                  <button
                    key={ds.path}
                    type="button"
                    onClick={() => pickDataset(ds)}
                    style={{
                      padding: '6px 12px', borderRadius: 8, fontSize: 12, cursor: 'pointer',
                      border: `1px solid ${form.dataset_path === ds.path ? COLORS.accent : COLORS.border}`,
                      background: form.dataset_path === ds.path ? COLORS.accent + '18' : '#fff',
                      color: form.dataset_path === ds.path ? COLORS.accent : COLORS.textMid,
                      fontWeight: form.dataset_path === ds.path ? 600 : 400,
                    }}
                  >
                    <FolderOpen size={12} style={{ marginRight: 5, verticalAlign: 'middle' }} />
                    {ds.name}
                    {ds.total_rows && <span style={{ color: COLORS.textLight, marginLeft: 4 }}>({ds.total_rows.toLocaleString()} rows)</span>}
                  </button>
                ))}
              </div>
            </div>
          )}

          <div style={field}>
            <label style={labelStyle}>Dataset Path *</label>
            <input
              style={input}
              placeholder="/dataset/csv-results/common_test.csv"
              value={form.dataset_path}
              onChange={e => set('dataset_path', e.target.value)}
            />
            <span style={{ fontSize: 11, color: COLORS.textLight, marginTop: 4, display: 'block' }}>
              Full path to a CSV file or directory on the server
            </span>
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
