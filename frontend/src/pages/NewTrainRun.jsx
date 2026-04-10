import { useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { Play, ChevronDown } from 'lucide-react'
import { api } from '../api'
import { COLORS } from '../theme'

export default function NewTrainRun() {
  const [form, setForm] = useState({
    name: '',
    base_model: 'openai/whisper-small',
    train_data_path: '',
    test_data_path: '',
    output_dir: 'output/',
    language: 'Chinese',
    task: 'transcribe',
    timestamps: false,
    num_train_epochs: 3,
    learning_rate: '0.001',
    per_device_train_batch_size: 8,
    per_device_eval_batch_size: 8,
    gradient_accumulation_steps: 1,
    use_adalora: true,
    fp16: true,
    use_8bit: false,
    use_compile: false,
    local_files_only: false,
    push_to_hub: false,
  })
  const [submitting, setSubmitting] = useState(false)
  const [error, setError] = useState('')
  const navigate = useNavigate()

  const set = (key, value) => setForm(prev => ({ ...prev, [key]: value }))

  const submit = async (e) => {
    e.preventDefault()
    if (!form.name.trim()) { setError('Run name is required'); return }
    if (!form.base_model.trim()) { setError('Base model is required'); return }
    if (!form.train_data_path.trim()) { setError('Train data path is required'); return }
    if (!form.test_data_path.trim()) { setError('Test data path is required'); return }

    setError('')
    setSubmitting(true)
    try {
      await api.createTrainRun({
        ...form,
        name: form.name.trim(),
        base_model: form.base_model.trim(),
        train_data_path: form.train_data_path.trim(),
        test_data_path: form.test_data_path.trim(),
        output_dir: form.output_dir.trim(),
        learning_rate: Number(form.learning_rate),
        num_train_epochs: Number(form.num_train_epochs),
        per_device_train_batch_size: Number(form.per_device_train_batch_size),
        per_device_eval_batch_size: Number(form.per_device_eval_batch_size),
        gradient_accumulation_steps: Number(form.gradient_accumulation_steps),
      })
      navigate('/train-runs')
    } catch (err) {
      setError(err.response?.data?.detail || 'Failed to create train run')
    } finally {
      setSubmitting(false)
    }
  }

  return (
    <div style={{ maxWidth: 860 }}>
      <div style={{ marginBottom: 28 }}>
        <h1 style={{ fontSize: 24, fontWeight: 700, color: COLORS.textDark }}>New Train Run</h1>
        <p style={{ color: COLORS.textLight, fontSize: 14, marginTop: 4 }}>
          Store a fine-tuning job configuration for Whisper-Finetune
        </p>
      </div>

      <form onSubmit={submit}>
        <div style={card}>
          <h2 style={sectionTitle}>Core Config</h2>

          <div style={grid2}>
            <div style={field}>
              <label style={labelStyle}>Run Name *</label>
              <input
                style={input}
                placeholder="e.g. tibetan-whisper-small-v2"
                value={form.name}
                onChange={e => set('name', e.target.value)}
              />
            </div>

            <div style={field}>
              <label style={labelStyle}>Base Model *</label>
              <input
                style={input}
                placeholder="openai/whisper-small"
                value={form.base_model}
                onChange={e => set('base_model', e.target.value)}
              />
            </div>
          </div>

          <div style={field}>
            <label style={labelStyle}>Train Data Path *</label>
            <input
              style={input}
              placeholder="/data/manifests/tibetan/train.json"
              value={form.train_data_path}
              onChange={e => set('train_data_path', e.target.value)}
            />
          </div>

          <div style={field}>
            <label style={labelStyle}>Test Data Path *</label>
            <input
              style={input}
              placeholder="/data/manifests/tibetan/test.json"
              value={form.test_data_path}
              onChange={e => set('test_data_path', e.target.value)}
            />
          </div>

          <div style={field}>
            <label style={labelStyle}>Output Directory</label>
            <input
              style={input}
              placeholder="output/"
              value={form.output_dir}
              onChange={e => set('output_dir', e.target.value)}
            />
          </div>
        </div>

        <div style={{ ...card, marginTop: 16 }}>
          <h2 style={sectionTitle}>Training Parameters</h2>

          <div style={grid3}>
            <div style={field}>
              <label style={labelStyle}>Language</label>
              <input
                style={input}
                value={form.language}
                onChange={e => set('language', e.target.value)}
              />
            </div>

            <div style={field}>
              <label style={labelStyle}>Task</label>
              <div style={{ position: 'relative' }}>
                <select
                  style={{ ...input, appearance: 'none', paddingRight: 36, cursor: 'pointer' }}
                  value={form.task}
                  onChange={e => set('task', e.target.value)}
                >
                  <option value="transcribe">transcribe</option>
                  <option value="translate">translate</option>
                </select>
                <ChevronDown size={15} style={selectChevron} />
              </div>
            </div>

            <div style={field}>
              <label style={labelStyle}>Epochs</label>
              <input
                style={input}
                type="number"
                min="1"
                value={form.num_train_epochs}
                onChange={e => set('num_train_epochs', e.target.value)}
              />
            </div>
          </div>

          <div style={grid3}>
            <div style={field}>
              <label style={labelStyle}>Learning Rate</label>
              <input
                style={input}
                type="number"
                min="0.000001"
                step="0.000001"
                value={form.learning_rate}
                onChange={e => set('learning_rate', e.target.value)}
              />
            </div>

            <div style={field}>
              <label style={labelStyle}>Train Batch Size</label>
              <input
                style={input}
                type="number"
                min="1"
                value={form.per_device_train_batch_size}
                onChange={e => set('per_device_train_batch_size', e.target.value)}
              />
            </div>

            <div style={field}>
              <label style={labelStyle}>Eval Batch Size</label>
              <input
                style={input}
                type="number"
                min="1"
                value={form.per_device_eval_batch_size}
                onChange={e => set('per_device_eval_batch_size', e.target.value)}
              />
            </div>
          </div>

          <div style={grid3}>
            <div style={field}>
              <label style={labelStyle}>Gradient Accumulation</label>
              <input
                style={input}
                type="number"
                min="1"
                value={form.gradient_accumulation_steps}
                onChange={e => set('gradient_accumulation_steps', e.target.value)}
              />
            </div>

            <div style={{ ...field, display: 'flex', alignItems: 'center', gap: 10, marginTop: 28 }}>
              <input
                id="timestamps"
                type="checkbox"
                checked={form.timestamps}
                onChange={e => set('timestamps', e.target.checked)}
              />
              <label htmlFor="timestamps" style={checkboxLabel}>Use timestamps</label>
            </div>

            <div />
          </div>
        </div>

        <div style={{ ...card, marginTop: 16 }}>
          <h2 style={sectionTitle}>Runtime Switches</h2>
          <div style={grid3}>
            <Checkbox id="use_adalora" label="Use AdaLora" checked={form.use_adalora} onChange={checked => set('use_adalora', checked)} />
            <Checkbox id="fp16" label="Use fp16" checked={form.fp16} onChange={checked => set('fp16', checked)} />
            <Checkbox id="use_8bit" label="Use 8-bit" checked={form.use_8bit} onChange={checked => set('use_8bit', checked)} />
            <Checkbox id="use_compile" label="Use compile" checked={form.use_compile} onChange={checked => set('use_compile', checked)} />
            <Checkbox id="local_files_only" label="Local files only" checked={form.local_files_only} onChange={checked => set('local_files_only', checked)} />
            <Checkbox id="push_to_hub" label="Push to Hub" checked={form.push_to_hub} onChange={checked => set('push_to_hub', checked)} />
          </div>
        </div>

        {error && (
          <div style={errorBox}>
            {error}
          </div>
        )}

        <div style={{ display: 'flex', gap: 12, marginTop: 20 }}>
          <button
            type="button"
            onClick={() => navigate('/train-runs')}
            style={ghostBtn}
          >
            Cancel
          </button>
          <button
            type="submit"
            disabled={submitting}
            style={{
              display: 'flex',
              alignItems: 'center',
              gap: 8,
              padding: '10px 28px',
              borderRadius: 8,
              border: 'none',
              background: submitting ? COLORS.textLight : COLORS.accent,
              color: '#fff',
              cursor: submitting ? 'not-allowed' : 'pointer',
              fontSize: 14,
              fontWeight: 600,
            }}
          >
            <Play size={15} />
            {submitting ? 'Submitting…' : 'Create Train Run'}
          </button>
        </div>
      </form>
    </div>
  )
}

function Checkbox({ id, label, checked, onChange }) {
  return (
    <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
      <input
        id={id}
        type="checkbox"
        checked={checked}
        onChange={e => onChange(e.target.checked)}
      />
      <label htmlFor={id} style={checkboxLabel}>{label}</label>
    </div>
  )
}

const card = {
  background: '#fff',
  borderRadius: 12,
  padding: '24px 28px',
  border: `1px solid ${COLORS.border}`,
  boxShadow: '0 2px 8px rgba(100,130,173,0.07)',
}

const sectionTitle = {
  fontSize: 15,
  fontWeight: 700,
  color: COLORS.textDark,
  marginBottom: 20,
  paddingBottom: 12,
  borderBottom: `1px solid ${COLORS.border}`,
}

const field = { marginBottom: 20 }

const labelStyle = {
  display: 'block',
  fontSize: 13,
  fontWeight: 600,
  color: COLORS.textMid,
  marginBottom: 6,
}

const input = {
  width: '100%',
  padding: '9px 12px',
  borderRadius: 8,
  border: `1px solid ${COLORS.border}`,
  fontSize: 14,
  color: COLORS.textDark,
  outline: 'none',
  background: '#fff',
  fontFamily: 'inherit',
}

const grid2 = {
  display: 'grid',
  gridTemplateColumns: '1fr 1fr',
  gap: 16,
}

const grid3 = {
  display: 'grid',
  gridTemplateColumns: 'repeat(3, 1fr)',
  gap: 16,
}

const errorBox = {
  marginTop: 16,
  padding: '12px 16px',
  borderRadius: 8,
  background: COLORS.danger + '18',
  border: `1px solid ${COLORS.danger}40`,
  color: COLORS.danger,
  fontSize: 13,
}

const ghostBtn = {
  padding: '10px 24px',
  borderRadius: 8,
  border: `1px solid ${COLORS.border}`,
  background: '#fff',
  color: COLORS.textMid,
  cursor: 'pointer',
  fontSize: 14,
}

const checkboxLabel = {
  fontSize: 13,
  color: COLORS.textMid,
  fontWeight: 600,
}

const selectChevron = {
  position: 'absolute',
  right: 12,
  top: '50%',
  transform: 'translateY(-50%)',
  color: COLORS.textLight,
  pointerEvents: 'none',
}
