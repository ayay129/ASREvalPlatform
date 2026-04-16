import { useEffect, useState } from 'react'
import { Cpu } from 'lucide-react'
import { api } from '../api'
import { COLORS } from '../theme'

/**
 * GPU 选择器组件。
 *
 * Props:
 *   value      当前选中的 gpu_id（如 "0"、"0,1"、""）
 *   onChange   (gpuId: string) => void
 *   label      显示标签
 */
export default function GpuPicker({ value, onChange, label = 'GPU' }) {
  const [gpus, setGpus] = useState([])
  const [available, setAvailable] = useState(false)
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    api.gpuStatus()
      .then(data => {
        setAvailable(data.available)
        setGpus(data.gpus || [])
      })
      .catch(() => {
        setAvailable(false)
        setGpus([])
      })
      .finally(() => setLoading(false))
  }, [])

  if (loading) {
    return (
      <div>
        <label style={labelStyle}>{label}</label>
        <div style={{ fontSize: 12, color: COLORS.textLight }}>Detecting GPUs...</div>
      </div>
    )
  }

  if (!available || gpus.length === 0) {
    return (
      <div>
        <label style={labelStyle}>{label}</label>
        <div style={warningBox}>
          <Cpu size={14} />
          <span>No GPU detected. Tasks will run on CPU.</span>
        </div>
      </div>
    )
  }

  return (
    <div>
      <label style={labelStyle}>{label}</label>
      <select style={selectStyle} value={value || ''} onChange={e => onChange(e.target.value)}>
        <option value="">Auto (all GPUs)</option>
        {gpus.map(g => (
          <option key={g.index} value={String(g.index)}>
            GPU {g.index}: {g.name} — {g.utilization_pct}% util, {Math.round(g.memory_used_mb)}MB / {Math.round(g.memory_total_mb)}MB
          </option>
        ))}
      </select>
      {/* Mini status bars */}
      <div style={{ display: 'flex', gap: 8, marginTop: 8, flexWrap: 'wrap' }}>
        {gpus.map(g => {
          const selected = value === String(g.index)
          return (
            <div
              key={g.index}
              onClick={() => onChange(selected ? '' : String(g.index))}
              style={{
                padding: '6px 10px', borderRadius: 8, cursor: 'pointer',
                border: `1.5px solid ${selected ? COLORS.accent : COLORS.border}`,
                background: selected ? COLORS.accent + '10' : COLORS.card,
                fontSize: 11, minWidth: 140, transition: 'all 0.15s',
              }}
            >
              <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 4 }}>
                <span style={{ fontWeight: 600, color: COLORS.textDark }}>GPU {g.index}</span>
                <span style={{ color: g.utilization_pct > 80 ? COLORS.danger : COLORS.textMid }}>
                  {g.utilization_pct}%
                </span>
              </div>
              {/* Util bar */}
              <div style={{ height: 4, borderRadius: 2, background: COLORS.border, overflow: 'hidden' }}>
                <div style={{
                  width: `${g.utilization_pct}%`, height: '100%', borderRadius: 2,
                  background: g.utilization_pct > 80 ? COLORS.danger : g.utilization_pct > 50 ? COLORS.warning : COLORS.success,
                  transition: 'width 0.3s',
                }} />
              </div>
              {/* Memory bar */}
              <div style={{ display: 'flex', justifyContent: 'space-between', marginTop: 4, fontSize: 10, color: COLORS.textLight }}>
                <span>Mem</span>
                <span>{g.memory_pct}%</span>
              </div>
              <div style={{ height: 3, borderRadius: 2, background: COLORS.border, overflow: 'hidden' }}>
                <div style={{
                  width: `${g.memory_pct}%`, height: '100%', borderRadius: 2,
                  background: g.memory_pct > 80 ? COLORS.danger : COLORS.secondary2,
                }} />
              </div>
              {g.temperature != null && (
                <div style={{ fontSize: 10, color: COLORS.textLight, marginTop: 3 }}>
                  {g.temperature}°C
                  {g.power_draw_w != null && ` · ${Math.round(g.power_draw_w)}W`}
                </div>
              )}
            </div>
          )
        })}
      </div>
    </div>
  )
}

const labelStyle = {
  display: 'block', fontSize: 13, fontWeight: 600,
  color: COLORS.textMid, marginBottom: 6,
}

const selectStyle = {
  width: '100%', padding: '9px 12px', borderRadius: 8,
  border: `1px solid ${COLORS.border}`, fontSize: 13,
  color: COLORS.textDark, outline: 'none', background: '#fff',
  fontFamily: 'inherit', boxSizing: 'border-box', cursor: 'pointer',
}

const warningBox = {
  display: 'flex', alignItems: 'center', gap: 8,
  padding: '8px 12px', borderRadius: 8,
  background: COLORS.warning + '18',
  border: `1px solid ${COLORS.warning}40`,
  color: COLORS.warning, fontSize: 12,
}
