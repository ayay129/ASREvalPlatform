import { useState, useEffect } from 'react'
import { useLocation, useNavigate } from 'react-router-dom'
import {
  BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer,
  Legend, RadarChart, PolarGrid, PolarAngleAxis, Radar,
} from 'recharts'
import { ArrowLeft, GitCompare } from 'lucide-react'
import { api } from '../api'
import { COLORS, CHART_PALETTE } from '../theme'

const pct = (v) => (v != null ? (v * 100).toFixed(2) + '%' : '—')

export default function Compare() {
  const location = useLocation()
  const navigate = useNavigate()
  const [allEvals, setAllEvals] = useState([])
  const [selected, setSelected] = useState(new Set(location.state?.ids || []))
  const [result, setResult] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')

  useEffect(() => {
    api.listEvals({ status: 'completed', limit: 100 }).then(setAllEvals).catch(() => {})
  }, [])

  useEffect(() => {
    if (location.state?.ids?.length >= 2) runCompare(location.state.ids)
  }, [])

  const toggle = (id) => setSelected(prev => {
    const s = new Set(prev); s.has(id) ? s.delete(id) : s.add(id); return s
  })

  const runCompare = async (ids) => {
    setError(''); setLoading(true)
    try {
      const data = await api.compare(ids || [...selected])
      setResult(data)
    } catch (e) {
      setError(e.response?.data?.detail || 'Compare failed')
    } finally { setLoading(false) }
  }

  const items = result?.items || []

  // Bar chart data: one entry per metric, each model is a bar
  const metrics = ['WER', 'CER', 'SER']
  const barData = metrics.map(m => {
    const key = `corpus_${m.toLowerCase()}`
    const entry = { metric: m }
    items.forEach(it => { entry[it.model_name] = +(it[key] * 100).toFixed(2) })
    return entry
  })

  // Radar data
  const radarData = ['WER', 'CER', 'SER'].map(m => {
    const key = `corpus_${m.toLowerCase()}`
    const entry = { metric: m }
    items.forEach(it => { entry[it.model_name] = +(it[key] * 100).toFixed(2) })
    return entry
  })

  return (
    <div>
      <div style={{ display: 'flex', alignItems: 'center', gap: 14, marginBottom: 28 }}>
        <button onClick={() => navigate('/dashboard')} style={{ background: 'none', border: 'none', cursor: 'pointer', color: COLORS.textMid }}>
          <ArrowLeft size={20} />
        </button>
        <div>
          <h1 style={{ fontSize: 24, fontWeight: 700, color: COLORS.textDark }}>Model Comparison</h1>
          <p style={{ color: COLORS.textLight, fontSize: 14, marginTop: 4 }}>Compare multiple evaluation results side by side</p>
        </div>
      </div>

      {/* Selector */}
      <div style={{ background: '#fff', borderRadius: 12, border: `1px solid ${COLORS.border}`, padding: '20px 24px', marginBottom: 20 }}>
        <h3 style={{ fontSize: 14, fontWeight: 700, color: COLORS.textDark, marginBottom: 14 }}>Select evaluations to compare</h3>
        <div style={{ display: 'flex', flexWrap: 'wrap', gap: 8, marginBottom: 16 }}>
          {allEvals.map(ev => {
            const sel = selected.has(ev.id)
            return (
              <button key={ev.id} onClick={() => toggle(ev.id)} style={{
                padding: '7px 14px', borderRadius: 8, fontSize: 13, cursor: 'pointer',
                border: `1px solid ${sel ? COLORS.accent : COLORS.border}`,
                background: sel ? COLORS.accent + '18' : '#fff',
                color: sel ? COLORS.accent : COLORS.textMid,
                fontWeight: sel ? 600 : 400,
              }}>
                #{ev.id} {ev.model_name}
                <span style={{ color: COLORS.textLight, marginLeft: 6, fontSize: 11 }}>
                  WER {pct(ev.corpus_wer)}
                </span>
              </button>
            )
          })}
          {allEvals.length === 0 && <span style={{ color: COLORS.textLight, fontSize: 13 }}>No completed evaluations yet</span>}
        </div>
        {error && <div style={{ color: COLORS.danger, fontSize: 13, marginBottom: 10 }}>{error}</div>}
        <button
          disabled={selected.size < 2 || loading}
          onClick={() => runCompare()}
          style={{
            display: 'flex', alignItems: 'center', gap: 7,
            padding: '9px 20px', borderRadius: 8, border: 'none',
            background: selected.size < 2 ? COLORS.textLight : COLORS.accent,
            color: '#fff', cursor: selected.size < 2 ? 'not-allowed' : 'pointer',
            fontSize: 13, fontWeight: 600,
          }}
        >
          <GitCompare size={15} />
          {loading ? 'Comparing…' : `Compare (${selected.size} selected)`}
        </button>
      </div>

      {items.length > 0 && (
        <>
          {/* Summary table */}
          <div style={{ background: '#fff', borderRadius: 12, border: `1px solid ${COLORS.border}`, overflow: 'hidden', marginBottom: 20 }}>
            <div style={{ padding: '16px 20px', borderBottom: `1px solid ${COLORS.border}` }}>
              <h3 style={{ fontSize: 15, fontWeight: 700, color: COLORS.textDark }}>Summary</h3>
            </div>
            <table style={{ width: '100%', borderCollapse: 'collapse' }}>
              <thead>
                <tr style={{ background: COLORS.secondary1 + '50' }}>
                  {['Model', 'Dataset', 'Sentences', 'WER', 'CER', 'SER', 'WER Mean', 'WER Median'].map(h => (
                    <th key={h} style={th}>{h}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {items.map((it, i) => (
                  <tr key={it.eval_id} style={{ borderTop: `1px solid ${COLORS.border}`, background: i % 2 === 0 ? '#fff' : COLORS.bg }}>
                    <td style={{ ...td, fontWeight: 700, color: CHART_PALETTE[i] }}>{it.model_name}</td>
                    <td style={{ ...td, color: COLORS.textMid, fontSize: 12 }}>{it.dataset_name}</td>
                    <td style={td}>{it.num_sentences?.toLocaleString()}</td>
                    <td style={{ ...td, fontWeight: 700 }}>{pct(it.corpus_wer)}</td>
                    <td style={td}>{pct(it.corpus_cer)}</td>
                    <td style={td}>{pct(it.corpus_ser)}</td>
                    <td style={td}>{pct(it.wer_mean)}</td>
                    <td style={td}>{pct(it.wer_median)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          {/* Charts */}
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 16, marginBottom: 20 }}>
            <ChartCard title="Error Rates Comparison">
              <ResponsiveContainer width="100%" height={240}>
                <BarChart data={barData} margin={{ top: 5, right: 20, bottom: 5, left: 0 }}>
                  <XAxis dataKey="metric" tick={{ fontSize: 12 }} />
                  <YAxis tickFormatter={v => v + '%'} tick={{ fontSize: 11 }} />
                  <Tooltip formatter={v => v + '%'} />
                  <Legend wrapperStyle={{ fontSize: 12 }} />
                  {items.map((it, i) => (
                    <Bar key={it.model_name} dataKey={it.model_name} fill={CHART_PALETTE[i]} radius={[4, 4, 0, 0]} />
                  ))}
                </BarChart>
              </ResponsiveContainer>
            </ChartCard>

            <ChartCard title="Radar Overview">
              <ResponsiveContainer width="100%" height={240}>
                <RadarChart data={radarData}>
                  <PolarGrid stroke={COLORS.border} />
                  <PolarAngleAxis dataKey="metric" tick={{ fontSize: 12 }} />
                  {items.map((it, i) => (
                    <Radar key={it.model_name} name={it.model_name} dataKey={it.model_name}
                      stroke={CHART_PALETTE[i]} fill={CHART_PALETTE[i]} fillOpacity={0.15} />
                  ))}
                  <Legend wrapperStyle={{ fontSize: 12 }} />
                  <Tooltip formatter={v => v + '%'} />
                </RadarChart>
              </ResponsiveContainer>
            </ChartCard>
          </div>

          {/* Edit ops comparison */}
          <ChartCard title="Edit Operations Comparison">
            <ResponsiveContainer width="100%" height={220}>
              <BarChart
                data={items.map(it => ({
                  name: it.model_name,
                  Substitutions: it.total_sub,
                  Insertions: it.total_ins,
                  Deletions: it.total_del,
                  Correct: it.total_cor,
                }))}
                margin={{ top: 5, right: 20, bottom: 5, left: 10 }}
              >
                <XAxis dataKey="name" tick={{ fontSize: 12 }} />
                <YAxis tick={{ fontSize: 11 }} />
                <Tooltip formatter={v => v.toLocaleString()} />
                <Legend wrapperStyle={{ fontSize: 12 }} />
                <Bar dataKey="Correct" stackId="a" fill={COLORS.success} />
                <Bar dataKey="Substitutions" stackId="a" fill={COLORS.danger} />
                <Bar dataKey="Insertions" stackId="a" fill={COLORS.warning} />
                <Bar dataKey="Deletions" stackId="a" fill={COLORS.secondary2} radius={[4, 4, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </ChartCard>
        </>
      )}
    </div>
  )
}

function ChartCard({ title, children }) {
  return (
    <div style={{ background: '#fff', borderRadius: 12, border: `1px solid ${COLORS.border}`, padding: '20px 20px 12px', boxShadow: '0 2px 8px rgba(100,130,173,0.06)' }}>
      <h3 style={{ fontSize: 13, fontWeight: 700, color: COLORS.textMid, marginBottom: 16, textTransform: 'uppercase', letterSpacing: '0.04em' }}>{title}</h3>
      {children}
    </div>
  )
}

const th = { padding: '10px 14px', textAlign: 'left', fontSize: 12, fontWeight: 600, color: COLORS.textMid }
const td = { padding: '11px 14px', fontSize: 13, color: COLORS.textDark }
