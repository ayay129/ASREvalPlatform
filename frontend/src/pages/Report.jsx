import { useState, useEffect } from 'react'
import { useParams, useNavigate } from 'react-router-dom'
import {
  BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer,
  PieChart, Pie, Cell, Legend, ScatterChart, Scatter,
  CartesianGrid,
} from 'recharts'
import { ArrowLeft, Download, FileText, ChevronsUpDown } from 'lucide-react'
import { api } from '../api'
import { COLORS, CHART_PALETTE } from '../theme'
import MetricCard from '../components/MetricCard'
import StatusBadge from '../components/StatusBadge'

const pct = (v, d = 2) => (v != null ? (v * 100).toFixed(d) + '%' : '—')

export default function Report() {
  const { id } = useParams()
  const navigate = useNavigate()
  const [ev, setEv] = useState(null)
  const [loading, setLoading] = useState(true)
  const [sortBy, setSortBy] = useState('idx')
  const [detailPage, setDetailPage] = useState(0)
  const PAGE = 50

  const load = async () => {
    setLoading(true)
    try {
      const data = await api.getEval(id, {
        sort_by: sortBy, detail_limit: PAGE, detail_offset: detailPage * PAGE,
      })
      setEv(data)
    } catch (e) { console.error(e) }
    finally { setLoading(false) }
  }

  useEffect(() => { load() }, [id, sortBy, detailPage])

  if (loading && !ev) return <div style={{ color: COLORS.textLight, padding: 48, textAlign: 'center' }}>Loading…</div>
  if (!ev) return <div style={{ color: COLORS.danger, padding: 48 }}>Evaluation not found</div>

  // Chart data
  const errorRates = [
    { name: 'WER', value: +(ev.corpus_wer * 100).toFixed(2) },
    { name: 'CER', value: +(ev.corpus_cer * 100).toFixed(2) },
    { name: 'SER', value: +(ev.corpus_ser * 100).toFixed(2) },
    { name: 'MER', value: +(ev.corpus_mer * 100).toFixed(2) },
    { name: 'WIL', value: +(ev.corpus_wil * 100).toFixed(2) },
  ]

  const editOps = ev.edit_ops ? [
    { name: 'Correct',       value: ev.edit_ops.correct,       fill: COLORS.success },
    { name: 'Substitution',  value: ev.edit_ops.substitutions, fill: COLORS.danger },
    { name: 'Insertion',     value: ev.edit_ops.insertions,    fill: COLORS.warning },
    { name: 'Deletion',      value: ev.edit_ops.deletions,     fill: COLORS.secondary2 },
  ] : []

  // WER bucket
  const details = ev.details || []
  const buckets = [
    { name: 'Perfect (0%)',   count: details.filter(d => d.wer === 0).length },
    { name: 'Low (0-10%)',    count: details.filter(d => d.wer > 0 && d.wer <= 0.1).length },
    { name: 'Medium (10-30%)', count: details.filter(d => d.wer > 0.1 && d.wer <= 0.3).length },
    { name: 'High (30-50%)',  count: details.filter(d => d.wer > 0.3 && d.wer <= 0.5).length },
    { name: 'Severe (>50%)',  count: details.filter(d => d.wer > 0.5).length },
  ]

  const acc = ev.num_sentences ? (ev.num_sentences - Math.round((ev.corpus_ser || 0) * ev.num_sentences)) / ev.num_sentences : 0

  return (
    <div>
      {/* Header */}
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: 28 }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 14 }}>
          <button onClick={() => navigate('/dashboard')} style={{ background: 'none', border: 'none', cursor: 'pointer', color: COLORS.textMid, display: 'flex', alignItems: 'center' }}>
            <ArrowLeft size={20} />
          </button>
          <div>
            <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
              <h1 style={{ fontSize: 22, fontWeight: 700, color: COLORS.textDark }}>{ev.model_name}</h1>
              <StatusBadge status={ev.status} />
            </div>
            <p style={{ color: COLORS.textLight, fontSize: 13, marginTop: 3 }}>
              {ev.dataset_name} · {ev.num_sentences?.toLocaleString()} sentences
            </p>
          </div>
        </div>
        {ev.status === 'completed' && (
          <a
            href={api.exportReport(id)}
            target="_blank" rel="noreferrer"
            style={{
              display: 'flex', alignItems: 'center', gap: 7,
              padding: '9px 18px', borderRadius: 8,
              background: COLORS.accent, color: '#fff',
              fontSize: 13, fontWeight: 600, textDecoration: 'none',
            }}
          >
            <Download size={15} /> Download PDF
          </a>
        )}
      </div>

      {ev.status !== 'completed' && (
        <div style={{ padding: '16px 20px', borderRadius: 10, background: COLORS.warning + '18', border: `1px solid ${COLORS.warning}40`, color: COLORS.warning, marginBottom: 24, fontSize: 14 }}>
          Evaluation is <strong>{ev.status}</strong>. Results will appear when complete.
        </div>
      )}

      {/* Metric cards */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(5, 1fr)', gap: 14, marginBottom: 24 }}>
        <MetricCard label="WER" value={pct(ev.corpus_wer)} color={COLORS.accent} />
        <MetricCard label="CER" value={pct(ev.corpus_cer)} color={COLORS.secondary2} />
        <MetricCard label="SER" value={pct(ev.corpus_ser)} color={COLORS.warning} />
        <MetricCard label="MER" value={pct(ev.corpus_mer)} color={COLORS.danger} />
        <MetricCard label="Sentence Acc." value={pct(acc)} color={COLORS.success} />
      </div>

      {/* Distribution stats */}
      {ev.wer_distribution && (
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: 14, marginBottom: 24 }}>
          <MetricCard label="WER Mean" value={pct(ev.wer_distribution.mean)} color={COLORS.secondary2} />
          <MetricCard label="WER Median" value={pct(ev.wer_distribution.median)} color={COLORS.secondary2} />
          <MetricCard label="WER Std Dev" value={pct(ev.wer_distribution.std)} color={COLORS.textLight} />
        </div>
      )}

      {/* Charts row 1 */}
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 16, marginBottom: 16 }}>
        <ChartCard title="Error Rates Overview">
          <ResponsiveContainer width="100%" height={220}>
            <BarChart data={errorRates} layout="vertical" margin={{ left: 10, right: 30, top: 5, bottom: 5 }}>
              <XAxis type="number" tickFormatter={v => v + '%'} tick={{ fontSize: 11 }} />
              <YAxis type="category" dataKey="name" tick={{ fontSize: 12 }} width={36} />
              <Tooltip formatter={v => v + '%'} />
              <Bar dataKey="value" radius={[0, 4, 4, 0]}>
                {errorRates.map((_, i) => <Cell key={i} fill={CHART_PALETTE[i]} />)}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </ChartCard>

        <ChartCard title="Edit Operation Distribution">
          <ResponsiveContainer width="100%" height={220}>
            <PieChart>
              <Pie
                data={editOps} dataKey="value" nameKey="name"
                cx="50%" cy="50%" innerRadius={55} outerRadius={80}
                paddingAngle={3}
              >
                {editOps.map((entry, i) => <Cell key={i} fill={entry.fill} />)}
              </Pie>
              <Tooltip formatter={v => v.toLocaleString()} />
              <Legend iconType="circle" iconSize={9} wrapperStyle={{ fontSize: 12 }} />
            </PieChart>
          </ResponsiveContainer>
        </ChartCard>
      </div>

      {/* Charts row 2 */}
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 16, marginBottom: 24 }}>
        <ChartCard title="WER Bucket Distribution">
          <ResponsiveContainer width="100%" height={200}>
            <BarChart data={buckets} margin={{ left: 0, right: 10, top: 5, bottom: 5 }}>
              <XAxis dataKey="name" tick={{ fontSize: 10 }} />
              <YAxis tick={{ fontSize: 11 }} />
              <Tooltip />
              <Bar dataKey="count" radius={[4, 4, 0, 0]}>
                {buckets.map((_, i) => <Cell key={i} fill={CHART_PALETTE[i]} />)}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </ChartCard>

        <ChartCard title="WER vs. Reference Length">
          <ResponsiveContainer width="100%" height={200}>
            <ScatterChart margin={{ left: 0, right: 10, top: 5, bottom: 5 }}>
              <CartesianGrid strokeDasharray="3 3" stroke={COLORS.border} />
              <XAxis dataKey="ref" name="Ref Length" type="number" tick={{ fontSize: 11 }} label={{ value: 'Ref words', position: 'insideBottom', offset: -2, fontSize: 11 }} />
              <YAxis dataKey="wer" name="WER" type="number" tickFormatter={v => v + '%'} tick={{ fontSize: 11 }} />
              <Tooltip formatter={(v, name) => name === 'WER' ? v.toFixed(1) + '%' : v} />
              <Scatter
                data={details.slice(0, 300).map(d => ({ ref: d.ref_syllables, wer: +(d.wer * 100).toFixed(1) }))}
                fill={COLORS.accent} opacity={0.5} r={3}
              />
            </ScatterChart>
          </ResponsiveContainer>
        </ChartCard>
      </div>

      {/* Sentence table */}
      <div style={{ background: '#fff', borderRadius: 12, border: `1px solid ${COLORS.border}`, overflow: 'hidden' }}>
        <div style={{ padding: '16px 20px', borderBottom: `1px solid ${COLORS.border}`, display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
          <h3 style={{ fontSize: 15, fontWeight: 700, color: COLORS.textDark }}>Sentence Details</h3>
          <div style={{ display: 'flex', gap: 8, alignItems: 'center' }}>
            <span style={{ fontSize: 12, color: COLORS.textLight }}>Sort by</span>
            {[['idx', 'Index'], ['wer_desc', 'WER ↓'], ['wer_asc', 'WER ↑']].map(([v, l]) => (
              <button key={v} onClick={() => { setSortBy(v); setDetailPage(0) }} style={{
                padding: '4px 10px', borderRadius: 6, fontSize: 12, cursor: 'pointer',
                border: `1px solid ${sortBy === v ? COLORS.accent : COLORS.border}`,
                background: sortBy === v ? COLORS.accent : 'transparent',
                color: sortBy === v ? '#fff' : COLORS.textMid,
              }}>{l}</button>
            ))}
          </div>
        </div>
        <div style={{ overflowX: 'auto' }}>
          <table style={{ width: '100%', borderCollapse: 'collapse' }}>
            <thead>
              <tr style={{ background: COLORS.secondary1 + '50' }}>
                {['#', 'Reference', 'Hypothesis', 'WER', 'CER', 'Sub', 'Ins', 'Del'].map(h => (
                  <th key={h} style={dth}>{h}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {details.map((d, i) => (
                <tr key={d.sentence_idx} style={{ borderTop: `1px solid ${COLORS.border}`, background: i % 2 === 0 ? '#fff' : COLORS.bg }}>
                  <td style={{ ...dtd, color: COLORS.textLight }}>{d.sentence_idx}</td>
                  <td style={{ ...dtd, maxWidth: 260, whiteSpace: 'pre-wrap', wordBreak: 'break-all' }}>{d.reference}</td>
                  <td style={{ ...dtd, maxWidth: 260, whiteSpace: 'pre-wrap', wordBreak: 'break-all', color: d.is_correct ? COLORS.success : COLORS.textDark }}>{d.hypothesis}</td>
                  <td style={{ ...dtd, fontWeight: 600, color: d.wer > 0.5 ? COLORS.danger : d.wer > 0.2 ? COLORS.warning : COLORS.success }}>{pct(d.wer)}</td>
                  <td style={dtd}>{pct(d.cer)}</td>
                  <td style={{ ...dtd, color: COLORS.danger }}>{d.word_sub}</td>
                  <td style={{ ...dtd, color: COLORS.warning }}>{d.word_ins}</td>
                  <td style={{ ...dtd, color: COLORS.secondary2 }}>{d.word_del}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        {ev.num_sentences > PAGE && (
          <div style={{ padding: '12px 20px', borderTop: `1px solid ${COLORS.border}`, display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
            <span style={{ fontSize: 12, color: COLORS.textLight }}>
              Showing {detailPage * PAGE + 1}–{Math.min((detailPage + 1) * PAGE, ev.num_sentences)} of {ev.num_sentences}
            </span>
            <div style={{ display: 'flex', gap: 8 }}>
              <button disabled={detailPage === 0} onClick={() => setDetailPage(p => p - 1)} style={pageBtn}>Prev</button>
              <button disabled={(detailPage + 1) * PAGE >= ev.num_sentences} onClick={() => setDetailPage(p => p + 1)} style={pageBtn}>Next</button>
            </div>
          </div>
        )}
      </div>
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

const dth = { padding: '10px 12px', textAlign: 'left', fontSize: 12, fontWeight: 600, color: COLORS.textMid, whiteSpace: 'nowrap' }
const dtd = { padding: '10px 12px', fontSize: 12, color: COLORS.textDark }
const pageBtn = { padding: '5px 14px', borderRadius: 6, border: `1px solid ${COLORS.border}`, background: '#fff', cursor: 'pointer', fontSize: 12, color: COLORS.textMid }
