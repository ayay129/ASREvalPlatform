import { useState, useEffect, useCallback } from 'react'
import { useNavigate } from 'react-router-dom'
import { Trash2, FileText, RefreshCw, PlusCircle, GitCompare } from 'lucide-react'
import { api } from '../api'
import { COLORS } from '../theme'
import StatusBadge from '../components/StatusBadge'
import MetricCard from '../components/MetricCard'

const pct = (v) => (v != null ? (v * 100).toFixed(2) + '%' : '—')
const fmt = (v) => (v != null ? v.toLocaleString() : '—')

export default function Evaluations() {
  const [evals, setEvals] = useState([])
  const [loading, setLoading] = useState(true)
  const [selected, setSelected] = useState(new Set())
  const [statusFilter, setStatusFilter] = useState('')
  const navigate = useNavigate()

  const load = useCallback(async () => {
    setLoading(true)
    try {
      const params = {}
      if (statusFilter) params.status = statusFilter
      const data = await api.listEvals(params)
      setEvals(data)
    } catch (e) {
      console.error(e)
    } finally {
      setLoading(false)
    }
  }, [statusFilter])

  useEffect(() => { load() }, [load])

  // Auto-refresh if any running/pending
  useEffect(() => {
    const hasActive = evals.some(e => e.status === 'running' || e.status === 'pending')
    if (!hasActive) return
    const t = setInterval(load, 3000)
    return () => clearInterval(t)
  }, [evals, load])

  const handleDelete = async (id, e) => {
    e.stopPropagation()
    if (!confirm('Delete this evaluation?')) return
    await api.deleteEval(id)
    load()
  }

  const toggleSelect = (id, e) => {
    e.stopPropagation()
    setSelected(prev => {
      const s = new Set(prev)
      s.has(id) ? s.delete(id) : s.add(id)
      return s
    })
  }

  const completed = evals.filter(e => e.status === 'completed')
  const avgWer = completed.length
    ? completed.reduce((s, e) => s + (e.corpus_wer || 0), 0) / completed.length
    : null

  return (
    <div>
      {/* Header */}
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 28 }}>
        <div>
          <h1 style={{ fontSize: 24, fontWeight: 700, color: COLORS.textDark }}>Evaluations</h1>
          <p style={{ color: COLORS.textLight, fontSize: 14, marginTop: 4 }}>
            All evaluation records
          </p>
        </div>
        <div style={{ display: 'flex', gap: 10 }}>
          <button onClick={load} style={btnStyle(COLORS.secondary2, true)}>
            <RefreshCw size={15} /> Refresh
          </button>
          {selected.size >= 2 && (
            <button
              onClick={() => navigate('/compare', { state: { ids: [...selected] } })}
              style={btnStyle(COLORS.secondary2, true)}
            >
              <GitCompare size={15} /> Compare ({selected.size})
            </button>
          )}
          <button onClick={() => navigate('/new')} style={btnStyle(COLORS.accent)}>
            <PlusCircle size={15} /> New Eval
          </button>
        </div>
      </div>

      {/* Summary cards */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: 16, marginBottom: 28 }}>
        <MetricCard label="Total Evals" value={fmt(evals.length)} color={COLORS.accent} />
        <MetricCard label="Completed" value={fmt(completed.length)} color={COLORS.success} />
        <MetricCard
          label="Running / Pending"
          value={fmt(evals.filter(e => e.status === 'running' || e.status === 'pending').length)}
          color={COLORS.warning}
        />
        <MetricCard
          label="Avg WER"
          value={avgWer != null ? pct(avgWer) : '—'}
          color={COLORS.secondary2}
        />
      </div>

      {/* Filter bar */}
      <div style={{ display: 'flex', gap: 10, marginBottom: 16 }}>
        {['', 'completed', 'running', 'pending', 'failed'].map(s => (
          <button
            key={s}
            onClick={() => setStatusFilter(s)}
            style={{
              padding: '5px 14px', borderRadius: 20, border: '1px solid',
              fontSize: 13, cursor: 'pointer', fontWeight: statusFilter === s ? 600 : 400,
              borderColor: statusFilter === s ? COLORS.accent : COLORS.border,
              background: statusFilter === s ? COLORS.accent : 'transparent',
              color: statusFilter === s ? '#fff' : COLORS.textMid,
            }}
          >
            {s || 'All'}
          </button>
        ))}
      </div>

      {/* Table */}
      <div style={{ background: COLORS.card, borderRadius: 12, border: `1px solid ${COLORS.border}`, overflow: 'hidden' }}>
        <table style={{ width: '100%', borderCollapse: 'collapse' }}>
          <thead>
            <tr style={{ background: COLORS.secondary1 + '60' }}>
              <th style={th}>
                <input type="checkbox" onChange={() => {
                  if (selected.size === completed.length) setSelected(new Set())
                  else setSelected(new Set(completed.map(e => e.id)))
                }} checked={selected.size === completed.length && completed.length > 0} />
              </th>
              {['ID', 'Model', 'Dataset', 'Status', 'WER', 'CER', 'SER', 'Sentences', 'Created', 'Actions'].map(h => (
                <th key={h} style={th}>{h}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {loading ? (
              <tr><td colSpan={11} style={{ textAlign: 'center', padding: 48, color: COLORS.textLight }}>Loading…</td></tr>
            ) : evals.length === 0 ? (
              <tr><td colSpan={11} style={{ textAlign: 'center', padding: 48, color: COLORS.textLight }}>No evaluations yet</td></tr>
            ) : evals.map((ev, i) => (
              <tr
                key={ev.id}
                onClick={() => ev.status === 'completed' && navigate(`/report/${ev.id}`)}
                style={{
                  borderTop: `1px solid ${COLORS.border}`,
                  cursor: ev.status === 'completed' ? 'pointer' : 'default',
                  background: i % 2 === 0 ? '#fff' : COLORS.bg,
                  transition: 'background 0.1s',
                }}
                onMouseEnter={e => ev.status === 'completed' && (e.currentTarget.style.background = COLORS.secondary1 + '40')}
                onMouseLeave={e => e.currentTarget.style.background = i % 2 === 0 ? '#fff' : COLORS.bg}
              >
                <td style={td} onClick={e => e.stopPropagation()}>
                  {ev.status === 'completed' && (
                    <input type="checkbox" checked={selected.has(ev.id)} onChange={e => toggleSelect(ev.id, e)} />
                  )}
                </td>
                <td style={{ ...td, color: COLORS.textLight, fontSize: 12 }}>#{ev.id}</td>
                <td style={{ ...td, fontWeight: 600, color: COLORS.textDark }}>{ev.model_name}</td>
                <td style={{ ...td, maxWidth: 160, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap', color: COLORS.textMid, fontSize: 12 }}>{ev.dataset_name}</td>
                <td style={td}><StatusBadge status={ev.status} /></td>
                <td style={{ ...td, fontWeight: 600, color: ev.corpus_wer > 0.5 ? COLORS.danger : COLORS.accent }}>{pct(ev.corpus_wer)}</td>
                <td style={td}>{pct(ev.corpus_cer)}</td>
                <td style={td}>{pct(ev.corpus_ser)}</td>
                <td style={td}>{fmt(ev.num_sentences)}</td>
                <td style={{ ...td, fontSize: 12, color: COLORS.textLight }}>{ev.created_at ? new Date(ev.created_at).toLocaleString() : '—'}</td>
                <td style={td}>
                  <div style={{ display: 'flex', gap: 6 }}>
                    {ev.status === 'completed' && (
                      <a
                        href={api.exportReport(ev.id)}
                        target="_blank"
                        rel="noreferrer"
                        onClick={e => e.stopPropagation()}
                        style={{ color: COLORS.secondary2, display: 'flex', alignItems: 'center' }}
                        title="Download PDF"
                      >
                        <FileText size={15} />
                      </a>
                    )}
                    <button
                      onClick={e => handleDelete(ev.id, e)}
                      style={{ background: 'none', border: 'none', cursor: 'pointer', color: COLORS.danger, display: 'flex', alignItems: 'center' }}
                      title="Delete"
                    >
                      <Trash2 size={15} />
                    </button>
                  </div>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  )
}

const th = {
  padding: '12px 14px', textAlign: 'left', fontSize: 12,
  fontWeight: 600, color: COLORS.textMid, whiteSpace: 'nowrap',
}
const td = { padding: '12px 14px', fontSize: 13, color: COLORS.textDark }

function btnStyle(bg, outline = false) {
  return {
    display: 'flex', alignItems: 'center', gap: 6,
    padding: '8px 16px', borderRadius: 8, fontSize: 13, fontWeight: 600,
    cursor: 'pointer', border: `1px solid ${bg}`,
    background: outline ? 'transparent' : bg,
    color: outline ? bg : '#fff',
  }
}
