import { useCallback, useEffect, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { PlusCircle, RefreshCw } from 'lucide-react'
import { api } from '../api'
import { COLORS } from '../theme'
import StatusBadge from '../components/StatusBadge'

const fmtDate = (value) => (value ? new Date(value).toLocaleString() : '—')

export default function TrainRuns() {
  const [runs, setRuns] = useState([])
  const [loading, setLoading] = useState(true)
  const [statusFilter, setStatusFilter] = useState('')
  const [error, setError] = useState('')
  const navigate = useNavigate()

  const load = useCallback(async () => {
    setLoading(true)
    setError('')
    try {
      const params = {}
      if (statusFilter) params.status = statusFilter
      const data = await api.listTrainRuns(params)
      setRuns(data)
    } catch (err) {
      console.error(err)
      setError('Failed to load training runs')
    } finally {
      setLoading(false)
    }
  }, [statusFilter])

  useEffect(() => {
    load()
  }, [load])

  useEffect(() => {
    const hasActive = runs.some(run => run.status === 'queued' || run.status === 'running')
    if (!hasActive) return

    const timer = setInterval(load, 2000)
    return () => clearInterval(timer)
  }, [runs, load])

  return (
    <div>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 28 }}>
        <div>
          <h1 style={{ fontSize: 24, fontWeight: 700, color: COLORS.textDark }}>Train Runs</h1>
          <p style={{ color: COLORS.textLight, fontSize: 14, marginTop: 4 }}>
            Training job configurations stored by the platform
          </p>
        </div>
        <div style={{ display: 'flex', gap: 10 }}>
          <button onClick={load} style={btnStyle(COLORS.secondary2, true)}>
            <RefreshCw size={15} /> Refresh
          </button>
          <button onClick={() => navigate('/train-runs/new')} style={btnStyle(COLORS.accent)}>
            <PlusCircle size={15} /> New Train Run
          </button>
        </div>
      </div>

      <div style={{ marginBottom: 16, display: 'flex', gap: 10 }}>
        {['', 'queued', 'running', 'completed', 'failed'].map(status => (
          <button
            key={status || 'all'}
            onClick={() => setStatusFilter(status)}
            style={{
              padding: '5px 14px',
              borderRadius: 20,
              border: '1px solid',
              fontSize: 13,
              cursor: 'pointer',
              fontWeight: statusFilter === status ? 600 : 400,
              borderColor: statusFilter === status ? COLORS.accent : COLORS.border,
              background: statusFilter === status ? COLORS.accent : 'transparent',
              color: statusFilter === status ? '#fff' : COLORS.textMid,
            }}
          >
            {status || 'All'}
          </button>
        ))}
      </div>

      {error && (
        <div style={errorBox}>
          {error}
        </div>
      )}

      <div style={{ background: COLORS.card, borderRadius: 12, border: `1px solid ${COLORS.border}`, overflow: 'hidden' }}>
        <table style={{ width: '100%', borderCollapse: 'collapse' }}>
          <thead>
            <tr style={{ background: COLORS.secondary1 + '60' }}>
              {['ID', 'Name', 'Base Model', 'Train Data', 'Test Data', 'Status', 'Created'].map(h => (
                <th key={h} style={th}>{h}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {loading ? (
              <tr>
                <td colSpan={7} style={emptyCell}>Loading…</td>
              </tr>
            ) : runs.length === 0 ? (
              <tr>
                <td colSpan={7} style={emptyCell}>No training runs yet</td>
              </tr>
            ) : runs.map((run, index) => (
              <tr
                key={run.id}
                onClick={() => navigate(`/train-runs/${run.id}`)}
                style={{
                  borderTop: `1px solid ${COLORS.border}`,
                  background: index % 2 === 0 ? '#fff' : COLORS.bg,
                  cursor: 'pointer',
                }}
              >
                <td style={{ ...td, color: COLORS.textLight, fontSize: 12 }}>#{run.id}</td>
                <td style={{ ...td, fontWeight: 700, color: COLORS.textDark }}>{run.name}</td>
                <td style={td}>{run.base_model}</td>
                <td style={pathCell}>{run.train_data_path}</td>
                <td style={pathCell}>{run.test_data_path}</td>
                <td style={td}>
                  <StatusBadge status={run.status} />
                  {run.phase && run.status === 'running' && (
                    <span style={phaseBadge}>{run.phase}</span>
                  )}
                </td>
                <td style={{ ...td, color: COLORS.textLight, fontSize: 12 }}>{fmtDate(run.created_at)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  )
}

const th = {
  padding: '12px 14px',
  textAlign: 'left',
  fontSize: 12,
  fontWeight: 600,
  color: COLORS.textMid,
  whiteSpace: 'nowrap',
}

const td = {
  padding: '12px 14px',
  fontSize: 13,
  color: COLORS.textDark,
  verticalAlign: 'top',
}

const pathCell = {
  ...td,
  maxWidth: 220,
  whiteSpace: 'pre-wrap',
  wordBreak: 'break-all',
  color: COLORS.textMid,
  fontSize: 12,
}

const emptyCell = {
  textAlign: 'center',
  padding: 48,
  color: COLORS.textLight,
}

const phaseBadge = {
  marginLeft: 6,
  padding: '2px 8px',
  borderRadius: 10,
  fontSize: 11,
  fontWeight: 600,
  background: COLORS.secondary1 + '80',
  color: COLORS.secondary2,
}

const errorBox = {
  marginBottom: 16,
  padding: '12px 16px',
  borderRadius: 8,
  background: COLORS.danger + '18',
  border: `1px solid ${COLORS.danger}40`,
  color: COLORS.danger,
  fontSize: 13,
}

function btnStyle(bg, outline = false) {
  return {
    display: 'flex',
    alignItems: 'center',
    gap: 6,
    padding: '8px 16px',
    borderRadius: 8,
    fontSize: 13,
    fontWeight: 600,
    cursor: 'pointer',
    border: `1px solid ${bg}`,
    background: outline ? 'transparent' : bg,
    color: outline ? bg : '#fff',
  }
}
