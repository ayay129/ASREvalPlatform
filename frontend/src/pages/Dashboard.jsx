import { useState, useEffect, useCallback } from 'react'
import { useNavigate } from 'react-router-dom'
import {
  RefreshCw, FlaskConical, Database, Activity, Trophy, Zap,
  ArrowRight, BarChart3, Brain, Cpu,
} from 'lucide-react'
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer,
} from 'recharts'
import { api } from '../api'
import { COLORS, CHART_PALETTE } from '../theme'
import MetricCard from '../components/MetricCard'
import StatusBadge from '../components/StatusBadge'

const pct = (v) => (v != null ? (v * 100).toFixed(2) + '%' : '—')

export default function Dashboard() {
  const [evals, setEvals] = useState([])
  const [trains, setTrains] = useState([])
  const [datasets, setDatasets] = useState([])
  const [gpuStatus, setGpuStatus] = useState(null)
  const [loading, setLoading] = useState(true)
  const navigate = useNavigate()

  const load = useCallback(async () => {
    setLoading(true)
    try {
      const [e, t, d, g] = await Promise.all([
        api.listEvals(),
        api.listTrainRuns(),
        api.listDatasets(),
        api.gpuStatus().catch(() => null),
      ])
      setEvals(e || [])
      setTrains(t || [])
      setDatasets(d || [])
      setGpuStatus(g)
    } catch (err) {
      console.error(err)
    } finally {
      setLoading(false)
    }
  }, [])

  useEffect(() => { load() }, [load])

  // Auto-refresh if active jobs
  useEffect(() => {
    const hasActive =
      evals.some(e => e.status === 'running' || e.status === 'pending') ||
      trains.some(t => t.status === 'running' || t.status === 'queued')
    if (!hasActive) return
    const t = setInterval(load, 4000)
    return () => clearInterval(t)
  }, [evals, trains, load])

  // ── derived data ──
  const completedEvals = evals.filter(e => e.status === 'completed')
  const completedTrains = trains.filter(t => t.status === 'completed')
  const activeJobs =
    evals.filter(e => e.status === 'running' || e.status === 'pending').length +
    trains.filter(t => t.status === 'running' || t.status === 'queued').length

  // Best WER
  const bestEval = completedEvals.length > 0
    ? completedEvals.reduce((best, e) =>
        (e.corpus_wer != null && (best == null || e.corpus_wer < best.corpus_wer)) ? e : best,
      null)
    : null

  // Recent activity: merge evals + trains, sort by created_at desc, take 8
  const activity = [
    ...evals.map(e => ({ type: 'eval', id: e.id, name: e.model_name, status: e.status, date: e.created_at })),
    ...trains.map(t => ({ type: 'train', id: t.id, name: t.name, status: t.status, date: t.created_at })),
  ]
    .sort((a, b) => new Date(b.date) - new Date(a.date))
    .slice(0, 8)

  // Leaderboard: top 5 evals by corpus_wer ascending
  const leaderboard = [...completedEvals]
    .filter(e => e.corpus_wer != null)
    .sort((a, b) => a.corpus_wer - b.corpus_wer)
    .slice(0, 5)

  // Chart data: recent 8 completed evals for WER/CER bar chart
  const chartData = [...completedEvals]
    .slice(0, 8)
    .reverse()
    .map(e => ({
      name: e.model_name.length > 16 ? e.model_name.slice(0, 14) + '...' : e.model_name,
      fullName: e.model_name,
      WER: e.corpus_wer != null ? +(e.corpus_wer * 100).toFixed(2) : 0,
      CER: e.corpus_cer != null ? +(e.corpus_cer * 100).toFixed(2) : 0,
    }))

  // Active training runs
  const activeTrains = trains.filter(t => t.status === 'running' || t.status === 'queued')

  if (loading) {
    return <div style={{ padding: 48, textAlign: 'center', color: COLORS.textLight }}>Loading dashboard...</div>
  }

  return (
    <div>
      {/* Header */}
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 28 }}>
        <div>
          <h1 style={{ fontSize: 24, fontWeight: 700, color: COLORS.textDark }}>Dashboard</h1>
          <p style={{ color: COLORS.textLight, fontSize: 14, marginTop: 4 }}>
            Platform overview
          </p>
        </div>
        <button onClick={load} style={refreshBtn}>
          <RefreshCw size={15} /> Refresh
        </button>
      </div>

      {/* Row 1: KPI Cards */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(5, 1fr)', gap: 14, marginBottom: 24 }}>
        <MetricCard
          label="Models Trained"
          value={completedTrains.length}
          color={COLORS.accent}
          icon={Brain}
        />
        <MetricCard
          label="Evaluations"
          value={completedEvals.length}
          color={COLORS.success}
          icon={FlaskConical}
        />
        <MetricCard
          label="Datasets"
          value={datasets.length}
          color={COLORS.secondary2}
          icon={Database}
        />
        <MetricCard
          label="Active Jobs"
          value={activeJobs}
          color={activeJobs > 0 ? COLORS.warning : COLORS.textLight}
          icon={Zap}
        />
        <MetricCard
          label="Best WER"
          value={bestEval ? pct(bestEval.corpus_wer) : '—'}
          color={COLORS.primary}
          icon={Trophy}
        />
      </div>

      {/* GPU Status */}
      {gpuStatus && gpuStatus.available && gpuStatus.gpus.length > 0 && (
        <div style={{ ...card, marginBottom: 16, padding: '16px 22px' }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 12 }}>
            <h2 style={{ ...sectionTitle, marginBottom: 0 }}>
              <Cpu size={16} /> GPU Status
            </h2>
            {gpuStatus.driver_version && (
              <span style={{ fontSize: 11, color: COLORS.textLight }}>
                Driver {gpuStatus.driver_version}
                {gpuStatus.cuda_version && ` · CUDA ${gpuStatus.cuda_version}`}
              </span>
            )}
          </div>
          <div style={{ display: 'flex', gap: 12, flexWrap: 'wrap' }}>
            {gpuStatus.gpus.map(g => (
              <div key={g.index} style={{
                flex: '1 1 200px', padding: '12px 14px', borderRadius: 10,
                background: COLORS.bg, border: `1px solid ${COLORS.border}`,
                minWidth: 200,
              }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 8 }}>
                  <span style={{ fontSize: 13, fontWeight: 700, color: COLORS.textDark }}>
                    GPU {g.index}
                  </span>
                  <span style={{ fontSize: 11, color: COLORS.textLight }}>{g.name}</span>
                </div>
                {/* Utilization */}
                <div style={{ marginBottom: 6 }}>
                  <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 11, marginBottom: 3 }}>
                    <span style={{ color: COLORS.textMid }}>Utilization</span>
                    <span style={{
                      fontWeight: 600,
                      color: g.utilization_pct > 80 ? COLORS.danger : g.utilization_pct > 50 ? COLORS.warning : COLORS.success,
                    }}>
                      {g.utilization_pct}%
                    </span>
                  </div>
                  <div style={{ height: 6, borderRadius: 3, background: COLORS.border, overflow: 'hidden' }}>
                    <div style={{
                      width: `${g.utilization_pct}%`, height: '100%', borderRadius: 3,
                      background: g.utilization_pct > 80 ? COLORS.danger : g.utilization_pct > 50 ? COLORS.warning : COLORS.success,
                      transition: 'width 0.4s',
                    }} />
                  </div>
                </div>
                {/* Memory */}
                <div style={{ marginBottom: 4 }}>
                  <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 11, marginBottom: 3 }}>
                    <span style={{ color: COLORS.textMid }}>Memory</span>
                    <span style={{ color: COLORS.textMid }}>
                      {Math.round(g.memory_used_mb)} / {Math.round(g.memory_total_mb)} MB
                    </span>
                  </div>
                  <div style={{ height: 6, borderRadius: 3, background: COLORS.border, overflow: 'hidden' }}>
                    <div style={{
                      width: `${g.memory_pct}%`, height: '100%', borderRadius: 3,
                      background: g.memory_pct > 85 ? COLORS.danger : COLORS.secondary2,
                      transition: 'width 0.4s',
                    }} />
                  </div>
                </div>
                {/* Temperature & Power */}
                <div style={{ display: 'flex', gap: 12, fontSize: 11, color: COLORS.textLight, marginTop: 6 }}>
                  {g.temperature != null && <span>{g.temperature}°C</span>}
                  {g.power_draw_w != null && g.power_limit_w != null && (
                    <span>{Math.round(g.power_draw_w)}W / {Math.round(g.power_limit_w)}W</span>
                  )}
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Row 2: Activity + Leaderboard */}
      <div style={{ display: 'grid', gridTemplateColumns: '1.5fr 1fr', gap: 16, marginBottom: 16 }}>
        {/* Recent Activity */}
        <div style={card}>
          <div style={cardHeader}>
            <h2 style={sectionTitle}><Activity size={16} /> Recent Activity</h2>
            <button onClick={() => navigate('/evaluations')} style={linkBtn}>
              View all <ArrowRight size={13} />
            </button>
          </div>
          {activity.length === 0 ? (
            <div style={emptyState}>No activity yet. Start by training a model or running an evaluation.</div>
          ) : (
            <div style={{ display: 'flex', flexDirection: 'column', gap: 0 }}>
              {activity.map((item, i) => (
                <div
                  key={`${item.type}-${item.id}`}
                  onClick={() => {
                    if (item.type === 'eval' && item.status === 'completed') navigate(`/report/${item.id}`)
                    else if (item.type === 'train') navigate(`/train-runs/${item.id}`)
                  }}
                  style={{
                    display: 'flex', alignItems: 'center', gap: 12,
                    padding: '10px 0',
                    borderTop: i > 0 ? `1px solid ${COLORS.border}` : 'none',
                    cursor: 'pointer',
                    transition: 'background 0.1s',
                  }}
                >
                  <div style={{
                    width: 30, height: 30, borderRadius: 8, display: 'flex',
                    alignItems: 'center', justifyContent: 'center', flexShrink: 0,
                    background: item.type === 'train' ? COLORS.accent + '18' : COLORS.success + '18',
                    color: item.type === 'train' ? COLORS.accent : COLORS.success,
                  }}>
                    {item.type === 'train' ? <Brain size={15} /> : <FlaskConical size={15} />}
                  </div>
                  <div style={{ flex: 1, minWidth: 0 }}>
                    <div style={{
                      fontSize: 13, fontWeight: 600, color: COLORS.textDark,
                      overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap',
                    }}>
                      {item.name}
                    </div>
                    <div style={{ fontSize: 11, color: COLORS.textLight }}>
                      {item.type === 'train' ? 'Training' : 'Evaluation'} #{item.id}
                    </div>
                  </div>
                  <StatusBadge status={item.status} />
                  <div style={{ fontSize: 11, color: COLORS.textLight, whiteSpace: 'nowrap', minWidth: 70, textAlign: 'right' }}>
                    {timeAgo(item.date)}
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>

        {/* Leaderboard */}
        <div style={card}>
          <div style={cardHeader}>
            <h2 style={sectionTitle}><Trophy size={16} /> Leaderboard</h2>
          </div>
          {leaderboard.length === 0 ? (
            <div style={emptyState}>No completed evaluations yet.</div>
          ) : (
            <div style={{ display: 'flex', flexDirection: 'column', gap: 0 }}>
              {leaderboard.map((ev, i) => {
                const werPct = (ev.corpus_wer || 0) * 100
                return (
                  <div
                    key={ev.id}
                    onClick={() => navigate(`/report/${ev.id}`)}
                    style={{
                      display: 'flex', alignItems: 'center', gap: 10,
                      padding: '10px 0',
                      borderTop: i > 0 ? `1px solid ${COLORS.border}` : 'none',
                      cursor: 'pointer',
                    }}
                  >
                    <div style={{
                      width: 26, height: 26, borderRadius: '50%', display: 'flex',
                      alignItems: 'center', justifyContent: 'center', flexShrink: 0,
                      fontSize: 12, fontWeight: 700,
                      background: i === 0 ? COLORS.warning + '30' : COLORS.bg,
                      color: i === 0 ? COLORS.warning : COLORS.textMid,
                      border: `1px solid ${i === 0 ? COLORS.warning + '60' : COLORS.border}`,
                    }}>
                      {i + 1}
                    </div>
                    <div style={{ flex: 1, minWidth: 0 }}>
                      <div style={{
                        fontSize: 13, fontWeight: 600, color: COLORS.textDark,
                        overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap',
                      }}>
                        {ev.model_name}
                      </div>
                      <div style={{ fontSize: 11, color: COLORS.textLight }}>{ev.dataset_name}</div>
                    </div>
                    <div style={{ textAlign: 'right', flexShrink: 0 }}>
                      <div style={{ fontSize: 15, fontWeight: 700, color: werPct > 50 ? COLORS.danger : COLORS.accent }}>
                        {werPct.toFixed(2)}%
                      </div>
                      <div style={{ fontSize: 10, color: COLORS.textLight }}>WER</div>
                    </div>
                    {/* mini bar */}
                    <div style={{ width: 50, height: 6, borderRadius: 3, background: COLORS.border, flexShrink: 0 }}>
                      <div style={{
                        width: `${Math.min(100, werPct)}%`, height: '100%', borderRadius: 3,
                        background: werPct > 50 ? COLORS.danger : COLORS.accent,
                      }} />
                    </div>
                  </div>
                )
              })}
            </div>
          )}
        </div>
      </div>

      {/* Row 3: Chart + Training Monitor */}
      <div style={{ display: 'grid', gridTemplateColumns: '1.5fr 1fr', gap: 16 }}>
        {/* WER Comparison Chart */}
        <div style={card}>
          <h2 style={sectionTitle}><BarChart3 size={16} /> WER / CER Comparison</h2>
          {chartData.length === 0 ? (
            <div style={emptyState}>No completed evaluations to chart.</div>
          ) : (
            <div style={{ width: '100%', height: 280 }}>
              <ResponsiveContainer>
                <BarChart data={chartData} margin={{ top: 10, right: 10, left: 0, bottom: 0 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke={COLORS.border} />
                  <XAxis dataKey="name" tick={{ fontSize: 11 }} stroke={COLORS.textMid} interval={0} angle={-15} textAnchor="end" height={50} />
                  <YAxis tick={{ fontSize: 11 }} stroke={COLORS.textMid} unit="%" />
                  <Tooltip
                    formatter={(val, name) => [`${val}%`, name]}
                    labelFormatter={(label, payload) => payload?.[0]?.payload?.fullName || label}
                  />
                  <Legend wrapperStyle={{ fontSize: 12 }} />
                  <Bar dataKey="WER" fill={CHART_PALETTE[0]} radius={[4, 4, 0, 0]} />
                  <Bar dataKey="CER" fill={CHART_PALETTE[1]} radius={[4, 4, 0, 0]} />
                </BarChart>
              </ResponsiveContainer>
            </div>
          )}
        </div>

        {/* Training Monitor */}
        <div style={card}>
          <div style={cardHeader}>
            <h2 style={sectionTitle}><Brain size={16} /> Training Monitor</h2>
            <button onClick={() => navigate('/train-runs')} style={linkBtn}>
              All runs <ArrowRight size={13} />
            </button>
          </div>
          {activeTrains.length === 0 ? (
            <div style={{ ...emptyState, paddingTop: 40, paddingBottom: 40 }}>
              <Brain size={36} style={{ color: COLORS.border, marginBottom: 12 }} />
              <div>No active training</div>
              <button
                onClick={() => navigate('/train-runs/new')}
                style={{ ...linkBtn, marginTop: 12, fontSize: 13, color: COLORS.accent }}
              >
                Start new training <ArrowRight size={13} />
              </button>
            </div>
          ) : (
            <div style={{ display: 'flex', flexDirection: 'column', gap: 14 }}>
              {activeTrains.map(t => {
                const p = t.total_steps > 0 ? Math.min(100, (t.current_step / t.total_steps) * 100) : 0
                return (
                  <div
                    key={t.id}
                    onClick={() => navigate(`/train-runs/${t.id}`)}
                    style={{ cursor: 'pointer' }}
                  >
                    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 6 }}>
                      <div style={{ fontSize: 13, fontWeight: 600, color: COLORS.textDark }}>
                        {t.name}
                      </div>
                      <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                        <StatusBadge status={t.status} />
                        {t.phase && (
                          <span style={{
                            padding: '2px 8px', borderRadius: 12, fontSize: 10, fontWeight: 600,
                            background: COLORS.secondary1 + '80', color: COLORS.secondary2,
                          }}>
                            {t.phase}
                          </span>
                        )}
                      </div>
                    </div>
                    <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 11, color: COLORS.textLight, marginBottom: 4 }}>
                      <span>Step {t.current_step} / {t.total_steps || '?'}</span>
                      <span>{p.toFixed(1)}%</span>
                    </div>
                    <div style={{ height: 6, borderRadius: 3, background: COLORS.secondary1 + '80', overflow: 'hidden' }}>
                      <div style={{
                        height: '100%', borderRadius: 3,
                        background: `linear-gradient(90deg, ${COLORS.primary}, ${COLORS.accent})`,
                        width: `${p}%`, transition: 'width 0.4s ease',
                      }} />
                    </div>
                    {t.current_loss != null && (
                      <div style={{ fontSize: 11, color: COLORS.textMid, marginTop: 4 }}>
                        Loss: {Number(t.current_loss).toFixed(4)}
                        {t.current_epoch != null && ` · Epoch ${Number(t.current_epoch).toFixed(1)}`}
                      </div>
                    )}
                  </div>
                )
              })}
            </div>
          )}
        </div>
      </div>
    </div>
  )
}

// ──────────────────── helpers ────────────────────

function timeAgo(dateStr) {
  if (!dateStr) return '—'
  const diff = Date.now() - new Date(dateStr).getTime()
  const mins = Math.floor(diff / 60000)
  if (mins < 1) return 'just now'
  if (mins < 60) return `${mins}m ago`
  const hrs = Math.floor(mins / 60)
  if (hrs < 24) return `${hrs}h ago`
  const days = Math.floor(hrs / 24)
  return `${days}d ago`
}

// ──────────────────── styles ────────────────────

const card = {
  background: COLORS.card,
  borderRadius: 12,
  padding: '20px 22px',
  border: `1px solid ${COLORS.border}`,
}

const cardHeader = {
  display: 'flex', justifyContent: 'space-between', alignItems: 'center',
  marginBottom: 14,
}

const sectionTitle = {
  fontSize: 15, fontWeight: 700, color: COLORS.textDark,
  margin: 0, display: 'flex', alignItems: 'center', gap: 8,
}

const refreshBtn = {
  display: 'flex', alignItems: 'center', gap: 6,
  padding: '8px 16px', borderRadius: 8,
  border: `1px solid ${COLORS.secondary2}`,
  background: 'transparent', color: COLORS.secondary2,
  cursor: 'pointer', fontSize: 13, fontWeight: 600,
}

const linkBtn = {
  display: 'flex', alignItems: 'center', gap: 4,
  background: 'none', border: 'none', cursor: 'pointer',
  color: COLORS.secondary2, fontSize: 12, fontWeight: 600,
}

const emptyState = {
  padding: 24, textAlign: 'center',
  color: COLORS.textLight, fontSize: 13,
  display: 'flex', flexDirection: 'column', alignItems: 'center',
}
