import { Fragment, useCallback, useEffect, useState } from 'react'
import {
  RefreshCw, Download, Copy, Eye, Trash2, X, Cloud, HardDrive,
  FileText, FileJson, AlertTriangle, ChevronDown, ChevronRight,
} from 'lucide-react'
import { api } from '../api'
import { COLORS } from '../theme'
import StatusBadge from '../components/StatusBadge'

const KIND_LABEL = {
  eval_csv: 'Eval CSV',
  train_manifest: 'Train Manifest',
}

const KIND_ICON = {
  eval_csv: FileText,
  train_manifest: FileJson,
}

const fmtBytes = (n) => {
  if (!n && n !== 0) return '—'
  if (n < 1024) return `${n} B`
  if (n < 1024 ** 2) return `${(n / 1024).toFixed(1)} KB`
  if (n < 1024 ** 3) return `${(n / 1024 ** 2).toFixed(1)} MB`
  return `${(n / 1024 ** 3).toFixed(2)} GB`
}

const fmtDuration = (s) => {
  if (s == null) return '—'
  if (s < 60) return `${s.toFixed(0)}s`
  if (s < 3600) return `${(s / 60).toFixed(1)}m`
  return `${(s / 3600).toFixed(2)}h`
}

const fmtDate = (v) => (v ? new Date(v).toLocaleString() : '—')

export default function Datasets() {
  const [datasets, setDatasets] = useState([])
  const [pulls, setPulls] = useState([])
  const [kindFilter, setKindFilter] = useState('')
  const [loading, setLoading] = useState(true)
  const [scanning, setScanning] = useState(false)
  const [toast, setToast] = useState('')

  const [pullOpen, setPullOpen] = useState(false)
  const [previewing, setPreviewing] = useState(null) // {ds, data}
  const [expandedPullId, setExpandedPullId] = useState(null)

  const load = useCallback(async () => {
    setLoading(true)
    try {
      const [ds, pl] = await Promise.all([
        api.listDatasets(kindFilter ? { kind: kindFilter } : {}),
        api.listDatasetPulls(),
      ])
      setDatasets(ds)
      setPulls(pl)
    } finally {
      setLoading(false)
    }
  }, [kindFilter])

  useEffect(() => { load() }, [load])

  // 如果有正在进行的 pull，每 3s 刷一次
  useEffect(() => {
    const active = pulls.some(p => p.status === 'queued' || p.status === 'running')
    if (!active) return
    const t = setInterval(load, 3000)
    return () => clearInterval(t)
  }, [pulls, load])

  const runScan = async () => {
    setScanning(true)
    try {
      const res = await api.scanDatasets()
      setToast(`Scan: +${res.added} added, ${res.updated} updated, ${res.removed} missing`)
      setTimeout(() => setToast(''), 3000)
      await load()
    } catch (err) {
      setToast(err.response?.data?.detail || 'Scan failed')
    } finally {
      setScanning(false)
    }
  }

  const copyPath = async (path) => {
    try {
      await navigator.clipboard.writeText(path)
      setToast('Path copied')
      setTimeout(() => setToast(''), 1500)
    } catch {
      setToast('Copy failed')
    }
  }

  const remove = async (id) => {
    if (!confirm('Remove this dataset from registry? (file on disk is NOT deleted)')) return
    await api.deleteDataset(id)
    await load()
  }

  const openPreview = async (ds) => {
    setPreviewing({ ds, data: null, loading: true })
    try {
      const data = await api.previewDataset(ds.id, 5)
      setPreviewing({ ds, data, loading: false })
    } catch (err) {
      setPreviewing({ ds, data: null, loading: false, error: err.message })
    }
  }

  const activePulls = pulls.filter(p => p.status === 'queued' || p.status === 'running')
  const recentPulls = pulls.slice(0, 5)

  return (
    <div>
      {/* Header */}
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 24 }}>
        <div>
          <h1 style={{ fontSize: 24, fontWeight: 700, color: COLORS.textDark }}>Datasets</h1>
          <p style={{ color: COLORS.textLight, fontSize: 14, marginTop: 4 }}>
            Registry of local and HuggingFace-pulled datasets
          </p>
        </div>
        <div style={{ display: 'flex', gap: 10 }}>
          <button onClick={runScan} style={btnOutline} disabled={scanning}>
            <RefreshCw size={15} className={scanning ? 'spin' : ''} />
            {scanning ? 'Scanning…' : 'Scan local'}
          </button>
          <button onClick={() => setPullOpen(true)} style={btnAccent}>
            <Cloud size={15} /> Pull from HuggingFace
          </button>
        </div>
      </div>

      {/* Active pulls banner */}
      {activePulls.length > 0 && (
        <div style={pullBanner}>
          <Download size={16} />
          <div style={{ flex: 1 }}>
            <div style={{ fontWeight: 600 }}>
              {activePulls.length} HF pull{activePulls.length > 1 ? 's' : ''} in progress
            </div>
            <div style={{ fontSize: 12, color: COLORS.textLight, marginTop: 2 }}>
              {activePulls.map(p => p.repo_id).join(', ')}
            </div>
          </div>
        </div>
      )}

      {/* Kind filter tabs */}
      <div style={{ display: 'flex', gap: 10, marginBottom: 16 }}>
        {[
          { val: '', label: 'All', count: datasets.length },
          { val: 'eval_csv', label: 'Eval CSV' },
          { val: 'train_manifest', label: 'Train Manifest' },
        ].map(tab => (
          <button
            key={tab.val || 'all'}
            onClick={() => setKindFilter(tab.val)}
            style={{
              padding: '6px 14px', borderRadius: 20, fontSize: 13, cursor: 'pointer',
              border: '1px solid',
              fontWeight: kindFilter === tab.val ? 600 : 400,
              borderColor: kindFilter === tab.val ? COLORS.accent : COLORS.border,
              background: kindFilter === tab.val ? COLORS.accent : 'transparent',
              color: kindFilter === tab.val ? '#fff' : COLORS.textMid,
            }}
          >
            {tab.label}
          </button>
        ))}
      </div>

      {/* Datasets table */}
      <div style={card}>
        <table style={{ width: '100%', borderCollapse: 'collapse' }}>
          <thead>
            <tr style={{ background: COLORS.secondary1 + '60' }}>
              {['Name', 'Kind', 'Rows', 'Size', 'Duration', 'Source', 'Status', 'Actions'].map(h => (
                <th key={h} style={th}>{h}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {loading ? (
              <tr><td colSpan={8} style={emptyCell}>Loading…</td></tr>
            ) : datasets.length === 0 ? (
              <tr><td colSpan={8} style={emptyCell}>
                No datasets yet. Click <strong>Scan local</strong> or pull one from HuggingFace.
              </td></tr>
            ) : datasets.map((ds, i) => {
              const Icon = KIND_ICON[ds.kind] || FileText
              return (
                <tr key={ds.id} style={{ borderTop: `1px solid ${COLORS.border}`, background: i % 2 === 0 ? '#fff' : COLORS.bg }}>
                  <td style={td}>
                    <div style={{ display: 'flex', alignItems: 'flex-start', gap: 8 }}>
                      <Icon size={14} style={{ color: COLORS.secondary2, marginTop: 2, flexShrink: 0 }} />
                      <div>
                        <div style={{ fontWeight: 600, color: COLORS.textDark }}>{ds.name}</div>
                        <div style={{ fontSize: 11, color: COLORS.textLight, fontFamily: 'ui-monospace,Menlo,monospace', marginTop: 2, wordBreak: 'break-all' }}>
                          {ds.path}
                        </div>
                      </div>
                    </div>
                  </td>
                  <td style={td}><span style={kindChip}>{KIND_LABEL[ds.kind] || ds.kind}</span></td>
                  <td style={td}>{ds.rows?.toLocaleString() || '—'}</td>
                  <td style={td}>{fmtBytes(ds.size_bytes)}</td>
                  <td style={td}>{fmtDuration(ds.duration_sec)}</td>
                  <td style={td}>
                    {ds.source === 'huggingface' ? (
                      <span style={{ ...sourceChip, color: COLORS.accent, background: COLORS.accent + '14' }}>
                        <Cloud size={11} /> {ds.source_repo || 'hf'}
                      </span>
                    ) : (
                      <span style={{ ...sourceChip, color: COLORS.textMid, background: COLORS.secondary1 + '60' }}>
                        <HardDrive size={11} /> local
                      </span>
                    )}
                  </td>
                  <td style={td}><StatusBadge status={ds.status} /></td>
                  <td style={td}>
                    <div style={{ display: 'flex', gap: 6 }}>
                      <IconBtn title="Preview" onClick={() => openPreview(ds)}><Eye size={14} /></IconBtn>
                      <IconBtn title="Copy path" onClick={() => copyPath(ds.path)}><Copy size={14} /></IconBtn>
                      <IconBtn title="Remove from registry" danger onClick={() => remove(ds.id)}><Trash2 size={14} /></IconBtn>
                    </div>
                  </td>
                </tr>
              )
            })}
          </tbody>
        </table>
      </div>

      {/* Recent pulls */}
      {recentPulls.length > 0 && (
        <div style={{ ...card, marginTop: 20 }}>
          <h2 style={sectionTitle}>Recent pulls</h2>
          <table style={{ width: '100%', borderCollapse: 'collapse' }}>
            <thead>
              <tr style={{ background: COLORS.secondary1 + '40' }}>
                <th style={{ ...th, fontSize: 11, width: 30 }}></th>
                {['Repo', 'Revision', 'Status', 'Registered', 'Local dir', 'Finished'].map(h => (
                  <th key={h} style={{ ...th, fontSize: 11 }}>{h}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {recentPulls.map(p => {
                const hasDetail = !!(p.error_message || p.log_tail)
                const isFailed = p.status === 'failed'
                const expanded = expandedPullId === p.id
                return (
                  <Fragment key={p.id}>
                    <tr
                      style={{
                        borderTop: `1px solid ${COLORS.border}`,
                        background: isFailed ? COLORS.danger + '08' : 'transparent',
                        cursor: hasDetail ? 'pointer' : 'default',
                      }}
                      onClick={() => hasDetail && setExpandedPullId(expanded ? null : p.id)}
                    >
                      <td style={{ ...td, color: COLORS.textLight }}>
                        {hasDetail
                          ? (expanded ? <ChevronDown size={14} /> : <ChevronRight size={14} />)
                          : null}
                      </td>
                      <td style={td}>
                        <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
                          {isFailed && <AlertTriangle size={13} style={{ color: COLORS.danger }} />}
                          <code>{p.repo_id}</code>
                        </div>
                      </td>
                      <td style={td}>{p.revision || 'main'}</td>
                      <td style={td}><StatusBadge status={p.status} /></td>
                      <td style={td}>{p.registered_count}</td>
                      <td style={{ ...td, fontFamily: 'ui-monospace,Menlo,monospace', fontSize: 11, color: COLORS.textLight }}>
                        {p.local_dir || '—'}
                      </td>
                      <td style={{ ...td, fontSize: 11, color: COLORS.textLight }}>{fmtDate(p.completed_at)}</td>
                    </tr>
                    {expanded && (
                      <tr style={{ background: isFailed ? COLORS.danger + '06' : COLORS.bg }}>
                        <td />
                        <td colSpan={6} style={{ padding: '10px 14px 16px' }}>
                          {p.error_message && (
                            <div style={pullErrorBox}>
                              <div style={{ fontWeight: 600, marginBottom: 4 }}>Error</div>
                              <div style={{ whiteSpace: 'pre-wrap' }}>{p.error_message}</div>
                            </div>
                          )}
                          {p.log_tail && (
                            <div>
                              <div style={{ fontSize: 11, fontWeight: 600, color: COLORS.textMid, marginBottom: 4 }}>
                                Log tail
                              </div>
                              <pre style={pullLogPre}>{p.log_tail}</pre>
                            </div>
                          )}
                        </td>
                      </tr>
                    )}
                  </Fragment>
                )
              })}
            </tbody>
          </table>
        </div>
      )}

      {toast && <div style={toastStyle}>{toast}</div>}

      {pullOpen && (
        <PullModal
          onClose={() => setPullOpen(false)}
          onSubmitted={() => { setPullOpen(false); load() }}
        />
      )}

      {previewing && (
        <PreviewModal info={previewing} onClose={() => setPreviewing(null)} />
      )}

      <style>{`
        .spin { animation: spin 0.9s linear infinite; }
        @keyframes spin { to { transform: rotate(360deg); } }
      `}</style>
    </div>
  )
}

// ──────────────────── sub-components ────────────────────

function IconBtn({ children, onClick, title, danger }) {
  return (
    <button
      title={title}
      onClick={onClick}
      style={{
        display: 'inline-flex', alignItems: 'center', justifyContent: 'center',
        width: 28, height: 28, borderRadius: 6,
        background: 'transparent',
        border: `1px solid ${danger ? COLORS.danger + '40' : COLORS.border}`,
        color: danger ? COLORS.danger : COLORS.textMid,
        cursor: 'pointer',
      }}
    >
      {children}
    </button>
  )
}

function PullModal({ onClose, onSubmitted }) {
  const [repoId, setRepoId] = useState('')
  const [revision, setRevision] = useState('')
  const [allowPatterns, setAllowPatterns] = useState('')
  const [submitting, setSubmitting] = useState(false)
  const [error, setError] = useState('')

  const submit = async (e) => {
    e.preventDefault()
    if (!repoId.trim()) { setError('repo_id is required'); return }
    setSubmitting(true)
    setError('')
    try {
      await api.createDatasetPull({
        repo_id: repoId.trim(),
        revision: revision.trim() || undefined,
        allow_patterns: allowPatterns.trim() || undefined,
      })
      onSubmitted()
    } catch (err) {
      setError(err.response?.data?.detail || 'Failed to enqueue pull')
    } finally {
      setSubmitting(false)
    }
  }

  return (
    <div style={modalBackdrop} onClick={onClose}>
      <div style={modalBody} onClick={e => e.stopPropagation()}>
        <div style={modalHeader}>
          <h3 style={{ margin: 0, fontSize: 16, fontWeight: 700 }}>Pull from HuggingFace</h3>
          <button style={closeBtn} onClick={onClose}><X size={16} /></button>
        </div>
        <form onSubmit={submit} style={{ padding: 24 }}>
          <div style={{ marginBottom: 16 }}>
            <label style={labelStyle}>Repository ID *</label>
            <input
              style={input}
              placeholder="user/my-dataset"
              value={repoId}
              onChange={e => setRepoId(e.target.value)}
              autoFocus
            />
          </div>
          <div style={{ marginBottom: 16 }}>
            <label style={labelStyle}>Revision (optional)</label>
            <input
              style={input}
              placeholder="main, a branch name, or a commit SHA"
              value={revision}
              onChange={e => setRevision(e.target.value)}
            />
          </div>
          <div style={{ marginBottom: 16 }}>
            <label style={labelStyle}>Subset / allow patterns (optional)</label>
            <input
              style={input}
              placeholder="transcript/mn/**,audio/mn/**"
              value={allowPatterns}
              onChange={e => setAllowPatterns(e.target.value)}
            />
            <div style={{ fontSize: 11, color: COLORS.textLight, marginTop: 4, lineHeight: 1.5 }}>
              Comma-separated globs. Leave empty to pull the whole repo.<br />
              For Common Voice Mongolian try <code>transcript/mn/**,audio/mn/**</code>
            </div>
          </div>
          <div style={{ fontSize: 12, color: COLORS.textLight, marginBottom: 16 }}>
            Downloads via <code>snapshot_download</code> into the server's
            dataset directory, then scans and registers any CSV / JSONL files found.
          </div>
          {error && <div style={errorBox}>{error}</div>}
          <div style={{ display: 'flex', justifyContent: 'flex-end', gap: 10, marginTop: 20 }}>
            <button type="button" style={btnGhost} onClick={onClose}>Cancel</button>
            <button type="submit" disabled={submitting} style={btnAccent}>
              <Download size={14} />
              {submitting ? 'Enqueuing…' : 'Start pull'}
            </button>
          </div>
        </form>
      </div>
    </div>
  )
}

function PreviewModal({ info, onClose }) {
  const { ds, data, loading: l, error } = info

  return (
    <div style={modalBackdrop} onClick={onClose}>
      <div style={{ ...modalBody, width: 720 }} onClick={e => e.stopPropagation()}>
        <div style={modalHeader}>
          <h3 style={{ margin: 0, fontSize: 16, fontWeight: 700 }}>
            Preview · {ds.name}
          </h3>
          <button style={closeBtn} onClick={onClose}><X size={16} /></button>
        </div>
        <div style={{ padding: 20, maxHeight: '70vh', overflow: 'auto' }}>
          {l && <div style={{ color: COLORS.textLight }}>Loading…</div>}
          {error && <div style={errorBox}>{error}</div>}
          {data && (
            <>
              {data.columns && (
                <div style={{ marginBottom: 12, fontSize: 12 }}>
                  <strong style={{ color: COLORS.textMid }}>Columns:</strong>{' '}
                  {data.columns.map(c => (
                    <code key={c} style={{ marginRight: 6, padding: '1px 6px', background: COLORS.bg, borderRadius: 4, fontSize: 11 }}>
                      {c}
                    </code>
                  ))}
                </div>
              )}
              {data.rows.length === 0 ? (
                <div style={{ color: COLORS.textLight }}>Empty.</div>
              ) : (
                <pre style={previewPre}>{JSON.stringify(data.rows, null, 2)}</pre>
              )}
            </>
          )}
        </div>
      </div>
    </div>
  )
}

// ──────────────────── styles ────────────────────

const card = {
  background: COLORS.card,
  borderRadius: 12,
  border: `1px solid ${COLORS.border}`,
  overflow: 'hidden',
}

const sectionTitle = {
  fontSize: 14, fontWeight: 700, color: COLORS.textDark,
  padding: '16px 20px', margin: 0,
  borderBottom: `1px solid ${COLORS.border}`,
}

const th = {
  padding: '12px 14px', textAlign: 'left',
  fontSize: 12, fontWeight: 600, color: COLORS.textMid,
  whiteSpace: 'nowrap',
}

const td = {
  padding: '12px 14px', fontSize: 13,
  color: COLORS.textDark, verticalAlign: 'top',
}

const emptyCell = {
  textAlign: 'center', padding: 48, color: COLORS.textLight,
}

const btnAccent = {
  display: 'flex', alignItems: 'center', gap: 6,
  padding: '8px 16px', borderRadius: 8,
  border: `1px solid ${COLORS.accent}`,
  background: COLORS.accent, color: '#fff',
  cursor: 'pointer', fontSize: 13, fontWeight: 600,
}

const btnOutline = {
  display: 'flex', alignItems: 'center', gap: 6,
  padding: '8px 16px', borderRadius: 8,
  border: `1px solid ${COLORS.secondary2}`,
  background: 'transparent', color: COLORS.secondary2,
  cursor: 'pointer', fontSize: 13, fontWeight: 600,
}

const btnGhost = {
  padding: '8px 16px', borderRadius: 8,
  border: `1px solid ${COLORS.border}`,
  background: '#fff', color: COLORS.textMid,
  cursor: 'pointer', fontSize: 13,
}

const kindChip = {
  display: 'inline-block',
  padding: '3px 8px', borderRadius: 4,
  fontSize: 11, fontWeight: 600,
  background: COLORS.secondary1 + '80',
  color: COLORS.secondary2,
}

const sourceChip = {
  display: 'inline-flex', alignItems: 'center', gap: 5,
  padding: '3px 9px', borderRadius: 20,
  fontSize: 11, fontWeight: 600,
}

const pullBanner = {
  display: 'flex', alignItems: 'center', gap: 12,
  padding: '12px 16px', marginBottom: 16,
  background: COLORS.warning + '15',
  border: `1px solid ${COLORS.warning}40`,
  borderRadius: 8, color: COLORS.textDark,
}

const modalBackdrop = {
  position: 'fixed', inset: 0, zIndex: 1000,
  background: 'rgba(30,42,58,0.45)',
  display: 'flex', alignItems: 'center', justifyContent: 'center',
}

const modalBody = {
  background: '#fff', borderRadius: 12, width: 480,
  boxShadow: '0 10px 40px rgba(0,0,0,0.2)',
  maxHeight: '90vh', overflow: 'hidden',
  display: 'flex', flexDirection: 'column',
}

const modalHeader = {
  display: 'flex', justifyContent: 'space-between', alignItems: 'center',
  padding: '16px 20px', borderBottom: `1px solid ${COLORS.border}`,
}

const closeBtn = {
  width: 28, height: 28, borderRadius: 6,
  border: 'none', background: 'transparent', cursor: 'pointer',
  color: COLORS.textMid,
}

const labelStyle = {
  display: 'block', fontSize: 13, fontWeight: 600,
  color: COLORS.textMid, marginBottom: 6,
}

const input = {
  width: '100%', padding: '9px 12px', borderRadius: 8,
  border: `1px solid ${COLORS.border}`, fontSize: 14,
  color: COLORS.textDark, outline: 'none', background: '#fff',
  fontFamily: 'inherit', boxSizing: 'border-box',
}

const errorBox = {
  padding: '10px 14px', borderRadius: 8,
  background: COLORS.danger + '18',
  border: `1px solid ${COLORS.danger}40`,
  color: COLORS.danger, fontSize: 13,
  marginBottom: 12,
}

const toastStyle = {
  position: 'fixed', bottom: 24, right: 24, zIndex: 2000,
  padding: '10px 16px', borderRadius: 8,
  background: COLORS.textDark, color: '#fff',
  fontSize: 13, fontWeight: 500,
  boxShadow: '0 4px 20px rgba(0,0,0,0.25)',
}

const previewPre = {
  fontSize: 12, lineHeight: 1.5,
  background: COLORS.bg, padding: 12, borderRadius: 8,
  overflow: 'auto', fontFamily: 'ui-monospace,Menlo,monospace',
  color: COLORS.textDark,
}

const pullErrorBox = {
  padding: '10px 12px', borderRadius: 6,
  background: COLORS.danger + '12',
  border: `1px solid ${COLORS.danger}30`,
  color: COLORS.danger,
  fontSize: 12, marginBottom: 10,
  fontFamily: 'ui-monospace,Menlo,monospace',
}

const pullLogPre = {
  fontSize: 11, lineHeight: 1.5,
  background: '#1e2a3a', color: '#cfe3ff',
  padding: 12, borderRadius: 6, margin: 0,
  overflow: 'auto', maxHeight: 240,
  fontFamily: 'ui-monospace,Menlo,monospace',
  whiteSpace: 'pre-wrap', wordBreak: 'break-all',
}
