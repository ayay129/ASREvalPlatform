import { STATUS_COLOR } from '../theme'

export default function StatusBadge({ status }) {
  const color = STATUS_COLOR[status] || '#999'
  return (
    <span style={{
      display: 'inline-flex', alignItems: 'center', gap: 5,
      padding: '3px 10px', borderRadius: 20, fontSize: 12, fontWeight: 600,
      background: color + '1A', color,
      border: `1px solid ${color}40`,
    }}>
      <span style={{
        width: 6, height: 6, borderRadius: '50%', background: color,
        animation: status === 'running' ? 'pulse 1.2s infinite' : 'none',
      }} />
      {status}
      <style>{`@keyframes pulse{0%,100%{opacity:1}50%{opacity:.3}}`}</style>
    </span>
  )
}
