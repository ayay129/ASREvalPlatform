import { COLORS } from '../theme'

export default function MetricCard({ label, value, sub, color, icon: Icon }) {
  const c = color || COLORS.accent
  return (
    <div style={{
      background: COLORS.card, borderRadius: 12, padding: '20px 24px',
      border: `1px solid ${COLORS.border}`,
      boxShadow: '0 2px 8px rgba(100,130,173,0.08)',
      display: 'flex', flexDirection: 'column', gap: 8,
    }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start' }}>
        <span style={{ fontSize: 12, color: COLORS.textLight, fontWeight: 500, textTransform: 'uppercase', letterSpacing: '0.05em' }}>
          {label}
        </span>
        {Icon && (
          <div style={{
            width: 32, height: 32, borderRadius: 8,
            background: c + '18', display: 'flex', alignItems: 'center', justifyContent: 'center',
          }}>
            <Icon size={16} color={c} />
          </div>
        )}
      </div>
      <div style={{ fontSize: 28, fontWeight: 700, color: c, lineHeight: 1 }}>
        {value}
      </div>
      {sub && <div style={{ fontSize: 12, color: COLORS.textLight }}>{sub}</div>}
    </div>
  )
}
