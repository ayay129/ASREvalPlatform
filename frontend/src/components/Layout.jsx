import { Outlet, NavLink, useLocation } from 'react-router-dom'
import { LayoutDashboard, GitCompare, Mic2, FlaskConical, Database, ClipboardList } from 'lucide-react'
import { COLORS } from '../theme'

const NAV = [
  { to: '/dashboard',   label: 'Dashboard',   Icon: LayoutDashboard },
  { to: '/datasets',    label: 'Datasets',    Icon: Database },
  { to: '/train-runs',  label: 'Train Runs',  Icon: FlaskConical },
  { to: '/evaluations', label: 'Evaluations', Icon: ClipboardList },
  { to: '/compare',     label: 'Compare',     Icon: GitCompare },
]

const S = {
  shell: {
    display: 'flex', minHeight: '100vh', background: COLORS.bg,
  },
  sidebar: {
    width: 220, background: COLORS.accent, display: 'flex',
    flexDirection: 'column', padding: '0',
    boxShadow: '2px 0 12px rgba(62,84,172,0.15)',
    position: 'fixed', top: 0, bottom: 0, left: 0, zIndex: 100,
  },
  logoBox: {
    padding: '28px 24px 24px',
    borderBottom: `1px solid rgba(255,255,255,0.15)`,
    display: 'flex', alignItems: 'center', gap: 10,
  },
  logoText: {
    color: '#fff', fontWeight: 700, fontSize: 16, lineHeight: 1.2,
  },
  logoSub: {
    color: COLORS.secondary1, fontSize: 11, marginTop: 2,
  },
  nav: { padding: '16px 0', flex: 1 },
  link: (active) => ({
    display: 'flex', alignItems: 'center', gap: 10,
    padding: '11px 24px', textDecoration: 'none',
    color: active ? '#fff' : COLORS.secondary1,
    background: active ? 'rgba(255,255,255,0.15)' : 'transparent',
    borderLeft: active ? `3px solid ${COLORS.primary}` : '3px solid transparent',
    fontWeight: active ? 600 : 400,
    fontSize: 14, transition: 'all 0.15s',
  }),
  main: {
    marginLeft: 220, flex: 1, padding: '32px 36px', minWidth: 0,
  },
}

export default function Layout() {
  const loc = useLocation()

  return (
    <div style={S.shell}>
      <aside style={S.sidebar}>
        <div style={S.logoBox}>
          <Mic2 size={22} color={COLORS.primary} />
          <div>
            <div style={S.logoText}>ASR Eval</div>
            <div style={S.logoSub}>Evaluation Platform</div>
          </div>
        </div>
        <nav style={S.nav}>
          {NAV.map(({ to, label, Icon }) => {
            const active = loc.pathname.startsWith(to)
            return (
              <NavLink key={to} to={to} style={S.link(active)}>
                <Icon size={17} />
                {label}
              </NavLink>
            )
          })}
        </nav>
        <div style={{ padding: '16px 24px', color: COLORS.secondary1, fontSize: 11 }}>
          v1.0.0
        </div>
      </aside>
      <main style={S.main}>
        <Outlet />
      </main>
    </div>
  )
}
