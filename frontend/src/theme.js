export const COLORS = {
  primary:    '#A0BFE0',   // foggy blue
  secondary1: '#C5DFF8',   // light gray-blue
  secondary2: '#6482AD',   // dark gray-blue
  accent:     '#3E54AC',   // blue-purple

  bg:         '#EEF3F9',   // page background
  card:       '#FFFFFF',
  border:     '#D6E4F0',

  textDark:   '#1E2A3A',
  textMid:    '#4A5F78',
  textLight:  '#8BA4BF',

  success:    '#4CAF84',
  warning:    '#E8A838',
  danger:     '#D85C5C',
  pending:    '#A0BFE0',
}

export const STATUS_COLOR = {
  queued:      COLORS.secondary2,
  completed:   COLORS.success,
  running:     COLORS.warning,
  downloading: COLORS.warning,
  pending:     COLORS.secondary2,
  failed:      COLORS.danger,
  ready:       COLORS.success,
  missing:     COLORS.danger,
}

// Recharts palette that fits the theme
export const CHART_PALETTE = [
  '#3E54AC', '#6482AD', '#A0BFE0', '#C5DFF8',
  '#4CAF84', '#E8A838', '#D85C5C', '#8BA4BF',
]
