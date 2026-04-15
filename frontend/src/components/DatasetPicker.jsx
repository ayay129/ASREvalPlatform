import { useEffect, useState } from 'react'
import { ChevronDown, Cloud, HardDrive } from 'lucide-react'
import { api } from '../api'
import { COLORS } from '../theme'

/**
 * 数据集选择器。
 *
 * 逻辑：
 *   - 下拉选择注册表里的 dataset（按 kind 过滤）
 *   - 下方始终保留一个手输框作为兜底；选了下拉会自动填到手输框
 *   - 用户手改路径不会清掉下拉选中项，方便基于模板微调
 *
 * Props:
 *   kind          筛选 kind: 'eval_csv' | 'train_manifest'
 *   value         当前路径字符串
 *   onChange      (path: string) => void
 *   placeholder   手输框的 placeholder
 *   label         顶部 label
 *   required      * 号
 */
export default function DatasetPicker({
  kind, value, onChange,
  placeholder = '', label = 'Dataset', required = false,
}) {
  const [list, setList] = useState([])
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    setLoading(true)
    api.listDatasets({ kind })
      .then(data => setList(data || []))
      .catch(() => setList([]))
      .finally(() => setLoading(false))
  }, [kind])

  const selected = list.find(d => d.path === value) || null

  return (
    <div>
      <label style={labelStyle}>
        {label}{required && ' *'}
      </label>

      <div style={{ position: 'relative', marginBottom: 8 }}>
        <select
          style={{ ...input, appearance: 'none', paddingRight: 36, cursor: 'pointer' }}
          value={selected?.id || ''}
          onChange={e => {
            const id = e.target.value
            if (!id) return
            const ds = list.find(d => String(d.id) === id)
            if (ds) onChange(ds.path)
          }}
        >
          <option value="">
            {loading
              ? 'Loading…'
              : list.length === 0
                ? `No ${kind.replace('_', ' ')} registered yet — scan or pull from HF`
                : `Select from registry (${list.length})`}
          </option>
          {list.map(ds => (
            <option key={ds.id} value={ds.id}>
              {ds.name}
              {ds.rows ? ` · ${ds.rows.toLocaleString()} rows` : ''}
              {ds.source === 'huggingface' ? ` · ☁ ${ds.source_repo}` : ''}
              {ds.status === 'missing' ? ' · ⚠ missing' : ''}
            </option>
          ))}
        </select>
        <ChevronDown size={15} style={chevron} />
      </div>

      <input
        style={input}
        placeholder={placeholder}
        value={value}
        onChange={e => onChange(e.target.value)}
      />

      {selected && (
        <div style={selectedHint}>
          {selected.source === 'huggingface'
            ? <Cloud size={11} />
            : <HardDrive size={11} />}
          <span>{selected.source === 'huggingface' ? selected.source_repo : 'local'}</span>
          <span style={{ color: COLORS.textLight }}>·</span>
          <span>{selected.kind}</span>
          {selected.rows != null && (
            <>
              <span style={{ color: COLORS.textLight }}>·</span>
              <span>{selected.rows.toLocaleString()} rows</span>
            </>
          )}
        </div>
      )}
    </div>
  )
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

const chevron = {
  position: 'absolute', right: 12, top: '50%',
  transform: 'translateY(-50%)',
  color: COLORS.textLight, pointerEvents: 'none',
}

const selectedHint = {
  marginTop: 6,
  display: 'flex', alignItems: 'center', gap: 6,
  fontSize: 11, color: COLORS.textMid,
}
