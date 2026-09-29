import { useEffect, useState } from 'react'
import {
  CartesianGrid,
  Legend,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts'

type Row = Record<string, number | null>
type Replay = { file: string; mornings: number; finished: boolean; mismatch: string | null }
type Metrics = { days: Row[]; now: Row; replay: Replay | null }
type Unit = 'count' | 'percent' | 'money' | 'seconds'
type ChartSpec = { title: string; note?: string; unit?: Unit; series: [key: string, label: string][] }

// Categorical slots 1-3, stepped for a dark surface and validated on the panel
// (#171720): adjacent colour-blind separation ΔE 9.4, every slot at least 3:1.
const SERIES = ['#3987e5', '#d95926', '#199e70']
// One step off the panel for the grid, the edge token for the baseline.
const CHROME = { surface: '#171720', grid: '#21212c', axis: '#272733', muted: '#8a8a99', ink: '#e8e8f0' }

const FORMAT: Record<Unit, (v: number) => string> = {
  count: (v) => Math.round(v).toLocaleString(),
  percent: (v) => `${Math.round(v * 100)}%`,
  money: (v) => `$${Math.round(v).toLocaleString()}`,
  seconds: (v) => `${v.toFixed(1)}s`,
}

// Three series at most per chart, and one unit per chart: never two y-axes.
const CHARTS: ChartSpec[] = [
  { title: 'Who is doing what', series: [['employed', 'employed'], ['students', 'studying'], ['looking', 'looking']] },
  { title: 'Attendance', note: 'Shifts worked of shifts called, since the start', unit: 'percent', series: [['attendance', 'attendance']] },
  { title: 'Job market', note: 'Running totals', series: [['hires', 'hired'], ['rejections', 'turned down'], ['let go', 'let go']] },
  { title: 'Careers', note: 'Running totals', series: [['promotions', 'promoted'], ['sponsored', 'sponsored courses'], ['dismissals', 'dismissed']] },
  { title: 'Median money', unit: 'money', series: [['median money', 'median money']] },
  { title: 'Warm ties', note: 'Relationships at affinity 20 or more', series: [['warm ties', 'warm ties']] },
  { title: 'Money trouble', note: 'People, each morning', series: [['overdrawn', 'overdrawn'], ['borrowers', 'owe the bank'], ['porters', 'porters']] },
  { title: 'Public money', note: 'Balances each morning', unit: 'money', series: [['treasury', 'treasury'], ['university fund', 'university fund']] },
  { title: "The model's day", note: 'Per sim-day', series: [['plans a day', 'plan requests'], ['chats a day', 'chats'], ['reflections a day', 'reflections']] },
  { title: 'Written by the model', note: 'Share of reviews and interviews so far', unit: 'percent', series: [['review share', 'reviews'], ['interview share', 'interviews']] },
  { title: 'Model latency', note: 'Seconds per call, smoothed', unit: 'seconds', series: [['latency fast', 'plans'], ['latency smart', 'talk']] },
]

// The backend stores running totals; per-day rates and shares are worked out here.
function derive(days: Row[]): Row[] {
  return days.map((r, i) => {
    const prev = i > 0 ? days[i - 1] : null
    const perDay = (k: string) => (prev ? (r[k] ?? 0) - (prev[k] ?? 0) : null)
    const share = (part: number | null, whole: number) => (whole > 0 ? (part ?? 0) / whole : null)
    return {
      ...r,
      'plans a day': perDay('plan requests'),
      'chats a day': perDay('chats'),
      'reflections a day': perDay('reflections'),
      'review share': share(r['reviews by model'], r['reviews'] ?? 0),
      'interview share': share(r['interviews by model'], (r['hires'] ?? 0) + (r['rejections'] ?? 0)),
    }
  })
}

type TipRow = { name?: string | number; value?: unknown; color?: string }

// Values lead and names follow, each keyed by a short stroke of its series colour.
function ChartTooltip({ active, label, payload, fmt }: {
  active?: boolean
  label?: string | number
  payload?: readonly TipRow[]
  fmt: (v: number) => string
}) {
  if (!active || !payload?.length) return null
  return (
    <div className="rounded border border-edge bg-panel px-2.5 py-1.5 text-xs shadow-lg">
      <div className="mb-1 text-muted">day {label}</div>
      {payload.map((p) => (
        <div key={String(p.name)} className="flex items-center gap-2">
          <span className="inline-block h-0.5 w-3 rounded" style={{ background: p.color }} />
          <span className="font-semibold tabular-nums">
            {typeof p.value === 'number' ? fmt(p.value) : '—'}
          </span>
          <span className="text-muted">{p.name}</span>
        </div>
      ))}
    </div>
  )
}

function Chart({ spec, data }: { spec: ChartSpec; data: Row[] }) {
  const fmt = FORMAT[spec.unit ?? 'count']
  const empty = data.every((r) => spec.series.every(([key]) => r[key] == null))
  return (
    // min-w-0: a grid item will not shrink below its content by default, so a
    // chart drawn wide would hold its card wide and spill into the next one.
    <figure className="m-0 min-w-0 rounded-md border border-edge bg-panel p-3" aria-label={spec.title}>
      <figcaption className="mb-1">
        <div className="text-sm">{spec.title}</div>
        {spec.note && <div className="text-[11px] text-muted">{spec.note}</div>}
      </figcaption>
      {/* The height includes the x-axis band, so the card never scrolls. */}
      {empty ? (
        <div className="flex h-48 items-center justify-center text-xs text-muted">No data yet</div>
      ) : (
        <div className="h-48">
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={data} margin={{ top: 6, right: 12, bottom: 0, left: 0 }}>
              <CartesianGrid vertical={false} stroke={CHROME.grid} />
              <XAxis dataKey="day" tick={{ fill: CHROME.muted, fontSize: 11 }} stroke={CHROME.axis} tickLine={false} />
              <YAxis
                width={44}
                tick={{ fill: CHROME.muted, fontSize: 11 }}
                axisLine={false}
                tickLine={false}
                tickFormatter={fmt}
                domain={spec.unit === 'percent' ? [0, 1] : ['auto', 'auto']}
              />
              <Tooltip
                cursor={{ stroke: CHROME.muted, strokeWidth: 1 }}
                content={({ active, label, payload }) => (
                  <ChartTooltip active={active} label={label} payload={payload} fmt={fmt} />
                )}
              />
              {/* A legend for two or more series; a single series is named by the title. */}
              {spec.series.length > 1 && (
                <Legend
                  verticalAlign="top"
                  align="left"
                  iconType="plainline"
                  wrapperStyle={{ paddingBottom: 14 }}
                  itemSorter={(item) => spec.series.findIndex(([key]) => key === item.dataKey)}
                  formatter={(v) => <span style={{ color: CHROME.ink, fontSize: 11 }}>{v}</span>}
                />
              )}
              {spec.series.map(([key, label], i) => (
                <Line
                  key={key}
                  dataKey={key}
                  name={label}
                  stroke={SERIES[i]}
                  strokeWidth={2}
                  dot={false}
                  activeDot={{ r: 4, stroke: CHROME.surface, strokeWidth: 2 }}
                  isAnimationActive={false}
                />
              ))}
            </LineChart>
          </ResponsiveContainer>
        </div>
      )}
    </figure>
  )
}

function Tile({ label, value }: { label: string; value: string }) {
  return (
    <div className="rounded-md border border-edge bg-panel px-3 py-2">
      <div className="text-[11px] text-muted">{label}</div>
      <div className="text-2xl font-semibold">{value}</div>
    </div>
  )
}

export default function Dashboard({ api }: { api: string }) {
  const [metrics, setMetrics] = useState<Metrics | null>(null)

  // Polled while open. A failed fetch keeps the last good render rather than
  // blanking the page.
  useEffect(() => {
    let cancelled = false
    const load = () =>
      fetch(`${api}/metrics`)
        .then((r) => r.json())
        .then((m: Metrics) => {
          if (!cancelled) setMetrics(m)
        })
        .catch(() => {})
    load()
    const timer = setInterval(load, 5000)
    return () => {
      cancelled = true
      clearInterval(timer)
    }
  }, [api])

  if (!metrics) return <div className="p-6 text-muted">Loading metrics…</div>
  const rows = derive(metrics.days)
  const now = metrics.now
  const n = (k: string) => now[k] ?? 0
  // Rates come from the last morning: mid-day, a shift under way has been
  // called but not yet worked, so attendance would dip until it finishes.
  const morning = metrics.days[metrics.days.length - 1]

  return (
    <div className="p-5">
      {/* A replay looks exactly like a live city, so say which it is. */}
      {metrics.replay && (
        <div className="mb-3 rounded-md border border-edge bg-panel px-3 py-2 text-xs text-muted">
          Replaying <span className="text-[#e8e8f0]">{metrics.replay.file}</span> — no model calls;
          every answer comes from the recording.{' '}
          {metrics.replay.mismatch
            ? `It stopped matching the recording at ${metrics.replay.mismatch}.`
            : `${metrics.replay.mornings} mornings match${metrics.replay.finished ? ', and the recording has ended.' : ' so far.'}`}
        </div>
      )}
      <div className="mb-4 grid grid-cols-[repeat(auto-fill,minmax(9rem,1fr))] gap-3">
        <Tile label="Employed" value={`${n('employed')} of ${n('employed') + n('students') + n('looking')}`} />
        <Tile
          label="Attendance"
          value={morning?.['attendance'] == null ? '—' : FORMAT.percent(morning['attendance'])}
        />
        <Tile label="Hired so far" value={FORMAT.count(n('hires'))} />
        <Tile label="Promoted so far" value={FORMAT.count(n('promotions'))} />
        <Tile label="Reflections so far" value={FORMAT.count(n('reflections'))} />
      </div>

      {rows.length < 2 ? (
        <div className="text-muted">The charts start at 06:00 on day 1, the first morning after startup.</div>
      ) : (
        <div className="grid grid-cols-[repeat(auto-fill,minmax(20rem,1fr))] gap-3">
          {CHARTS.map((spec) => (
            <Chart key={spec.title} spec={spec} data={rows} />
          ))}
        </div>
      )}

      {/* The table twin: every value, reachable without colour or hovering. */}
      <details className="mt-4 rounded-md border border-edge bg-panel p-3 text-xs">
        <summary className="cursor-pointer text-muted">Every morning as a table</summary>
        <div className="mt-2 overflow-x-auto">
          <table className="tabular-nums">
            <thead>
              <tr>
                {Object.keys(metrics.days[0] ?? {}).map((k) => (
                  <th key={k} className="px-2 py-1 text-left font-normal whitespace-nowrap text-muted">{k}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {metrics.days.map((r) => (
                <tr key={r.day} className="border-t border-edge">
                  {Object.keys(metrics.days[0]).map((k) => (
                    <td key={k} className="px-2 py-1 text-right">
                      {r[k] === null ? '—' : Number(r[k]).toLocaleString(undefined, { maximumFractionDigits: 3 })}
                    </td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </details>
    </div>
  )
}
