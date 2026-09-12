import {
  useCallback,
  useEffect,
  useState,
  useSyncExternalStore,
  type ReactNode,
} from 'react'
import { ACTION_COLORS } from './render/colors'
import CityScene from './render/CityScene'
import { cityStore } from './state/socket'

const API = 'http://127.0.0.1:8000'
const SPEEDS = [0, 0.5, 1, 2, 4, 8]

const BTN =
  'flex-1 cursor-pointer rounded border border-edge bg-chip py-1.5 text-xs hover:bg-[#2b2b38]'

type AgentDetail = {
  id: string
  name: string
  age: number
  traits: string[]
  home: string
  pos: [number, number]
  action: string
  needs: Record<string, number>
  skills: Record<string, number>
  money: number
  relationships: {
    name: string
    affinity: number
    label: string
    timesMet: number
    note: string
  }[]
}

function Panel({ children, className = '' }: { children: ReactNode; className?: string }) {
  return <section className={`border-b border-edge px-3.5 py-3 ${className}`}>{children}</section>
}

function Title({ children }: { children: ReactNode }) {
  return (
    <div className="mt-3.5 mb-1.5 flex items-center justify-between text-[11px] tracking-wider text-muted uppercase first:mt-0">
      {children}
    </div>
  )
}

function Bar({ label, value }: { label: string; value: number }) {
  // Red under 20 (the sim's critical threshold), amber under 50, else green.
  const hue = value < 20 ? 0 : value < 50 ? 38 : 145
  return (
    <div className="my-[3px] flex items-center gap-2">
      <span className="w-22 text-[11px] text-muted">{label}</span>
      <div className="h-1.5 flex-1 overflow-hidden rounded-full bg-chip">
        <div
          className="h-full transition-[width] duration-300"
          style={{ width: `${value}%`, background: `hsl(${hue} 62% 50%)` }}
        />
      </div>
      <span className="w-7 text-right text-[11px] tabular-nums text-muted">
        {Math.round(value)}
      </span>
    </div>
  )
}

function Relation({ r }: { r: AgentDetail['relationships'][number] }) {
  // Same hue vocabulary as Bar, but keyed to the sim's own label thresholds
  // rather than a 0-100 scale: affinity runs -100..100 and the boundaries that
  // matter are the ones Relationship.label uses.
  const hue = r.affinity < -25 ? 0 : r.affinity > 15 ? 145 : 38
  return (
    <div className="border-b border-[#1e1e28] py-1.5">
      <div className="flex items-baseline gap-2">
        <span className="flex-1 truncate text-xs">{r.name}</span>
        <span className="text-[11px] text-muted">{r.timesMet}×</span>
        <span
          className="w-9 text-right text-[11px] tabular-nums"
          style={{ color: `hsl(${hue} 62% 55%)` }}
        >
          {r.affinity > 0 ? '+' : ''}
          {r.affinity.toFixed(0)}
        </span>
      </div>
      <div className="text-[11px] text-muted">{r.label}</div>
      {r.note && (
        <div className="mt-0.5 truncate text-[11px] text-muted italic">“{r.note}”</div>
      )}
    </div>
  )
}

export default function App() {
  const [selected, setSelected] = useState<number | null>(null)
  const [detail, setDetail] = useState<AgentDetail | null>(null)
  const [speed, setSpeed] = useState(1)

  // One HUD re-render per tick. The canvas reads the store directly at 60fps
  // and never passes through React.
  useSyncExternalStore(cityStore.subscribe, cityStore.getSnapshot)

  // Clearing here rather than inside the effect below: this runs from an event
  // handler, so it triggers no cascading render. It also drops the previous
  // agent's panel immediately instead of showing stale data for up to a second.
  const select = useCallback((index: number | null) => {
    setSelected(index)
    setDetail(null)
  }, [])

  // R3F owns the canvas lifecycle now, so this only manages the socket.
  useEffect(() => {
    cityStore.connect()
    return () => cityStore.disconnect()
  }, [])

  // Inspector detail is pulled, not streamed: full needs, skills and money for
  // fifty agents every tick would be bandwidth for a panel usually closed.
  useEffect(() => {
    if (selected === null) return
    const id = cityStore.roster[selected]?.id
    if (!id) return

    let cancelled = false
    const load = () =>
      fetch(`${API}/agent/${id}`)
        .then((r) => r.json())
        .then((d) => {
          if (!cancelled) setDetail(d)
        })
        .catch(() => {})

    load()
    const timer = setInterval(load, 1000)
    return () => {
      cancelled = true
      clearInterval(timer)
    }
  }, [selected])

  const changeSpeed = useCallback((v: number) => {
    setSpeed(v)
    fetch(`${API}/control/speed?value=${v}`, { method: 'POST' }).catch(() => {})
  }, [])

  const { status, clock, events, tick } = cityStore
  const statusColor =
    status === 'open' ? 'bg-[#4ad78b]' : status === 'connecting' ? 'bg-[#d7a24a]' : 'bg-[#d76b6b]'

  return (
    <div className="flex h-full">
      <div className="relative min-w-0 flex-1">
        <CityScene store={cityStore} onSelect={select} />
      </div>

      <aside className="flex w-[340px] flex-none flex-col overflow-y-auto border-l border-edge bg-panel">
        <Panel>
          <div className="text-xl tabular-nums">{clock || '—'}</div>
          <div className="mt-0.5 flex items-center gap-1.5 text-xs text-muted">
            <span className={`inline-block size-1.5 rounded-full ${statusColor}`} />
            {status} · tick {tick}
          </div>
        </Panel>

        <Panel>
          <Title>Speed</Title>
          <div className="flex gap-1">
            {SPEEDS.map((s) => (
              <button
                key={s}
                onClick={() => changeSpeed(s)}
                className={`${BTN} ${s === speed ? 'border-[#4f60b0] bg-[#3d4a8c]' : ''}`}
              >
                {s === 0 ? '❚❚' : `${s}×`}
              </button>
            ))}
          </div>
        </Panel>

        {detail && (
          <Panel>
            <Title>
              {detail.name}
              <button
                className="cursor-pointer text-muted hover:text-white"
                onClick={() => select(null)}
              >
                ✕
              </button>
            </Title>
            <div className="text-xs text-muted">
              {detail.age} · {detail.traits.join(', ')}
            </div>
            <div className="text-xs text-muted">
              {detail.home} · ${detail.money.toFixed(0)}
            </div>
            <div
              className="mt-2 font-semibold"
              style={{ color: ACTION_COLORS[detail.action.split(' ')[0]] ?? '#ffffff' }}
            >
              {detail.action}
            </div>

            <Title>Needs</Title>
            {Object.entries(detail.needs).map(([k, v]) => (
              <Bar key={k} label={k} value={v} />
            ))}

            <Title>Skills</Title>
            {Object.entries(detail.skills).map(([k, v]) => (
              <Bar key={k} label={k} value={v} />
            ))}

            {/* Guarded on length: the inspector re-fetches every second, and an
                agent who has met nobody returns [], which would otherwise flash
                a heading over an empty list. */}
            {detail.relationships?.length ? (
              <>
                <Title>Knows</Title>
                {detail.relationships.map((r) => (
                  <Relation key={r.name} r={r} />
                ))}
              </>
            ) : null}
          </Panel>
        )}

        <Panel className="flex-1 border-b-0">
          <Title>Events</Title>
          {events.length === 0 && <div className="text-xs text-muted">nothing yet…</div>}
          {events.map((e, i) => (
            <div key={i} className="border-b border-[#1e1e28] py-1 text-xs leading-snug">
              {e}
            </div>
          ))}
        </Panel>
      </aside>
    </div>
  )
}