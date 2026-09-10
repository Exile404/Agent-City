/**
 * WebSocket client and snapshot store.
 *
 * Per-tick data intentionally does not live in React state: fifty agents at one
 * frame a second would re-render the tree for data only the canvas consumes.
 * Positions land in plain typed arrays the renderer reads directly at 60fps;
 * React subscribes purely for the cheap HUD fields.
 */

export type WorldMeta = {
  width: number
  height: number
  minutesPerTick: number
  tilesPerTick: number
  /** Road pitch and carriageway width, for placing traffic lights. */
  block: number
  roadWidth: number
}

export type Building = {
  id: string
  kind: string
  name: string
  x: number
  y: number
  w: number
  h: number
  door: [number, number]
}

export type AgentIdentity = {
  id: string
  name: string
  age: number
  traits: string[]
}

export type Status = 'connecting' | 'open' | 'closed'

function base64ToBytes(b64: string): Uint8Array {
  const bin = atob(b64)
  const out = new Uint8Array(bin.length)
  for (let i = 0; i < bin.length; i++) out[i] = bin.charCodeAt(i)
  return out
}

type HelloMessage = {
  type: 'hello'
  world: WorldMeta
  tiles: string
  buildings: Building[]
  agents: AgentIdentity[]
}

type TickMessage = {
  type: 'tick'
  t: number
  clock: string
  minuteOfDay: number
  /** [x, y, action], index-aligned with the hello roster. */
  agents: [number, number, string][]
  events: string[]
}

/** The `type` field discriminates the union, so each branch narrows. */
type ServerMessage = HelloMessage | TickMessage

export class CityStore {
  status: Status = 'connecting'
  world: WorldMeta | null = null
  tiles: Uint8Array | null = null
  buildings: Building[] = []
  roster: AgentIdentity[] = []

  /**
   * Positions at the previous tick and at the current one, as x,y pairs. The
   * renderer interpolates between them so agents glide rather than jumping
   * twelve tiles once a second.
   */
  prev = new Int16Array(0)
  curr = new Int16Array(0)
  actions: string[] = []

  tick = 0
  clock = ''
  /** Drives the sun's angle and colour. 480 = 08:00, a sane pre-connect value. */
  minuteOfDay = 480
  events: string[] = []

  /**
   * performance.now() when the current tick landed, and the measured gap to the
   * one before. Measured rather than assumed, so changing sim speed does not
   * desynchronise interpolation.
   */
  tickAt = 0
  tickMs = 1000

  /** Bumped on every change; useSyncExternalStore reads it as the snapshot. */
  version = 0

  private ws: WebSocket | null = null
  private listeners = new Set<() => void>()
  private retry = 0
  private stopped = false

  private readonly url: string

  constructor(url: string) {
    this.url = url
  }

  connect(): void {
    this.stopped = false
    this.status = 'connecting'
    this.emit()

    const ws = new WebSocket(this.url)
    this.ws = ws

    ws.onopen = () => {
      this.status = 'open'
      this.retry = 0
      // The server blocks on receive_text(); one message keeps its read side
      // alive and gives it a prompt disconnect signal when this tab closes.
      ws.send('hello')
      this.emit()
    }

    ws.onmessage = (ev) => this.onMessage(JSON.parse(ev.data) as ServerMessage)
    ws.onerror = () => ws.close()

    ws.onclose = () => {
      this.status = 'closed'
      this.emit()
      if (this.stopped) return
      // uvicorn --reload restarts constantly in dev. Back off to a 2s ceiling
      // instead of hammering it.
      const delay = Math.min(2000, 250 * 2 ** this.retry++)
      setTimeout(() => this.connect(), delay)
    }
  }

  disconnect(): void {
    this.stopped = true
    this.ws?.close()
  }

  subscribe = (fn: () => void): (() => void) => {
    this.listeners.add(fn)
    return () => {
      this.listeners.delete(fn)
    }
  }

  getSnapshot = (): number => this.version

  private emit(): void {
    this.version++
    for (const fn of this.listeners) fn()
  }

  private onMessage(msg: ServerMessage): void {
    if (msg.type === 'hello') {
      this.world = msg.world
      this.tiles = base64ToBytes(msg.tiles)
      this.buildings = msg.buildings
      this.roster = msg.agents

      const n = msg.agents.length
      this.prev = new Int16Array(n * 2)
      this.curr = new Int16Array(n * 2)
      this.actions = new Array(n).fill('idle')
      this.tickAt = 0
      this.emit()
      return
    }

    if (msg.type !== 'tick') return

    const now = performance.now()
    const first = this.tickAt === 0
    if (!first) this.tickMs = Math.max(16, now - this.tickAt)
    this.tickAt = now

    this.prev.set(this.curr)

    const rows = msg.agents
    for (let i = 0; i < rows.length; i++) {
      this.curr[i * 2] = rows[i][0]
      this.curr[i * 2 + 1] = rows[i][1]
      this.actions[i] = rows[i][2]
    }

    // On the very first frame `prev` is all zeros, so agents would fly in from
    // the top-left corner. Collapse the gap instead.
    if (first) this.prev.set(this.curr)

    this.tick = msg.t
    this.clock = msg.clock
    this.minuteOfDay = msg.minuteOfDay
    if (msg.events.length) {
      this.events = [...msg.events, ...this.events].slice(0, 80)
    }
    this.emit()
  }
}

const WS_URL =
  (import.meta.env.VITE_WS_URL as string | undefined) ?? 'ws://127.0.0.1:8000/ws'

export const cityStore = new CityStore(WS_URL)