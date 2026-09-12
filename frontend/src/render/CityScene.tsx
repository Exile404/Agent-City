/**
 * Three.js scene for the city — emissive/neon rather than physically lit.
 *
 * Nothing here is sun-lit. Buildings and agents emit their own colour and bloom
 * does the rest, which is why the city reads at night. Buildings are drawn
 * translucent with depthWrite off so the agents inside them stay visible.
 */

import { Grid, Html, OrbitControls } from '@react-three/drei'
import { Canvas, useFrame } from '@react-three/fiber'
import { Bloom, EffectComposer } from '@react-three/postprocessing'
import { memo, useEffect, useMemo, useRef, useSyncExternalStore } from 'react'
import * as THREE from 'three'
import type { Building, CityStore } from '../state/socket'
import { ACTION_COLORS } from './colors'

/** Building height in world units. One tile = one unit. */
const HEIGHTS: Record<string, number> = {
  university: 9,
  power: 8,
  office: 7,
  hospital: 7,
  gas: 5,
  bank: 6,
  gym: 4.5,
  library: 5,
  home: 4,
  market: 4,
  cafe: 3,
  park: 0.15,
}

const BUILDING_COLORS: Record<string, string> = {
  university: '#3b82f6',
  office: '#f0a020',
  cafe: '#ec4899',
  park: '#22c55e',
  gym: '#a855f7',
  home: '#06b6d4',
  power: '#facc15',
  gas: '#f97316',
  hospital: '#f43f5e',
  market: '#84cc16',
  library: '#38bdf8',
  bank: '#e2e8f0',
}

const dummy = new THREE.Object3D()
const scratch = new THREE.Color()

/**
 * Occupants per building, recomputed every frame by <Agents> and read by
 * <BuildingMesh>. Module-level on purpose: this is renderer scratch state, not
 * React state, and threading a ref through props to mutate it is precisely what
 * react-hooks/immutability exists to stop.
 */
const occupancy = { counts: new Float32Array(0) }

/**
 * Last facing angle per agent. Kept between frames so someone standing still
 * keeps facing the way they arrived rather than snapping back to north.
 */
const headings = { yaw: new Float32Array(0) }

/** Lit windows, tiled across every facade. Built once and shared. */
const WINDOW_TEXTURE = (() => {
  const canvas = document.createElement('canvas')
  canvas.width = 32
  canvas.height = 32
  const ctx = canvas.getContext('2d')!
  ctx.fillStyle = '#000'
  ctx.fillRect(0, 0, 32, 32)
  ctx.fillStyle = '#fff'
  for (let y = 3; y < 30; y += 7) {
    for (let x = 3; x < 30; x += 7) {
      if (Math.random() > 0.3) ctx.fillRect(x, y, 3, 4)
    }
  }
  const tex = new THREE.CanvasTexture(canvas)
  tex.wrapS = THREE.RepeatWrapping
  tex.wrapT = THREE.RepeatWrapping
  return tex
})()

function BuildingMesh({
  b,
  index,
  ox,
  oz,
}: {
  b: Building
  index: number
  ox: number
  oz: number
}) {
  const matRef = useRef<THREE.MeshStandardMaterial>(null)
  const h = HEIGHTS[b.kind] ?? 4
  const color = BUILDING_COLORS[b.kind] ?? '#64748b'
  const isHome = b.kind === 'home'
  const w = b.w - 0.4
  const d = b.h - 0.4

  const geometry = useMemo(() => new THREE.BoxGeometry(w, h, d), [w, h, d])
  const edges = useMemo(() => new THREE.EdgesGeometry(geometry), [geometry])
  useEffect(() => () => {
    geometry.dispose()
    edges.dispose()
  }, [geometry, edges])

  // Occupancy drives the glow: a building with people inside lights its
  // windows. The simulation state *is* the visual effect.
  useFrame(() => {
    if (!matRef.current) return
    const people = occupancy.counts[index] ?? 0
    const target = 0.15 + Math.min(1, people / 5) * 1.5
    matRef.current.emissiveIntensity += (target - matRef.current.emissiveIntensity) * 0.08
  })

  return (
    <group position={[b.x + b.w / 2 + ox, h / 2, b.y + b.h / 2 + oz]}>
      <mesh geometry={geometry}>
        <meshStandardMaterial
          ref={matRef}
          color="#0a0d18"
          emissive={color}
          emissiveMap={WINDOW_TEXTURE}
          emissiveIntensity={0.2}
          transparent
          opacity={0.55}
          // Off so agents standing inside are not clipped by the near wall.
          depthWrite={false}
          roughness={0.4}
        />
      </mesh>

      <lineSegments geometry={edges}>
        <lineBasicMaterial color={color} toneMapped={false} />
      </lineSegments>

      {/* Every building is named, but homes are smaller and dimmer so ten of
          them read as background rather than competing with the landmarks. */}
      <Html
        position={[0, h / 2 + 1.4, 0]}
        center
        distanceFactor={isHome ? 58 : 42}
        style={{ pointerEvents: 'none' }}
      >
        <div
          style={{
            whiteSpace: 'nowrap',
            fontFamily: 'ui-monospace, monospace',
            fontSize: isHome ? 10 : 11,
            letterSpacing: '0.12em',
            textTransform: 'uppercase',
            opacity: isHome ? 0.5 : 1,
            color,
            background: 'rgba(6,8,18,0.75)',
            border: `1px solid ${color}`,
            borderRadius: 3,
            padding: '2px 7px',
          }}
        >
          {b.name}
        </div>
      </Html>
    </group>
  )
}

function Agents({
  store,
  lookup,
  onSelect,
  ox,
  oz,
}: {
  store: CityStore
  lookup: Int16Array | null
  onSelect: (index: number) => void
  ox: number
  oz: number
}) {
  const torsoRef = useRef<THREE.InstancedMesh>(null)
  const headRef = useRef<THREE.InstancedMesh>(null)
  const legLRef = useRef<THREE.InstancedMesh>(null)
  const legRRef = useRef<THREE.InstancedMesh>(null)
  const count = store.roster.length

  // Pre-translated so the capsule hangs *below* its origin. That puts the pivot
  // at the hip, which is what makes a rotation read as a stride rather than the
  // whole leg spinning about its middle.
  const legGeometry = useMemo(() => {
    const g = new THREE.CapsuleGeometry(0.065, 0.2, 3, 6)
    g.translate(0, -0.165, 0)
    return g
  }, [])
  useEffect(() => () => legGeometry.dispose(), [legGeometry])

  useFrame(() => {
    const torso = torsoRef.current
    const head = headRef.current
    const legL = legLRef.current
    const legR = legRRef.current
    const world = store.world
    if (!torso || !head || !legL || !legR || !world) return

    if (headings.yaw.length !== count) headings.yaw = new Float32Array(count)
    occupancy.counts.fill(0)

    // Interpolate between the last two ticks so agents walk rather than jump.
    const t = Math.min(1, (performance.now() - store.tickAt) / store.tickMs)
    const clock = performance.now() / 95

    for (let i = 0; i < count; i++) {
      const px = store.prev[i * 2]
      const pz = store.prev[i * 2 + 1]
      const cx = store.curr[i * 2]
      const cz = store.curr[i * 2 + 1]
      const dx = cx - px
      const dz = cz - pz

      const x = px + dx * t + 0.5 + ox
      const z = pz + dz * t + 0.5 + oz

      // Face the direction of travel, and hold that heading once stopped.
      const moving = dx !== 0 || dz !== 0
      if (moving) headings.yaw[i] = Math.atan2(dx, dz)
      const yaw = headings.yaw[i]

      // One phase drives everything: legs swing in opposition, and the torso
      // bobs at double rate — the body rises once per step, not once per cycle.
      const phase = clock + i
      const stride = moving ? Math.sin(phase) * 0.65 : 0
      const bob = moving ? Math.abs(Math.sin(phase)) * 0.045 : 0

      scratch.set(ACTION_COLORS[store.actions[i]] ?? '#ffffff')

      dummy.rotation.set(0, yaw, 0)
      dummy.position.set(x, 0.66 + bob, z)
      dummy.updateMatrix()
      torso.setMatrixAt(i, dummy.matrix)
      torso.setColorAt(i, scratch)

      dummy.position.set(x, 1.12 + bob, z)
      dummy.updateMatrix()
      head.setMatrixAt(i, dummy.matrix)
      head.setColorAt(i, scratch)

      // Hips sit either side of the body along its local right vector, which
      // for yaw = atan2(dx, dz) is (cos yaw, 0, -sin yaw).
      const hipX = Math.cos(yaw) * 0.1
      const hipZ = -Math.sin(yaw) * 0.1
      const hipY = 0.36 + bob

      // YXZ order: yaw the figure first, then swing the leg within that frame.
      dummy.rotation.set(stride, yaw, 0, 'YXZ')
      dummy.position.set(x - hipX, hipY, z - hipZ)
      dummy.updateMatrix()
      legL.setMatrixAt(i, dummy.matrix)
      legL.setColorAt(i, scratch)

      dummy.rotation.set(-stride, yaw, 0, 'YXZ')
      dummy.position.set(x + hipX, hipY, z + hipZ)
      dummy.updateMatrix()
      legR.setMatrixAt(i, dummy.matrix)
      legR.setColorAt(i, scratch)

      // Tally who is inside which building, for the window glow.
      if (lookup) {
        const b = lookup[cz * world.width + cx]
        if (b >= 0) occupancy.counts[b] += 1
      }
    }

    dummy.rotation.order = 'XYZ'

    for (const mesh of [torso, head, legL, legR]) {
      mesh.instanceMatrix.needsUpdate = true
      if (mesh.instanceColor) mesh.instanceColor.needsUpdate = true
    }
  })

  if (count === 0) return null
  return (
    <group>
      {/* Clicks land on the torso — the biggest target. */}
      <instancedMesh
        ref={torsoRef}
        args={[undefined, undefined, count]}
        onClick={(e) => {
          e.stopPropagation()
          if (e.instanceId != null) onSelect(e.instanceId)
        }}
      >
        <capsuleGeometry args={[0.18, 0.28, 4, 8]} />
        {/* Basic + toneMapped off: colours stay at full brightness so bloom
            catches them, instead of being dimmed by the lighting model. */}
        <meshBasicMaterial toneMapped={false} />
      </instancedMesh>

      <instancedMesh ref={headRef} args={[undefined, undefined, count]}>
        <sphereGeometry args={[0.17, 10, 10]} />
        <meshBasicMaterial toneMapped={false} />
      </instancedMesh>

      <instancedMesh ref={legLRef} args={[undefined, undefined, count]} geometry={legGeometry}>
        <meshBasicMaterial toneMapped={false} />
      </instancedMesh>

      <instancedMesh ref={legRRef} args={[undefined, undefined, count]} geometry={legGeometry}>
        <meshBasicMaterial toneMapped={false} />
      </instancedMesh>
    </group>
  )
}

// No offset args: the plane is width x height centred on the origin, which is
// already where the city sits.
function Ground({ store }: { store: CityStore }) {
  const { world, tiles } = store

  // The tile grid as one small image, nearest-filtered. Roads glow faintly;
  // open lots stay near black so the buildings carry the image.
  const texture = useMemo(() => {
    if (!world || !tiles) return null
    const canvas = document.createElement('canvas')
    canvas.width = world.width
    canvas.height = world.height
    const ctx = canvas.getContext('2d')!
    const img = ctx.createImageData(world.width, world.height)
    for (let i = 0; i < tiles.length; i++) {
      const road = tiles[i] === 1
      img.data[i * 4] = road ? 34 : 8
      img.data[i * 4 + 1] = road ? 52 : 11
      img.data[i * 4 + 2] = road ? 92 : 22
      img.data[i * 4 + 3] = 255
    }
    ctx.putImageData(img, 0, 0)
    const tex = new THREE.CanvasTexture(canvas)
    tex.magFilter = THREE.NearestFilter
    tex.colorSpace = THREE.SRGBColorSpace
    return tex
  }, [world, tiles])

  if (!world || !texture) return null
  return (
    <mesh rotation={[-Math.PI / 2, 0, 0]} position={[0, 0, 0]}>
      <planeGeometry args={[world.width, world.height]} />
      <meshBasicMaterial map={texture} toneMapped={false} />
    </mesh>
  )
}

const Scene = memo(function Scene({
  store,
  onSelect,
  buildings,
}: {
  store: CityStore
  onSelect: (index: number) => void
  /** Passed rather than read off `store`: see CityScene below. */
  buildings: Building[]
}) {
  const { world, roster } = store
  const ox = world ? -world.width / 2 : 0
  const oz = world ? -world.height / 2 : 0

  useEffect(() => {
    occupancy.counts = new Float32Array(buildings.length)
  }, [buildings.length])

  // Tile -> building index, so occupancy is one array read per agent per frame
  // instead of testing fifty agents against eighteen rectangles.
  const lookup = useMemo(() => {
    if (!world || buildings.length === 0) return null
    const arr = new Int16Array(world.width * world.height).fill(-1)
    buildings.forEach((b, i) => {
      for (let y = b.y; y < b.y + b.h; y++) {
        for (let x = b.x; x < b.x + b.w; x++) arr[y * world.width + x] = i
      }
    })
    return arr
  }, [world, buildings])

  return (
    <>
      <color attach="background" args={['#05060d']} />
      <fog attach="fog" args={['#05060d', 70, 210]} />
      {/* Emissive materials carry the image; this only keeps faces from being
          pure black where the window map is dark. */}
      <ambientLight intensity={0.7} />

      <Grid
        position={[0, -0.03, 0]}
        args={[240, 240]}
        cellSize={1}
        cellThickness={0.4}
        cellColor="#16204a"
        sectionSize={10}
        sectionThickness={1.1}
        sectionColor="#7c3aed"
        fadeDistance={170}
        fadeStrength={1.4}
        infiniteGrid
      />

      <Ground store={store} />

      {buildings.map((b, i) => (
        <BuildingMesh key={b.id} b={b} index={i} ox={ox} oz={oz} />
      ))}

      {roster.length > 0 && (
        <Agents store={store} lookup={lookup} onSelect={onSelect} ox={ox} oz={oz} />
      )}

      <OrbitControls
        target={[0, 0, 0]}
        maxPolarAngle={Math.PI / 2.2}
        minDistance={12}
        maxDistance={170}
        enableDamping
      />

      <EffectComposer>
        <Bloom luminanceThreshold={0.22} luminanceSmoothing={0.9} intensity={1.15} mipmapBlur />
      </EffectComposer>
    </>
  )
})

export default function CityScene({
  store,
  onSelect,
}: {
  store: CityStore
  onSelect: (index: number) => void
}) {
  // Scene is memoised, and both `store` (a module singleton) and `onSelect` (a
  // useCallback with no deps) are permanently stable references — so without a
  // prop that actually changes, Scene renders once against an empty world and
  // never again. Subscribing to the buildings array rather than the version
  // counter keeps that to a single re-render when `hello` lands instead of one
  // per tick: the canvas still reads positions straight from the store at 60fps
  // without passing through React.
  const buildings = useSyncExternalStore(store.subscribe, () => store.buildings)

  return (
    <Canvas camera={{ position: [0, 34, 44], fov: 45 }}>
      <Scene store={store} onSelect={onSelect} buildings={buildings} />
    </Canvas>
  )
}
