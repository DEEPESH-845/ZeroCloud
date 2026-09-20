"use client";
import { Canvas, useFrame, useThree } from "@react-three/fiber";
import { ContactShadows, Edges, Html, Line, type EdgesRef } from "@react-three/drei";
import { useEffect, useMemo, useRef, type RefObject } from "react";
import * as THREE from "three";
import { BUDGET, lengthsAt, STAGES } from "./stages";
import s from "./Budget.module.css";

export type Progress = { v: number };
export type Pointer = { x: number; y: number };
type Props = {
  progress: Progress; // 0..1, written by the section (GSAP), read here per frame
  pointer: Pointer; // -1..1, written by the section
  active: boolean; // section in view
  reduced: boolean; // prefers-reduced-motion
  invalidateRef: RefObject<(() => void) | null>;
};

// 1 world unit = 1 GiB. The bar is the budget; blocks sit inside it.
const X0 = -BUDGET / 2;
const BAR_H = 1.15, BAR_D = 1.15, BLOCK = 0.82;
const INK = "#13161B", OK = "#2E8B57", AMBER = "#B7791A", KV_COLOR = "#8FA3B8";

// Faint lines sweeping along the weights: decode reads the active weights
// once per token, so the surface reads left to right. ≤0.12 contrast.
const sweep = new THREE.ShaderMaterial({
  transparent: true,
  depthWrite: false,
  uniforms: { uTime: { value: 0 }, uLen: { value: 1 } },
  vertexShader: `varying vec2 vUv; void main(){ vUv = uv; gl_Position = projectionMatrix * modelViewMatrix * vec4(position,1.0); }`,
  fragmentShader: `uniform float uTime; uniform float uLen; varying vec2 vUv;
    void main(){ float x = fract(vUv.x * uLen * 1.5 - uTime * 0.9);
      float l = smoothstep(0.0, 0.5, x) * (1.0 - smoothstep(0.5, 1.0, x));
      gl_FragColor = vec4(1.0, 1.0, 1.0, l * 0.12); }`,
});

function Fit() {
  const { camera, size } = useThree();
  useEffect(() => {
    const c = camera as THREE.OrthographicCamera;
    // fit ~16.5 GiB across the box width, but never taller than the box
    c.zoom = Math.min(size.width / 13.6, size.height / 4.6);
    c.updateProjectionMatrix();
  }, [camera, size]);
  return null;
}

function Instrument({ progress, pointer, active, reduced, invalidateRef }: Props) {
  const invalidate = useThree((st) => st.invalidate);
  const compact = useThree((st) => st.size.width < 480);
  useEffect(() => {
    invalidateRef.current = invalidate;
    return () => { invalidateRef.current = null; };
  }, [invalidate, invalidateRef]);

  const group = useRef<THREE.Group>(null);
  const weights = useRef<THREE.Mesh>(null);
  const kv = useRef<THREE.Mesh>(null);
  const sweepMesh = useRef<THREE.Mesh>(null);
  const edges = useRef<EdgesRef>(null);
  const wLabel = useRef<THREE.Group>(null);
  const kLabel = useRef<THREE.Group>(null);
  const wDiv = useRef<HTMLDivElement>(null);
  const kDiv = useRef<HTMLDivElement>(null);
  const overDiv = useRef<HTMLDivElement>(null);
  const overMark = useRef<THREE.Mesh>(null);
  const cur = useRef({ w: 0, kv: 0, over: 0, rx: 0, ry: 0 });
  const edgeColor = useMemo(() => ({ ok: new THREE.Color(OK), amber: new THREE.Color(AMBER), ink: new THREE.Color(INK), tmp: new THREE.Color() }), []);

  const ticks = useMemo(() => {
    const pts: [number, number, number][] = [];
    for (let i = 0; i <= BUDGET; i++) {
      const len = i % 4 === 0 ? BAR_D : BAR_D * 0.45;
      pts.push([X0 + i, BAR_H / 2 + 0.002, BAR_D / 2], [X0 + i, BAR_H / 2 + 0.002, BAR_D / 2 - len]);
    }
    return pts;
  }, []);

  useFrame((_, dt) => {
    const c = cur.current;
    const t = lengthsAt(progress.v);
    const k = reduced ? 1 : 1 - Math.exp(-dt * 9);
    c.w += (t.w - c.w) * k;
    c.kv += (t.kv - c.kv) * k;
    // over-budget emphasis: ramps in as the last stage forms
    const overT = THREE.MathUtils.smoothstep(progress.v, 0.78, 0.86);
    c.over += (overT - c.over) * k;
    c.rx += ((reduced ? 0 : pointer.y) * 0.02 - c.rx) * k;
    c.ry += ((reduced ? 0 : pointer.x) * 0.035 - c.ry) * k;

    const w = Math.max(c.w, 1e-4), kvl = Math.max(c.kv, 1e-4);
    if (weights.current) { weights.current.scale.x = w; weights.current.position.x = X0 + w / 2; }
    if (sweepMesh.current) { sweepMesh.current.scale.x = w; sweepMesh.current.position.x = X0 + w / 2; sweep.uniforms.uLen.value = w; }
    if (kv.current) { kv.current.scale.x = kvl; kv.current.position.x = X0 + w + kvl / 2; }
    if (wLabel.current) wLabel.current.position.x = X0 + w / 2;
    if (kLabel.current) kLabel.current.position.x = X0 + w + kvl / 2;
    // compact (phones): the readout below carries every number; labels off
    if (wDiv.current) {
      wDiv.current.style.opacity = String(compact ? 0 : Math.min(1, c.w / 1.5));
      const txt = `weights ${c.w.toFixed(2)} GiB`;
      if (wDiv.current.textContent !== txt) wDiv.current.textContent = txt;
    }
    if (kDiv.current) {
      kDiv.current.style.opacity = String(compact ? 0 : Math.min(1, c.kv / 1.5));
      const txt = `KV ${c.kv.toFixed(2)} GiB`;
      if (kDiv.current.textContent !== txt) kDiv.current.textContent = txt;
    }
    if (overDiv.current) overDiv.current.style.opacity = String(compact ? 0 : c.over);
    if (overMark.current) overMark.current.scale.setScalar(Math.max(c.over, 1e-4));
    if (edges.current) {
      const loaded = Math.min(1, c.w / 1.5);
      edgeColor.tmp.copy(edgeColor.ink).lerp(edgeColor.ok, loaded).lerp(edgeColor.amber, c.over);
      edges.current.material.color.copy(edgeColor.tmp);
    }
    if (group.current) { group.current.rotation.x = c.rx; group.current.rotation.y = c.ry; }
    if (!reduced && active) sweep.uniforms.uTime.value += dt;

    const settled = Math.abs(t.w - c.w) + Math.abs(t.kv - c.kv) + Math.abs(overT - c.over) < 1e-3
      && Math.abs(pointer.y * 0.02 - c.rx) + Math.abs(pointer.x * 0.035 - c.ry) < 1e-4;
    if (!settled) invalidate();
  });

  const last = STAGES[STAGES.length - 1];
  return (
    <group ref={group}>
      {/* the budget: a translucent volume with etched graduations */}
      <mesh position={[0, 0, 0]}>
        <boxGeometry args={[BUDGET, BAR_H, BAR_D]} />
        <meshStandardMaterial color="#ffffff" transparent opacity={0.16} roughness={0.9} depthWrite={false} />
        <Edges ref={edges} lineWidth={1.25} color={INK} />
      </mesh>
      <Line points={ticks} segments color="#4B525C" lineWidth={1} transparent opacity={0.8} />
      {/* end labels */}
      <Html position={[X0, -BAR_H / 2 - 0.1, BAR_D / 2]} transform={false} zIndexRange={[10, 0]} pointerEvents="none" wrapperClass={s.labelEnd}>
        <div className={s.labelEnd} style={{ transform: "translate(-50%,0)" }}>0</div>
      </Html>
      <Html position={[X0 + BUDGET, -BAR_H / 2 - 0.1, BAR_D / 2]} transform={false} zIndexRange={[10, 0]} pointerEvents="none">
        <div className={s.labelEnd} style={{ transform: "translate(-100%,0)" }}>12.80 GiB · budget</div>
      </Html>

      {/* weights block */}
      <mesh ref={weights} position={[X0, 0, 0]} castShadow>
        <boxGeometry args={[1, BLOCK, BLOCK]} />
        <meshStandardMaterial color={INK} roughness={0.35} metalness={0.5} />
      </mesh>
      <mesh ref={sweepMesh} position={[X0, BLOCK / 2 + 0.002, 0]} rotation={[-Math.PI / 2, 0, 0]} material={sweep}>
        <planeGeometry args={[1, BLOCK]} />
      </mesh>
      {/* KV block */}
      <mesh ref={kv} position={[X0, 0, 0]}>
        <boxGeometry args={[1, BLOCK, BLOCK]} />
        <meshStandardMaterial color={KV_COLOR} roughness={0.6} metalness={0.1} />
      </mesh>
      {/* over-budget marker at the bar's end face */}
      <mesh ref={overMark} position={[X0 + BUDGET, 0, 0]}>
        <boxGeometry args={[0.06, BAR_H + 0.2, BAR_D + 0.2]} />
        <meshBasicMaterial color={AMBER} />
      </mesh>

      {/* floating labels tracking the blocks */}
      <group ref={wLabel} position={[X0, BAR_H / 2 + 0.32, 0]}>
        <Html transform={false} zIndexRange={[10, 0]} pointerEvents="none">
          <div ref={wDiv} className={s.labelBlock} style={{ opacity: 0 }}>weights</div>
        </Html>
      </group>
      <group ref={kLabel} position={[X0, BAR_H / 2 + 0.32, 0]}>
        <Html transform={false} zIndexRange={[10, 0]} pointerEvents="none">
          <div ref={kDiv} className={s.labelBlock} style={{ opacity: 0 }}>KV</div>
        </Html>
      </group>
      <Html position={[X0 + BUDGET + 0.05, BAR_H / 2 + 0.8, BAR_D / 2]} transform={false} zIndexRange={[10, 0]} pointerEvents="none">
        <div ref={overDiv} className={s.labelOver} style={{ opacity: 0 }}>{last.verdict}</div>
      </Html>

      <ContactShadows position={[0, -BAR_H / 2 - 0.02, 0]} opacity={0.3} scale={[22, 7]} blur={2.4} far={2.5} resolution={256} color="#13161B" />
    </group>
  );
}

export default function BudgetScene(props: Props) {
  return (
    <Canvas
      orthographic
      camera={{ position: [2.6, 4.6, 10], zoom: 40, near: 0.1, far: 100 }}
      dpr={[1, 1.5]}
      gl={{ alpha: true, antialias: true }}
      frameloop={props.active && !props.reduced ? "always" : "demand"}
      onCreated={({ camera }) => camera.lookAt(0, 0, 0)}
      style={{ position: "absolute", inset: 0 }}
    >
      <Fit />
      <hemisphereLight args={["#ffffff", "#9aa3ad", 0.9]} />
      <directionalLight position={[4, 8, 6]} intensity={1.4} />
      <directionalLight position={[-6, 3, -4]} intensity={0.35} />
      <Instrument {...props} />
    </Canvas>
  );
}
