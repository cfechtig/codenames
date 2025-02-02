import { useRef, useState } from "react";
import { useFrame } from "@react-three/fiber";
import { Text } from "@react-three/drei";

export function Card({ text, position, highlighted = false }) {
  const groupRef = useRef(null);
  const [hovered, setHovered] = useState(false);

  useFrame(() => {
    if (groupRef.current) {
      groupRef.current.rotation.x = hovered ? -0.2 : -0.1;
      groupRef.current.position.y = hovered ? 0.5 : 0;
    }
  });

  return (
    <group
      ref={groupRef}
      position={position}
      onPointerEnter={() => setHovered(true)}
      onPointerLeave={() => setHovered(false)}
    >
      <mesh castShadow receiveShadow position={[0, 0, 0]} rotation={[0, 0, 0]}>
        <boxGeometry args={[6, 0, 4]} />
        <meshStandardMaterial color="#f5f5dc" />
      </mesh>

      {/* Border (if highlighted) */}
      {highlighted && (
        <mesh position={[0, 0.05, 0]} rotation={[-0.1, 0, 0]}>
          <boxGeometry args={[3.2, 0.11, 2.2]} />
          <meshStandardMaterial color="#4CAF50" />
        </mesh>
      )}

      {/* Text */}
      <Text
        position={[0, 1, 0]}
        rotation={[-1.8, 0, 0]}
        fontSize={0.8}
        color="black"
        // font="/fonts/Inter-Bold.ttf"
        anchorX="center"
        anchorY="middle"
      >
        {text}
      </Text>
    </group>
  );
}
