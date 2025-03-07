import { c } from "/src/lib/c";
import { useForm } from "react-hook-form";
import { useEffect, useState } from "react";
import { Canvas } from "@react-three/fiber";
import { Environment, OrbitControls } from "@react-three/drei";
import { Card } from "/src/components/Card";
import BgImg from "/src/assets/bg.png";
import LogoImg from "/src/assets/logo.png";

const wordsTMP = [
  { value: "BRICK", type: "A" },
  { value: "ANT", type: "B" },
  { value: "VAMPIRE", type: "F" },
  { value: "SKATES", type: "B" },
  { value: "CRAFT", type: "A" },
  { value: "RIFLE", type: "F" },
  { value: "VIRUS", type: "A" },
  { value: "IGLOO", type: "B" },
  { value: "RANCH", type: "A" },
  { value: "WOLF", type: "F" },
  { value: "DOLL", type: "A" },
  { value: "LUNCH", type: "B" },
  { value: "TATTOO", type: "F" },
  { value: "PEW", type: "B" },
  { value: "PINE", type: "A" },
];

function App() {
  const [board, setBoard] = useState({});
  const [turn, setTurn] = useState("HUMAN");
  const [hint, setHint] = useState("");
  const [hintsRemaining, setHintsRemaining] = useState(9);
  const [strikes, setStrikes] = useState(0);

  const { register, handleSubmit, setValue, resetField } = useForm();

  const onHint = async ({ hint }) => {
    setValue("turn", turn);
    const response = await fetch("/hint", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({ data: hint }),
    });
    const result = await response.json();
    console.log(result);
    setHintsRemaining(hintsRemaining - 1);
  };

  const onGuess = async (data) => {
    setValue("guess", data);
    const response = await fetch("/guess", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({ data }),
    });
    const result = await response.json();
    if (!result.correct) {
      setStrikes((value) => value + 1);
      setTurn("AI");
    }
    resetField("guess");
  };

  const init = async () => {
    const response = await fetch("/board");
    // const data = await response.json();
    // setBoard(data);
  };

  useEffect(() => {
    init();
  }, []);

  const words = Object.entries(board).flatMap(([type, words]) =>
    words.map((word) => ({ value: word, type }))
  );

  const onKeyDown = (e) => {
    if (e.key === "Enter") {
      handleSubmit(onHint)();
    }
  };

  if (hintsRemaining === 0 || strikes === 3) {
    return (
      <div>
        <h1>Game Over</h1>
        <p>Final Score: {hintsRemaining - strikes}</p>
      </div>
    );
  }

  return (
    <div className="flex flex-col items-center h-screen">
      <img src={BgImg} alt="bg" className="absolute h-screen w-screen -z-50" />
      <div className="flex flex-col items-center justify-center m-2">
        <img src={LogoImg} alt="logo" className="h-20" />
        <h2 className="text-white text-lg font-semibold">
          Current Turn <span className="text-purple-300">({turn})</span>
        </h2>
        <div className="flex justify-between w-full m-2">
          <div className="flex items-center gap-2 bg-[#2a1f3d] rounded-lg px-4 py-2">
            <span className="text-purple-300 text-sm">Hints Remaining:</span>
            <span className="text-white font-bold">{hintsRemaining}</span>
          </div>
          <div className="flex items-center gap-2 bg-[#2a1f3d] rounded-lg px-4 py-2">
            <span className="text-purple-300 text-sm">Strikes:</span>
            <span className="text-white font-bold">{strikes}</span>
          </div>
        </div>
        {hint && <p>Hint: {hint}</p>}
      </div>
      <Canvas camera={{ position: [0, 15, 20], fov: 90 }} className="h-screen">
        <ambientLight intensity={0.5} />
        <directionalLight
          position={[5, 5, 5]}
          castShadow
          shadow-mapSize-width={1024}
          shadow-mapSize-height={1024}
        />
        <group position={[-16, 0, 0]}>
          {wordsTMP.map((word, i) => (
            <Card
              key={word.value + i}
              text={word.value}
              position={[(i % 5) * 7, 0, Math.floor(i / 5) * 7]}
              type={word.type}
            />
          ))}
        </group>
        <Environment preset="studio" />
        <OrbitControls
          enableZoom={false}
          enableRotate={false}
          enablePan={false}
          maxPolarAngle={Math.PI / 2}
          minPolarAngle={Math.PI / 3}
        />
      </Canvas>
      <form onSubmit={handleSubmit(onHint)}>
        {/* <div className="grid grid-cols-5 gap-4">
          {Object.entries(board)
            .flatMap(([type, words]) =>
              words.map((word) => ({ value: word, type }))
            )
            .map((word) => (
              <div
                key={word.value}
                className={c(
                  turn === "AI" && "cursor-pointer",
                  turn === "HUMAN" && "cursor-not-allowed",
                  "p-4 flex justify-center items-center rounded-xl border-8",
                  word.type === "A" && "border-black",
                  word.type === "F" && "border-[#348E47]",
                  word.type === "B" && "border-[#D4C297]"
                )}
                onClick={() => turn === "AI" && onGuess(word.value)}
              >
                {word.value}
              </div>
            ))}
        </div> */}

        {turn === "HUMAN" && (
          <div className="flex items-center gap-2">
            <input
              className="border"
              {...register("hint")}
              onKeyDown={onKeyDown}
            />
            <input
              className={`
        relative px-6 py-2 rounded-lg
        bg-gradient-to-b from-purple-900/90 to-purple-950/90
        text-purple-300 font-medium
        shadow-[0_0_15px_rgba(147,51,234,0.5)]
        transition-all duration-200
        hover:shadow-[0_0_20px_rgba(147,51,234,0.7)]
        hover:text-purple-200
        active:scale-95
        backdrop-blur-sm
        border border-purple-700/50
        cursor-pointer
      `}
              type="submit"
              value="Hint"
            />
          </div>
        )}
      </form>
    </div>
  );
}

export default App;
