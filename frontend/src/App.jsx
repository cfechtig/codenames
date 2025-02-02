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
      <div className="flex flex-col items-center justify-center">
        <img src={LogoImg} alt="logo" className="h-20" />
        <p>Current Turn: {turn}</p>
        <p>Hints Remaining: {hintsRemaining}</p>
        <p>Strikes: {strikes}</p>
        {hint && <p>Hint: {hint}</p>}
      </div>
      <Canvas
        shadows
        camera={{ position: [0, 15, 20], fov: 90 }}
        className="h-screen"
      >
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
              highlighted={i % 3 === 0}
            />
          ))}
        </group>
        <Environment preset="studio" />
        <OrbitControls
          enableZoom={false}
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
          <div>
            <input
              className="border"
              {...register("hint")}
              onKeyDown={onKeyDown}
            />
            <input className="cursor-pointer" type="submit" hidden />
          </div>
        )}
      </form>
    </div>
  );
}

export default App;
