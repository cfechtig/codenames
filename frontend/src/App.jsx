import { c } from "/src/lib/c";
import { useForm } from "react-hook-form";
import { useEffect, useState } from "react";

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
    const data = await response.json();
    setBoard(data);
  };

  useEffect(() => {
    init();
  }, []);

  if (hintsRemaining === 0 || strikes === 3) {
    return (
      <div>
        <h1>Game Over</h1>
        <p>Final Score: {hintsRemaining - strikes}</p>
      </div>
    );
  }

  return (
    <div>
      <h1>Code Names</h1>
      <p>Current Turn: {turn}</p>
      <p>Hints Remaining: {hintsRemaining}</p>
      <p>Strikes: {strikes}</p>
      {hint && <p>Hint: {hint}</p>}
      <form onSubmit={handleSubmit(onHint)}>
        <div className="grid grid-cols-5 gap-4">
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
        </div>
        {turn === "HUMAN" && (
          <div>
            <input className="border" {...register("hint")} />
            <input className="cursor-pointer" type="submit" />
          </div>
        )}
      </form>
    </div>
  );
}

export default App;
