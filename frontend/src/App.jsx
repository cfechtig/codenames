import { c } from "/src/lib/c";
import { useForm } from "react-hook-form";
import { useEffect, useState } from "react";

const CardTypeColorMap = {
  F: "bg-[#348E47]",
  B: "bg-[#D4C297]",
  A: "bg-gray-500",
};

function App() {
  const { register, handleSubmit, setValue, resetField } = useForm();

  const onHint = async (data) => {
    setValue("turn", turn);
    fetch("/hint", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({ data }),
    });
    setHintsRemaining(hintsRemaining - 1);
  };

  const [board, setBoard] = useState({});
  const [turn, setTurn] = useState("HUMAN");
  const [hintsRemaining, setHintsRemaining] = useState(9);
  const [strikes, setStrikes] = useState(0);

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

  return (
    <div>
      <h1>Code Names</h1>
      <p>Current Turn: {turn}</p>
      <p>Hints Remaining: {hintsRemaining}</p>
      <p>Strikes: {strikes}</p>
      <form onSubmit={handleSubmit(onHint)}>
        <input {...register("turn")} type="hidden" />
        <div className="grid grid-cols-5 gap-4">
          {Object.entries(board)
            .flatMap(([type, words]) =>
              words.map((word) => ({ value: word, type }))
            )
            .map((word) => (
              <div
                key={word.value}
                className={c(CardTypeColorMap[word.type], "cursor-pointer")}
                onClick={() => onGuess(word.value)}
              >
                {word.value}
              </div>
            ))}
        </div>
        <input className="border" {...register("hint")} />
        <input className="cursor-pointer" type="submit" />
      </form>
    </div>
  );
}

export default App;
