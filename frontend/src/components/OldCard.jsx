import PersonImg from "/src/assets/person.png";

export function Card() {
  return (
    <div className="rounded-2xl bg-[#CDB798] p-6 max-w-md mx-auto">
      <div className="rounded-2xl border-4 border-[#9F8261] flex flex-col p-4">
        <div className="flex items-center">
          <div className="flex flex-col justify-end flex-1 mx-4">
            <span className="text-3xl font-bold text-[#9B8C79] uppercase italic">
              Heart
            </span>
            <div className="border-y-2 border-[#AF9D81] rounded-full" />
          </div>
          <div className="border-4 border-[#E5DCD1]">
            <img src={PersonImg} />
          </div>
        </div>
        <div className="bg-white flex flex-col items-center rounded-b-2xl py-4">
          <span className="uppercase font-bold text-4xl">Heart</span>
        </div>
      </div>
    </div>
  );
}
