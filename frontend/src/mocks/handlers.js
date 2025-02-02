import { http, HttpResponse } from "msw";

export const handlers = [
  http.get("/board", () => {
    return HttpResponse.json({
      F: [
        "GENIE",
        "MILL",
        "LACE",
        "SPURS",
        "CAPTAIN",
        "BARBECUE",
        "MINUTE",
        "CHRISTMAS",
        "FOG",
      ],
      B: [
        "PAINT",
        "RUBBER",
        "BOIL",
        "TEAM",
        "FUEL",
        "DISK",
        "RAIL",
        "COACH",
        "THUNDER",
        "CLOUD",
        "RAZOR",
        "BENCH",
        "ARMOR",
      ],
      A: ["LIP", "JUDGE", "ANTHEM"],
    });
  }),
  http.post("/guess", async ({ request }) => {
    const requestBody = await request.json();
    return HttpResponse.json(`guess: ${requestBody.data}`);
  }),
];
