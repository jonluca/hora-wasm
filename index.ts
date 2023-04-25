// In pure ESM web bundles, you must call init() and wait for the promised result before you can
// call any module methods. To make that as easy as possible, this module directly exposes the
// init() promise result, and returns the methods at the end of the promise.
import init, * as horaJs from "./pkg/horajs.js";

let input: undefined | Buffer = undefined;
let initialized = false;
export type HoraJsType = typeof horaJs;
export const getHora = async (): Promise<HoraJsType> => {
  if (initialized) {
    return horaJs;
  }
  if (typeof window === "undefined") {
    const fs = await import("fs");
    input = fs.readFileSync("./pkg/horajs_bg.wasm");
  }
  await init(input);
  initialized = true;
  return horaJs;
};

export default getHora;
