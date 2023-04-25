import { describe, beforeAll, expect, it } from "vitest";
import { getHora, HoraJsType } from "../index.ts";

describe("hora-wasm", () => {
  let hora: HoraJsType;
  beforeAll(async () => {
    // hora = await horaPromise;
    hora = await getHora();
  });

  it("can compress data", () => {
    const dimension = 50;
    const bf_idx = hora.BruteForceIndexUsize.new(dimension);
    // var hnsw_idx = horajs.HNSWIndexUsize.new(dimension, 1000000, 32, 64, 20, 500, 16, false);
    for (let i = 0; i < 1000; i++) {
      const feature: number[] = [];
      for (let j = 0; j < dimension; j++) {
        feature.push(Math.random());
      }
      bf_idx.add(Float32Array.from(feature), i); // add point
    }
    bf_idx.build("euclidean"); // build index
    let feature = [];
    for (let j = 0; j < dimension; j++) {
      feature.push(Math.random());
    }
    console.log("bf result", bf_idx.search(Float32Array.from(feature), 10)); //bf result Uint32Array(10) [704, 113, 358, 835, 408, 379, 117, 414, 808, 826]
  });
});
