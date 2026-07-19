import { describe, expect, it } from "vitest";
import horaPromise from "../index.ts";

describe("public API regressions", () => {
  it("exports the initialization promise", async () => {
    expect(horaPromise).toBeInstanceOf(Promise);
    const hora = await horaPromise;
    expect(hora.BruteForceIndexUsize.new).toBeTypeOf("function");
  });

  it("tracks size and clears every index implementation", async () => {
    const hora = await horaPromise;
    const dimension = 4;
    const indexes = [
      hora.BruteForceIndexUsize.new(dimension),
      hora.HNSWIndexUsize.new(dimension, 100, 4, 8, 4, 10, false),
      hora.PQIndexUsize.new(dimension, 2, 1, 1),
      hora.IVFPQIndexUsize.new(dimension, 2, 1, 1, 1, 1),
      hora.SSGIndexUsize.new(dimension, 4, 2, 4, 60, 1),
    ];

    for (const index of indexes) {
      expect(index.size(), `${index.name()} initial size`).toBe(0);
      index.add(Float32Array.from([1, 2, 3, 4]), 7);
      expect(index.size(), `${index.name()} size after add`).toBe(1);
      expect(index.clear(), `${index.name()} clear result`).toBe(true);
      expect(index.size(), `${index.name()} size after clear`).toBe(0);
    }

    const bruteForce = indexes[0];
    bruteForce.build("euclidean");
    expect(bruteForce.search(Float32Array.from([1, 2, 3, 4]), 10)).toHaveLength(
      0
    );
  });

  it("preserves reset metadata when serializing an index", async () => {
    const hora = await horaPromise;
    const index = hora.BruteForceIndexUsize.new(4);
    index.add(Float32Array.from([4, 3, 2, 1]), 9);

    const restored = hora.BruteForceIndexUsize.load_index(index.dump_index());
    expect(restored.size()).toBe(1);
    expect(restored.clear()).toBe(true);
    expect(restored.size()).toBe(0);
    restored.build("euclidean");
    expect(restored.search(Float32Array.from([4, 3, 2, 1]), 10)).toHaveLength(
      0
    );
  });
});
