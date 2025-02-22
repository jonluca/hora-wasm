import { describe, expect, it } from "vitest";
import { getHora, HoraJsType } from "../index.ts";
import seedrandom from "seedrandom";

const DIMENSION = 10;
describe("hora-wasm", async () => {
  let hora: HoraJsType = await getHora();
  await hora.init_env();

  const SEARCH_INDEXES = [
    "euclidean",
    "manhattan",
    "dot_product",
    "cosine_similarity",
    "angular",
  ];

  Error.stackTraceLimit = 30;
  for (const searchIndex of SEARCH_INDEXES) {
    // recreate index for every search type
    const INDEXES = [
      hora.BruteForceIndexUsize.new(DIMENSION),
      hora.HNSWIndexUsize.new(DIMENSION, 1_000_000, 32, 64, 20, 500, false),
      // TODO - these indexes don't work currently
      // hora.IVFPQIndexUsize.new(DIMENSION, 1000000, 32, 64, 20, 500),
      // hora.PQIndexUsize.new(DIMENSION, 1000000, 32, 64),
      // hora.SSGIndexUsize.new(DIMENSION, 1000000, 32, 64, 20, 500),
    ];

    for (let i = 0; i < INDEXES.length; i++) {
      const idx = INDEXES[i];
      idx.clear();

      let name = idx.name();
      const rng = seedrandom.alea(
        `horajs-consistent-rng-${name}-${searchIndex}`
      );

      const getFeatures = (count: number) => {
        const features: number[][] = [];
        for (let i = 0; i < count; i++) {
          const feature: number[] = [];
          for (let j = 0; j < DIMENSION; j++) {
            feature.push(rng());
          }
          features.push(feature);
        }
        return features;
      };

      it(`${name} - creates an index and searches over it using ${searchIndex}`, () => {
        const now = performance.now();
        const features = getFeatures(10_000);
        const now2 = performance.now();
        for (let i = 0; i < features.length; i++) {
          const feature = features[i];
          idx.add(Float32Array.from(feature), i); // add point
        }
        idx.build(searchIndex); // build index
        const featureToSearch = getFeatures(1)[0];
        const search_result = idx.search(
          Float32Array.from(featureToSearch),
          10
        );
        expect(search_result).toBeInstanceOf(Uint32Array);
        expect(search_result.length).toBe(10);
        expect(search_result).toMatchSnapshot();
      });

      it(`${name} - bulk inserts and indexes using ${searchIndex}`, () => {
        idx.clear();
        const features = getFeatures(10_000);
        const allVecs = Float32Array.from(features.flat());
        idx.bulk_add(
          allVecs,
          DIMENSION,
          Uint32Array.from(features.map((_, i) => i))
        );
        idx.build(searchIndex); // build index
        const featureToSearch = getFeatures(1)[0];
        const search_result = idx.search(
          Float32Array.from(featureToSearch),
          10
        );
        expect(search_result).toBeInstanceOf(Uint32Array);
        expect(search_result.length).toBe(10);
        expect(search_result).toMatchSnapshot();
      });
      it(`${name} - serializes indexes properly`, () => {
        const features = getFeatures(10_000);
        for (let i = 0; i < features.length; i++) {
          const feature = features[i];
          idx.add(Float32Array.from(feature), i); // add point
        }
        idx.build(searchIndex); // build index
        const featureToSearch = getFeatures(1)[0];
        let searchVector = Float32Array.from(featureToSearch);
        const search_result = idx.search(searchVector, 10);
        const serialized = idx.dump_index();
        const className = idx.constructor.name;
        const reconstructedIdx = (
          hora[className as keyof typeof hora] as any
        ).load_index(serialized);
        const new_search_result = reconstructedIdx.search(searchVector, 10);

        expect(search_result).toEqual(new_search_result);
      });
    }
  }
});
