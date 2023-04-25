<div align="center">
  <img src="asset/logo.svg" width="70%"/>
</div>

# hora-wasm

**[[Homepage](http://horasearch.com/)]** **[[Document](https://horasearch.com/doc)]** **[[Examples](https://horasearch.com/doc/example.html)]** **[[Hora](https://github.com/hora-search/hora)]**

Javascript bidding for the **`Hora Approximate Nearest Neighbor Search`**, in **WebAssembly** way.

## Node Versions

The rust `getrandom` library, when used in ESM, requires a polyfill into `globalThis.crypto`. This is done automatically if you use Node 18 or lower, but you should know that this package overrides the globalThis.crypto.

```ts
import { webcrypto } from "node:crypto";
globalThis.crypto = webcrypto;
```

If using Node 19, 20 or greater this works out of the box.

## Features

- **Performant** ⚡️

  - **SIMD-Accelerated ([packed_simd](https://github.com/rust-lang/packed_simd))**
  - **Stable algorithm implementation**
  - **Multiple threads design**

- **Multiple Indexes Support** 🚀

  - `Hierarchical Navigable Small World Graph Index(HNSWIndex)` ([detail](https://arxiv.org/abs/1603.09320))
  - `Satellite System Graph (SSGIndex)` ([detail](https://arxiv.org/abs/1907.06146))
  - `Product Quantization Inverted File(PQIVFIndex)` ([detail](https://lear.inrialpes.fr/pubs/2011/JDS11/jegou_searching_with_quantization.pdf))
  - `Random Projection Tree(RPTIndex)` (LSH, WIP)
  - `BruteForce (BruteForceIndex)` (naive implementation with SIMD)

- **Reliability** 🔒

  - `Rust` compiler secure all code
  - Memory managed by `Rust` for all language libs such as `Python lib`
  - Broad testing coverage

- **Multiple Distances Support** 🧮

  - `Dot Product Distance`
    - ![equation](https://latex.codecogs.com/gif.latex?D%28x%2Cy%29%20%3D%20%5Csum%7B%28x*y%29%7D)
  - `Euclidean Distance`
    - ![equation](https://latex.codecogs.com/gif.latex?D%28x%2Cy%29%20%3D%20%5Csqrt%7B%5Csum%7B%28x-y%29%5E2%7D%7D)
  - `Manhattan Distance`
    - ![equation](https://latex.codecogs.com/gif.latex?D%28x%2Cy%29%20%3D%20%5Csum%7B%7C%28x-y%29%7C%7D)
  - `Cosine Similarity`
    - ![equation](https://latex.codecogs.com/gif.latex?D%28x%2Cy%29%20%3D%20%5Cfrac%7Bx%20*y%7D%7B%7C%7Cx%7C%7C*%7C%7Cy%7C%7C%7D)

- **Productive** ⭐
  - Well documented
  - Elegant and simple API, easy to learn

# Benchmark

<img src="asset/fashion-mnist-784-euclidean_10_euclidean.png"/>

by `aws t2.medium (CPU: Intel(R) Xeon(R) CPU E5-2686 v4 @ 2.30GHz)` [more information](https://github.com/hora-search/ann-benchmarks)

## Installation

```bash
yarn add horajs
```

## Example

```Javascript
import horaPromise from "horajs";

const demo = async () => {
    const hora = await horaPromise;
    await hora.init_env();
    await horajs.init_env();
    const dimension = 50;
    var bf_idx = horajs.BruteForceIndexUsize.new(dimension);
    // var hnsw_idx = horajs.HNSWIndexUsize.new(dimension, 1000000, 32, 64, 20, 500, 16, false);
    for (var i = 0; i < 1000; i++) {
        var feature = [];
        for (var j = 0; j < dimension; j++) {
            feature.push(Math.random());
        }
        bf_idx.add(feature, i); // add point
    }
    bf_idx.build("euclidean"); // build index
    var feature = [];
    for (var j = 0; j < dimension; j++) {
        feature.push(Math.random());
    }
    console.log("bf result", bf_idx.search(feature, 10)); //bf result Uint32Array(10) [704, 113, 358, 835, 408, 379, 117, 414, 808, 826]
}

demo();
```

# License

The entire repo is under [Apache License](https://github.com/hora-search/hora/blob/main/LICENSE).

## Contribution

Build scripts borrowed from [brotli-wasm](https://github.com/httptoolkit/brotli-wasm)
