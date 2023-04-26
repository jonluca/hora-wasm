extern crate console_error_panic_hook;
extern crate wasm_bindgen;

use hora_new::core::ann_index::{SerializableIndex, ANNIndex};
use hora_new::core::metrics;
use std::panic;
use wasm_bindgen::prelude::*;

fn metrics_transform(s: &str) -> metrics::Metric {
    match s {
        "angular" => metrics::Metric::Angular,
        "manhattan" => metrics::Metric::Manhattan,
        "dot_product" => metrics::Metric::DotProduct,
        "euclidean" => metrics::Metric::Euclidean,
        "cosine_similarity" => metrics::Metric::CosineSimilarity,
        _ => metrics::Metric::Unknown,
    }
}

#[macro_export]
macro_rules! inherit_ann_index_method {
    (  $ann_idx:ident, $type_expr: ty, $idx_type_expr: ty) => {
        #[wasm_bindgen]
        pub struct $ann_idx {
            _idx: Box<$type_expr>,
        }

        #[wasm_bindgen]
        impl $ann_idx {
            pub fn build(&mut self, s: String) -> bool {
                self._idx.build(metrics_transform(&s)).unwrap();
                true
            }
            pub fn add(&mut self, vs: &[f32], idx: $idx_type_expr) -> bool {
                self._idx.add(&vs, idx).unwrap();
                true
            }

            pub fn clear(&mut self) -> bool {
                self._idx.clear();
                true
            }

            pub fn size(&mut self) -> $idx_type_expr {
               return self._idx.nodes_size();
            }

            pub fn bulk_add(&mut self, flat_vs: &[f32], length: usize, idx: &[$idx_type_expr]) -> bool {
               // extract the vectors from the flat array, where each vector has the same length length
                let num_vecs = flat_vs.len() / length;
                let mut vs = Vec::with_capacity(num_vecs);
                for i in 0..num_vecs {
                    let start = i * length;
                    let end = start + length;
                    vs.push(&flat_vs[start..end]);
                }
                self._idx.madd(&vs, &idx).unwrap(); // Replace this with the actual implementation
                true
            }

            pub fn search(&self, vs: &[f32], k: usize) -> Vec<$idx_type_expr> {
                self._idx.search(&vs, k)
            }

            pub fn name(&self) -> String {
                self._idx.name().to_string()
            }

            pub fn dump_index(&mut self) -> Vec<u8> {
                self._idx.dump_bin().unwrap()
            }
            pub fn load_index(serialized: &[u8]) -> Self {
                $ann_idx {
                    _idx: Box::new(<$type_expr>::load_bin(&serialized.to_vec()).unwrap())
                }
            }

        }
    };
}

inherit_ann_index_method!(BruteForceIndexUsize, hora_new::index::bruteforce_idx::BruteForceIndex<f32,usize>, usize);
#[wasm_bindgen]
impl BruteForceIndexUsize {
    pub fn new(dimension: usize) -> Self {
        console_error_panic_hook::set_once();
        BruteForceIndexUsize {
            _idx: Box::new(hora_new::index::bruteforce_idx::BruteForceIndex::<
                f32,
                usize,
            >::new(
                dimension,
                &hora_new::index::bruteforce_params::BruteForceParams::default(),
            )),
        }
    }
}

inherit_ann_index_method!(HNSWIndexUsize, hora_new::index::hnsw_idx::HNSWIndex<f32, usize>,usize);
#[wasm_bindgen]
impl HNSWIndexUsize {
    pub fn new(
        dimension: usize,
        max_item: usize,
        n_neigh: usize,
        n_neigh0: usize,
        ef_build: usize,
        ef_search: usize,
        has_deletion: bool,
    ) -> Self {
        console_error_panic_hook::set_once();

        HNSWIndexUsize {
            _idx: Box::new(hora_new::index::hnsw_idx::HNSWIndex::<f32, usize>::new(
                dimension,
                &hora_new::index::hnsw_params::HNSWParams::<f32>::default()
                    .max_item(max_item)
                    .n_neighbor(n_neigh)
                    .n_neighbor0(n_neigh0)
                    .ef_build(ef_build)
                    .ef_search(ef_search)
                    .has_deletion(has_deletion),
            )),
        }
    }
}

inherit_ann_index_method!(PQIndexUsize, hora_new::index::pq_idx::PQIndex<f32, usize>,usize);
#[wasm_bindgen]
impl PQIndexUsize {
    pub fn new(dimension: usize, n_sub: usize, sub_bits: usize, train_epoch: usize) -> Self {
        console_error_panic_hook::set_once();

        PQIndexUsize {
            _idx: Box::new(hora_new::index::pq_idx::PQIndex::<f32, usize>::new(
                dimension,
                &hora_new::index::pq_params::PQParams::default()
                    .n_sub(n_sub)
                    .sub_bits(sub_bits)
                    .train_epoch(train_epoch),
            )),
        }
    }
}

inherit_ann_index_method!(IVFPQIndexUsize, hora_new::index::pq_idx::IVFPQIndex<f32, usize>,usize);
#[wasm_bindgen]
impl IVFPQIndexUsize {
    pub fn new(
        dimension: usize,
        n_sub: usize,
        sub_bits: usize,
        n_kmeans_center: usize,
        search_n_center: usize,
        train_epoch: usize,
    ) -> Self {
        console_error_panic_hook::set_once();
        IVFPQIndexUsize {
            _idx: Box::new(hora_new::index::pq_idx::IVFPQIndex::<f32, usize>::new(
                dimension,
                &hora_new::index::pq_params::IVFPQParams::default()
                    .n_sub(n_sub)
                    .sub_bits(sub_bits)
                    .n_kmeans_center(n_kmeans_center)
                    .search_n_center(search_n_center)
                    .train_epoch(train_epoch),
            )),
        }
    }
}

inherit_ann_index_method!(SSGIndexUsize, hora_new::index::ssg_idx::SSGIndex<f32, usize>,usize);
#[wasm_bindgen]
impl SSGIndexUsize {
    pub fn new(
        dimension: usize,
        neighbor_neighbor_size: usize,
        init_k: usize,
        index_size: usize,
        angle: f32,
        root_size: usize,
    ) -> Self {
        console_error_panic_hook::set_once();

        SSGIndexUsize {
            _idx: Box::new(hora_new::index::ssg_idx::SSGIndex::<f32, usize>::new(
                dimension,
                &hora_new::index::ssg_params::SSGParams::default()
                    .neighbor_neighbor_size(neighbor_neighbor_size)
                    .init_k(init_k)
                    .index_size(index_size)
                    .angle(angle)
                    .root_size(root_size),
            )),
        }
    }
}

#[wasm_bindgen]
pub fn init_env() {
    panic::set_hook(Box::new(console_error_panic_hook::hook));
}
