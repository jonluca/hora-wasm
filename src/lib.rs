extern crate console_error_panic_hook;
extern crate wasm_bindgen;

use hora_new::core::ann_index::{ANNIndex, SerializableIndex};
use hora_new::core::metrics;
use std::panic;
use wasm_bindgen::prelude::*;

const SERIALIZATION_MAGIC: &[u8; 8] = b"HORAJS01";

fn encode_index(mut serialized: Vec<u8>, config: &[u64], count: usize) -> Vec<u8> {
    for value in config {
        serialized.extend_from_slice(&value.to_le_bytes());
    }
    serialized.extend_from_slice(&(config.len() as u64).to_le_bytes());
    serialized.extend_from_slice(&(count as u64).to_le_bytes());
    serialized.extend_from_slice(SERIALIZATION_MAGIC);
    serialized
}

fn decode_index(serialized: &[u8]) -> (Vec<u8>, Option<(Vec<u64>, usize)>) {
    const METADATA_LEN: usize = 24;
    if serialized.len() < METADATA_LEN
        || &serialized[serialized.len() - SERIALIZATION_MAGIC.len()..] != SERIALIZATION_MAGIC
    {
        return (serialized.to_vec(), None);
    }

    let metadata_start = serialized.len() - METADATA_LEN;
    let config_len = u64::from_le_bytes(
        serialized[metadata_start..metadata_start + 8]
            .try_into()
            .unwrap(),
    ) as usize;
    let count = u64::from_le_bytes(
        serialized[metadata_start + 8..metadata_start + 16]
            .try_into()
            .unwrap(),
    );
    let config_bytes = match config_len.checked_mul(8) {
        Some(config_bytes) if config_bytes <= metadata_start => config_bytes,
        _ => return (serialized.to_vec(), None),
    };
    if count > usize::MAX as u64 {
        return (serialized.to_vec(), None);
    }

    let config_start = metadata_start - config_bytes;
    let config = serialized[config_start..metadata_start]
        .chunks_exact(8)
        .map(|bytes| u64::from_le_bytes(bytes.try_into().unwrap()))
        .collect();
    (
        serialized[..config_start].to_vec(),
        Some((config, count as usize)),
    )
}

trait ResettableIndex {
    fn new_empty(config: &[u64]) -> Self;
}

impl ResettableIndex for hora_new::index::bruteforce_idx::BruteForceIndex<f32, usize> {
    fn new_empty(config: &[u64]) -> Self {
        Self::new(
            config[0] as usize,
            &hora_new::index::bruteforce_params::BruteForceParams::default(),
        )
    }
}

impl ResettableIndex for hora_new::index::hnsw_idx::HNSWIndex<f32, usize> {
    fn new_empty(config: &[u64]) -> Self {
        Self::new(
            config[0] as usize,
            &hora_new::index::hnsw_params::HNSWParams::<f32>::default()
                .max_item(config[1] as usize)
                .n_neighbor(config[2] as usize)
                .n_neighbor0(config[3] as usize)
                .ef_build(config[4] as usize)
                .ef_search(config[5] as usize)
                .has_deletion(config[6] != 0),
        )
    }
}

impl ResettableIndex for hora_new::index::pq_idx::PQIndex<f32, usize> {
    fn new_empty(config: &[u64]) -> Self {
        Self::new(
            config[0] as usize,
            &hora_new::index::pq_params::PQParams::default()
                .n_sub(config[1] as usize)
                .sub_bits(config[2] as usize)
                .train_epoch(config[3] as usize),
        )
    }
}

impl ResettableIndex for hora_new::index::pq_idx::IVFPQIndex<f32, usize> {
    fn new_empty(config: &[u64]) -> Self {
        Self::new(
            config[0] as usize,
            &hora_new::index::pq_params::IVFPQParams::default()
                .n_sub(config[1] as usize)
                .sub_bits(config[2] as usize)
                .n_kmeans_center(config[3] as usize)
                .search_n_center(config[4] as usize)
                .train_epoch(config[5] as usize),
        )
    }
}

impl ResettableIndex for hora_new::index::ssg_idx::SSGIndex<f32, usize> {
    fn new_empty(config: &[u64]) -> Self {
        Self::new(
            config[0] as usize,
            &hora_new::index::ssg_params::SSGParams::default()
                .neighbor_neighbor_size(config[1] as usize)
                .init_k(config[2] as usize)
                .index_size(config[3] as usize)
                .angle(f32::from_bits(config[4] as u32))
                .root_size(config[5] as usize),
        )
    }
}

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
            _config: Option<Vec<u64>>,
            _count: usize,
        }

        #[wasm_bindgen]
        impl $ann_idx {
            pub fn build(&mut self, s: String) -> bool {
                self._idx.build(metrics_transform(&s)).unwrap();
                true
            }
            pub fn add(&mut self, vs: &[f32], idx: $idx_type_expr) -> bool {
                self._idx.add(&vs, idx).unwrap();
                self._count += 1;
                true
            }

            pub fn clear(&mut self) -> bool {
                if let Some(config) = &self._config {
                    self._idx = Box::new(<$type_expr>::new_empty(config));
                } else {
                    self._idx.clear();
                }
                self._count = 0;
                true
            }

            pub fn size(&self) -> $idx_type_expr {
                self._count
            }

            pub fn bulk_add(
                &mut self,
                flat_vs: &[f32],
                length: usize,
                idx: &[$idx_type_expr],
            ) -> bool {
                // extract the vectors from the flat array, where each vector has the same length length
                let num_vecs = flat_vs.len() / length;
                let mut vs = Vec::with_capacity(num_vecs);
                for i in 0..num_vecs {
                    let start = i * length;
                    let end = start + length;
                    vs.push(&flat_vs[start..end]);
                }
                self._idx.madd(&vs, &idx).unwrap();
                self._count += num_vecs;
                true
            }

            pub fn search(&self, vs: &[f32], k: usize) -> Vec<$idx_type_expr> {
                self._idx.search(&vs, k)
            }

            pub fn name(&self) -> String {
                self._idx.name().to_string()
            }

            pub fn dump_index(&mut self) -> Vec<u8> {
                let serialized = self._idx.dump_bin().unwrap();
                match &self._config {
                    Some(config) => encode_index(serialized, config, self._count),
                    None => serialized,
                }
            }
            pub fn load_index(serialized: &[u8]) -> Self {
                let (serialized, metadata) = decode_index(serialized);
                let idx = <$type_expr>::load_bin(&serialized).unwrap();
                let (config, count) = metadata
                    .map(|(config, count)| (Some(config), count))
                    .unwrap_or_else(|| (None, idx.nodes_size()));
                $ann_idx {
                    _idx: Box::new(idx),
                    _config: config,
                    _count: count,
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
        let config = vec![dimension as u64];
        BruteForceIndexUsize {
            _idx: Box::new(ResettableIndex::new_empty(&config)),
            _config: Some(config),
            _count: 0,
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
        let config = vec![
            dimension as u64,
            max_item as u64,
            n_neigh as u64,
            n_neigh0 as u64,
            ef_build as u64,
            ef_search as u64,
            has_deletion as u64,
        ];
        HNSWIndexUsize {
            _idx: Box::new(ResettableIndex::new_empty(&config)),
            _config: Some(config),
            _count: 0,
        }
    }
}

inherit_ann_index_method!(PQIndexUsize, hora_new::index::pq_idx::PQIndex<f32, usize>,usize);
#[wasm_bindgen]
impl PQIndexUsize {
    pub fn new(dimension: usize, n_sub: usize, sub_bits: usize, train_epoch: usize) -> Self {
        console_error_panic_hook::set_once();
        let config = vec![
            dimension as u64,
            n_sub as u64,
            sub_bits as u64,
            train_epoch as u64,
        ];
        PQIndexUsize {
            _idx: Box::new(ResettableIndex::new_empty(&config)),
            _config: Some(config),
            _count: 0,
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
        let config = vec![
            dimension as u64,
            n_sub as u64,
            sub_bits as u64,
            n_kmeans_center as u64,
            search_n_center as u64,
            train_epoch as u64,
        ];
        IVFPQIndexUsize {
            _idx: Box::new(ResettableIndex::new_empty(&config)),
            _config: Some(config),
            _count: 0,
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
        let config = vec![
            dimension as u64,
            neighbor_neighbor_size as u64,
            init_k as u64,
            index_size as u64,
            angle.to_bits() as u64,
            root_size as u64,
        ];
        SSGIndexUsize {
            _idx: Box::new(ResettableIndex::new_empty(&config)),
            _config: Some(config),
            _count: 0,
        }
    }
}

#[wasm_bindgen]
pub fn init_env() {
    panic::set_hook(Box::new(console_error_panic_hook::hook));
}
