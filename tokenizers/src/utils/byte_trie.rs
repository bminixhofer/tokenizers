use rand::Rng;
use std::collections::HashMap;

/// A node in a byte-level trie storing token IDs.
#[derive(Clone)]
pub struct ByteTrieNode {
    pub is_token: bool,
    pub token_id: Option<u32>,
    pub children: [Option<Box<ByteTrieNode>>; 256],
    /// Number of stoppable nodes (nodes with < 256 children) in this subtree, including self.
    num_stoppable: usize,
}

impl ByteTrieNode {
    fn new() -> Self {
        ByteTrieNode {
            is_token: false,
            token_id: None,
            children: std::array::from_fn(|_| None),
            num_stoppable: 0,
        }
    }
}

/// A byte-level trie for mapping byte sequences to token IDs.
#[derive(Clone)]
pub struct ByteTrie {
    root: ByteTrieNode,
}

impl ByteTrie {
    pub fn new() -> Self {
        ByteTrie {
            root: ByteTrieNode::new(),
        }
    }

    /// Compute `num_stoppable` on every node (bottom-up).
    fn compute_num_stoppable(node: &mut ByteTrieNode) {
        let num_children = node.children.iter().filter(|c| c.is_some()).count();
        let self_stoppable = if num_children < 256 { 1 } else { 0 };

        let mut child_stoppable: usize = 0;
        for child in &mut node.children {
            if let Some(c) = child {
                Self::compute_num_stoppable(c);
                child_stoppable += c.num_stoppable;
            }
        }
        node.num_stoppable = self_stoppable + child_stoppable;
    }

    pub fn insert(&mut self, bytes: &[u8], token_id: u32) {
        let mut node = &mut self.root;
        for &b in bytes {
            let idx = b as usize;
            if node.children[idx].is_none() {
                node.children[idx] = Some(Box::new(ByteTrieNode::new()));
            }
            node = node.children[idx].as_mut().unwrap();
        }
        node.is_token = true;
        node.token_id = Some(token_id);
    }

    pub fn lookup(&self, prefix: &[u8]) -> Option<&ByteTrieNode> {
        let mut node = &self.root;
        for &b in prefix {
            let idx = b as usize;
            match &node.children[idx] {
                Some(child) => node = child,
                None => return None,
            }
        }
        Some(node)
    }

    pub fn from_vocab(vocab: &HashMap<u32, Vec<u8>>) -> Self {
        let mut trie = ByteTrie::new();
        for (&token_id, token_bytes) in vocab {
            if !token_bytes.is_empty() {
                trie.insert(token_bytes, token_id);
            }
        }
        Self::compute_num_stoppable(&mut trie.root);
        trie
    }

    /// Sample a single uniform random stoppable node via weighted random descent,
    /// returning its path plus one random invalid continuation byte.
    /// Cost: O(depth * 256) per sample.
    fn sample_one_noise_path(&self, rng: &mut impl Rng) -> Vec<u8> {
        let mut path = Vec::new();
        let mut node = &self.root;

        loop {
            let num_children = node.children.iter().filter(|c| c.is_some()).count();
            let self_stoppable: usize = if num_children < 256 { 1 } else { 0 };

            let idx = rng.random_range(0..node.num_stoppable);
            if idx < self_stoppable {
                // Select this node: pick a random invalid continuation byte
                let num_invalid = 256 - num_children;
                let inv_idx = rng.random_range(0..num_invalid);
                let invalid_byte = (0..=255u8)
                    .filter(|&b| node.children[b as usize].is_none())
                    .nth(inv_idx)
                    .unwrap();
                path.push(invalid_byte);
                return path;
            }

            // Descend into the appropriate child based on cumulative stoppable counts
            let mut remaining = idx - self_stoppable;
            for b in 0u8..=255 {
                if let Some(child) = &node.children[b as usize] {
                    if remaining < child.num_stoppable {
                        path.push(b);
                        node = child;
                        break;
                    }
                    remaining -= child.num_stoppable;
                }
            }
        }
    }

    /// Sample noise paths until they fill `maxlen` bytes.
    fn sample_noise_chunk(&self, rng: &mut impl Rng, maxlen: usize) -> Vec<Vec<u8>> {
        if self.root.num_stoppable == 0 {
            return Vec::new();
        }

        let mut sequences = Vec::new();
        let mut current_len = 0;

        loop {
            let path = self.sample_one_noise_path(rng);
            if current_len + path.len() > maxlen {
                break;
            }
            current_len += path.len();
            sequences.push(path);
        }
        sequences
    }
}

/// Result of prepare_batch_metadata: layout information needed to allocate arrays.
#[derive(Clone)]
pub struct BatchMetadata {
    pub num_chunks: usize,
    pub total_len: usize,
    pub num_tokens: usize,
    /// Real token chunks (noise chunks are generated during fill_batch).
    pub chunks: Vec<Vec<Vec<u8>>>,
    /// Number of extra chunks to fill with noise during fill_batch.
    pub num_noise_chunks: usize,
}

/// Greedy bin-pack a list of byte sequences into fixed-size chunks.
fn bin_pack(sequences: &[Vec<u8>], maxlen: usize) -> Vec<Vec<Vec<u8>>> {
    let mut chunks: Vec<Vec<Vec<u8>>> = Vec::new();
    let mut current_chunk: Vec<Vec<u8>> = Vec::new();
    let mut current_len = 0;

    for seq in sequences {
        if current_len + seq.len() > maxlen {
            chunks.push(std::mem::take(&mut current_chunk));
            current_len = 0;
        }
        current_len += seq.len();
        current_chunk.push(seq.clone());
    }
    if !current_chunk.is_empty() {
        chunks.push(current_chunk);
    }
    chunks
}

/// Compute batch metadata: sort vocab by token_id, filter, greedy bin-pack.
/// Noise chunks are only counted here; their content is generated during fill_batch.
pub fn prepare_batch_metadata(
    vocab: &HashMap<u32, Vec<u8>>,
    maxlen: usize,
    noise_fraction: f32,
) -> Result<BatchMetadata, String> {
    let mut sorted_tokens: Vec<(u32, &Vec<u8>)> =
        vocab.iter().map(|(&id, bytes)| (id, bytes)).collect();
    sorted_tokens.sort_by_key(|(id, _)| *id);

    let mut all_bytes: Vec<Vec<u8>> = Vec::new();
    let mut too_long_tokens: Vec<(u32, usize)> = Vec::new();

    for (token_id, token_bytes) in &sorted_tokens {
        let token_len = token_bytes.len();
        if token_len == 0 {
            continue;
        }
        if token_len > maxlen {
            too_long_tokens.push((*token_id, token_len));
            continue;
        }
        all_bytes.push((*token_bytes).clone());
    }

    if !too_long_tokens.is_empty() {
        let max_token_len = too_long_tokens.iter().map(|(_, l)| *l).max().unwrap();
        return Err(format!(
            "maxlen={} is too small: {} tokens exceed maxlen (longest token length={}). \
             Increase maxlen to at least {}.",
            maxlen,
            too_long_tokens.len(),
            max_token_len,
            max_token_len
        ));
    }

    let num_tokens = all_bytes.len();
    let chunks = bin_pack(&all_bytes, maxlen);
    let num_real_chunks = chunks.len();
    let num_noise_chunks = ((num_real_chunks as f32) * noise_fraction).ceil() as usize;
    let num_chunks = num_real_chunks + num_noise_chunks;
    let total_len = num_chunks * maxlen;

    Ok(BatchMetadata {
        num_chunks,
        total_len,
        num_tokens,
        chunks,
        num_noise_chunks,
    })
}

/// Fill pre-allocated arrays with batch data.
///
/// Real token chunks are filled from metadata.chunks. Noise chunks are generated
/// on the fly by random-walking the trie and appending an invalid byte.
///
/// Array shapes (flattened as 1-D slices):
/// - input_bytes:     [num_chunks * maxlen]        u8
/// - position_ids:    [num_chunks * maxlen]        u8
/// - edge_labels:     [num_chunks * maxlen * 256]  bool
/// - is_token_labels: [num_chunks * maxlen]        bool
/// - attention_mask:  [num_chunks * maxlen]        bool
/// - segment_ids:     [num_chunks * maxlen]        i32
pub fn fill_batch(
    trie: &ByteTrie,
    metadata: &BatchMetadata,
    maxlen: usize,
    fill_true: bool,
    fill_noise: bool,
    rng: &mut impl Rng,
    input_bytes: &mut [u8],
    position_ids: &mut [u8],
    edge_labels: &mut [bool],
    is_token_labels: &mut [bool],
    attention_mask: &mut [bool],
    segment_ids: &mut [i32],
) {
    let mut global_seg_id: i32 = 0;

    // Helper: fill one chunk given its sequences
    let mut fill_chunk =
        |chunk_idx: usize, sequences: &[Vec<u8>], global_seg_id: &mut i32| {
            let chunk_offset = chunk_idx * maxlen;
            let mut pos = 0;

            for token_bytes in sequences {
                for (j, &b) in token_bytes.iter().enumerate() {
                    let flat_idx = chunk_offset + pos;

                    input_bytes[flat_idx] = b;
                    position_ids[flat_idx] = j as u8;
                    segment_ids[flat_idx] = *global_seg_id;
                    attention_mask[flat_idx] = true;

                    let prefix = &token_bytes[..j + 1];
                    if let Some(node) = trie.lookup(prefix) {
                        let edge_base = flat_idx * 256;
                        for (child_byte, child) in node.children.iter().enumerate() {
                            if child.is_some() {
                                edge_labels[edge_base + child_byte] = true;
                            }
                        }
                        if node.is_token {
                            is_token_labels[flat_idx] = true;
                        }
                    }
                    pos += 1;
                }
                *global_seg_id += 1;
            }

            // Padding positions get a unique segment id
            let pad_seg_id = *global_seg_id;
            *global_seg_id += 1;
            for p in pos..maxlen {
                segment_ids[chunk_offset + p] = pad_seg_id;
            }
        };

    // Fill real token chunks
    if fill_true {
        for (chunk_idx, chunk_tokens) in metadata.chunks.iter().enumerate() {
            fill_chunk(chunk_idx, chunk_tokens, &mut global_seg_id);
        }
    }

    // Fill noise chunks by sampling uniform random trie nodes until each chunk is full
    if fill_noise && metadata.num_noise_chunks > 0 {
        let noise_start = metadata.chunks.len();

        for i in 0..metadata.num_noise_chunks {
            let sequences = trie.sample_noise_chunk(rng, maxlen);
            fill_chunk(noise_start + i, &sequences, &mut global_seg_id);
        }
    }
}
