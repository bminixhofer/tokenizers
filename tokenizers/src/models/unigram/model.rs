use itertools::Itertools;
use rand::seq::SliceRandom;
use std::cmp::Reverse;

use super::{
    lattice::Lattice,
    trainer::UnigramTrainer,
    trie::{Trie, TrieBuilder},
};
use crate::utils::cache::Cache;
use crate::{
    tokenizer::{Model, Result, Token},
    OffsetReferential, OffsetType, PreTokenizedString, PreTokenizer,
};

use std::collections::{HashMap, HashSet};
use std::convert::TryInto;
use std::fs::read_to_string;
use std::path::{Path, PathBuf};

type TokenMap = HashMap<String, u32>;
type Vocab = Vec<(String, f64)>;

/// A `Unigram` model to encode sentences.
pub struct Unigram {
    token_to_ids: TokenMap,
    pub(crate) vocab: Vocab,
    cache: Cache<String, Vec<String>>,
    trie: Trie<u8>,
    pub min_score: f64,
    pub(super) unk_id: Option<usize>,
    pub(super) bos_id: usize,
    pub(super) eos_id: usize,

    fuse_unk: bool,
    is_optimized: bool,
    byte_fallback: bool,

    original_vocab: Vocab,
    original_indices: Vec<usize>,

    subsample_cache: once_cell::sync::OnceCell<(Vec<f64>, Vec<(usize, (String, f64))>, HashMap<String, usize>)>,
}
impl PartialEq for Unigram {
    fn eq(&self, other: &Self) -> bool {
        self.unk_id == other.unk_id && self.vocab == other.vocab
    }
}

impl Clone for Unigram {
    // `Clone` can't be derive because it's not implemented for `Cache`.
    // To keep things simple when we clone, the new Unigram will start with a fresh cache.
    fn clone(&self) -> Self {
        let fresh_cache = self.cache.fresh();
        Self {
            vocab: self.vocab.clone(),
            cache: fresh_cache,
            token_to_ids: self.token_to_ids.clone(),
            trie: self.trie.clone(),
            min_score: self.min_score,
            unk_id: self.unk_id,
            bos_id: self.bos_id,
            eos_id: self.eos_id,
            fuse_unk: self.fuse_unk,
            is_optimized: self.is_optimized,
            byte_fallback: self.byte_fallback,
            original_vocab: self.original_vocab.clone(),
            original_indices: self.original_indices.clone(),
            subsample_cache: once_cell::sync::OnceCell::new(),
        }
    }
}

impl std::fmt::Debug for Unigram {
    fn fmt(&self, fmt: &mut std::fmt::Formatter) -> std::fmt::Result {
        fmt.debug_struct("Unigram")
            .field("vocab", &self.vocab.len())
            .field("unk_id", &self.unk_id)
            .field("byte_fallback", &self.byte_fallback)
            .finish()
    }
}

static K_UNK_PENALTY: f64 = 10.0;
static LARGE_NUMBER: f64 = 10e12;

#[derive(thiserror::Error, Debug)]
pub enum UnigramError {
    #[error("The vocabulary is empty but at least <unk> is needed")]
    EmptyVocabulary,
    #[error("The `unk_id` is larger than vocabulary size")]
    UnkIdNotInVocabulary,
    #[error("Encountered an unknown token but `unk_id` is missing")]
    MissingUnkId,
}

impl Default for Unigram {
    fn default() -> Self {
        let vocab = vec![("<unk>".to_string(), 0.0)];
        Self::from(vocab, Some(0), false).unwrap()
    }
}

impl Unigram {
    /// Create a `Unigram` model from a given vocabulary.
    /// Vocabulary are the various tokens and their associated score which is a sort of a logprob of
    /// their frequency, which will enable tokenization and sampling.
    /// unk_id, is the index within the vocabulary.
    /// For now `Unigram` *requires* at least `unk` because we might find a never seen char.
    /// Further versions might allow that part to be hidden.
    pub fn from(
        vocab: Vec<(String, f64)>,
        unk_id: Option<usize>,
        byte_fallback: bool,
    ) -> Result<Self> {
        let n = vocab.len();
        let mut token_to_ids: TokenMap = HashMap::new();
        let mut builder = TrieBuilder::default();

        if let Some(unk_id) = unk_id {
            if vocab.is_empty() {
                return Err(Box::new(UnigramError::EmptyVocabulary));
            }
            if unk_id >= vocab.len() {
                return Err(Box::new(UnigramError::UnkIdNotInVocabulary));
            }
        }
        let bos_id = n + 1;
        let eos_id = n + 2;

        let mut min_score = f64::INFINITY;
        for (id, (token, score)) in vocab.iter().enumerate() {
            token_to_ids.insert(token.to_string(), id as u32);
            let bytes: Vec<u8> = token.bytes().collect();
            builder.push(&bytes);
            if score < &min_score {
                min_score = *score;
            }
        }
        let trie = builder.build();
        let fuse_unk = true;
        let is_optimized = true;

        Ok(Self {
            vocab: vocab.clone(),
            token_to_ids,
            trie,
            min_score,
            bos_id,
            eos_id,
            unk_id,
            fuse_unk,
            cache: Cache::default(),
            is_optimized,
            byte_fallback,

            original_indices: (0..vocab.len()).collect(),
            original_vocab: vocab,
            subsample_cache: once_cell::sync::OnceCell::new(),
        })
    }

    #[cfg(test)]
    pub(super) fn set_fuse_unk(&mut self, fuse_unk: bool) {
        self.fuse_unk = fuse_unk;
        self.cache = self.cache.fresh();
    }

    #[cfg(test)]
    pub(super) fn set_optimized(&mut self, is_optimized: bool) {
        self.is_optimized = is_optimized;
    }
    pub fn byte_fallback(&self) -> bool {
        self.byte_fallback
    }
    pub(super) fn len(&self) -> usize {
        self.vocab.len()
    }

    pub fn encode_with_regularization<T: PreTokenizer>(&self, pre_tokenizer: T, text: String, top_n: usize, temperature: f64) -> Vec<usize> {
        let mut rng = rand::thread_rng();

        let mut pretokenized: PreTokenizedString = text.into();
        pre_tokenizer.pre_tokenize(&mut pretokenized).unwrap();
        let mut input_ids: Vec<usize> = Vec::new();

        let mut pre_token_to_tokenization: HashMap<String, Vec<usize>> = HashMap::new();

        for (pretoken, _, _) in pretokenized.get_splits(OffsetReferential::Original, OffsetType::Byte).iter() {
            if let Some(sequence) = pre_token_to_tokenization.get(*pretoken) {
                input_ids.extend(sequence)
            } else {
                let tokenizations = self.get_top_n_encodings(pretoken, top_n);

                let scores: Vec<_> = tokenizations
                    .iter()
                    .map(|t| {
                        t.0.iter().map(|x| self.vocab[*x].1).sum::<f64>() / temperature
                    })
                    .collect();
                let max_score =
                    scores.iter().fold(
                        f64::NEG_INFINITY,
                        |max, &val| if val > max { val } else { max },
                    );
                let exp_scores: Vec<_> = scores.iter().map(|x| (x - max_score).exp()).collect();
                let exp_scores_sum: f64 = exp_scores.iter().sum();
    
                let probs: Vec<f64> = exp_scores
                    .iter()
                    .map(|x| x / exp_scores_sum * LARGE_NUMBER)
                    .collect();
                let index = *(0..probs.len())
                    .collect::<Vec<_>>()
                    .choose_weighted(&mut rng, |i| probs[*i])
                    .unwrap();
                input_ids.extend(tokenizations[index].0.iter().cloned());
                pre_token_to_tokenization.insert((*pretoken).to_owned(), tokenizations[index].0.clone());
            }
        }

        input_ids
    }

    pub fn encode_bpe_style<T: PreTokenizer>(&self, pre_tokenizer: T, texts: Vec<String>, block_size: usize, top_n: usize) -> Vec<Vec<usize>> {
        let mut all_input_ids: Vec<Vec<usize>> = Vec::with_capacity(texts.len());
        let mut cache: HashMap<String, Vec<usize>> = HashMap::new();

        // tokenize batch with the sampled vocab
        for text in texts {
            let mut pretokenized: PreTokenizedString = text.into();
            pre_tokenizer.pre_tokenize(&mut pretokenized).unwrap();
            let mut input_ids: Vec<usize> = Vec::with_capacity(block_size);

            for (pretoken, _, _) in pretokenized.get_splits(OffsetReferential::Original, OffsetType::Byte).iter() {
                let sequence = if let Some(sequence) = cache.get(*pretoken) {
                    sequence.clone()
                } else {
                    let tokenizations = self.get_top_n_encodings(&pretoken, top_n);
                    let mut surface_forms: Vec<_> = tokenizations
                        .iter()
                        .enumerate()
                        .map(|(i, t)| {
                            (
                                i,
                                t.0.iter()
                                    .map(|x| self.id_to_token(*x as u32).unwrap())
                                    .collect::<Vec<_>>(),
                            )
                        })
                        .collect();
                    surface_forms.sort_by_key(|(_, s)| {
                        Reverse((
                            s.iter().map(|x| 11 * x.chars().count() - 1).sum::<usize>(),
                            s.iter().cloned().collect::<Vec<_>>(),
                        ))
                    });
                    let sequence = tokenizations[surface_forms[0].0].0.clone();
                    cache.insert((*pretoken).to_owned(), sequence.clone());
                    sequence
                };

                for id in sequence.iter() {
                    if input_ids.len() < block_size {
                        input_ids.push(*id);
                    }
                }

                if input_ids.len() == block_size {
                    break;
                }
            }

            while input_ids.len() < block_size {
                // also convention? :|
                input_ids.push(0);
            }

            all_input_ids.push(input_ids);
        }

        all_input_ids
    }

    pub fn make_seed_sentence_pieces(
        &self,
        map: HashMap<String, u32>,
        seed_size: usize,
        max_length: usize
    ) -> Vec<(String, f64)> {
        let sentences: Vec<(Vec<char>, u32)> = map.iter().map(|(s, i)| (s.chars().collect(), *i)).collect();

        let mut all_chars: HashSet<char> = HashSet::new();

        for (string, _) in &sentences {
            for c in string {
                all_chars.insert(*c);
            }
        }

        //  Basic chars need to be in sentence pieces.
        let mut seed_sentencepieces: Vec<(String, f64)> = vec![];

        println!("Constructing prefixes / suffixes...");

        let suffixes: Vec<_> = sentences.iter().map(|(string, _)| {
            let mut pieces = Vec::with_capacity(string.len() * 2);
            for i in 0..string.len() {
                pieces.push(&string[i..]);
            }
            for i in 1..(string.len() + 1) {
                pieces.push(&string[..i]);
            }
            pieces
        }).collect();

        println!("Computing scores...");

        let mut substr_index: HashMap<String, u32> = HashMap::new();

        for ((_, n), suffix) in sentences.iter().zip(suffixes.iter()) {
            for string in suffix.iter() {
                let freq = n;

                if string.is_empty() {
                    continue;
                }
                if string.len() > max_length {
                    continue;
                }
                let score = freq * string.len() as u32;

                substr_index.entry(string.iter().collect()).and_modify(|e| {*e += score}).or_insert(score);
            }
        }

        println!("Filling & sorting...");

        // Fill seed_sentencepieces
        for character in all_chars {
            let string = character.to_string();
            let count = *substr_index.get(&string).unwrap_or(&1);

            seed_sentencepieces.push((string, count.into()));
        }

        let mut substr_index = substr_index.into_iter().collect::<Vec<_>>();

        // sort by decreasing score
        substr_index.sort_by_key(|a| Reverse(a.1));

        for (string, score) in substr_index {
            if string.chars().count() == 1 {
                // already added
                continue;
            }
            seed_sentencepieces.push((string, score.into()));
            if seed_sentencepieces.len() >= seed_size {
                break;
            }
        }
        super::to_log_prob(&mut seed_sentencepieces);
        seed_sentencepieces
    }

    pub fn subsample(
        &mut self,
        subsample_size: usize,
        temperature: f64,
        ignore_pieces: Vec<String>,
        add_pieces: Vec<String>,
        add_pieces_ids: Vec<usize>,
    ) {
        let mut rng = rand::thread_rng();

        let ignore_pieces: HashSet<_> = ignore_pieces
            .into_iter()
            .chain(add_pieces.clone().into_iter())
            .collect();

        // NB: this assumes ignore_pieces and temperature are the same across all calls!! very bad pattern btw
        // but it should speed things up a lot
        let (probs, pieces_with_indices, piece_to_original_index) = self.subsample_cache.get_or_init(|| {
            let exp_logprobs: Vec<f64> = self
            .original_vocab
            .iter()
            .map(|x| (x.1 / temperature).exp())
            .collect();
            let exp_logprobs_sum = exp_logprobs.iter().sum::<f64>();
            let probs: Vec<_> = exp_logprobs
                .iter()
                .map(|x| x / exp_logprobs_sum * LARGE_NUMBER)
                .collect();

            let pieces_with_indices: Vec<_> = self.original_vocab.iter().enumerate().map(|x| (x.0, x.1.to_owned())).collect();
            let piece_to_original_index: HashMap<_, _> = pieces_with_indices
                .iter()
                .map(|(i, (p, _))| (p.to_owned(), *i))
                .collect();

            (probs, pieces_with_indices, piece_to_original_index)
        });

        // sampling is wrong (samples low prob. too much) without the large multiplier
        // maybe f64 imprecisions?
        let to_sample: Vec<_> = pieces_with_indices
            .into_iter()
            .filter(|(_, x)| !ignore_pieces.contains(&x.0))
            .collect();
        let sampled: Vec<_> = to_sample
            .choose_multiple_weighted(&mut rng, subsample_size - add_pieces.len(), |piece| {
                probs[piece.0]
            })
            .unwrap()
            .cloned()
            .collect();

        self.original_indices = sampled.iter().map(|(i, _)| *i).collect::<Vec<_>>();
        self.vocab = sampled
            .into_iter()
            .map(|(_, p)| p.clone())
            .collect::<Vec<_>>();

        for (piece, index) in add_pieces
            .iter()
            .zip(add_pieces_ids.iter())
            .sorted_by_key(|x| x.1)
        {
            let original_index = piece_to_original_index
                .get(piece)
                .cloned();

            self.vocab.insert(*index, (piece.clone(), original_index.map_or(0.0, |i| self.original_vocab[i].1)));
            self.original_indices.insert(
                *index,
                original_index.unwrap_or(usize::MAX),
            );
        }

        // this is problematic :|
        // may be fine as implicit convention?
        self.unk_id = Some(0);

        self.cache = self.cache.fresh();
        self.update_trie();

        self.token_to_ids = HashMap::with_capacity(self.vocab.len());
        for (id, (token, _)) in self.vocab.iter().enumerate() {
            self.token_to_ids.insert(token.to_string(), id as u32);
        }
    }

    pub fn get_original_indices(&self) -> Vec<usize> {
        self.original_indices.clone()
    }

    pub fn set_vocab(&mut self, vocab: HashMap<String, u32>) {
        self.vocab.sort_by_key(|x| vocab.get(&x.0).unwrap());
        self.token_to_ids = vocab;

        self.cache = self.cache.fresh();
    }

    pub fn set_pieces(&mut self, pieces: Vec<(String, f64)>) {
        self.vocab = pieces;
    }

    pub fn get_pieces(&self) -> Vec<(String, f64)> {
        self.vocab.clone()
    }

    pub fn set_scores(&mut self, scores: Vec<f64>) {
        for (i, score) in scores.into_iter().enumerate() {
            self.vocab[i].1 = score;
        }
        self.cache = self.cache.fresh();
    }

    pub fn get_scores(&self) -> Vec<f64> {
        self.vocab.iter().map(|(_, score)| *score).collect()
    }

    pub fn update_trie(&mut self) {
        let mut builder = TrieBuilder::default();

        for (token, _) in self.vocab.iter() {
            let bytes: Vec<u8> = token.bytes().collect();
            builder.push(&bytes);
        }
        self.trie = builder.build();
    }

    pub(super) fn populate_nodes(&self, lattice: &mut Lattice) {
        let unk_score = self.min_score - K_UNK_PENALTY;

        let len = lattice.len();

        let mut begin_pos = 0;
        while begin_pos < len {
            let mblen = lattice.sentence[begin_pos..]
                .chars()
                .next()
                .unwrap()
                .len_utf8();

            let mut has_single_node = false;

            for bytes in self
                .trie
                .common_prefix_search(lattice.sentence.bytes().skip(begin_pos))
            {
                let n = bytes.len();
                let tok = String::from_utf8(bytes).unwrap();
                let id = *self.token_to_ids.get(&tok).unwrap();

                let item = &self.vocab[id as usize];
                assert_eq!(item.0, tok);
                let score: f64 = item.1;
                lattice.insert(begin_pos, n, score, id.try_into().unwrap());
                if !has_single_node && n == mblen {
                    has_single_node = true;
                }
            }

            if !has_single_node {
                if let Some(unk_id) = self.unk_id {
                    lattice.insert(begin_pos, mblen, unk_score, unk_id);
                }
            }
            begin_pos += mblen
        }
    }

    /// This functions take a String, and will encode it in a Vec of Strings,
    /// of the best tokenization available to the current model.
    /// ```
    /// use tokenizers::models::unigram::Unigram;
    ///
    /// let pieces = vec![
    ///     ("<unk>".to_string(), 0.0),
    ///     ("a".to_string(), 0.0),
    ///     ("b".to_string(), 0.0),
    ///     ("c".to_string(), 0.0),
    ///     ("d".to_string(), 0.0),
    ///     ("cd".to_string(), 1.0),
    ///     ("ab".to_string(), 2.0),
    ///     ("abc".to_string(), 5.0),
    ///     ("abcd".to_string(), 10.0),
    /// ];
    /// let model = Unigram::from(pieces, Some(0), false).unwrap();
    /// let result = model.encode("abcdacdxx").unwrap();
    /// assert_eq!(result, vec!["abcd", "a", "cd", "xx"]);
    /// ```
    pub fn encode(&self, sentence: &str) -> Result<Vec<String>> {
        if sentence.is_empty() {
            return Ok(vec![]);
        }
        if let Some(result) = self.cache.get(sentence) {
            Ok(result.to_vec())
        } else {
            let result = if self.is_optimized {
                self.encode_optimized(sentence)?
            } else {
                self.encode_unoptimized(sentence)?
            };
            self.cache.set(sentence.to_owned(), result.clone());
            Ok(result)
        }
    }

    pub fn get_top_n_encodings(&self, sentence: &str, n: usize) -> Vec<(Vec<usize>, f64)> {
        let mut lattice = Lattice::from(sentence, self.bos_id, self.eos_id);
        self.populate_nodes(&mut lattice);
        lattice
            .nbest(n)
            .iter()
            .map(|x| {
                let ids: Vec<_> = x
                    .iter()
                    .map(|node| {
                        let tok = lattice.piece_str(&node.borrow());
                        *self.token_to_ids.get(tok).unwrap() as usize
                    })
                    .collect();

                let score = x.iter().fold(0.0, |acc, node| acc + node.borrow().score);
                (ids, score)
            })
            .collect()
    }

    fn encode_optimized(&self, sentence: &str) -> Result<Vec<String>> {
        // https://github.com/google/sentencepiece/blob/d48247191a6d50e469ed1a4a36e877befffd1851/src/unigram_model.cc#L600
        #[derive(Debug, Clone)]
        struct BestPathNode {
            /// The vocab id. (maybe UNK)
            id: usize,
            /// The total score of the best path ending at this node.
            best_path_score: f64,
            /// The starting position (in utf-8) of this node. The entire best
            /// path can be constructed by backtracking along this link.
            starts_at: Option<usize>,
        }
        impl Default for BestPathNode {
            fn default() -> Self {
                Self {
                    id: 0,
                    best_path_score: 0.0,
                    starts_at: None,
                }
            }
        }
        let size = sentence.len();
        let unk_score = self.min_score - K_UNK_PENALTY;

        let mut best_path_ends_at = vec![BestPathNode::default(); size + 1];
        let mut starts_at = 0;
        while starts_at < size {
            let best_path_score_till_here = best_path_ends_at[starts_at].best_path_score;
            let mut has_single_node = false;
            let mblen = sentence[starts_at..].chars().next().unwrap().len_utf8();
            for tok_bytes in self
                .trie
                .common_prefix_search(sentence.bytes().skip(starts_at))
            {
                let key_pos = starts_at + tok_bytes.len();
                let token: String = String::from_utf8(tok_bytes).unwrap();
                let target_node = &mut best_path_ends_at[key_pos];
                let length = key_pos - starts_at;
                let id = self.token_to_ids.get(&token).unwrap();
                let score = self.vocab.get(*id as usize).unwrap().1;
                let candidate_best_path_score = score + best_path_score_till_here;
                if target_node.starts_at.is_none()
                    || candidate_best_path_score > target_node.best_path_score
                {
                    target_node.best_path_score = candidate_best_path_score;
                    target_node.starts_at = Some(starts_at);
                    target_node.id = *id as usize;
                }
                if !has_single_node && length == mblen {
                    has_single_node = true;
                }
            }
            if !has_single_node {
                let target_node = &mut best_path_ends_at[starts_at + mblen];
                let candidate_best_path_score = unk_score + best_path_score_till_here;
                if target_node.starts_at.is_none()
                    || candidate_best_path_score > target_node.best_path_score
                {
                    target_node.best_path_score = candidate_best_path_score;
                    target_node.starts_at = Some(starts_at);
                    target_node.id = self.unk_id.ok_or(UnigramError::MissingUnkId)?;
                }
            }
            starts_at += mblen
        }
        let mut ends_at = size;
        let mut results: Vec<String> = vec![];
        let mut token = vec![];
        while ends_at > 0 {
            let node = &best_path_ends_at[ends_at];
            let starts_at = node.starts_at.unwrap();
            if self.fuse_unk
                && self.unk_id.is_some()
                && node.id == self.unk_id.ok_or(UnigramError::MissingUnkId)?
            {
                token.push(
                    String::from_utf8(sentence[starts_at..ends_at].as_bytes().to_vec()).unwrap(),
                );
            } else {
                if !token.is_empty() {
                    token.reverse();
                    results.push(token.concat());
                    token = vec![];
                }
                results.push(
                    String::from_utf8(sentence[starts_at..ends_at].as_bytes().to_vec()).unwrap(),
                );
            }
            ends_at = starts_at;
        }
        if !token.is_empty() {
            token.reverse();
            results.push(token.concat());
        }
        results.reverse();
        Ok(results)
    }

    fn encode_unoptimized(&self, sentence: &str) -> Result<Vec<String>> {
        let mut lattice = Lattice::from(sentence, self.bos_id, self.eos_id);
        self.populate_nodes(&mut lattice);
        if self.fuse_unk {
            let mut results = vec![];
            let mut token = String::new();
            for node in lattice.viterbi().iter() {
                let item = lattice.piece(&node.borrow());
                if node.borrow().id == self.unk_id.ok_or(UnigramError::MissingUnkId)? {
                    token.push_str(&item);
                } else {
                    if !token.is_empty() {
                        results.push(token);
                        token = String::new();
                    }
                    results.push(item.to_string());
                }
            }
            if !token.is_empty() {
                results.push(token);
            }
            Ok(results)
        } else {
            Ok(lattice.tokens())
        }
    }

    /// Iterate of vocabulary of the model as a pair of `(token, score)`.
    pub fn iter(&self) -> UnigramIterator {
        UnigramIterator { model: self, i: 0 }
    }

    /// Loads a SentencePiece output model after being trained by tokenizers.
    /// After that you can use the model with tokenizers library.
    /// ```no_run
    /// use tokenizers::models::unigram::Unigram;
    /// use std::path::Path;
    ///
    /// let model = Unigram::load("mymodel-unigram.json").unwrap();
    /// ```
    pub fn load<P: AsRef<Path>>(path: P) -> Result<Unigram> {
        let string = read_to_string(path)?;
        Ok(serde_json::from_str(&string)?)
    }
}

/// Iterator to iterate of vocabulary of the model, and their relative score.
pub struct UnigramIterator<'a> {
    model: &'a Unigram,
    i: usize,
}

impl<'a> Iterator for UnigramIterator<'a> {
    type Item = &'a (String, f64);

    fn next(&mut self) -> Option<Self::Item> {
        let i = self.i;
        if i < self.model.len() {
            let r = Some(&self.model.vocab[i]);
            self.i += 1;
            r
        } else {
            None
        }
    }
}

impl Model for Unigram {
    type Trainer = UnigramTrainer;

    fn get_vocab(&self) -> HashMap<String, u32> {
        self.token_to_ids.clone()
    }

    fn get_vocab_size(&self) -> usize {
        self.vocab.len()
    }

    fn tokenize(&self, sentence: &str) -> Result<Vec<Token>> {
        let str_tokens = self.encode(sentence)?;
        let mut offset = 0;
        let mut tokens = Vec::with_capacity(str_tokens.len());
        for string in str_tokens {
            let len = string.len();
            let offsets = (offset, offset + len);
            let id: u32 = match self.token_to_ids.get(&string) {
                Some(id) => *id,
                None => {
                    if self.byte_fallback {
                        let byte_tokens: Option<Vec<_>> = string
                            .bytes()
                            .map(|byte| -> Option<Token> {
                                let byte_string = format!("<0x{:02X}>", byte);
                                let id = self.token_to_ids.get(&byte_string);
                                id.map(|id| Token::new(*id, byte_string, (offset, offset + len)))
                            })
                            .collect();
                        if let Some(byte_tokens) = byte_tokens {
                            for token in byte_tokens {
                                tokens.push(token);
                            }
                            offset += len;
                            continue;
                        }
                    }
                    self.unk_id.ok_or(UnigramError::MissingUnkId)? as u32
                }
            };
            offset += len;
            tokens.push(Token::new(id, string, offsets));
        }
        Ok(tokens)
    }

    fn token_to_id(&self, token: &str) -> Option<u32> {
        self.token_to_ids.get(token).copied()
    }

    fn id_to_token(&self, id: u32) -> Option<String> {
        self.vocab.get(id as usize).map(|item| item.0.clone())
    }

    fn save(&self, folder: &Path, name: Option<&str>) -> Result<Vec<PathBuf>> {
        let name = match name {
            Some(name) => format!("{}-unigram.json", name),
            None => "unigram.json".to_string(),
        };
        let mut fullpath = PathBuf::new();
        fullpath.push(folder);
        fullpath.push(name);
        let string = serde_json::to_string_pretty(self)?;
        std::fs::write(&fullpath, string)?;
        Ok(vec![fullpath])
    }

    fn get_trainer(&self) -> Self::Trainer {
        UnigramTrainer::default()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_populate_nodes_unk() {
        let pieces = vec![("<unk>".to_string(), 0.0)];
        let model = Unigram::from(pieces, Some(0), false).unwrap();

        let mut lattice = Lattice::from("abc", model.bos_id, model.eos_id);
        model.populate_nodes(&mut lattice);

        assert_eq!(lattice.begin_nodes[0].len(), 1);
        assert_eq!(lattice.begin_nodes[1].len(), 1);
        assert_eq!(lattice.begin_nodes[2].len(), 1);
        assert_eq!(lattice.begin_nodes[0][0].borrow().id, 0);
        assert_eq!(lattice.begin_nodes[1][0].borrow().id, 0);
        assert_eq!(lattice.begin_nodes[2][0].borrow().id, 0);
        assert_eq!(lattice.begin_nodes[0][0].borrow().node_id, 2);
        assert_eq!(lattice.begin_nodes[1][0].borrow().node_id, 3);
        assert_eq!(lattice.begin_nodes[2][0].borrow().node_id, 4);
    }

    #[test]
    fn test_populate_nodes() {
        let pieces = vec![
            ("<unk>".to_string(), 0.0),
            ("a".to_string(), 0.1),
            ("b".to_string(), 0.2),
            ("ab".to_string(), 0.3),
            ("bc".to_string(), 0.4),
        ];
        let model = Unigram::from(pieces, Some(0), false).unwrap();

        let mut lattice = Lattice::from("abc", model.bos_id, model.eos_id);
        model.populate_nodes(&mut lattice);

        assert_eq!(lattice.begin_nodes[0].len(), 2); // a, ab
        assert_eq!(lattice.begin_nodes[1].len(), 2); // b, bc
        assert_eq!(lattice.begin_nodes[2].len(), 1); // c(unk)

        // Id is the vocabulary id from Unigram model
        // node_id is simply the rank of the given node in the lattice.
        assert_eq!(lattice.begin_nodes[0][0].borrow().id, 1);
        assert_eq!(lattice.begin_nodes[0][1].borrow().id, 3);
        assert_eq!(lattice.begin_nodes[1][0].borrow().id, 2);
        assert_eq!(lattice.begin_nodes[1][1].borrow().id, 4);
        assert_eq!(lattice.begin_nodes[2][0].borrow().id, 0);
        assert_eq!(lattice.begin_nodes[0][0].borrow().node_id, 2);
        assert_eq!(lattice.begin_nodes[0][1].borrow().node_id, 3);
        assert_eq!(lattice.begin_nodes[1][0].borrow().node_id, 4);
        assert_eq!(lattice.begin_nodes[1][1].borrow().node_id, 5);
        assert_eq!(lattice.begin_nodes[2][0].borrow().node_id, 6);
    }

    #[test]
    fn test_encode() {
        let sentencepieces = vec![
            ("<unk>".to_string(), 0.0),
            ("a".to_string(), 0.0),
            ("b".to_string(), 0.0),
            ("c".to_string(), 0.0),
            ("d".to_string(), 0.0),
            ("cd".to_string(), 1.0),
            ("ab".to_string(), 2.0),
            ("abc".to_string(), 5.0),
            ("abcd".to_string(), 10.0),
        ];

        let model = Unigram::from(sentencepieces, Some(0), false).unwrap();
        let result = model.encode("abcd").unwrap();
        assert_eq!(result, vec!["abcd"]);
    }

    #[test]
    fn test_encode2() {
        let sentencepieces = vec![
            ("<unk>".to_string(), 0.0),
            ("ab".to_string(), 0.0),
            ("cd".to_string(), -0.1),
            ("abc".to_string(), -0.2),
            ("a".to_string(), -0.3),
            ("b".to_string(), -0.4),
            ("c".to_string(), -0.5),
            ("ABC".to_string(), -0.5),
            ("abcdabcd".to_string(), 20.0), // User defined just max the scores.
            ("q".to_string(), 20.5),
            ("r".to_string(), 20.5),
            ("qr".to_string(), -0.5),
        ];

        let mut model = Unigram::from(sentencepieces, Some(0), false).unwrap();

        for is_optimized in &[true, false] {
            model.set_optimized(*is_optimized);
            println!("IsOptimized {:?}", is_optimized);
            assert_eq!(model.encode("abc").unwrap(), vec!["abc"]);
            assert_eq!(model.encode("AB").unwrap(), vec!["AB"]);

            model.set_fuse_unk(false);
            assert_eq!(model.encode("AB").unwrap(), vec!["A", "B"]);
            model.set_fuse_unk(true);
            assert_eq!(model.encode("AB").unwrap(), vec!["AB"]);

            assert_eq!(model.encode("abcd").unwrap(), vec!["ab", "cd"]);
            assert_eq!(model.encode("abcc").unwrap(), vec!["abc", "c"]);
            assert_eq!(
                model.encode("xabcabaabcdd").unwrap(),
                vec!["x", "abc", "ab", "a", "ab", "cd", "d"]
            );
            model.set_fuse_unk(false);
            assert_eq!(
                model.encode("xyz東京").unwrap(),
                vec!["x", "y", "z", "東", "京"]
            );
            model.set_fuse_unk(true);
            assert_eq!(model.encode("xyz東京").unwrap(), vec!["xyz東京"]);

            // User encoded in original version
            assert_eq!(model.encode("ABC").unwrap(), vec!["ABC"]);
            assert_eq!(model.encode("abABCcd").unwrap(), vec!["ab", "ABC", "cd"]);
            assert_eq!(
                model.encode("ababcdabcdcd").unwrap(),
                vec!["ab", "abcdabcd", "cd"]
            );
            assert_eq!(model.encode("abqrcd").unwrap(), vec!["ab", "q", "r", "cd"]);
        }
    }

    #[test]
    fn test_unigram_bytefallback() {
        // In [97]: processor.encode_as_pieces("⅐⅛⅑ ")
        // Out[97]: ['▁', '<0xE2>', '<0x85>', '<0x90>', '⅛', '<0xE2>', '<0x85>', '<0x91>', '▁']
        let sentencepieces = vec![
            ("<unk>".to_string(), 0.0),
            ("<0xC3>".to_string(), -0.01),
            ("<0xA9>".to_string(), -0.03),
        ];
        let unigram = Unigram::from(sentencepieces, Some(0), true).unwrap();
        let tokens: Vec<Token> = unigram.tokenize("é").unwrap();
        assert_eq!(
            tokens,
            [
                Token {
                    id: 1,
                    value: "<0xC3>".to_string(),
                    offsets: (0, 2)
                },
                Token {
                    id: 2,
                    value: "<0xA9>".to_string(),
                    offsets: (0, 2)
                }
            ]
        );

        let tokens = unigram.tokenize("?é").unwrap();
        assert_eq!(tokens[0].id, 0);
    }
}
