pub mod bpe;
pub mod byte_fallback;
pub mod ctc;
pub mod fuse;
pub mod sequence;
pub mod strip;
pub mod wordpiece;

// Re-export these as decoders
pub use super::pre_tokenizers::byte_level;
pub use super::pre_tokenizers::metaspace;

use serde::{Deserialize, Deserializer, Serialize};

use crate::decoders::bpe::BPEDecoder;
use crate::decoders::byte_fallback::ByteFallback;
use crate::decoders::ctc::CTC;
use crate::decoders::fuse::Fuse;
use crate::decoders::sequence::Sequence;
use crate::decoders::strip::Strip;
use crate::decoders::wordpiece::WordPiece;
use crate::normalizers::replace::Replace;
use crate::pre_tokenizers::byte_level::ByteLevel;
use crate::pre_tokenizers::metaspace::Metaspace;
use crate::pre_tokenizers::byte_level::CHAR_BYTES;
use crate::{Decoder, Result};

#[derive(Serialize, Clone, Debug)]
#[serde(untagged)]
pub enum DecoderWrapper {
    BPE(BPEDecoder),
    ByteLevel(ByteLevel),
    WordPiece(WordPiece),
    Metaspace(Metaspace),
    CTC(CTC),
    Sequence(Sequence),
    Replace(Replace),
    Fuse(Fuse),
    Strip(Strip),
    ByteFallback(ByteFallback),
}

impl<'de> Deserialize<'de> for DecoderWrapper {
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        #[derive(Deserialize)]
        pub struct Tagged {
            #[serde(rename = "type")]
            variant: EnumType,
            #[serde(flatten)]
            rest: serde_json::Value,
        }
        #[derive(Serialize, Deserialize)]
        pub enum EnumType {
            BPEDecoder,
            ByteLevel,
            WordPiece,
            Metaspace,
            CTC,
            Sequence,
            Replace,
            Fuse,
            Strip,
            ByteFallback,
        }

        #[derive(Deserialize)]
        #[serde(untagged)]
        pub enum DecoderHelper {
            Tagged(Tagged),
            Legacy(serde_json::Value),
        }

        #[derive(Deserialize)]
        #[serde(untagged)]
        pub enum DecoderUntagged {
            BPE(BPEDecoder),
            ByteLevel(ByteLevel),
            WordPiece(WordPiece),
            Metaspace(Metaspace),
            CTC(CTC),
            Sequence(Sequence),
            Replace(Replace),
            Fuse(Fuse),
            Strip(Strip),
            ByteFallback(ByteFallback),
        }

        let helper = DecoderHelper::deserialize(deserializer).expect("Helper");
        Ok(match helper {
            DecoderHelper::Tagged(model) => {
                let mut values: serde_json::Map<String, serde_json::Value> =
                    serde_json::from_value(model.rest).map_err(serde::de::Error::custom)?;
                values.insert(
                    "type".to_string(),
                    serde_json::to_value(&model.variant).map_err(serde::de::Error::custom)?,
                );
                let values = serde_json::Value::Object(values);
                match model.variant {
                    EnumType::BPEDecoder => DecoderWrapper::BPE(
                        serde_json::from_value(values).map_err(serde::de::Error::custom)?,
                    ),
                    EnumType::ByteLevel => DecoderWrapper::ByteLevel(
                        serde_json::from_value(values).map_err(serde::de::Error::custom)?,
                    ),
                    EnumType::WordPiece => DecoderWrapper::WordPiece(
                        serde_json::from_value(values).map_err(serde::de::Error::custom)?,
                    ),
                    EnumType::Metaspace => DecoderWrapper::Metaspace(
                        serde_json::from_value(values).map_err(serde::de::Error::custom)?,
                    ),
                    EnumType::CTC => DecoderWrapper::CTC(
                        serde_json::from_value(values).map_err(serde::de::Error::custom)?,
                    ),
                    EnumType::Sequence => DecoderWrapper::Sequence(
                        serde_json::from_value(values).map_err(serde::de::Error::custom)?,
                    ),
                    EnumType::Replace => DecoderWrapper::Replace(
                        serde_json::from_value(values).map_err(serde::de::Error::custom)?,
                    ),
                    EnumType::Fuse => DecoderWrapper::Fuse(
                        serde_json::from_value(values).map_err(serde::de::Error::custom)?,
                    ),
                    EnumType::Strip => DecoderWrapper::Strip(
                        serde_json::from_value(values).map_err(serde::de::Error::custom)?,
                    ),
                    EnumType::ByteFallback => DecoderWrapper::ByteFallback(
                        serde_json::from_value(values).map_err(serde::de::Error::custom)?,
                    ),
                }
            }
            DecoderHelper::Legacy(value) => {
                let untagged = serde_json::from_value(value).map_err(serde::de::Error::custom)?;
                match untagged {
                    DecoderUntagged::BPE(dec) => DecoderWrapper::BPE(dec),
                    DecoderUntagged::ByteLevel(dec) => DecoderWrapper::ByteLevel(dec),
                    DecoderUntagged::WordPiece(dec) => DecoderWrapper::WordPiece(dec),
                    DecoderUntagged::Metaspace(dec) => DecoderWrapper::Metaspace(dec),
                    DecoderUntagged::CTC(dec) => DecoderWrapper::CTC(dec),
                    DecoderUntagged::Sequence(dec) => DecoderWrapper::Sequence(dec),
                    DecoderUntagged::Replace(dec) => DecoderWrapper::Replace(dec),
                    DecoderUntagged::Fuse(dec) => DecoderWrapper::Fuse(dec),
                    DecoderUntagged::Strip(dec) => DecoderWrapper::Strip(dec),
                    DecoderUntagged::ByteFallback(dec) => DecoderWrapper::ByteFallback(dec),
                }
            }
        })
    }
}

impl Decoder for DecoderWrapper {
    fn decode_chain(&self, tokens: Vec<String>) -> Result<Vec<String>> {
        match self {
            Self::BPE(bpe) => bpe.decode_chain(tokens),
            Self::ByteLevel(bl) => bl.decode_chain(tokens),
            Self::Metaspace(ms) => ms.decode_chain(tokens),
            Self::WordPiece(wp) => wp.decode_chain(tokens),
            Self::CTC(ctc) => ctc.decode_chain(tokens),
            Self::Sequence(seq) => seq.decode_chain(tokens),
            Self::Replace(seq) => seq.decode_chain(tokens),
            Self::ByteFallback(bf) => bf.decode_chain(tokens),
            Self::Strip(bf) => bf.decode_chain(tokens),
            Self::Fuse(bf) => bf.decode_chain(tokens),
        }
    }

    fn decode_single_token_to_bytes(&self, token: &str) -> Result<Vec<u8>> {
        match self {
            Self::ByteLevel(_) => {
                // Each char in a byte-level token maps to one byte via CHAR_BYTES
                token
                    .chars()
                    .map(|c| {
                        CHAR_BYTES.get(&c).copied().ok_or_else(|| {
                            format!("unexpected char '{}' in byte-level token", c).into()
                        })
                    })
                    .collect()
            }
            Self::ByteFallback(_) => {
                // <0xHH> → single byte; otherwise UTF-8 bytes of the token string
                if token.len() == 6 && token.starts_with("<0x") && token.ends_with('>') {
                    let byte = u8::from_str_radix(&token[3..5], 16).map_err(|e| {
                        Box::new(e) as Box<dyn std::error::Error + Send + Sync>
                    })?;
                    Ok(vec![byte])
                } else {
                    Ok(token.as_bytes().to_vec())
                }
            }
            Self::Sequence(_) => {
                Err("decode_single_token_to_bytes is not supported for Sequence decoders".into())
            }
            Self::WordPiece(wp) => {
                // Continuation token (e.g. "##a"): strip prefix → "a"
                // Word-start token (e.g. "a"): prepend space → " a"
                // This mirrors decode_chain's behavior at i != 0, so that
                // word-start and continuation tokens are distinguishable.
                if let Some(stripped) = token.strip_prefix(&wp.prefix) {
                    Ok(stripped.as_bytes().to_vec())
                } else {
                    let mut bytes = vec![b' '];
                    bytes.extend_from_slice(token.as_bytes());
                    Ok(bytes)
                }
            }
            Self::Metaspace(ms) => {
                // Replace the metaspace replacement char (e.g. '▁') with a space.
                // decode_chain drops leading '▁' at i == 0, but for single-token
                // byte extraction we always want the space.
                let replacement = ms.get_replacement();
                let decoded: String = token
                    .chars()
                    .map(|c| if c == replacement { ' ' } else { c })
                    .collect();
                Ok(decoded.into_bytes())
            }
            _ => {
                // For all other decoders, run the normal decode and return UTF-8 bytes
                let decoded = self.decode(vec![token.to_string()])?;
                Ok(decoded.into_bytes())
            }
        }
    }
}

impl_enum_from!(BPEDecoder, DecoderWrapper, BPE);
impl_enum_from!(ByteLevel, DecoderWrapper, ByteLevel);
impl_enum_from!(ByteFallback, DecoderWrapper, ByteFallback);
impl_enum_from!(Fuse, DecoderWrapper, Fuse);
impl_enum_from!(Strip, DecoderWrapper, Strip);
impl_enum_from!(Metaspace, DecoderWrapper, Metaspace);
impl_enum_from!(WordPiece, DecoderWrapper, WordPiece);
impl_enum_from!(CTC, DecoderWrapper, CTC);
impl_enum_from!(Sequence, DecoderWrapper, Sequence);
impl_enum_from!(Replace, DecoderWrapper, Replace);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn decoder_serialization() {
        let oldjson = r#"{"type":"Sequence","decoders":[{"type":"ByteFallback"},{"type":"Metaspace","replacement":"▁","add_prefix_space":true,"prepend_scheme":"always"}]}"#;
        let olddecoder: DecoderWrapper = serde_json::from_str(oldjson).unwrap();
        let oldserialized = serde_json::to_string(&olddecoder).unwrap();
        let json = r#"{"type":"Sequence","decoders":[{"type":"ByteFallback"},{"type":"Metaspace","replacement":"▁","prepend_scheme":"always","split":true}]}"#;
        assert_eq!(oldserialized, json);

        let decoder: DecoderWrapper = serde_json::from_str(json).unwrap();
        let serialized = serde_json::to_string(&decoder).unwrap();
        assert_eq!(serialized, json);
    }
    #[test]
    fn decoder_serialization_other_no_arg() {
        let json = r#"{"type":"Sequence","decoders":[{"type":"Fuse"},{"type":"Metaspace","replacement":"▁","prepend_scheme":"always","split":true}]}"#;
        let decoder: DecoderWrapper = serde_json::from_str(json).unwrap();
        let serialized = serde_json::to_string(&decoder).unwrap();
        assert_eq!(serialized, json);
    }

    #[test]
    fn decoder_serialization_no_decode() {
        let json = r#"{"type":"Sequence","decoders":[{},{"type":"Metaspace","replacement":"▁","prepend_scheme":"always"}]}"#;
        let parse = serde_json::from_str::<DecoderWrapper>(json);
        match parse {
            Err(err) => assert_eq!(
                format!("{err}"),
                "data did not match any variant of untagged enum DecoderUntagged"
            ),
            _ => panic!("Expected error"),
        }

        let json = r#"{"replacement":"▁","prepend_scheme":"always"}"#;
        let parse = serde_json::from_str::<DecoderWrapper>(json);
        match parse {
            Err(err) => assert_eq!(
                format!("{err}"),
                "data did not match any variant of untagged enum DecoderUntagged"
            ),
            _ => panic!("Expected error"),
        }

        let json = r#"{"type":"Sequence","prepend_scheme":"always"}"#;
        let parse = serde_json::from_str::<DecoderWrapper>(json);
        match parse {
            Err(err) => assert_eq!(format!("{err}"), "missing field `decoders`"),
            _ => panic!("Expected error"),
        }
    }

    #[test]
    fn byte_level_decode_single_token_to_bytes() {
        use crate::Decoder;

        let decoder = DecoderWrapper::ByteLevel(crate::pre_tokenizers::byte_level::ByteLevel::default());

        // ASCII 'H' maps to 'H' in byte-level encoding
        let bytes = decoder.decode_single_token_to_bytes("H").unwrap();
        assert_eq!(bytes, vec![b'H']);

        // 'Ġ' is the byte-level encoding of 0x20 (space)
        let bytes = decoder.decode_single_token_to_bytes("Ġ").unwrap();
        assert_eq!(bytes, vec![0x20]);

        // Test a multi-char token
        let bytes = decoder.decode_single_token_to_bytes("Hello").unwrap();
        assert_eq!(bytes, b"Hello");
    }

    #[test]
    fn byte_fallback_decode_single_token_to_bytes() {
        use crate::Decoder;

        let decoder = DecoderWrapper::ByteFallback(ByteFallback::default());

        // <0xC3> → [0xC3]
        let bytes = decoder.decode_single_token_to_bytes("<0xC3>").unwrap();
        assert_eq!(bytes, vec![0xC3]);

        // <0x00> → [0x00]
        let bytes = decoder.decode_single_token_to_bytes("<0x00>").unwrap();
        assert_eq!(bytes, vec![0x00]);

        // <0xFF> → [0xFF]
        let bytes = decoder.decode_single_token_to_bytes("<0xFF>").unwrap();
        assert_eq!(bytes, vec![0xFF]);

        // Regular token → UTF-8 bytes
        let bytes = decoder.decode_single_token_to_bytes("hello").unwrap();
        assert_eq!(bytes, b"hello");
    }

    #[test]
    fn sequence_with_byte_fallback_decode_single_token_to_bytes() {
        use crate::Decoder;

        let json = r#"{"type":"Sequence","decoders":[{"type":"ByteFallback"},{"type":"Fuse"}]}"#;
        let decoder: DecoderWrapper = serde_json::from_str(json).unwrap();

        // Byte fallback token in a sequence
        let bytes = decoder.decode_single_token_to_bytes("<0xC3>").unwrap();
        assert_eq!(bytes, vec![0xC3]);

        // Regular token in a sequence
        let bytes = decoder.decode_single_token_to_bytes("hello").unwrap();
        assert_eq!(bytes, b"hello");
    }

    #[test]
    fn wordpiece_decode_single_token_to_bytes() {
        use crate::Decoder;

        let decoder = DecoderWrapper::WordPiece(WordPiece::default());

        // Word-start token → prepend space to distinguish from continuation
        let bytes = decoder.decode_single_token_to_bytes("hello").unwrap();
        assert_eq!(bytes, b" hello");

        // Continuation token: ## prefix stripped, no space
        let bytes = decoder.decode_single_token_to_bytes("##onste").unwrap();
        assert_eq!(bytes, b"onste");

        // Word-start and continuation of same text are distinguishable
        let bytes_start = decoder.decode_single_token_to_bytes("a").unwrap();
        let bytes_cont = decoder.decode_single_token_to_bytes("##a").unwrap();
        assert_eq!(bytes_start, b" a");
        assert_eq!(bytes_cont, b"a");
        assert_ne!(bytes_start, bytes_cont);

        // Token starting with # but not ## → word-start, gets space
        let bytes = decoder.decode_single_token_to_bytes("#tag").unwrap();
        assert_eq!(bytes, b" #tag");
    }

    #[test]
    fn metaspace_decode_single_token_to_bytes() {
        use crate::Decoder;
        use crate::pre_tokenizers::metaspace::{Metaspace, PrependScheme};

        let decoder = DecoderWrapper::Metaspace(Metaspace::new('▁', PrependScheme::Always, true));

        // ▁hello → " hello" (▁ replaced with space, even at position 0)
        let bytes = decoder.decode_single_token_to_bytes("▁hello").unwrap();
        assert_eq!(bytes, b" hello");

        // Plain token without ▁ → UTF-8 bytes unchanged
        let bytes = decoder.decode_single_token_to_bytes("world").unwrap();
        assert_eq!(bytes, b"world");
    }

    #[test]
    fn sequence_with_byte_fallback_and_metaspace() {
        use crate::Decoder;

        // LLaMA-style: [ByteFallback, Metaspace]
        let json = r#"{"type":"Sequence","decoders":[{"type":"ByteFallback"},{"type":"Metaspace","replacement":"▁","prepend_scheme":"always","split":true}]}"#;
        let decoder: DecoderWrapper = serde_json::from_str(json).unwrap();

        // Byte token → raw byte (short-circuits before Metaspace)
        let bytes = decoder.decode_single_token_to_bytes("<0xC3>").unwrap();
        assert_eq!(bytes, vec![0xC3]);

        // ▁hello → " hello" (Metaspace converts ▁ to space)
        let bytes = decoder.decode_single_token_to_bytes("▁hello").unwrap();
        assert_eq!(bytes, b" hello");

        // Plain token
        let bytes = decoder.decode_single_token_to_bytes("world").unwrap();
        assert_eq!(bytes, b"world");
    }
}
