//! The DXBC container: a header, then a table of FourCC-tagged parts.
//!
//! ```text
//! 0x00  "DXBC"
//! 0x04  digest[16]
//! 0x14  version major u16, minor u16
//! 0x18  total size u32
//! 0x1C  part count u32
//! 0x20  part offsets u32[count]
//! part: fourcc[4], size u32, data[size]
//! ```

use crate::ShaderError;

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct FourCC(pub [u8; 4]);

impl FourCC {
    pub const DXIL: FourCC = FourCC(*b"DXIL");
    pub const STAT: FourCC = FourCC(*b"STAT");
    pub const RTS0: FourCC = FourCC(*b"RTS0");
    pub const PSV0: FourCC = FourCC(*b"PSV0");
    pub const ISG1: FourCC = FourCC(*b"ISG1");
    pub const OSG1: FourCC = FourCC(*b"OSG1");
    /// Per-primitive outputs of a mesh shader.
    pub const PSG1: FourCC = FourCC(*b"PSG1");
    pub const SFI0: FourCC = FourCC(*b"SFI0");
    pub const HASH: FourCC = FourCC(*b"HASH");
    pub const ILDB: FourCC = FourCC(*b"ILDB");
    pub const ILDN: FourCC = FourCC(*b"ILDN");
    /// ACCC chunk holding an FSPX precompiled compute payload.
    pub const ACCC: FourCC = FourCC(*b"ACCC");
}

impl std::fmt::Debug for FourCC {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self)
    }
}

impl std::fmt::Display for FourCC {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        for &b in &self.0 {
            let c = if b.is_ascii_graphic() { b as char } else { '.' };
            write!(f, "{c}")?;
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Copy)]
pub struct Part<'a> {
    pub fourcc: FourCC,
    /// Offset of the part header within the container.
    pub offset: usize,
    pub data: &'a [u8],
}

#[derive(Debug, Clone)]
pub struct Container<'a> {
    pub digest: [u8; 16],
    pub version: (u16, u16),
    pub parts: Vec<Part<'a>>,
}

const HEADER_SIZE: usize = 0x20;
const MAX_PARTS: usize = 256;

impl<'a> Container<'a> {
    pub fn parse(data: &'a [u8]) -> Result<Self, ShaderError> {
        if data.len() < HEADER_SIZE || &data[0..4] != b"DXBC" {
            return Err(ShaderError::NotDxbc);
        }
        let digest = data[4..20].try_into().expect("16 bytes");
        let version = (u16_at(data, 0x14)?, u16_at(data, 0x16)?);
        let total = u32_at(data, 0x18)? as usize;
        if total > data.len() || total < HEADER_SIZE {
            return Err(ShaderError::Malformed("container size disagrees with data"));
        }
        let data = &data[..total];
        let count = u32_at(data, 0x1C)? as usize;
        if count > MAX_PARTS {
            return Err(ShaderError::Malformed("implausible part count"));
        }
        let mut parts = Vec::with_capacity(count);
        for i in 0..count {
            let offset = u32_at(data, HEADER_SIZE + 4 * i)? as usize;
            let fourcc = FourCC(
                data.get(offset..offset + 4)
                    .ok_or(ShaderError::Malformed("part header out of bounds"))?
                    .try_into()
                    .expect("4 bytes"),
            );
            let size = u32_at(data, offset + 4)? as usize;
            let body = data
                .get(offset + 8..offset + 8 + size)
                .ok_or(ShaderError::Malformed("part data out of bounds"))?;
            parts.push(Part {
                fourcc,
                offset,
                data: body,
            });
        }
        Ok(Container {
            digest,
            version,
            parts,
        })
    }

    pub fn part(&self, fourcc: FourCC) -> Option<&Part<'a>> {
        self.parts.iter().find(|p| p.fourcc == fourcc)
    }

    /// If this container holds an FSPX compute payload in its ACCC part.
    pub fn fspx(&self) -> Option<Fspx<'a>> {
        let part = self.part(FourCC::ACCC)?;
        if part.data.starts_with(Fspx::MAGIC) {
            Fspx::parse(part.data).ok()
        } else {
            None
        }
    }
}

/// An FSPX payload header and its directory of entries.
#[derive(Debug, Clone)]
pub struct Fspx<'a> {
    pub version: u32,
    pub count: u32,
    pub total_size: u32,
    /// Kind 0: dispatch state (28 bytes).
    pub dispatch_state: Option<&'a [u8]>,
    /// Kind 1: unknown (20 bytes).
    pub kind1: Option<&'a [u8]>,
    /// Kind 6: precompiled RDNA2 compute program (with 36-byte prologue).
    pub program: Option<&'a [u8]>,
    /// Kind 7: resource and hash records.
    pub records: Option<&'a [u8]>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DispatchInfo {
    pub threads: [u32; 3],
    pub wave32: bool,
    pub rsrc1: u32,
    pub rsrc2: u32,
}

impl<'a> Fspx<'a> {
    pub const MAGIC: &'static [u8; 4] = b"FSPX";

    pub fn parse(data: &'a [u8]) -> Result<Self, ShaderError> {
        if data.len() < 16 || &data[0..4] != Self::MAGIC {
            return Err(ShaderError::Malformed("not an FSPX payload"));
        }
        let version = u32_at(data, 4)?;
        let count = u32_at(data, 8)?;
        let total_size = u32_at(data, 12)?;
        if total_size as usize > data.len() || total_size < 16 {
            return Err(ShaderError::Malformed("FSPX total size out of bounds"));
        }
        if 16 + (count as usize) * 16 > total_size as usize {
            return Err(ShaderError::Malformed("FSPX entry directory out of bounds"));
        }
        let mut dispatch_state = None;
        let mut kind1 = None;
        let mut program = None;
        let mut records = None;
        for i in 0..count as usize {
            let entry_offset = 16 + i * 16;
            let kind = u32_at(data, entry_offset)?;
            let offset = u32_at(data, entry_offset + 4)? as usize;
            let size = u32_at(data, entry_offset + 8)? as usize;
            if offset + size > total_size as usize {
                continue;
            }
            let entry_data = &data[offset..offset + size];
            match kind {
                0 => dispatch_state = Some(entry_data),
                1 => kind1 = Some(entry_data),
                6 => program = Some(entry_data),
                7 => records = Some(entry_data),
                _ => {}
            }
        }
        Ok(Fspx {
            version,
            count,
            total_size,
            dispatch_state,
            kind1,
            program,
            records,
        })
    }

    /// Parsed dispatch state (kind 0): (threads_x, threads_y, threads_z, wave32, rsrc1, rsrc2).
    pub fn dispatch_info(&self) -> Option<DispatchInfo> {
        let state = self.dispatch_state?;
        if state.len() < 28 {
            return None;
        }
        let w3 = u32::from_le_bytes(state[12..16].try_into().ok()?);
        let w4 = u32::from_le_bytes(state[16..20].try_into().ok()?);
        let threads_x = w3 & 0xffff;
        let threads_y = (w3 >> 16) & 0xffff;
        let threads_z = w4 & 0xffff;
        let wave32 = (w4 & (1 << 20)) != 0;
        let rsrc1 = u32::from_le_bytes(state[20..24].try_into().ok()?);
        let rsrc2 = u32::from_le_bytes(state[24..28].try_into().ok()?);
        Some(DispatchInfo {
            threads: [threads_x, threads_y, threads_z],
            wave32,
            rsrc1,
            rsrc2,
        })
    }
}

/// Checks whether a byte slice is an FSPX payload or a DXBC container containing an FSPX ACCC chunk.
pub fn inspect_fspx(bytes: &[u8]) -> Option<Fspx<'_>> {
    if bytes.starts_with(Fspx::MAGIC) {
        return Fspx::parse(bytes).ok();
    }
    if let Ok(c) = Container::parse(bytes) {
        return c.fspx();
    }
    None
}

pub(crate) fn u16_at(data: &[u8], at: usize) -> Result<u16, ShaderError> {
    data.get(at..at + 2)
        .map(|b| u16::from_le_bytes(b.try_into().expect("2 bytes")))
        .ok_or(ShaderError::Malformed("truncated"))
}

pub(crate) fn u32_at(data: &[u8], at: usize) -> Result<u32, ShaderError> {
    data.get(at..at + 4)
        .map(|b| u32::from_le_bytes(b.try_into().expect("4 bytes")))
        .ok_or(ShaderError::Malformed("truncated"))
}

pub(crate) fn u64_at(data: &[u8], at: usize) -> Result<u64, ShaderError> {
    data.get(at..at + 8)
        .map(|b| u64::from_le_bytes(b.try_into().expect("8 bytes")))
        .ok_or(ShaderError::Malformed("truncated"))
}

/// Builds a container from parts (test helper).
#[cfg(test)]
pub fn build(parts: &[(FourCC, &[u8])]) -> Vec<u8> {
    let table = HEADER_SIZE + 4 * parts.len();
    let total = table + parts.iter().map(|(_, d)| 8 + d.len()).sum::<usize>();
    let mut out = Vec::with_capacity(total);
    out.extend_from_slice(b"DXBC");
    out.extend_from_slice(&[0; 16]);
    out.extend_from_slice(&1u16.to_le_bytes());
    out.extend_from_slice(&0u16.to_le_bytes());
    out.extend_from_slice(&(total as u32).to_le_bytes());
    out.extend_from_slice(&(parts.len() as u32).to_le_bytes());
    let mut offset = table;
    for (_, d) in parts {
        out.extend_from_slice(&(offset as u32).to_le_bytes());
        offset += 8 + d.len();
    }
    for (fourcc, d) in parts {
        out.extend_from_slice(&fourcc.0);
        out.extend_from_slice(&(d.len() as u32).to_le_bytes());
        out.extend_from_slice(d);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trips_parts() {
        let bytes = build(&[(FourCC::SFI0, &[1, 2, 3, 4]), (FourCC::HASH, &[9; 20])]);
        let c = Container::parse(&bytes).unwrap();
        assert_eq!(c.parts.len(), 2);
        assert_eq!(c.part(FourCC::SFI0).unwrap().data, &[1, 2, 3, 4]);
        assert_eq!(c.part(FourCC::HASH).unwrap().data.len(), 20);
        assert!(c.part(FourCC::DXIL).is_none());
    }

    #[test]
    fn rejects_bad_input() {
        assert_eq!(Container::parse(b"nope").unwrap_err(), ShaderError::NotDxbc);
        let mut bytes = build(&[(FourCC::SFI0, &[0; 8])]);
        let last = bytes.len() - 1;
        bytes[0x20] = 0xFF; // corrupt the first part offset
        assert!(Container::parse(&bytes).is_err());
        bytes.truncate(last);
        assert!(Container::parse(&bytes).is_err());
    }

    #[test]
    fn parses_fspx_payload_and_dispatch_state() {
        let mut fspx = Vec::new();
        fspx.extend_from_slice(b"FSPX");
        fspx.extend_from_slice(&1u32.to_le_bytes()); // version
        fspx.extend_from_slice(&4u32.to_le_bytes()); // count
        let total_size = 0x140u32;
        fspx.extend_from_slice(&total_size.to_le_bytes());
        // 4 directory entries:
        // kind 0 at 0x50, size 28
        fspx.extend_from_slice(&0u32.to_le_bytes());
        fspx.extend_from_slice(&0x50u32.to_le_bytes());
        fspx.extend_from_slice(&28u32.to_le_bytes());
        fspx.extend_from_slice(&0u32.to_le_bytes());
        // kind 1 at 0x70, size 20
        fspx.extend_from_slice(&1u32.to_le_bytes());
        fspx.extend_from_slice(&0x70u32.to_le_bytes());
        fspx.extend_from_slice(&20u32.to_le_bytes());
        fspx.extend_from_slice(&0u32.to_le_bytes());
        // kind 6 at 0x100, size 48
        fspx.extend_from_slice(&6u32.to_le_bytes());
        fspx.extend_from_slice(&0x100u32.to_le_bytes());
        fspx.extend_from_slice(&48u32.to_le_bytes());
        fspx.extend_from_slice(&0u32.to_le_bytes());
        // kind 7 at 0x130, size 16
        fspx.extend_from_slice(&7u32.to_le_bytes());
        fspx.extend_from_slice(&0x130u32.to_le_bytes());
        fspx.extend_from_slice(&16u32.to_le_bytes());
        fspx.extend_from_slice(&0u32.to_le_bytes());

        // Pad to 0x50:
        fspx.resize(0x50, 0);
        // Kind 0 dispatch state (28 bytes):
        // w0, w1, w2 = 0
        fspx.extend_from_slice(&[0u8; 12]);
        // w3: numthreads_x = 8, numthreads_y = 16
        fspx.extend_from_slice(&(8u32 | (16u32 << 16)).to_le_bytes());
        // w4: numthreads_z = 1, bit 20 set = wave32
        fspx.extend_from_slice(&(1u32 | (1u32 << 20)).to_le_bytes());
        // w5: rsrc1
        fspx.extend_from_slice(&0x000000C0u32.to_le_bytes());
        // w6: rsrc2
        fspx.extend_from_slice(&0x00000400u32.to_le_bytes());

        // Pad to 0x70:
        fspx.resize(0x70, 0);
        // Kind 1: 20 bytes
        fspx.extend_from_slice(&[1u8; 20]);

        // Pad to 0x100:
        fspx.resize(0x100, 0);
        // Kind 6: 48 bytes
        fspx.extend_from_slice(&[0xaa; 48]);

        // Pad to 0x130:
        fspx.resize(0x130, 0);
        // Kind 7: 16 bytes
        fspx.extend_from_slice(&[0xbb; 16]);

        let parsed = Fspx::parse(&fspx).expect("valid FSPX");
        assert_eq!(parsed.version, 1);
        assert_eq!(parsed.count, 4);
        assert_eq!(parsed.total_size, 0x140);
        assert_eq!(parsed.program.unwrap(), &[0xaa; 48]);
        assert_eq!(parsed.records.unwrap(), &[0xbb; 16]);

        let info = parsed.dispatch_info().expect("valid dispatch info");
        assert_eq!(info.threads, [8, 16, 1]);
        assert!(info.wave32);
        assert_eq!(info.rsrc1, 0x000000C0);
        assert_eq!(info.rsrc2, 0x00000400);

        // Test inspect_fspx on bare payload
        assert!(inspect_fspx(&fspx).is_some());

        // Test inspect_fspx on DXBC container holding ACCC part
        let dxbc_container = build(&[(FourCC::ACCC, &fspx), (FourCC::HASH, &[7; 16])]);
        let container = Container::parse(&dxbc_container).expect("DXBC container");
        let inner_fspx = container.fspx().expect("ACCC holds FSPX");
        assert_eq!(inner_fspx.dispatch_info().unwrap().threads, [8, 16, 1]);

        let inspected = inspect_fspx(&dxbc_container).expect("inspect finds ACCC FSPX");
        assert_eq!(inspected.dispatch_info().unwrap().threads, [8, 16, 1]);
    }
}
