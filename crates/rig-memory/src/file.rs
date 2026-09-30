//! A [`ConversationMemory`] kept on disk: one JSON Lines file per
//! conversation, one message per line, appended as the conversation grows.
//! Native targets only, behind the `file` feature.
//!
//! ```no_run
//! # async fn run() -> Result<(), rig_core::memory::MemoryError> {
//! use rig_core::completion::Message;
//! use rig_memory::{ConversationMemory, FileConversationMemory};
//!
//! let memory = FileConversationMemory::new("sessions");
//! memory.append(&"thread-1".into(), vec![Message::user("hello")]).await?;
//! assert_eq!(memory.load(&"thread-1".into()).await?.len(), 1);
//! # Ok(()) }
//! ```

use std::fs::{self, File, OpenOptions};
use std::io::{self, BufRead, BufReader, Write};
use std::path::{Path, PathBuf};
use std::sync::Mutex;
use std::time::SystemTime;

use rig_core::completion::Message;
use rig_core::id::ConversationId;
use rig_core::memory::{ConversationMemory, MemoryError};
use rig_core::wasm_compat::WasmBoxedFuture;

const EXTENSION: &str = "jsonl";

/// Conversations stored as `<dir>/<id>.jsonl`.
///
/// Appends write whole lines and sync them before returning, so a crash can
/// at worst leave a torn final line, which [`load`](ConversationMemory::load)
/// skips. File I/O is synchronous inside the returned futures. Writers in one
/// process are serialized; separate processes must not append to the same
/// conversation concurrently.
#[derive(Debug)]
pub struct FileConversationMemory {
    dir: PathBuf,
    lock: Mutex<()>,
}

impl FileConversationMemory {
    /// A store in `dir`, which is created on the first write.
    pub fn new(dir: impl Into<PathBuf>) -> Self {
        Self {
            dir: dir.into(),
            lock: Mutex::new(()),
        }
    }

    /// The directory holding the conversation files.
    pub fn dir(&self) -> &Path {
        &self.dir
    }

    /// The file backing `conversation_id`. Characters other than ASCII
    /// letters, digits, `-` and `_` are percent-encoded, so any id is a safe
    /// file name.
    pub fn path(&self, conversation_id: &ConversationId) -> PathBuf {
        self.dir
            .join(format!("{}.{EXTENSION}", encode(conversation_id.as_str())))
    }

    /// Every stored conversation, most recently written first.
    pub fn conversations(&self) -> Result<Vec<ConversationId>, MemoryError> {
        let entries = match fs::read_dir(&self.dir) {
            Ok(entries) => entries,
            Err(error) if error.kind() == io::ErrorKind::NotFound => return Ok(Vec::new()),
            Err(error) => return Err(MemoryError::backend(error)),
        };
        let mut found: Vec<(SystemTime, ConversationId)> = Vec::new();
        for entry in entries {
            let path = entry.map_err(MemoryError::backend)?.path();
            if path.extension().and_then(|ext| ext.to_str()) != Some(EXTENSION) {
                continue;
            }
            let Some(id) = path
                .file_stem()
                .and_then(|stem| stem.to_str())
                .and_then(decode)
            else {
                continue;
            };
            let modified = fs::metadata(&path)
                .and_then(|metadata| metadata.modified())
                .unwrap_or(SystemTime::UNIX_EPOCH);
            found.push((modified, ConversationId::new(id)));
        }
        found.sort_by(|a, b| b.0.cmp(&a.0).then_with(|| a.1.as_str().cmp(b.1.as_str())));
        Ok(found.into_iter().map(|(_, id)| id).collect())
    }

    /// Replace the whole history of `conversation_id` atomically, for
    /// example with a compacted one. Readers see the old or the new history,
    /// never a mix.
    pub async fn replace(
        &self,
        conversation_id: &ConversationId,
        messages: Vec<Message>,
    ) -> Result<(), MemoryError> {
        let _guard = self.guard()?;
        fs::create_dir_all(&self.dir).map_err(MemoryError::backend)?;
        let path = self.path(conversation_id);
        let tmp = path.with_extension(format!("{EXTENSION}.tmp"));
        let mut file = File::create(&tmp).map_err(MemoryError::backend)?;
        write_lines(&mut file, &messages)?;
        fs::rename(&tmp, &path).map_err(MemoryError::backend)
    }

    fn guard(&self) -> Result<std::sync::MutexGuard<'_, ()>, MemoryError> {
        self.lock
            .lock()
            .map_err(|error| MemoryError::Internal(error.to_string()))
    }

    fn read(&self, conversation_id: &ConversationId) -> Result<Vec<Message>, MemoryError> {
        let file = match File::open(self.path(conversation_id)) {
            Ok(file) => file,
            Err(error) if error.kind() == io::ErrorKind::NotFound => return Ok(Vec::new()),
            Err(error) => return Err(MemoryError::backend(error)),
        };
        let mut messages = Vec::new();
        let mut reader = BufReader::new(file);
        let mut line = String::new();
        loop {
            line.clear();
            if reader.read_line(&mut line).map_err(MemoryError::backend)? == 0 {
                break;
            }
            let complete = line.ends_with('\n');
            if line.trim().is_empty() {
                continue;
            }
            match serde_json::from_str(&line) {
                Ok(message) => messages.push(message),
                // A final line without its newline is a write the process did
                // not finish; everything before it is intact.
                Err(_) if !complete => break,
                Err(error) => return Err(MemoryError::backend(error)),
            }
        }
        Ok(messages)
    }
}

fn write_lines(file: &mut File, messages: &[Message]) -> Result<(), MemoryError> {
    let mut buffer = Vec::new();
    for message in messages {
        serde_json::to_writer(&mut buffer, message).map_err(MemoryError::backend)?;
        buffer.push(b'\n');
    }
    file.write_all(&buffer).map_err(MemoryError::backend)?;
    file.sync_data().map_err(MemoryError::backend)
}

impl ConversationMemory for FileConversationMemory {
    fn load<'a>(
        &'a self,
        conversation_id: &'a ConversationId,
    ) -> WasmBoxedFuture<'a, Result<Vec<Message>, MemoryError>> {
        Box::pin(async move {
            let _guard = self.guard()?;
            self.read(conversation_id)
        })
    }

    fn append<'a>(
        &'a self,
        conversation_id: &'a ConversationId,
        messages: Vec<Message>,
    ) -> WasmBoxedFuture<'a, Result<(), MemoryError>> {
        Box::pin(async move {
            if messages.is_empty() {
                return Ok(());
            }
            let _guard = self.guard()?;
            fs::create_dir_all(&self.dir).map_err(MemoryError::backend)?;
            let mut file = OpenOptions::new()
                .create(true)
                .read(true)
                .append(true)
                .open(self.path(conversation_id))
                .map_err(MemoryError::backend)?;
            drop_torn_tail(&mut file).map_err(MemoryError::backend)?;
            write_lines(&mut file, &messages)
        })
    }

    fn clear<'a>(
        &'a self,
        conversation_id: &'a ConversationId,
    ) -> WasmBoxedFuture<'a, Result<(), MemoryError>> {
        Box::pin(async move {
            let _guard = self.guard()?;
            match fs::remove_file(self.path(conversation_id)) {
                Err(error) if error.kind() != io::ErrorKind::NotFound => {
                    Err(MemoryError::backend(error))
                }
                _ => Ok(()),
            }
        })
    }
}

/// Cuts a final line an interrupted write left without its newline, so new
/// messages do not join it.
fn drop_torn_tail(file: &mut File) -> io::Result<()> {
    use std::io::{Read, Seek, SeekFrom};
    let length = file.metadata()?.len();
    let mut end = length;
    let mut chunk = [0u8; 4096];
    while end > 0 {
        let start = end.saturating_sub(chunk.len() as u64);
        let size = usize::try_from(end - start).unwrap_or(chunk.len());
        let window = chunk.get_mut(..size).unwrap_or_default();
        file.seek(SeekFrom::Start(start))?;
        file.read_exact(window)?;
        if let Some(newline) = window.iter().rposition(|&byte| byte == b'\n') {
            end = start + newline as u64 + 1;
            break;
        }
        end = start;
    }
    if end < length {
        file.set_len(end)?;
    }
    Ok(())
}

fn encode(id: &str) -> String {
    let mut name = String::with_capacity(id.len());
    for byte in id.bytes() {
        if byte.is_ascii_alphanumeric() || byte == b'-' || byte == b'_' {
            name.push(char::from(byte));
        } else {
            name.push_str(&format!("%{byte:02X}"));
        }
    }
    name
}

fn decode(name: &str) -> Option<String> {
    let mut bytes = Vec::with_capacity(name.len());
    let mut rest = name.as_bytes();
    while let Some((&first, tail)) = rest.split_first() {
        if first == b'%' {
            let hex = std::str::from_utf8(tail.get(..2)?).ok()?;
            bytes.push(u8::from_str_radix(hex, 16).ok()?);
            rest = tail.get(2..)?;
        } else {
            bytes.push(first);
            rest = tail;
        }
    }
    String::from_utf8(bytes).ok()
}

#[cfg(test)]
mod tests;
