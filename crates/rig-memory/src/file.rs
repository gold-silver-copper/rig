//! A [`ConversationMemory`] that keeps each conversation in an append-only
//! JSON Lines file, so history survives process restarts.
//!
//! Every message is one line of `serde_json`. An append writes and syncs its
//! lines before returning; a load reads them back in order. A line cut short
//! by a crash is skipped, so a torn write loses only the message it was
//! writing. Wrap the store in [`PolicyMemory`](crate::PolicyMemory) or
//! [`CompactingMemory`](crate::CompactingMemory) to shape what loads return;
//! the file keeps everything.
//!
//! ```no_run
//! # async fn run() -> Result<(), rig_core::memory::MemoryError> {
//! use rig_core::{completion::Message, memory::ConversationMemory};
//! use rig_memory::FileConversationMemory;
//!
//! let memory = FileConversationMemory::new("sessions")?;
//! memory.append(&"thread-1".into(), vec![Message::user("Hello")]).await?;
//! let history = memory.load(&"thread-1".into()).await?;
//! assert_eq!(history.len(), 1);
//! # Ok(()) }
//! ```

use std::fs::{File, OpenOptions};
use std::io::{ErrorKind, Write};
use std::path::{Path, PathBuf};
use std::sync::Mutex;

use rig_core::completion::Message;
use rig_core::id::ConversationId;
use rig_core::memory::{ConversationMemory, MemoryError};
use rig_core::wasm_compat::WasmBoxedFuture;

const EXTENSION: &str = "jsonl";

/// Conversation history stored as one JSON Lines file per conversation in a
/// directory.
///
/// File names are the percent-encoded conversation ID, so any ID maps to one
/// file and back. Operations use blocking file I/O inside the returned
/// futures and are serialized by an internal lock, so one store must not be
/// shared by processes writing the same conversation.
#[derive(Debug)]
pub struct FileConversationMemory {
    dir: PathBuf,
    lock: Mutex<()>,
}

impl FileConversationMemory {
    /// A store in `dir`, which is created if missing.
    ///
    /// # Errors
    ///
    /// [`MemoryError::Backend`] when the directory cannot be created.
    pub fn new(dir: impl Into<PathBuf>) -> Result<Self, MemoryError> {
        let dir = dir.into();
        std::fs::create_dir_all(&dir).map_err(MemoryError::backend)?;
        Ok(Self {
            dir,
            lock: Mutex::new(()),
        })
    }

    /// The directory holding the conversation files.
    pub fn dir(&self) -> &Path {
        &self.dir
    }

    /// The file that holds `conversation_id`.
    pub fn path(&self, conversation_id: &ConversationId) -> PathBuf {
        self.dir
            .join(format!("{}.{EXTENSION}", encode(conversation_id.as_str())))
    }

    /// Every conversation with a file in the directory, most recently
    /// modified first.
    ///
    /// # Errors
    ///
    /// [`MemoryError::Backend`] when the directory cannot be read.
    pub fn conversations(&self) -> Result<Vec<ConversationId>, MemoryError> {
        let mut found = Vec::new();
        for entry in std::fs::read_dir(&self.dir).map_err(MemoryError::backend)? {
            let entry = entry.map_err(MemoryError::backend)?;
            let path = entry.path();
            if path.extension().and_then(|e| e.to_str()) != Some(EXTENSION) {
                continue;
            }
            let Some(id) = path
                .file_stem()
                .and_then(|stem| stem.to_str())
                .and_then(decode)
            else {
                continue;
            };
            let modified = entry.metadata().and_then(|m| m.modified()).ok();
            found.push((modified, ConversationId::new(id)));
        }
        found.sort_by_key(|(modified, _)| std::cmp::Reverse(*modified));
        Ok(found.into_iter().map(|(_, id)| id).collect())
    }

    fn guard(&self) -> Result<std::sync::MutexGuard<'_, ()>, MemoryError> {
        self.lock
            .lock()
            .map_err(|_| MemoryError::Internal("file memory lock poisoned".into()))
    }

    fn read(&self, conversation_id: &ConversationId) -> Result<Vec<Message>, MemoryError> {
        let _guard = self.guard()?;
        let bytes = match std::fs::read(self.path(conversation_id)) {
            Ok(bytes) => bytes,
            Err(error) if error.kind() == ErrorKind::NotFound => return Ok(Vec::new()),
            Err(error) => return Err(MemoryError::backend(error)),
        };
        let lines: Vec<&[u8]> = bytes.split(|byte| *byte == b'\n').collect();
        let last = lines.len().saturating_sub(1);
        let mut messages = Vec::with_capacity(lines.len());
        for (index, line) in lines.into_iter().enumerate() {
            if line.iter().all(u8::is_ascii_whitespace) {
                continue;
            }
            match serde_json::from_slice(line) {
                Ok(message) => messages.push(message),
                // A crash mid-append leaves one line cut short: it ends early,
                // possibly inside a UTF-8 sequence if it is the file's last.
                Err(error) if error.is_eof() || index == last => {}
                Err(error) => {
                    return Err(MemoryError::backend(format!(
                        "{} line {}: {error}",
                        self.path(conversation_id).display(),
                        index + 1
                    )));
                }
            }
        }
        Ok(messages)
    }

    fn write(
        &self,
        conversation_id: &ConversationId,
        messages: &[Message],
    ) -> Result<(), MemoryError> {
        let mut buffer = Vec::new();
        for message in messages {
            serde_json::to_writer(&mut buffer, message).map_err(MemoryError::backend)?;
            buffer.push(b'\n');
        }
        let _guard = self.guard()?;
        let path = self.path(conversation_id);
        let mut file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&path)
            .map_err(MemoryError::backend)?;
        // A previous torn write may have left a partial final line; start a
        // fresh line so the new messages stay readable.
        if file.metadata().map_err(MemoryError::backend)?.len() > 0 && !ends_with_newline(&path)? {
            buffer.insert(0, b'\n');
        }
        file.write_all(&buffer).map_err(MemoryError::backend)?;
        file.sync_data().map_err(MemoryError::backend)
    }

    fn remove(&self, conversation_id: &ConversationId) -> Result<(), MemoryError> {
        let _guard = self.guard()?;
        match std::fs::remove_file(self.path(conversation_id)) {
            Err(error) if error.kind() != ErrorKind::NotFound => Err(MemoryError::backend(error)),
            _ => Ok(()),
        }
    }
}

impl ConversationMemory for FileConversationMemory {
    fn load<'a>(
        &'a self,
        conversation_id: &'a ConversationId,
    ) -> WasmBoxedFuture<'a, Result<Vec<Message>, MemoryError>> {
        Box::pin(async move { self.read(conversation_id) })
    }

    fn append<'a>(
        &'a self,
        conversation_id: &'a ConversationId,
        messages: Vec<Message>,
    ) -> WasmBoxedFuture<'a, Result<(), MemoryError>> {
        Box::pin(async move { self.write(conversation_id, &messages) })
    }

    fn clear<'a>(
        &'a self,
        conversation_id: &'a ConversationId,
    ) -> WasmBoxedFuture<'a, Result<(), MemoryError>> {
        Box::pin(async move { self.remove(conversation_id) })
    }
}

fn ends_with_newline(path: &Path) -> Result<bool, MemoryError> {
    use std::io::{Read, Seek, SeekFrom};
    let mut file = File::open(path).map_err(MemoryError::backend)?;
    file.seek(SeekFrom::End(-1)).map_err(MemoryError::backend)?;
    let mut last = [0u8];
    file.read_exact(&mut last).map_err(MemoryError::backend)?;
    Ok(last[0] == b'\n')
}

/// Percent-encode every byte outside `[A-Za-z0-9_-]`, so no ID can name a
/// path outside the directory and distinct IDs get distinct names.
fn encode(id: &str) -> String {
    let mut out = String::with_capacity(id.len());
    for byte in id.bytes() {
        if byte.is_ascii_alphanumeric() || byte == b'_' || byte == b'-' {
            out.push(char::from(byte));
        } else {
            out.push_str(&format!("%{byte:02X}"));
        }
    }
    out
}

fn decode(name: &str) -> Option<String> {
    let bytes = name.as_bytes();
    let mut out = Vec::with_capacity(bytes.len());
    let mut index = 0;
    while let Some(&byte) = bytes.get(index) {
        if byte == b'%' {
            let hex = name.get(index + 1..index + 3)?;
            out.push(u8::from_str_radix(hex, 16).ok()?);
            index += 3;
        } else {
            out.push(byte);
            index += 1;
        }
    }
    String::from_utf8(out).ok()
}

#[cfg(test)]
mod tests;
