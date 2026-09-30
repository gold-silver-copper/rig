use std::io::Write;

use rig_core::completion::Message;
use rig_core::id::ConversationId;
use rig_core::memory::{ConversationMemory, MemoryError};

use super::FileConversationMemory;

fn store() -> (assert_fs::TempDir, FileConversationMemory) {
    let dir = assert_fs::TempDir::new().expect("temporary directory");
    let memory = FileConversationMemory::new(dir.path().join("sessions")).expect("store");
    (dir, memory)
}

#[tokio::test]
async fn appends_load_back_in_order_across_store_instances() {
    let (dir, memory) = store();
    let id = ConversationId::from("thread-1");
    memory
        .append(&id, vec![Message::user("one"), Message::assistant("two")])
        .await
        .expect("append");
    memory
        .append(&id, vec![Message::user("three")])
        .await
        .expect("append");

    let reopened = FileConversationMemory::new(dir.path().join("sessions")).expect("reopen");
    let history = reopened.load(&id).await.expect("load");
    assert_eq!(
        history,
        vec![
            Message::user("one"),
            Message::assistant("two"),
            Message::user("three")
        ]
    );
}

#[tokio::test]
async fn an_unknown_conversation_loads_empty() {
    let (_dir, memory) = store();
    assert!(
        memory
            .load(&"nobody".into())
            .await
            .expect("load")
            .is_empty()
    );
}

#[tokio::test]
async fn clear_removes_only_that_conversation() {
    let (_dir, memory) = store();
    let (a, b) = (ConversationId::from("a"), ConversationId::from("b"));
    memory
        .append(&a, vec![Message::user("a")])
        .await
        .expect("append");
    memory
        .append(&b, vec![Message::user("b")])
        .await
        .expect("append");
    memory.clear(&a).await.expect("clear");
    memory.clear(&a).await.expect("clearing twice is fine");

    assert!(memory.load(&a).await.expect("load").is_empty());
    assert_eq!(memory.load(&b).await.expect("load").len(), 1);
}

#[tokio::test]
async fn ids_with_path_characters_stay_inside_the_directory_and_list_back() {
    let (dir, memory) = store();
    let id = ConversationId::from("../escape/ü x%.jsonl");
    memory
        .append(&id, vec![Message::user("hi")])
        .await
        .expect("append");

    let path = memory.path(&id);
    assert_eq!(path.parent(), Some(dir.path().join("sessions").as_path()));
    assert!(path.exists());
    assert_eq!(memory.conversations().expect("list"), vec![id]);
}

#[tokio::test]
async fn a_torn_line_is_skipped_and_later_appends_still_load() {
    let (_dir, memory) = store();
    let id = ConversationId::from("torn");
    memory
        .append(&id, vec![Message::user("kept")])
        .await
        .expect("append");
    // What a crash in the middle of an append leaves behind.
    let mut file = std::fs::OpenOptions::new()
        .append(true)
        .open(memory.path(&id))
        .expect("open");
    file.write_all(br#"{"role":"user","content":[{"type":"te"#)
        .expect("tear");
    drop(file);
    assert_eq!(
        memory.load(&id).await.expect("load"),
        vec![Message::user("kept")]
    );

    memory
        .append(&id, vec![Message::user("after")])
        .await
        .expect("append");
    assert_eq!(
        memory.load(&id).await.expect("load"),
        vec![Message::user("kept"), Message::user("after")]
    );
}

#[tokio::test]
async fn a_corrupt_line_is_an_error_not_silent_loss() {
    let (_dir, memory) = store();
    let id = ConversationId::from("corrupt");
    std::fs::write(
        memory.path(&id),
        "{\"role\":\"user\",\"content\":[{\"type\":\"text\",\"text\":\"a\"}]}\nnot json\n{}\n",
    )
    .expect("write");
    assert!(matches!(
        memory.load(&id).await,
        Err(MemoryError::Backend(_))
    ));
}

#[tokio::test]
async fn conversations_lists_most_recently_written_first() {
    let (_dir, memory) = store();
    memory
        .append(&"old".into(), vec![Message::user("x")])
        .await
        .expect("append");
    // File modification times can share a coarse timestamp.
    std::thread::sleep(std::time::Duration::from_millis(20));
    memory
        .append(&"new".into(), vec![Message::user("y")])
        .await
        .expect("append");
    assert_eq!(
        memory.conversations().expect("list"),
        vec![ConversationId::from("new"), ConversationId::from("old")]
    );
}
