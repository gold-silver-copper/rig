use super::*;
use rig_core::message::{AssistantContent, ToolName};

fn store() -> (assert_fs::TempDir, FileConversationMemory) {
    let dir = assert_fs::TempDir::new().expect("temp dir");
    let memory = FileConversationMemory::new(dir.path().join("sessions"));
    (dir, memory)
}

#[tokio::test]
async fn appends_accumulate_and_survive_a_new_store() {
    let (_dir, memory) = store();
    let id = ConversationId::from("thread-1");
    assert!(memory.load(&id).await.expect("load").is_empty());
    memory
        .append(&id, vec![Message::user("hi"), Message::assistant("hello")])
        .await
        .expect("append");
    let call = AssistantContent::tool_call(
        "c1",
        ToolName::new("bash").expect("tool name"),
        serde_json::json!({"command": "ls"}),
    );
    memory
        .append(
            &id,
            vec![Message::Assistant {
                id: None,
                content: vec![call],
            }],
        )
        .await
        .expect("append");
    let reopened = FileConversationMemory::new(memory.dir());
    let history = reopened.load(&id).await.expect("load");
    assert_eq!(history.len(), 3);
    assert_eq!(history.first(), Some(&Message::user("hi")));
}

#[tokio::test]
async fn a_torn_final_line_is_skipped_and_later_appends_start_fresh() {
    let (_dir, memory) = store();
    let id = ConversationId::from("torn");
    memory
        .append(&id, vec![Message::user("kept")])
        .await
        .expect("append");
    let mut file = OpenOptions::new()
        .append(true)
        .open(memory.path(&id))
        .expect("open");
    file.write_all(br#"{"role":"user","con"#)
        .expect("torn write");
    assert_eq!(
        memory.load(&id).await.expect("load"),
        vec![Message::user("kept")]
    );
    memory
        .append(&id, vec![Message::user("next")])
        .await
        .expect("append");
    let history = memory.load(&id).await.expect("load");
    assert_eq!(history, vec![Message::user("kept"), Message::user("next")]);
}

#[tokio::test]
async fn a_corrupt_complete_line_is_an_error() {
    let (_dir, memory) = store();
    let id = ConversationId::from("corrupt");
    fs::create_dir_all(memory.dir()).expect("dir");
    fs::write(memory.path(&id), "not json\n").expect("write");
    assert!(matches!(
        memory.load(&id).await,
        Err(MemoryError::Backend(_))
    ));
}

#[tokio::test]
async fn replace_and_clear() {
    let (_dir, memory) = store();
    let id = ConversationId::from("r");
    memory
        .append(&id, vec![Message::user("a"), Message::user("b")])
        .await
        .expect("append");
    memory
        .replace(&id, vec![Message::user("summary")])
        .await
        .expect("replace");
    assert_eq!(
        memory.load(&id).await.expect("load"),
        vec![Message::user("summary")]
    );
    memory.clear(&id).await.expect("clear");
    assert!(memory.load(&id).await.expect("load").is_empty());
    memory.clear(&id).await.expect("clearing twice is fine");
}

#[tokio::test]
async fn any_id_maps_to_a_file_and_back() {
    let (_dir, memory) = store();
    for id in ["a/b", "../escape", "spaces and ünïcode", "%41"] {
        memory
            .append(&id.into(), vec![Message::user(id)])
            .await
            .expect("append");
        let path = memory.path(&id.into());
        assert_eq!(path.parent(), Some(memory.dir()), "{id} stays in the store");
    }
    let mut listed: Vec<String> = memory
        .conversations()
        .expect("list")
        .into_iter()
        .map(ConversationId::into_string)
        .collect();
    listed.sort();
    assert_eq!(listed, ["%41", "../escape", "a/b", "spaces and ünïcode"]);
}
