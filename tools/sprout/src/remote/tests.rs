use super::*;
use bevy::ecs::system::RunSystemOnce;

#[test]
fn brp_registration_dispatch_and_result() -> Result<()> {
    let mut world = World::new();
    let extensions = Extensions::default();
    world.insert_resource(extensions.clone());
    let definition = Some(
        json!({"name":"uppercase", "description":"Uppercase text", "parameters":{"type":"object"}}),
    );
    assert!(
        world
            .run_system_once_with(register, definition.clone())
            .map_err(|e| anyhow!("{e:?}"))?
            .is_ok()
    );
    assert!(
        world
            .run_system_once_with(register, definition)
            .map_err(|e| anyhow!("{e:?}"))?
            .is_err()
    );
    assert_eq!(extensions.definitions()?.len(), 1);
    let caller = std::thread::spawn(move || extensions.call("uppercase", json!({"text":"sprout"})));
    let deadline = std::time::Instant::now() + Duration::from_secs(2);
    let id = loop {
        let calls = world
            .run_system_once_with(poll, Some(json!({"name":"uppercase"})))
            .map_err(|e| anyhow!("{e:?}"))?
            .map_err(|e| anyhow!("{e:?}"))?;
        if let Some(id) = calls
            .get(0)
            .and_then(|c| c.get("id"))
            .and_then(Value::as_u64)
        {
            break id;
        }
        ensure!(std::time::Instant::now() < deadline, "call was not queued");
        std::thread::yield_now();
    };
    assert!(
        world
            .run_system_once_with(result, Some(json!({"id":id, "text":"SPROUT"})))
            .map_err(|e| anyhow!("{e:?}"))?
            .is_ok()
    );
    let output = caller.join().map_err(|_| anyhow!("worker panicked"))??;
    assert_eq!(output, "SPROUT");
    assert!(
        world
            .run_system_once_with(result, Some(json!({"id":id, "text":"duplicate"})))
            .map_err(|e| anyhow!("{e:?}"))?
            .is_err()
    );
    assert!(
        world
            .resource::<Extensions>()
            .call("unknown", Value::Null)
            .is_err()
    );
    Ok(())
}

#[test]
fn reserved_tool_is_rejected() -> Result<()> {
    let mut world = World::new();
    world.init_resource::<Extensions>();
    assert!(
        world
            .run_system_once_with(
                register,
                Some(json!({"name":"shell", "description":"x", "parameters":{}}))
            )
            .map_err(|e| anyhow!("{e:?}"))?
            .is_err()
    );
    Ok(())
}
