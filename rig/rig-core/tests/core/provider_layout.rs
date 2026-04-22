//! Provider-local test layout checks to keep the provider-first tree normalized.

use anyhow::{Context, Result, anyhow};
use std::fs;
use std::path::PathBuf;

fn provider_dirs() -> Result<Vec<PathBuf>> {
    let tests_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests");
    let mut dirs = fs::read_dir(tests_dir)?
        .filter_map(|entry| entry.ok())
        .map(|entry| entry.path())
        .filter(|path| path.is_dir())
        .filter(|path| {
            !matches!(
                path.file_name().and_then(|name| name.to_str()),
                Some("common" | "core" | "data")
            )
        })
        .collect::<Vec<_>>();
    dirs.sort();
    Ok(dirs)
}

#[test]
fn provider_local_tests_do_not_use_legacy_prefixes() -> Result<()> {
    for dir in provider_dirs()? {
        let provider = dir
            .file_name()
            .and_then(|name| name.to_str())
            .context("provider directory name should be valid utf-8")?;

        for entry in fs::read_dir(&dir)?.filter_map(|entry| entry.ok()) {
            let path = entry.path();
            if path.extension().and_then(|ext| ext.to_str()) != Some("rs") {
                continue;
            }

            let stem = path
                .file_stem()
                .and_then(|stem| stem.to_str())
                .ok_or_else(|| anyhow!("test file should have a valid utf-8 stem"))?;
            if stem == "mod" {
                continue;
            }

            assert!(
                !stem.starts_with(&format!("{provider}_")),
                "provider-local test file {:?} still uses a provider prefix",
                path
            );
            assert!(
                !stem.starts_with("agent_with_"),
                "provider-local test file {:?} still uses an agent_with_* name",
                path
            );
        }
    }
    Ok(())
}
