//! A native Bevy plugin with one hotpatchable system and persistent host state.
use bevy_app::{App, Plugin, Update};
use bevy_ecs::{prelude::*, system::SystemParamFunction};

mod behavior;

#[derive(Resource, Default)]
pub struct NativeState {
    pub ticks: u64,
    pub label: String,
}

pub struct NativePlugin;
impl Plugin for NativePlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<NativeState>().add_systems(Update, tick);
    }
}

fn tick(mut state: ResMut<NativeState>) {
    state.ticks += 1;
    behavior::update(&mut state);
}

/// Address of exactly the trampoline Bevy uses for this system.
pub fn system_address() -> u64 {
    fn address<M, F: SystemParamFunction<M>>(_: F) -> u64 {
        subsecond::HotFn::current(<F as SystemParamFunction<M>>::run)
            .ptr_address()
            .0
    }
    address(tick)
}

/// Fingerprint of the fixed Rust ABI boundary, checked before installing a patch.
pub fn abi() -> u64 {
    let source = concat!(
        env!("SPROUT_RUSTC"),
        include_str!("lib.rs"),
        include_str!("../Cargo.toml"),
        include_str!("../../Cargo.toml"),
        include_str!("../../Cargo.lock")
    );
    source.bytes().fold(0xcbf29ce484222325, |h, b| {
        (h ^ u64::from(b)).wrapping_mul(0x100000001b3)
    })
}
