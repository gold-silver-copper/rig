// Edit this function while Sprout runs. The host retains NativeState across patches.
pub fn update(state: &mut super::NativeState) {
    state.label = "native: seedling".into();
}
