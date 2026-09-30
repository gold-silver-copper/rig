//! `rig-pi` starts a supervisor that runs the agent as a child process (see
//! `supervisor.rs`); `rig-pi eval` runs the evals headless.

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let code = match args.first().map(String::as_str) {
        Some("--child") => rig_pi::run_child(&args[1..]),
        Some("eval") => rig_pi::plugins::evals::run(&args[1..]),
        _ => rig_pi::supervisor::run(&args),
    };
    std::process::exit(code);
}
