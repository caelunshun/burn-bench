curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- --default-toolchain nightly
source /root/.cargo/env
export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/usr/local/lib/python3.10/dist-packages/torch/lib"
export LIBTORCH_USE_PYTORCH=1
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
cargo run --release
