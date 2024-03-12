# otherwise use 'maturin develop --release'
set -e

maturin build --release
mv $PWD/target/wheels/tokenizers-0.14.1.dev0-cp38-cp38-manylinux_2_31_x86_64.whl $PWD/target/wheels/tokenizers-0.14.1.dev0-cp38-none-any.whl
pip install --force-reinstall $PWD/target/wheels/tokenizers-0.14.1.dev0-cp38-none-any.whl
pip install fsspec==2023.9.2
pip install --upgrade huggingface_hub datasets