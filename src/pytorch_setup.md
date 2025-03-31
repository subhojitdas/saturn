This document is 54% show-off , 36% for my future reference and rest is for random-ness.
It will try to explain the steps to build pytorch from source code and setup debugger to get to cpp files.

Install some random library,

```
xcode-select --install

/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

brew install cmake ninja libomp
```

Clone pytorch repo,

```
git clone --recursive https://github.com/pytorch/pytorch.git
cd pytorch
```

Install conda,

Install dependencies,

```
pip install -r requirements.txt
```

setup some env var
```
export CMAKE_PREFIX_PATH=${CONDA_PREFIX:-"$(dirname $(which conda))/../"}
export USE_MPS=1
export USE_CUDA=0  # CUDA not supported on Apple Silicon
```

Verify installation,
```
import torch
print(torch.__version__)
2.8.0a0+git3a8171e # will get something like this where the end in the HEAD of the pytorch repo
```

