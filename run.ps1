docker run `
    --rm `
    --gpus all `
    --ipc=host `
    -p 8888:8888 `
    -p 6006:6006 `
    -v ${PWD}:/app `
    schms27/computer-vision