docker run `
    --rm `
    --gpus all `
    --ipc=host `
    -p 8888:8888 `
    -v ${PWD}:/app `
    schms27/computer-vision