# computer-vision
A collection of classical and modern computer vision scripts and notebooks


## Getting started


### Docker
1. Install docker
2. Create an image from dockerfile, run:
```powershell
./build.ps1
```
3. To run the container:
```powershell
./run.ps1
```
4. Navigate to the link provided by the output in the terminal, i.e.  
http://localhost:8888/lab?token=abcdefg123456

### Locally (not encouraged, its better to use docker...)
1. Install anaconda
2. Create conda environment:  
    ``` 
    conda env create -f environment.yml
    ```
3. List envs:
    ``` 
    conda info --envs 
    ```
4. Activate environment (if not already active):
    ``` 
    conda activate computer-vision
    ```
5. Start Jupyterlab:
    ``` 
    jupyter lab    
    ```
6. Navigate to the link provided by the output in the terminal, i.e.  
http://localhost:8888/lab?token=abcdefg123456

