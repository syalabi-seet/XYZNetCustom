# XYZNet-TF
Implementation of CIEXYZNet using Tensorflow 2.0

```
Model 1: 
```

## Docker setup
### Create docker image and container
```
Volumes example --> C:\your\local\directory\XYZNet:/var/lib/docker/volumes/XYZNet
Volumes include the parent directory with the sRGB2XYZ dataset folder
```
> docker run --gpus all -v {VOLUMES} --name xyznet -it nvcr.io/nvidia/tensorflow:22.05-tf2-py3

### Rename docker image
> docker image tag nvcr.io/nvidia/tensorflow:22.05-tf2-py3 xyznet:latest

### Commit changes to docker image
> docker commit xyznet xyznet:latest

### Rebuild docker image and container
> docker run --gpus all {VOLUMES} --name xyznet -it xyznet:latest

## References
- https://github.com/mahmoudnafifi/CIE_XYZ_NET