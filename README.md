# Custom XYZNet

## Docker setup
### Create docker image and container
```
Volumes example --> C:\your\local\directory\XYZNet:/var/lib/docker/volumes/XYZNet
Volumes include the parent directory and the sRGB2XYZ dataset folder
```
> docker run --gpus all {VOLUMES} --name xyznet -it nvcr.io/nvidia/tensorflow:22.05-tf2-py3

### Rename docker image
> docker image tag nvcr.io/nvidia/tensorflow:22.05-tf2-py3 xyznet:latest

### Commit changes to docker image
> docker commit xyznet xyznet:latest

### Rebuild docker image and container
> docker run --gpus all {VOLUMES} --name xyznet -it xyznet:latest