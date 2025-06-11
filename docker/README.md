```bash
cd docker
docker build -t cuda-python:python3.10-cuda14.1 .
docker tag cuda-python:python3.10-cuda14.1 gbarbadillo/cuda-python:python3.10-cuda14.1
docker push gbarbadillo/cuda-python:python3.10-cuda14.1
```