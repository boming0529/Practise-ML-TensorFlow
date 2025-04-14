## using

if docker-compose v1

```
docker-compose -f docker_resNet50/docker-compose.yml up -d
```

docker-compose v2

```
docker compose -f docker_resNet50/docker-compose.yml up -d
```

project/
├── docker_resNet50/
│ ├── Dockerfile
│ └── docker-compose.yml
├── ResNet50_Cnn.py
└── tool/
