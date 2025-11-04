## Passo a passo:

### Link simbólico:
Criar o link simbolico da /home de cada ums dos pcs para `/home/lucas`

```
ls -l <caminho> /home/lucas
```

### Criar o container e rodar  
```
podman build . 

podman run \
    --name PIBIC_GPU \
    --device nvidia.com/gpu=all \
    --security-opt=label=disable \
    --env TF_FORCE_GPU_ALLOW_GROWTH=true \
    --volume /home/lucas/DeepLearning:/home/lucas/DeepLearning \
    --network host \
    --tty --interactive \
    ID_CONTAINER
```
