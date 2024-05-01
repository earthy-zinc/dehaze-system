docker run -d \
-v /home/earthyzinc/code/DeepLearning/dehaze-python:/code \
-p 92:80 \
--gpus all \
--restart=always \
--network earthy-net \
--name dehaze-python \
dehaze-python:1.1

sudo docker login --username=一克土味锌 registry.cn-hangzhou.aliyuncs.com

docker tag 45bde4af752a registry.cn-hangzhou.aliyuncs.com/earthyzinc/pei-blog:1.0
docker tag fdd041ed9bab registry.cn-hangzhou.aliyuncs.com/earthyzinc/earthy-delivery:1.0
docker tag 2f66e58b09d2 registry.cn-hangzhou.aliyuncs.com/earthyzinc/youlai-mall:1.0
docker tag 637ae51cfb06 registry.cn-hangzhou.aliyuncs.com/earthyzinc/dehaze-java:1.0
docker tag 7dbeea06b5f6 registry.cn-hangzhou.aliyuncs.com/earthyzinc/kob-backend-cloud:1.0
docker tag [ImageId] registry.cn-hangzhou.aliyuncs.com/earthyzinc/dehaze-python:1.0


docker push registry.cn-hangzhou.aliyuncs.com/earthyzinc/pei-blog:1.0
docker push registry.cn-hangzhou.aliyuncs.com/earthyzinc/earthy-delivery:1.0
docker push registry.cn-hangzhou.aliyuncs.com/earthyzinc/youlai-mall:1.0
docker push registry.cn-hangzhou.aliyuncs.com/earthyzinc/dehaze-java:1.0
docker push registry.cn-hangzhou.aliyuncs.com/earthyzinc/kob-backend-cloud:1.0
docker push registry.cn-hangzhou.aliyuncs.com/earthyzinc/dehaze-python:1.0

docker run -d --name pei-blog --restart always -p 93:80 registry.cn-hangzhou.aliyuncs.com/earthyzinc/pei-blog:1.0
docker run -d --name earthy-delivery --restart always -p 85:80 registry.cn-hangzhou.aliyuncs.com/earthyzinc/earthy-delivery:1.0
docker run -d --name youlai-mall --restart always -p 94:9999 registry.cn-hangzhou.aliyuncs.com/earthyzinc/youlai-mall:1.0
docker run -d --name dehaze-java --restart always -p 91:80 registry.cn-hangzhou.aliyuncs.com/earthyzinc/dehaze-java:1.0
docker run -d --name kob-backend-cloud --restart always -p 90:3000 registry.cn-hangzhou.aliyuncs.com/earthyzinc/kob-backend-cloud:1.0
docker run -d --name dehaze-python --restart always -p 92:80 registry.cn-hangzhou.aliyuncs.com/earthyzinc/dehaze-python:1.0

