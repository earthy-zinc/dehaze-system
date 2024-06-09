#!/bin/bash

docker_image_names=(
#  "dehaze-java"
#  "earthy-delivery"
#  "pei-blog"
   "blog"
#  "kob-backend-cloud"
#  "mall-system"
#  "mall-gateway"
#  "mall-auth"
#  "mall-ums"
#  "mall-sms"
#  "mall-pms"
#  "mall-oms"
#  "mall-lab"
)

for image_name in "${docker_image_names[@]}"
do
  docker save "${image_name}":1.0 | bzip2 | pv | ssh earthyzinc@192.168.31.2 'cat | docker load'
done
