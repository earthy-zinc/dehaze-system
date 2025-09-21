
docker run --rm -v es-plugins:/target -v E:/ProgramProject/dehaze-java-cloud/script/docker/elk/elasticsearch/plugins:/source alpine cp -a /source/. /target/

docker run --rm -v kibana-config:/target -v E:/ProgramProject/dehaze-java-cloud/script/docker/elk/kibana/config:/source alpine cp -a /source/. /target/

docker run --rm -v logstash-pipeline:/target -v E:/ProgramProject/dehaze-java-cloud/script/docker/elk/logstash/pipeline:/source alpine cp -a /source/. /target/

docker run --rm -v logstash-config:/target -v E:/ProgramProject/dehaze-java-cloud/script/docker/elk/logstash/config:/source alpine cp -a /source/. /target/

docker run --rm -v rocketmq-broker-config:/target -v E:/ProgramProject/dehaze-java-cloud/script/docker/rocketmq/broker1/conf:/source alpine cp -a /source/. /target/
