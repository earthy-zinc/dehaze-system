# Kafka 练习

## 概述

## 生产者

### 生产者消息发送流程

在消息发送的过程中，涉及到了两个线程，main线程和sender线程。我们在main线程中创建了一个双端队列RecordAccumulator，main线程将消息发送给RecordAccumulator这个双端队列，而发送线程sender不断地从双端队列中拉取消息到kafka的Broker
