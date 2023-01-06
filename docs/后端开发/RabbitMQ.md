# 一、 简介

RabbitMQ即（Rabbit Message Queue)消息队列中间件，用于接收和转发消息。具体流程是接收消息-> 存储-> 转发二进制数据。

相关概念：

* 生产：Producing，发送消息，生产者
* 消费：Consuming，接收消息，消费者
* 队列：Queue，本质上是一个消息缓冲区，生产者发送

## 1、发送消息

```java
public class Send{
    private final static String QUEUE_NAME = "hello";
    public static void main(String[] argv) throws Exception{
        ConnectionFactory factory = new ConnectionFactory();
        factory.setHost("设置RabbitMQ的服务器IP");
        factory.setUsername("安装时设置的管理员用户名");
        factory.setPassword("安装时设置的管理员密码");
        
        try(Connection connection = factory.newConnection();
           Channel channel = connection.createChannel()){
            // 表示生成一个队列。参数作用：1. 队列名称 2. 队列里面的消息是否持久化，默认情况下消息存储在内存中 3. 该队列是否只提供一个消费者进行消费，是否进行共享，如果为true表示可以由多个消费者消费 4. 是否自动删除，最后一个消费者断开连接以后，该队列是否会自动删除。
            channel.queueDeclare(QUEUE_NAME, false, false, false,  null);
            String message = "hello world";
            //表示发送一个消息。参数作用：1. 发送到哪一个交换机 2. 路由的key是哪一个 3. 其他参数信息 4. 发送消息的消息体
            channel.basicPublish("", QUEUE_NAME, null, message.getBytes());
            System.out.println("发送了消息");
        }
    }
}
```

## 2、接收消息

```java
public class Recv{
    private final static String QUEUE_NAME = "hello";
    public static void main(String[] argv) throws Exception {
        // 建立与RabbitMQ的连接，这里略
        Connection connection = factory.newConnection();
        Channel channel = connection.createChannel();
        
        // 声明从哪一个队列中接受消息
        channel.queueDeclare(QUEUE_NAME, false, false, false, null);
        
        // 定义接受信息和取消接受信息的回调窗口，目的是当接收到一条信息是，进行一些操作，比如可以在控制台打印输出
        DeliverCallback deliverCallback = (consumerTag, delivery) -> {
            String message = new String(delivery.getBody(), "UTF_8");
        }
        CancelCallback cancelCallback = (consumerTag) -> {
            System.out.println("消费被中断");
        }
        // 管道接收信息
        channel.basicConsume(QUEUE_NAME, true, deliverCallback, cancelCallback);
    }
}
```

# 二、工作队列/任务队列

对于生产者：避免必须立刻执行资源紧张的任务

对于消息队列：生产者想要做的任务会被封装成一个消息放在队列里

对于消费者：当你有多个工人时，这些任务就会被轮询分配个不同的工人

这些思想也被应用于那些需要处理不能再一个很短的HTTP请求窗口期间完成的复杂任务的网页程序中。

## 1、消息应答机制

### 轮流调度

默认情况下，RabbitMQ会把消息按照顺序传给下一个消费者，平均来看，每个消费者拿到的信息数量都是相同的，这种分发机制称为轮询调度。

### 消息确认

一般情况下，RabbitMQ把信息转发给消费者后就会马上把这个任务再队列里面删掉。而完成任务需要一定事件，如果消费者未能够完成任务，我们就丢失了这个信息。如果不希望丢失消息，那么我们需要开启RabbitMQ中的信息确认功能，消费者再接受并处理完成一个任务后，会给RabbitMQ发送一个确认消息。告诉他任务已经完成，如果RabbitMQ没有收到确认消息，就判定消息没有被处理，从而把这个消息再放进队列中，让其他消费者去处理。