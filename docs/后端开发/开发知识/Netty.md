# Netty

## BIO、NIO、AIO

IO操作，指的是和输入输出相关的操作，可以用来读写外部数据和文件。

* BIO即Blocking Input and Output，同步阻塞IO模式。其流程为：发起请求、阻塞等待、处理完成。
* NIO即New Input and
  Output，同步非阻塞模式，与BIO的区别在于数据打包和传输的方式不同，原来的IO是以流的形式处理数据，而NIO是以块的形式处理数据。面向流的IO一次处理一个字节的数据。而面向块的数据处理，每次处理一个数据块。
* AIO即Asynchronous IO，异步非阻塞模式，在NIO的基础上引入的异步通道的概念。

