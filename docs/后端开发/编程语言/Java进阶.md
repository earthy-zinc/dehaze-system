# Java进阶知识

## 一、并发

### 1、并发问题三要素

多线程是一种并发的操作，并发操作会出现一些问题，这些问题有三个特点：

* 可见性：由CPU的缓存引起，在多CPU的情况下，每个CPU都有自己的高速缓存，由于CPU和主存之间速度不匹配的问题，其中一个CPU对共享变量的修改并不会立即写回主存中，此时另一个线程从主存中读取该值并加载到另一个CPU高速缓存中。另一个线程并没有看到该变量已经被修改过了，这就是线程之间共享数据的可见性问题。
* 原子性：一个操作或多个操作，是不能被任何因素打断的执行。否则就不执行。由于操作系统分时复用CPU，就会导致这些操作可能会被打断。
* 有序性：程序执行按照代码的先后顺序执行。编译程序会优化指令执行次序，会导致有序性被破坏。发生指令重新排序
  * 编译器优化的重排序，编译器在不改变单线程程序语义情况下，重新安排语句执行次序
  * 指令级并行重排序，如果不存在数据依赖性，处理器可以改变语句对应的机器指令的执行顺序
  * 内存系统重排序，处理器有高速缓存和读写缓冲区，写回内存时间在微观上不确定，加载和存储操作可能是在乱序进行

### 2、Java解决并发问题方法

#### 1）关键字

##### volatile

保证变量的内存可见性。

synchronized负责给线程加锁，一把锁只能同时被一个线程获取，没有获取到锁的线程只能等待。每个实例对对应有自己的一把锁，不同实例之间互不影响。但是锁的如果时类级别的化，锁class、锁static，所有该类的实例对象共用一把锁。用synchronized修饰的方法，无论方法是正常执行完毕还是抛出异常，都会释放锁。

###### 对象锁

使用synchronized把代码块包裹起来，手动制定锁定对象，可以是this，也可以是自定义的锁

```java
public class SynchronizedObjectLock implements Runnable{
    static SynchronizedObjectLock instance = new SynchronizedObjectLock();
    @Override
    public void run(){
        synchronized (this){
            System.out.println(Thread.currentThread().getName());
            try{Thread.sleep(3000);}catch (InterruptedException e){}
        }
    }
    public static void main(String[] args){
        Thread t1 = new Thread(instance);
        Thread t2 = new Thread(instance);
        t1.start();
        t2.start();
    }
}
```

synchronized添加在普通方法上，锁的对象默认为this当前实例

###### 类锁

synchronized修饰静态的方法或者制定锁对象为class对象

```java
synchronized(SynchronizedObjectLock.class){
    // 代码
}
```

这里锁住的就是类级别的对象

###### 原理分析

**加锁和释放锁**

我们在加入synchronized关键字后，编译器会给对应的代码加入两个指令，分别是monitorenter 和 monitorexit，这两个指令会让对象在执行时使其锁计数器加减1，每个对象同时只与一个锁相关联。但是一个monitor（监视器）在同一时间只能被一个线程获得，一个对象在尝试获得与这个对象相关练的monitor锁的所有权时，需要进行如下判断：

* 监视器计数器为0，说明目前还没有被获得，那么这个线程就会立刻获得并把锁的计数器+1，其他线程就需要等待锁的释放。
* 如果监视器已经拿到了这个锁的所有权，又重入了这锁，锁的计数器就+1，随着重入次数增多，会一直增加。
* 锁被别的线程获取，需要等待释放。

对于释放监视器锁，计数器-1，再减完后如果计数器变为0，则代表该线程不再拥有锁，释放该锁。

**可重入原理**

可重入锁：同一个线程在外层方法获取锁的时候，再进入该线程的内层方法会自动获取锁。不会因为之前已经获得还没有释放而阻塞。在同一个锁线程中，每个对象拥有一个monitor计数器，当线程获取该对象锁，计数器加一，释放锁，计数器减一。

###### JVM锁优化

对于monitorenter和monitorexit字节码指令，依赖于底层操作系统的MutexLock实现的，但是由于使用MutexLock需要将当前线程挂起并且从用户态切换到内核态执行，切换代价大，现实中，同步方法时运行在单线程中，调用锁会严重影响程序性能，因此JVM对锁引入了优化。

* 锁粗化：减少不必要的解锁和加锁操作，这些紧连在一起的锁可以扩大成一个大锁。
* 锁消除：通过运行Java即时编译器的逃逸分析来消除一些锁。
* 轻量级锁：假设在真实情况下我们程序中大部分同步代码处于无锁竞争状态，在这种情况下避免调用操作系统的互斥锁，而是依靠一条原子CAS指令完成锁的获取和释放，在存在锁竞争时，执行CAS指令失败的线程调用操作系统的互斥锁进入阻塞状态。
* 偏向锁：是为了在无锁竞争的情况下，谜面在锁获取过程中执行不必要的CAS比较并交换指令
* 适应性自旋锁：当线程获取轻量级锁失败时，在进入重量级锁之前会进入忙等待然后再次尝试，尝试次数一定后如果还没有成功则调用互斥锁进入阻塞状态。

##### synchronized

synchronized修饰的实例方法可以保证对共享变量修改的原子性，也就是说这个实例方法实际保护的是同一个对象的方法调用。确保同时只能有一个线程执行，它保护的是当前实例对象，this，而这个对象有一个锁和等待条件，锁只能被一个线程持有，其他试图获得锁的线程需要等待。整个执行过程如下：

1. 线程尝试获得锁，如果能够获得，继续下一步，否则加入等待对列，阻塞并等待唤醒
2. 执行实例方法的代码
3. 释放锁，如果等待对列上有等待的线程，从中取出一个并唤醒。

synchronized保护的是对象而非代码，也不是某个变量，只要访问的是同一个对象的synchronized方法，即使是不同的代码，也会被同步顺序访问。因此在保护变量是，需要在所有访问该变量的方法上加上synchronized。

synchronized同样可以用于静态方法，对于实例方法，保护的是当前的实例对象this，对于静态方法，保护的是类对象，类对象也有锁和等待队列。

synchronized静态方法和synchronized实例方法保护的是不同的对象，不同的两个线程，可以同时执行。

除了用于修饰方法外，synchronized也可以用于包装代码块，用一个括号(X)来表示要保护的对象X，X可以是实例对象this，也可以是类对象 xxx.class，也可以是任意对象，类中的某个私有对象属性等。

synchronized是可重入的，对同一个执行线程，它在获得了锁之后，在调用其他需要同样锁的代码时可以直接调用。这种特性是通过记录锁的持有线程和持有数量来实现的，当调用被synchronized保护的代码时，检查对象是否已经被锁，如果被锁，检查是否被当前进程锁定，是的话增加持有数量，如果不是，则会加入等待对列，当释放锁时减少持有数量，数量变为0时才释放锁。

synchronized除了保证原子操作外，他还能保证内存可见性，在释放锁时，所有写入都会写回内存中，而获得锁后，都会从内存中读取最新数据。如果只是为了保证内存可见性，就给变量添加volatile修饰符。

在处理并发的时候，应该尽量避免在持有一个锁的同时去申请另一个锁。

##### final



#### 2）Happens-Before (在其之前要发生)规则

* 单一线程原则：在一个线程内，程序前面的操作要先于后面的操作发生
* 管程锁定原则：对于同一个锁，解锁操作要先于加锁操作
* 变量读写原则：对于有volatile关键字的变量，写操作要先于读操作完成。
* 线程启动原则：线程对象的start()方法调用先于此线程的每一个动作
* 线程加入规则：线程对象的结束先于其他线程的加入join()方法
* 线程中断规则：对某个线程调用中断方法先于被中断线程检测到中断时间的发生
* 对象终结原则：对象初始化完成先于它的结束方法finalize()
* 传递性：操作A先于操作B，操作B先于操作C，那么操作A先于操作C

### 3、线程安全

#### 1）概念

一个类可以被多个线程安全的调用时就是线程安全的，根据共享数据需要的安全程度可以分为五类：不可变、绝对线程安全、相对线程安全、线程兼容、线程对立。

* 不可变：不可变的对象一定是线程安全的，只要这个对象被正确的构建出来，在它整个生命周期中就不可变。不可变的类型有以下几种：

  * final关键字修饰的基本数据类型

  * String类 字符串

  * 枚举类型

  * Number的部分子类，Long Double BigInteger BigDecimal

  * 使用Collection.unmodifiableXX()方法获取的集合

* 绝对线程安全：对于某个对象，不管运行环境如何，调用者都不需要考虑线程安全的问题

* 相对线程安全：对于这个对象单独的操作时线程安全的，但是对于特定顺序的连续调用，需要使用额外的同步手段。Java语言中大部分线程安全类属于这种类型。

* 线程兼容：指的是对象本身不是线程安全的，但是通过正确使用同步手段可以保证对象在并发环境下安全的使用。

* 线程对立：指的是无论是否采取同步措施，都无法保证多线程环境中并发造成的问题。很少出现

#### 2）解决方法





### 4、Java多线程的实现

#### 1）线程状态的分类

![](E:\StudyDoc\6.阅读笔记\图片\java_thread_condition.png)

* New 新建状态：指的是一个线程被创建出来，但没有启动。
* Runnable 可运行的状态：指的是线程可能在运行，也可能在等待时间片轮转。
* Blocked 阻塞状态：等待获取一个排他锁
* Time Waiting 等待状态：在一定时间后会被系统自动唤醒，阻塞和等待的区别是阻塞是被动的，是需要获取一个资源。而等待是主动的，通过调用Object.wait() Thread.sleep()等进入
* Waiting 一直等待状态：等待其他线程唤醒
* Terminated 结束状态：结束了任务或者产生了异常

#### 2）执行多线程的方法

在Java中总得来说就是实现接口、继承Thread。实现接口更好一些，因为接口可以多实现，一个类可能要求并不高，单独为其创建一个线程开销过大。具体方法如下：

* 自定义的类继承（extends）Thread类，然后覆写它的run方法。然后定义并启动这个类
* 创建Thread实例时，传入一个Runnable实例。也就是自定义一个类实现Runnable接口，需要实现run方法
* 自定义类实现Callable接口，但它可以有返回值。创建Thread实例时，传入这个类的实例

#### 3）线程机制

* Excutor 线程执行器：能够管理多个异步任务的执行，主要有三种：缓冲线程池、固定线程池、单例线程池
* Daemon 守护线程：是程序运行时在后台提供服务的线程，所有非守护线程结束后，程序会终止。
* sleep方法：会休眠当前正在执行的线程
* yield方法：说明当前线程中涉及到线程安全重要的部分已经完成，接下来可以切换给其他线程运行
* interrupt方法：会中断当前线程，之后我们可以调用interrupted() 方法来判断线程是否处于中断状态。如果该线程处于阻塞或等待状态，会抛出中断异常从而提前结束该线程。这种中断不能中断因I/O请求阻塞和同步锁阻塞。
* join方法：可以让调用join方法的线程等待等待主线程结束。

#### wait/notify

Java在object类中定义了一些线程协作的基本方法，wait和notify

每个对象都有一把锁和等待对列，一个线程在进入synchronized代码块时，会尝试获取锁，获取不到的话会把当前线程加入等待队列中，除了用于锁的等待队列，每个对象还有另一个等待对列，该队列用于线程间的协作，调用wait就会把当前线程放到条件队列上并阻塞，表示当前线程执行不下去了，他需要等待一个条件，这个条件需要其他线程来改变，当其他线程改变了条件了以后，应该调用notify方法。因此notify做的事情就是从条件队列中选择一个线程，将其从队列中移除并唤醒，notifyAll能够移除条件队列中所有线程并且全部唤醒。

假设现在有两个线程，一个主线程和一个等待线程，协作的条件变量是fire，等待线程等待该变量变成True，在false时调用wait方法等待，主线程会负责设置该变量并且调用notify唤醒等待线程

这样两个线程都需要访问协作的变量fire，所以相关代码都需要被synchronized保护，而wait和notify方法只能在synchronized代码块内调用，如果调用这些方法时，当前线程没有对象锁，就会冒出异常。

wait的具体过程是：

1. 把当前线程放入等待队列，释放对象锁，阻塞等待，线程状态变为waiting
2. 等待时间到或者被其他线程唤醒，这时需要重新竞争对象锁，能够获得锁的话，线程就可继续，并从wait调用中返回

因此调用notify方法只是把线程从条件队列中移除，但是并不会释放对象锁。

#### 生产者消费者模式

在生产者消费者模式中协作的共享变量是队列，生产者往队列上放数据，如果满了就等待，而消费者从队列中取数据，队列为空也等待。

Java中提供了专门的阻塞队列实现

* 接口BlockingQueue BlockingDeque
* 基于数组的实现类
* 基于链表的实现类LinkedBlockingQueue LinkedBlockingDeque
* 基于堆的实现类PriorityBlockingQueue

### 5、并发容器

Java中普通的容器对象在多线程的环境下是不安全的，因此我们可以使用线程安全的容器，他们是给所有容器方法添加上synchronized来实现安全的。这样所有的方法调用就变成了原子操作。但是在调用的时候仍需要注意：

1. 复合操作——多次操作，而这些操作是需要原子化的
2. 伪同步——保护线程安全所做的处理实际上作用在了不同的对象上
3. 迭代——单个操作安全，迭代不安全

Java中有一些专门为并发设计的容器类。

#### CopyOnWriteArrayList

* 这个类是线程安全的，可以被多个线程并发访问、
* 迭代器不支持修改操作，但是也不会抛出ConcurrentModificationException
* 以原子方式支持一些复合操作

基于synchronized的同步容器，迭代时，需要对整个列表对象枷锁，否则会抛出异常，而这个类就没有这个问题。因为这个类的迭代器不支持修改，当然也就不能支持一些依赖迭代器修改方法的那些操作。

这个类的内部也是一个数组，但是这个数组是以原子的方式被整体更新的，每次修改操作，都会新建一个数组，赋值原数组的内容到新的数组，在新数组上进行需要的修改，然后以原子的方式设置内部的数组引用，这就是写时拷贝。

所有的读操作，都是先拿到当前引用的数组，然后直接访问该数组，在度的过程中，可能内部的数组引用已经被修改，但是不会影响读操作，仍然能够访问原数组的内容。

也就是说，数组的内容一直都是只读的，写操作都是通过新建数组，然后原子性的修改数组引用来实现的。

#### CopyOnWriteArraySet

#### ConcurrentHashMap

是HashMap的并发版本，这个有以下几个特点：

* 并发安全
* 直接支持一些原子复合操作
* 支持高并发、读操作完全并行、写操作一定程度的并行
* 与同步容器相比，迭代不需要加锁
* 弱一致性

同步容器使用的是synchronized，所有的方法，竞争同一个锁，但是ConcurrentHashMap采用分段锁技术，将数据分为多个端，而每一个端都有一个独立的锁，每一个段就相当与一个独立的哈希表。分段的依据也是哈希值。无论是保存键值对还是根据键来查找，都需要先根据哈希值映射到端，再在段对应的哈希表上进行操作。

采用分段锁，可以大大提高并发度，多个段之间可以并行读写，默认情况下，段时16个。对于写操作，需要获取锁，不能并行，但是读操作可以，多个读操作可以并行，写的同时也可以读。

### 6、异步执行任务

异步执行任务，也就是说，将任务的提交和任务的执行相分离。执行服务封装了任务执行的细节，对于任务提交者而言，它可以关注任务本身，如提交任务获取结果、取消任务。不需要关系任务执行的细节，如线程的创建、任务调度、线程关闭。也就是说，有两个角色，一个任务的提交者。一个是任务的执行者

* Runnable和Callable表示要执行的那个异步的任务
* Executor和ExecutorService表示执行任务的执行器
* Future表示异步任务执行的结果

#### Runnable Callable

都表示一个需要执行的任务，对于一个方法实现了RUnnable或者Callable接口，就能够变成能够异步执行的任务

#### Executor ExecutorService

Executor表示一个最简单的执行服务，可以执行一个实现了Runnable接口的方法，没有返回结果，接口没有限定任务应该如何执行，可能是创建一个新的线程，也可能是复用线程池中的某个线程。也可能是在调用者线程中执行。

ExecutorService 就扩展了Executor，定义了更多的服务，有提交任务的方法submit，返回值为Future，返回后表示任务已经提交，但不代表已经执行。

通过Future可以查询异步任务的状态、获取最终的结果、取消任务等。

#### Future

Future中有个get方法，用于返回异步任务最终的结果，如果任务还没有执行完成，就会阻塞等待。另一个get方法可以限定阻塞等待的时间

异步任务执行服务和Future，在异步任务执行服务中，用Callable或者Runnable表示服务。

```java
private static Random rnd = new Random();
// 用于模拟延时处理
static int delayRandom(int min, int max){
    int mili = max > min ? rnd.nextInt(max-min) : 0;
    try {
        Thread.sleep(min + milli);
    }catch(Exception e){}
    return milli;
}
// 是一个外部任务，会延时处理time的时间
static Callable<Integer> externalTask = () -> {
    int time = delayRandom(20, 2000);
    return time;
}
```

如果我们有一个异步任务执行服务，通过这个任务执行服务调用外部服务，一般就返回Future表示异步结果。

### 7、线程池

线程池主要由两个概念，一个是任务队列，一个是工作者线程，工作者线程主体就是一个循环，循环从队列中接受任务并执行，任务队列保存待执行的任务。

线程池可以重用线程，避免线程创建的开销。在任务过多的时候，通过排队避免创建过多的线程，减少系统资源消耗和竞争，确保任务有序完成。Java中线程池的实现类是ThreadPoolExecutor，继承字AbstractExecutorService，实现了ExecutorService接口。

```java
// 构造方法
public ThreadPoolExecutor(int corePoolSize, int maximumPoolSize, long keepAliveTime, TimeUnit unit, BlockingQueue<Runnable> workQueue, ThreadFactory threadFactory, RejectedExecutionHandler handler);
```

构造方法中的参数，用于控制线程池中的线程个数，workQueue表示任务队列，threadFactory用于对创建的线程进行一些配置，handler表示任务拒绝的策略。

实际中线程池的大小主要定义核心线程的个数，最大线程个数，空闲线程的存活时间。在运行中线程的个数会动态变化，但不会比最大值高。核心线程指的是，如果由新的任务到来，如果当前线程数小于核心线程数，就会创建一个新线程来执行任务。即使其他已经创建的线程也是空闲的。但是如果现有线程数已经大于核心显成熟了，就不会立即创建新的线程，他会尝试先排队。如果不能立即入队，在未达到最大线程数时，才会创建新的线程。

线程的存活时间是为了释放多余的线程资源，一个非核心的线程，在空闲等待任务时，会有一个最长等待时间，如果到时还没有新任务，就会被终止。

线程池中要求的队列类型是阻塞队列，阻塞队列有四种：

* 基于链表的阻塞队列
* 基于数组
* 基于堆的
* 没有实际存储空间的同步阻塞队列

在使用无界队列时，线程个数最多只能到核心线程数的大小。

当线程队列满了，新任务到来就会触发线程的任务拒绝策略。在任务量非常大的场景中，拒绝策略是非常重要的。



### 8、组合式异步编程

一个软件系统的很多功能可能会被切分为小的服务，在对外展示具体页面时，可能会调用多个服务。为了提高性能充分利用系统资源，这些对外部服务的调用一般是异步的，尽量使并发的。

CompleteableFuture是一个具体的类，实现了两个接口，一个使Future，另一个使CompletionStage，Future表示异步任务的结果，而CompletionStage字面意思就是完成任务的阶段。多个阶段可以用用流水线的方式组合起来，对于其中一个阶段，有一个计算任务，但是可能要等待其他一个或者多个阶段完成才能开始，等待它完成后，可能会触发其他阶段开始运行。

### 9、锁

#### 乐观锁和悲观锁

悲观锁：对于同一个数据的并发操作，总是认为自己在使用数据时会有别的线程来修改数据，因此在获取数据之前会先加锁，确保数据不会被别的线程所修改。synchronized和Lock实现类都是悲观锁

```java
//方法1 synchronized悲观锁
public synchronized void test(){
    //操作同步资源
}
//方法2 Lock悲观锁
private ReentrantLock lock = new ReentrantLock();//创建锁，多个线程中锁为同一个
public void modifyPublicResources(){
    lock.lock();
    //操作同步资源
    lock.unlock();
}
```

乐观锁：对于同一个数据的并发操作，认为自己在使用数据时不会有别的线程修改数据，因此不会添加锁，只有在更新数据时判断之前有没有别的线程更新了这个数据，如果没有被更新，则写入自己修改的数据。

```java
private AtomicInteger atomicInteger = new AtomicInteger();
atomicInteger.incrementAndGet();
```

因此，悲观锁适合写操作多的场景，先加锁可以保证写操作时数据正确。

乐观锁适合读操作多的场景，不加锁可以使其读操作性能大幅提升。

#### 自旋锁和适应性自旋锁

> 在没有加入锁优化的时候，Synchronized时一个非常庞大的家伙，在多线程竞争锁的场景下，当一个线程获取锁时，synchronized会阻塞所有正在竞争的线程，这样对性能造成了极大的影响。挂起线程和恢复线程的操作都需要在操作系统的内核态中完成，这些操作会给系统并发性能带来很大的压力。但是共享数据的锁定状态只会持续一段很短的时间。为了这段时间去挂起和恢复阻塞线程并不是很值得。在如今多处理器条件下，我们可以让另外一个没有获取到锁的线程自旋一会，不放弃CPU。大多数情况下。持有锁的线程会很快的释放锁。这样自旋的线程就可以结束自旋获得该共享数据的处理权了。

阻塞或者唤醒一个Java线程需要操作系统切换CPU状态，这种状态转换需要耗费处理器事件，如果同步代码块中的内容很简单，状态转换消耗的事件甚至会比执行代码的时间还要长。在许多场景中，同步资源的锁定时间很短，为了这一小段时间切换线程，线程挂起和回复现场的花费有可能让系统得不偿失。如果物理机器有多个处理器，能够让两个或者以上的线程同时并行执行，我们就可以让后面那个请求锁的线程不放弃CPU执行时间，看看持有锁的线程是否会很快地释放锁。

那么等待锁释放的那个线程，我们就需要让线程自旋，也就是循环询问是否有锁，如果在自旋完成后，前面锁定同步资源的线程已经释放了锁，那么当前线程就可以不必阻塞而是直接获取同步资源，从而避免切换线程的开销，这就是自旋锁。

**缺点**

自旋锁等待虽然避免了线程切换的开销，但是要占用处理器的时间。如果锁被占用的时间过长，那么自旋的线程就只能白白浪费处理器资源。所以自旋等待的时间必须有一定的限度，如果自旋超过了限定次数，就应该挂起线程，释放处理器资源。

自旋锁的实现同样采用CAS对比后交换原子指令完成。

#### 无锁、偏向锁、轻量级锁、重量级锁

#### 公平锁和非公平锁

公平锁指的是多个线程按照申请所得顺序来获取锁，线程直接进入队列中排队，队列中的第一个线程才能够获取锁。公平锁的优点是等待锁的线程不会饿死。缺点是整体吞吐效率相对非公平锁要低。等待队列中除了第一个线程以外的所有线程都会阻塞。CPU唤醒阻塞线程的开销比非公平锁要大。

非公平锁是多个线程加锁时直接尝试获取锁，获取不到时才会到队列队尾等待，但是如果此时的锁刚好可用，那么这个线程就可以无需阻塞直接获取到锁，所以非公平锁有可能出现这种情况：后申请锁的线程先获取到锁。非公平锁的有点事可以减少唤起线程的开销，整体吞吐效率高，因为线程有几率不阻塞直接获取到锁，CPU不必唤醒所有线程逐一询问。但是缺点是处于等待队列中的线程可能会饿死。

#### 可重入锁和非可重入锁

**显式锁**

```java
public interface Lock{
    // 普通的获取锁和释放锁方法，lock会阻塞直到成功
    void lock();
    void unLock();
    // 它可以响应中断，如果被其他线程中断了，会抛出异常
    void lockInterruptibly();
    // 尝试获取锁，会立即返回，不阻塞
    boolean tryLock();
    // 先尝试获取锁，能成功则立即返回，否则阻塞等待。但是等待时间最长为指定的参数。
    boolean tryLock(long time, TimeUnit u);
}
```

显式锁相比于synchronized它支持以非阻塞的方式获取锁、可以响应中断、限时。

Lock是一个接口，他的主要实现式ReentrantLock。但是可以重入每一个线程在持有一个锁的前提下，可以继续获得该锁。可以解决竟态条件的问题。可以保证内存的可见性。这个实现类有一个构造方法，传入boolean值，意思是是否保持公平。等待时间最长的线程会优先获得锁。保证公平会影响性能。使用显式锁，要记得调用unLock。

使用tryLock可以避免死锁，在持有一个锁获取另一个锁，获取不到的时候，释放自己已持有的锁。

可重入锁的底层，依赖LockSupport的一些方法。这个类中有一些基本方法。如park使得当前线程放弃CPU，进入等待状态。操作系统不再对他进行调度。当其他线程对他调用了unpark时，就会恢复运行状态。

可重入锁又叫做递归锁，是指同一个线程在外层方法获取锁的时候，在进入该线程内层方法时会自动获取锁。不会因为之气那已经获取过还没有释放就阻塞。ReentrantLock和synchronized都是可重入锁。可重入锁在一定程度上可以避免死锁。

比如说同一个类中的两个方法都是被内置的锁synchronized修饰的，在其中一个方法内部调用另一个被synchronized加锁的方法，进入过另一个方法是可以直接获得当前对象的锁，而不必等待刚刚的方法释放锁。而非可重入锁在调用另一个方法是，进入方法内部则无法获取到锁，因为该锁被外部方法占用，且无法释放，此时就会造成死锁。

重入锁和非重入锁中记录了一个同步状态status来计算重入次数，当线程尝试获取锁是，可重入锁先尝试并更新这个值，如果值不为0，则会判断当前线程是否是获取到了这个锁的线程，如果是的status+1.且当前线程可以尝试再次获取锁。而非重入锁在获取更新status值时，如果值不为0，则会导致其获取锁失败，当前线程阻塞。

#### 独享锁和共享锁

独享锁也叫做排他锁，是指该锁一次只能被一个线程所持有，如果线程对数据加上排他锁后，其他线程则不能再对数据加任何类型的锁。获取排他锁的线程既能够读取数据也能够修改数据。

共享锁指的是该锁可以被多个线程所持有，如果线程对数据加上共享锁后，其他线程只能对数据加上共享锁而不能加排他锁，获得共享锁的线程只能够读数据而不能够写数据。

### 10、CAS

#### 概念

CAS的全称时Compare and Swap，即对比后交换，是CPU的一条原子指令，其作用是让CPU先进行比较两个只是否相等，然后原子性的更新某个位置的值，它是基于硬件平台的汇编指令实现的，也就是说CAS靠硬件实现，JVM封装了汇编的调用。因为CAS操作是原子性的，所以在多线程并发的时候可以不用使用所。JDK中大量使用了CAS这个指令来更新数据而不是使用重量级锁synchronized来保证原子更新。

* java为我们提供了AtomicInteger原子类，该类底层基于CAS指令进行更新数据，不需要加锁就可以完成在多线程并发场景下的数据一致性。

```java
private AtomicInteger i = new AtomicInteger(0);
//原子操作
public int add(){
    return i.addAndGet(1);
}
```

* 在多线程环境下为了保持数据一致性，我们使用synchronized对变量的值进行加锁。

```java
private int i = 0;
public synchronized int add(){
    return i++;
}
```



#### 缺点

CAS的方式是乐观锁，而synchronized是悲观锁，乐观锁解决并发问题在通常情况下性能更优。但是CAS也存在一些问题。

##### ABA问题

CAS对比后交换指令，需要在操作值的时候，检查值有没有变化，如果没有发生变化则更新，但是如果一个值原来是A，变成了B，随后又变成了A，那么使用CAS进行检查时就会发现他的值没有发生变化，但是实际上却已经变化了。

ABA问题我们可以采用版本号来解决，在变量前面追加上版本号，每次变量更新的时候，把版本号加1。

##### 循环开销时间大

自旋的CAS如果长时间不能够成功，会给CPU带来很大的开销，如果JVM能够支持处理器提供的pause指令，那么效率则会有一定的提升。pause指令的作用是：1、延迟流水线执行命令，使CPU不会消耗过多的资源。2、避免在退出循环时因为存在内存顺序冲突而引起CPU流水线被清空，从而提高CPU执行效率。

##### 只能保证一个共享变量的原子操作

对于多个共享变量的操作，循环CAS无法保证原子性，需要使用锁。

#### 解决方案

##### AtomicStampedReference

该类主要维护包含一个对象引用以及一个可以自动更新的整数的pair对象来解决ABA问题。

```java
public class AtomicStampedReference<V> {
    private static class Pair<T> {
        final T reference;
        final int stamp;//用于标志版本
        static <T> Pair<T> of(T reference, int stamp){
            return new Pair<T>(reference, stamp);
        }
    }
    private volatile Pair<V> pair;
    
    public boolean compareAndSet(V expectedReference, V newReference, int expectedStamp, int newStamp){
        Pair<V> current = pair;
        return // 略
    }
    
    private boolean casPair(Pair<V> cmp, Pair<V> val){
        return UNSAFE.compareAndSwapObject(this, pairOffset, cmp, val);
    }
}
```

如果发现元素值和版本号都没有发生变化，和新的值也相同，则返回true表示成功；

如果元素值和版本号没有变化，和新的值不完全相同，则构造Pair对象，执行CAS更新

### 11、Unsafe类

#### 概念

Unsafe是JDK提供的一个类，用于提供一些执行低级别、不安全操作的方法，如直接访问系统内存资源、自主管理内存资源等。这些方法在提升Java运行效率、增加Java语言底层资源操作能力方面起到了很大的作用。Unsafe类使Java语言拥有了类似C语言指针一样的操作内存空间的能力。

Unsafe类提供的API可以分为内存操作、CAS、类操作、对象操作、系统信息获取、内存屏障、数组操作等几类。

```java
public final int getAndAddInt(Object paramObject, long paramLong, int paramInt){
    int i;
    do{
        i = getIntVolatile(paramObject, paramLong);
    }while(!compareAndSwapInt(paramObject, paramLong, i, i + paramInt));
    return i;
}
```

Unsafe类在内部使用自旋的方式进行CAS更新。也就是通过while循环，如果更新失败，则循环重试。

在底层方法中，使用C++语言来跟操作系统进行沟通，并直接调用汇编语言实现该功能

```c++
UNSAFE_ENTRY(jboolean, Unsafe_CompareAndSwapInt(JNIEnv *env, jobject unsafe, jobject offset, jint e, jint x))
    UnsafeWrapper("Unsafe_compareAndSwapInt");
	oop p = JNIHandles::resolve(obj);
	jint* addr = (jint *)index_oop_from_field_offset_long(p, offset);
	return (jint)(Atomic::cmpxchg(x, addr, e))==e;
UNSAFE_END
    
// 在windows中
inline jint Atomic::cmpxchg(jint exchange_value, volatile jint* dest, jint compare_value){
    int mp = os::isMP();
    _asm{
        mov edx, dest
        mov ecx, exchange_value
        mov eax, compare_value
        LOCK_IF_MP(mp)
        cmpxchg dword ptr [edx],ecx
    }
}
```

Unsafe类提供了硬件级别的操作，比如说获取某个属性在内存中的位置、修改对象的字段值。

### 12、AtomicInteger

| API                     | 说明                                                         |
| ----------------------- | ------------------------------------------------------------ |
| get()                   | 获取当前值                                                   |
| getAndSet(int newValue) | 获取并设置新的值                                             |
| getAndIncrement()       | 获取当前值并自增                                             |
| getAndDecrement()       | 获取当前值并自减                                             |
| getAndAdd(int delta)    | 获取当前值并加上预期的值                                     |
| lazySet(int newValue)   | 懒设置，最终会设置成新的值，但是会不够及时。其他线程可能在这段时间里读到旧值 |

在使用AtomicInteger之前，多线程要让变量自增需要这些写：

```java
private volatile int count = 0;
public synchronized void increment(){count++;}
public int getCount(){return count;}
```

在使用AtomicInteger后

```java
private AtomicInteger count = new AtomicInteger();
private void increment(){
    count.incrementAndGet();
}
public int getCount(){
    return count.get();
}
```

AtomicInteger在底层使用的是volatile的变量和CAS来进行更改数据的。

* volatile保证线程的可见性，多线程并发誓，一个线程修改数据，可以保证其他线程立马看到修改后的值
* CAS比较后对换原子指令保证数据更新时的原子性

### 13、原子更新类

使用原子方式更新基本类型，Atomic包提供了三类。

| 类名          | 说明             |
| ------------- | ---------------- |
| AtomicBoolean | 原子更新布尔类型 |
| AtomicInteger | 原子更新整型     |
| AtomicLong    | 原子更新长整型   |

使用原子方式更新数组中的某个元素，Atomi提供了三类。

| 类名                 | 说明                         |
| -------------------- | ---------------------------- |
| AtomicIntegerArray   | 原子更新整型数组里的元素     |
| AtomicLongArray      | 原子更新长整型数组里的元素   |
| AtomicReferenceArray | 原子更新引用类型数组里的元素 |

使用原子方式更新引用类型，也是三个类，这三个类首先需要构造一个引用对象，然后把引用对象设置进入Atomic类，然后调用compareAndSet等一些方法进行原子操作。原理都是基于Unsafe类实现。

| 类名                    | 说明                                               |
| ----------------------- | -------------------------------------------------- |
| AtomicReference         | 原子更新引用类型                                   |
| AtomicStampedReferece   | 原子更新引用类型，内部使用Pair来存储元素值及版本号 |
| AtomicMarkableReference | 原子更新带有标记为的引用类型                       |

使用原子更新字段类。需要两步：1、因为原子更新字段类都是抽象类，每次使用的时候必须使用静态方法创建一个更新器，并且设置想要更新的类和属性。2、更新的字段必须使用volatile修饰。

| 类名                        | 说明                         |
| --------------------------- | ---------------------------- |
| AtomicIntegerFieldUpdater   | 原子更新整型的字段更新其     |
| AtomicLongFieldUpdater      | 原子更新长整型字段的更新器   |
| AtomicReferenceFieldUpdater | 更新的字段需要用volatile修饰 |

### 14、LockSupport



### 15、AQS

AQS也就是Abstract Queued Synchronizer是一个用于构建锁和同步器的框架，使用AQS能够简单高效的构造出应用管饭的大量同步器，如ReentrantLock、Semaphore。

#### 核心思想

如果被请求的共享资源是空闲的，则将当前请求资源的线程设置为有效的工作线程，并且将共享资源设置为锁定状态。如果被请求的共享资源被占用，那么需要一套线程阻塞等待、线程被唤醒时锁如何分配的机制。这个机制AQS使用一种队列实现的。也就是将暂时获取不到锁的线程先加入到队列当中。

这个队列是一个虚拟的双向队列，不存在队列实例，而仅仅存在结点之间的关联关系。AQS是将每条请求共享资源的线程封装成一个队列中的一个结点，然后在队列中采用一定的逻辑进行锁的分配。

在队列内部，使用一个int成员变量表示同步状态state，使用volatile来对该变量进行修饰保证线程的可见性。

#### 资源共享方式

AQS定义了两种对这个共享资源（如共享变量、共享对象等）的共享方式

* 一种是独占的方式，只有一个线程能够持有这个共享资源。又分为公平锁和非公平锁。
  * 公平锁：按照线程在队列中的排队顺序，先到的就能够先拿到锁
  * 非公平锁：当线程想要获取锁是，能够五十队列顺序直接去抢锁，谁能抢到就是谁的
* 共享方式：多个线程可以同时执行，如Semaphore/CountDownLatch。

AQS的实现类在实现时只需要实现共享资源state的获取和释放方式。具体线程等待队列的维护，AQS上层已经帮我们实现了。

在AQS的底层，使用了模板方法的设计模式。使用者基层ABstractQueuedSynchronizer并重写器指定的方法。内容是关于对共享资源的获取和释放，将AQS组合在自定义的同步组件的实现中，并且调用其模板方法。而这些模板方法会调用使用者重写的方法。

AQS使用了一种虚拟的双向队列，每条请求共享资源的线程会被封装成一个CLH锁队列的一个结点。在这个队列中进行锁的分配，其中有一个同步队列是双向列表，包括头节点和为节点，

### 16、ReentrantLock

Reentrant意为可重入的，即可重入锁。

### 17、ReentrantReadWriteLock

### 18、fork join

Fork Join 框架是Java并发工具包中一种可以将一个大任务拆分成多个小任务来异步执行的工具

## 二、集合


Java标准库自带的`java.util`包提供了集合类`collection`，他是除了`Map`之外的所有集合类的根接口。Java主要提供了三种类型的集合。Java的集合接口和实现类分离，支持泛型。访问集合可以通过抽象的迭代器来实现，这样就无需知道集合内部元素存储方式。

* List 顺序表或者链表
* Set 没有重复元素的集合
* Map 键值对

<img src="D:\StudyDoc\4.阅读笔记\图片\java_collections_overview.png" alt="img" style="zoom:150%;" />

* Collection接口：List接口、Queue接口、Set接口继承了Collection接口，他们之间是泛化的关系。
  * List接口：ArrayList类、LinkedList类实现了List接口，他们之间是接口与类的实现关系
  * Queue接口：BlockingQueue接口、Deque接口继承了Queue接口，LinkedList、PriorityQueue类实现了Queue接口。
    * Deque接口：LinkedList类、ArrayDeque类实现了Deque接口，BlockingDeque接口继承了Deque接口
  * Set接口：HashSet类实现了Set接口。SortedSet继承了Set接口，NavigableSet又继承了SortedSet接口，TreeSet最终实现了SortedSet接口。
* Map接口：EnumMap HashMap实现了Map接口。SortedMap接口继承了Map接口，TreeMap类最终实现了SortedMap接口。

总体来讲，集合类所所实现的数据结构可由带颜色的圆点看出，一共有链表List、数组Array、红黑树Red-Black Tree、哈希表Hash Table、二叉堆Binary Tree几种数据结构

### 1、List

* `List<E>`接口有以下几种方法：
  * 在末尾添加一个元素

  * 在指定的索引后添加一种元素

  * 删除指定索引的元素

  * 删除某个元素

  * 获取指定索引的元素

  * 获取列表的大小
  * 返回某个元素的索引
  * 判断List是否存在某个元素

* `List<E>`接口常用的主要有两种实现：LinkedList 链表，但不仅可以用所链表，还可以用作栈、队列、双向队列。ArrayList 是动态数组

* 遍历访问 List 最好使用迭代器 Iterator 访问

* 转换成数组：使用`toArray(T[List.size()])`方法传递一个与LIst元素类型相同的数组，List会自动将所有元素复制到数组中。

#### ArrayList

| 方法               | 函数名称                       |
| ------------------ | ------------------------------ |
| 判断是否为空       | boolean isEmpty()              |
| 获取长度           | int size()                     |
| 访问指定位置的元素 | E get(int index)               |
| 查找元素           | int indexOf(Object o)          |
| 从后往前查找       | int lastIndexOf(Object o)      |
| 是否包含指定元素   | boolean contains(Object o)     |
| 删除元素           | E remove(int index)            |
| 删除所有元素       | boolean clear()                |
| 插入元素           | void add(int index, E element) |
| 修改元素           | E set(int index, E element)    |

ArrayList实现了List接口，是顺序容器，元素存放的数据与放进去的顺序相同，允许放入null元素，底层通过数组实现，除了该类未实现同步以外，其余都和Vector相同。每个ArrayList都有一个容量，表示底层数组的实际大小。容器内存储的元素个数不能多余当前容量。当向容器中添加元素时，如果容量不足，容器会自动增大底层数组的大小。Java中的泛型只是编译器提供的语法糖，实际上数组中存放的都是Object类型。以便能够容纳任何对象。为了追求效率，Array List没有实现同步。

自动扩容机制：每当向数组中添加元素时，都要去检查添加之后元素的个数是否会超出当前数组的长度，如果超出长度，数组将会进行扩容，以满足添加数据的需求，数组扩容通过一个公开的房吗来实现，在实际添加大量元素之前，我们也可以使用ensureCapacity方法来手动增加ArrayList的容量。数组扩容时，会将老数据中的元素重新拷贝一份到新的数据中，每次数组容量的增长大约是其原容量的1.5倍，这种操作的代价时很高的，因此在实际使用时，我们应该尽量避免数组容量的扩张，如果能偶与之保存的数据是多少，我们在构造Array List实例时，就指定其容量，或者通过调用ensureCapacity方法来手动增加容量。

```java
public void ensureCapacity(int minCapacity){
    int minExpand = (elementData != DEFAULTCAPACITY_EMPTY_ELEMENTDATA) ? 0 : DEFAULT_CAPACITY;
    if(minCapacity > minExpand){
        ensureExplicitCapacity(minCapacity);
    }
}

private void ensureCapacityInternal(int minCapacity){
    if(elementData == DEFAULTCAPACITY_EMPTY_ELEMENTDATA){
        minCapacity = Math.max(DEFAULT_CAPACITY, minCapacity);
    }
    ensureExplicitCapacity(minCapacity);
}
private void ensureExplicitCapacity(int minCapacity){
    modCount++;
    if(minCapacity- elementData.length > 0){
        grow(minCapacity);
    }
}
private void grow(int minCapacity){
    int oldCapacity = elementData.length;
    int newCapacity = oldCapacity + (oldCapacity >> 1);
    if (newCapacity - minCapacity > 0) newCapacity = minCapacity;
    if (newCapacity - MAX_ARRAY_SIZE > 0) newCapacity = hugecapacity(minCapacity);
    elementData = Arrays.copyOf(elementData, newCapacity);
}
```

add, addAll方法：添加元素可以是在数组最后面添加，也可以是在指定位置添加。这两种添加元素的方式，都有可能导致数组剩余空间不足，因此在添加元素之前都需要进行剩余空间检查。如有需要就进行自动扩容，扩容的操作最终是通过grow方法实现的。在指定位置添加的话，需要先对后面的元素将进行移动，然后才能完成插入操作。

#### LinkedList

LinkedList同时实现了List接口和Deque接口，也就是说它既可以看作一个顺序容器，又可以看作一个队列，同时也可以看作是一个栈。当我们需要使用栈或者队列时，我们就可以考虑使用LinkedList，不过也可以使用ArrayDeque，这个在当作栈或者队列使用时性能更好。

底层主要通过双向链表实现，双向链表的每个结点用内部类Node来表示，LinkedList通过first、last引用分别指向链表的第一个和最后一个元素，当链表为空的时候first、last指针都指向null；

```java
// 查看队头元素
public E peek(){
    final Node<E> f = first;
    return (f == null) ? null : f.item;
}
// 查看队头元素，没有会抛出异常
public E element(){
    return getFirst();
}
// 获取队头元素，会将队头元素弹出队列
public E poll(){
    final Node<E> f = first;
    return (f == null) ? null : unlinkFirst(f);
}
// 双端队列方法
public boolean offerFirst(E e){
    addFirst(e);
    return true;
}
public boolean offerLast(E e){
    addLast(e);
    return true;
}
public E peekFirst(){
    final Node<E> f = first;
    return (f == null) ? null: f.item;
}
```

#### Stack & Queue

Queue接口继承自Collection，除了最基本的Collection方法之外，他还额外支持insertion，extraction和inspection操作，每种类型有两个不同的方法，一种抛出异常的实现，另外一组没有就返回null。

| 类型    | 抛出异常的方法 | 返回特定值方法 |
| ------- | -------------- | -------------- |
| insert  | add(e)         | offer(e)       |
| remove  | remove()       | poll()         |
| examine | element()      | peek()         |

Deque时double ended queue 表示双向的队列，Deque继承自Queue接口，除了支持Queue的方法之外，还支持双向的insert、remove、examine操作。由于Deque是双向的，所以可以对队列的头和尾部都进行操作，他同时还支持两组格式，一组是抛出异常的实现，另外一组是返回值的实现。

从名字我们可以看出ArrayDeque底层通过数组来实现，为了满足可以同时在数组两段插入或者删除元素的需求，该数组还必须是循环的，即循环数组，也就是说数组的任何一点都可能被看作起点或者重点，ArrayDeque是非线程安全的，当多个线程同时使用的时候，需要程序员手动同步。

在Deque中head指向首段第一个有效的元素，而tail指向尾端第一个可以插入元素的空位。因为是循环数组，所以head不一定总等于0，tail也不一定总是比head大。

addFirst方法，作用是在Deque的首端插入元素，也就是在head指针的前面插入一个元素，在空间足够且下标没有越界的情况下，只需要将elements[--head] = e，在实际中，需要考虑空间是否够用，下标是否越界，而空间问题是在插入之后解决的，因为tail总是执行下一个可以插入的空位，也就意味着elements数组至少有一个空位，所以插入元素的时候不用考虑空间问题。针对下标越界的问题，head = (head - 1) & (elements.length - 1)相当于取余，同时解决了head为复制的情况，因为elements.length必须是2的指数倍，elements - 1就是二进制低位全1，跟head - 1 相与之后就起到了驱魔的作用。对于扩容函数，逻辑是申请一个原数组容量两倍的新数组，然后将原数组复制过去。

addLast方法，作用是在Deque的尾端插入元素，也就是在tail的位置插入元素，由于tail总是指向下一个可以插入的空位，因此只需要elements[tail] = e; 插入完成后再次检查空间，如果空间已经用光，则会调用doubleCapacity方法进行扩容。

```java
public void addLast(E e){
    if(e == null) throw new NullPointerException();
    elements[tail] = e;
    if( (tail = (tail + 1) & (element.length -1)) == head)
        doubleCapacity();
}
```

pollFrist()方法，这个方法的作用是删除并返回Deque首段的元素，也就是head位置上的元素，如果容器不空，只需要直接返回elements[head] 即可，然后处理下标的问题。由于ArrayDeque中不允许放入null，当elements[head] == null 的时候，也就意味着容器为空。

```java
public E pollFirst(){
    int h = head;
    E result = elements[head];
    if(result == null) return null;
    elements[h] = null;
    head = (head + 1) & (elements.length -1);
    return result;
}
```

pollLast()方法是删除并返回Deque尾端的元素，也就是tail位置前面的哪一个元素。

peekFirst，peekLast方法是返回但不删除首端、尾端的元素。

#### PriorityQueue

优先队列的作用是能够保证每次去除的元素都是队列中权值最小的，这里涉及到了大小关系，元素之间的大小可以通过元素本身的自然顺序，也可以通过构造时传入的比较器来判别。PriorityQueue也实现了Queue接口，不允许放入null元素。通过堆这个数据结构来实现。具体说是通过完全二叉树实现的小顶堆，任意一个非叶子节点的权值都不大于其左右子节点的权值。也就意味着可以通过数组来作为PriorityQueue的底层实现。在堆中父子结点的编号之间有一定的数学关系。在优先队列中，查看某个元素获取某个元素都是常数级的时间，而添加删除元素的时间复杂度为log(N)

add, offer方法，都是向优先队列中插入元素，只是两者在插入失败时的处理方法不同，add方法在插入失败后会抛出异常，offer方法在插入失败时会返回false。

每当添加一个元素，都有可能破坏小顶堆的性质，使得小顶堆不满足任意一个非叶子节点的权值都不大于其左右子节点。因此需要一定的调整。

```java
public boolean offer(E e){
    if(e == null) throw new NullPointerException();
    modCount++;
    int i = size;//size为当前小顶堆中存放元素的个数
    if(i >= queue.length) grow(i+1);//如果当前存放元素个数大于等于队列的长度了，就自动扩容
    size = i + 1;//添加一个元素 ，size增加1
    if(i == 0) queue[0] = e; //队列本身为空时，就在第一个位置处添加元素
    else siftUp(i, e); //否则就需要添加，然后调整堆操作
}

// 从k指定的位置开始，将x 逐层与当前点的父节点值比较并交换，知道满足x>=queue[parent]为止
private void siftUp(int k, E x){
    while(k > 0){
        int parent = (k - 1) >>> 1;//parentNo = (nodeNo - 1)/2
        Object e = queue[parent];
        if(comparator.compare(x, (E) e) >= 0) 
            break;
        queue[k] = e;
        k = parent;
    }
    queue[k] = x;
}
```

element, peek方法，获取但是不删除队首元素，也就是队列中权值最小的那个元素，前者会抛出异常，后者会返回null，根据小顶堆的性质，堆顶那个元素就是全局最小的元素，由于堆用数组表示，因此0下标处的元素就是对顶的元素。

remove，poll方法，获取并删除堆顶的元素，由于删除操作会改变队列的结构，因此删除需要对堆做出调整。

```java
public E poll(){
    if(size == 0) return null;
    int s = --size; // s是最后一个元素的下标
    modCount++;
    E result = (E) queue[0];
    E x = (E) queue[s]; // 先取出最后一个元素
    queue[s] = null; // 将最后一个位置置空
    if (s!= 0)//如果不是首个元素，那就需要对堆进行调整
        siftDown(0, x)
    return result;
}
// 首先记录0下标处的元素，并用最后一个元素替换0下标位置的元素，之后调用siftDown方法堆堆进行调整，最后返回原来0下标处的那个元素，这个方法作用是从k指定的位置开始将x逐层向下与当前点的左右孩子中较小的那个交换，知道x小于等于左右孩子中的任意一个为止。
```

remove方法用于删除队列中跟传入元素相等的某一个元素，由于删除操作会改变队列结构，因此需要进行调整，而删除元素的位置可能是任意的，所以调整过程比其他函数稍微繁琐。如果删除的是最后一个元素，直接删除。如果删除的不是最后一个元素，从删除点开始以最后一个元素为参照调用一次siftDown方法

### 2、Set



### 3、Map

* `Map<E>`接口主要有三种实现：EnumMap TreeMap HashMap
* Map有键值的概念，一个键映射到一个值，Map按照键存储和访问值，键不能重复，即一个键只会存储一份，给同一个键重复设值会覆盖原来的值，

#### HashMap

HashMap实现了Map接口，即允许放入key为null的元素，也允许插入value为空的元素，除了该类尚未实现同步以外，其他的和Hashtable大致相同。和TreeMap不同，该容器不保证元素的顺序，根据需要该容器可能会对元素重新哈希，元素的顺序也会倍重新打散，因此不同时间迭代同一个HashMap的顺序可能会不同，根据对冲突的处理方式不同，哈希表有两种实现方式，一种是开放地址方式，另一种是冲突链表方式。

如果选择合适的哈希函数，往Map中放数据和取数据都可以在常数时间内完成，但是对HashMap进行迭代时，需要遍历整个table以及后面跟的冲突链表，因此对于迭代比较频繁的场景。不宜将HashMap的初始大小设置的过大。有两个参数可以影响HashMap的性能：初始容量和负载系数。初始容量制定了初始table的大小，负载系数用来指定自动扩容的临界值。当entry的数量超过了capacity*load_factor时，容器将自动扩容并且重新计算哈希函数。对于插入元素较多的场景。将初始容量设置大点可以减少重新哈希的次数。

将对象放入到HashMap或者时HashSet中，有两个方法需要特别关心。hashCode方法决定了对象会被放到哪一个桶中，当多个对象的哈希值冲突时，equals方法决定了这些对象是不是同一个对象。所以如果将自定义对象放入到HashMap或者HashSet中，需要复写这两个方法。

get方法根据指定的key返回对应的value，该方法调用了getEntry(Object key)得到相应的entry，然后返回entry.getValue，因此getEntry是算法的核心。首先要通过hash函数得到对应的bucket下标，然后依次遍历冲突列表。通过key.equals(k)方法来判断是否是要找的那个entry

```java
final Entry<K, V> getEntry(Object key){
    int hash = (key == null) ? 0 : hash(key);
    for (Entry<K, V> e = table[hash&(table.length-1)]); e!=null; e = e.next){
        Object k;
        if(e.hash == hash && ((k = e.key) == key || (key != null && key.equals(k))))
            return e;
    }
    return null;
}
```

put方法，将指定的key，value添加到map中，该方法首先会对map做一次查找，看是否包含该元组，如果已经包含则直接返回，查找过程类似于getEntry方法。如果没有找到，则会通过addEntry方法插入新的entry，插入方式为头插法。

remove方法，作用是删除key值对应的entry，该方法的具体逻辑是在removeEntryForKey中实现的，这个方法会首先找到key值对应的entry，然后删除该entry。

```java
final Entry<K, V> removeEntryForKey(Object key){
    int hash = (key == null) ? 0 : hash(key);// 计算哈希值
    int i = indexFor(hash, table.length); //计算引用数组的下标
    Entry<K, V> prev = table[i];//得到下标对应的引用的冲突链表
    Entry<K, V> e = prev;
    while (e != null){
        Entry<K, V> next = e.next;
        Object k;
        if(e.hash == hash && ((k = e.key) == key || (key!=null && key.equals(k)))){
            modCount++; size--;
            if(prev==e) table[i] = next; //如果要删除的是冲突链表中的第一个元素
            else prev.next = next;
            return e;
        }
        prev = e;
        e = next;
    }
    return e;
}
```

在java8中，HashMap有了一些修改，最大的不同就是利用了红黑树，所以这是由数组+链表+红黑树组成。HashMap在查找的时候，根据hash值我们能够迅速的定位到数组的具体下标，但是在之后需要顺着链表一个个对比才能找到我们需要的元素。时间复杂度取决于链表的长度。为O(n)，为了降低这些开销，在Java8中当链表中的元素达到了8个时，会将链表转换为红黑树，在这些位置进行查找的时候可以降低时间复杂度为O(logN)

#### LinkedHashMap

LinkedHashSet和LinkedHashMap在java中由相同的实现。前者仅仅时对后者做了一层包赚，也就是说LinkedHashSet里面有个LinkedHashMap，适配器模式。LinkedHashMap实现了Map接口，即允许放入key为null的元素，也允许插入value为null的元素，从名字上可以看出该容器是LinkedList和HashMap的混合题。也即是说它同时满足HashMap和LinkedList的一些特性，可以将LinkedHashMap看作LinkedList增强的HashMap

事实上，LinkedHashMap是HashMap的直接子类，二者唯一的区别就是LinkedHashMap在HashMap的基础上采用了双向链表的形式将所有的entry连接起来，这样是为了保证元素的迭代顺序和插入顺序相同。LinkedHashMap在形式上和HashMap大致类似，多了个header指向双向链表的头部。该双向链表的迭代顺序是entry的插入顺序。

除了可以保存迭代顺序，这种结构还有一种好处。迭代LinkedHashMap时不需要像HashMap那么遍历整个table，而只需要遍历header指向的双向链表即可。也就是说LinkedHashMap的迭代时间就只跟entry的个数相关。而和table的大小无关。

有两个参数可以影响LinkedHashMap的性能，初始容量和负载系数，初始容量指定了初始table的大小，负载系数用来指定自动扩容的临界值。当entry数量超过capacity*load_factor时，容器将自动扩容并且重新Hash，对于插入较多的场景可以将初始容量设置大些。

get方法，根据指定的key获取对应的value，

put方法，将指定的key，value对添加到map中，该方法首先会对map做出一次查找，看是否包含该元组，如果包含就直接返回，查找过程类似于get方法。如果没有找到，则会通过addEntry方法插入新的Entry。这里的插入有两层含义。从table角度来看，新的entry需要插入到对应的bucket中，当有哈希冲突时，采用头插法将新的entry插入到冲突链表的头部。从header的加澳督来看，新的entry需要插入到双向链表的尾部。

```java
void addEntry(int hash, K key, V value, int bucketIndex){
    if((size >= threshold) && (table[bucketIndex] != null)){
        resize(2*tabke.length);
        hash = (null != key) ? hash(key) : 0;
        bucketIndex = hash & (table.length -1);
    }
    HashMap.Entry<K, V> old = table[bucketIndex];
    Entry<K, V> e = new Entry<>(hash, key, value, old);
    e.addBefore(header);
    size++;
}
```

在添加entry时，使用addBefore方法将新的entry插入到双向链表表头引用header的前面，这样e就成为了双向链表中最后一个元素。

remove方法。作用是删除key值对应的entry，该方法的具体逻辑是在removeEntryForKey中实现的，这个方法会首先找到key值对应的entry，然后删除该entry，修改链表相应的引用。查找过程和get方法类似。

## 三、IO


IO流以`byte`（字节）为最小单位，因此也称为*字节流*。在Java中，`InputStream`代表输入字节流，`OuputStream`代表输出字节流，这是最基本的两种IO流。字节流传输的最小数据单位是字节byte。

如果我们要读写的是字符，并且字符不全是以单字节表示的ASCII字符，那么按照char读写更方便，这种流称为*字符流*。Java提供了`Reader`和`Writer`表示字符流，字符流传输的最小数据单位是char。使用`Reader`和`Writer`，读取的数据源虽然是字节，但是他们内部对若干字节做了编码和解码，然后将字节转换成了字符。本质上能够自动编码和解码的`InputStream`和`OuputStream`。

因此实际上我们也能够自己编写逻辑，实现字节到字符的编码和解码，究竟使用`InputStream`和`OuputStream`还是`Reader`和`Writer`取决于数据源是文本还是其他文件。

Java提供一个FILE的类实现对文件的操作。FILE对象即可以表示文件，也可以表示目录。只有当我们调用FILE对象的某些方法时，才回真正执行磁盘操作。

所有的文件都是以二进制的形式保存的，但是为了便于理解和处理文件，文件就有了文件类型的概念，文件类型以后辍名来体现，每种文件类型都有一定的格式，代表着文件含义和二进制之间的映射关系。对于一种文件类型往往有一种或者多种应用程序可以解读它，进行查看和编辑，一个应用程序往往可以解读一种或者多种文件类型。在操作系统中，一种后缀名往往关联一个应用程序，打开文件时，操作系统查找相应关联的应用程序。

文本类型可以粗略的分为文本文件和二进制文件，基本上文本文件每个二进制字节都是某个可打印字符的一部分，但是二进制文件中，每个字节就不一定表示字符，可能表示的是颜色、字体、声音大小等等。

### 文本文件

文本文件包含的基本都是可打印字符，字符到二进制的映射称为编码，编码有多种方式，应用程序应该如何识别编码方式呢，对于utf-8文件，在文件最开头有三个特殊字节，称为BOM字节序标记。

### 文件系统

* 绝对路径：从根目录到当前文件的完整路径
* 相对路径：相对于当前目录而言

### 文件读写

硬盘访问延时，相比内存是比较慢的。一般读写文件需要两次数据拷贝，从用户态和内核态相互切换，然而这种谢欢是由开销的。

为了提高文件操作效率，应用程序使用缓冲区。读取文件时，即使目前只需要少量内容，在预知还会接着读取时，就会异地读取较多内容，放在读缓冲区，下次读取时，就可以直接从缓冲区读取，减少访问操作系统和硬盘。写文件时，先写到缓冲区，满了以后再一次性调用操作系统写入硬盘。

### Java中的文件概念

在Java中文件不是单独处理的，而是为输入输出设备的一种，Java使用同一的概念处理输入输出操作。这个概念叫做流。输入流可以获取数据，输出流可以到显示终端、文件、网络等。

Java中对流的操作有加密、压缩、计算信息摘要、计算检验和等，接收抽象的流，返回流。

而基本的流按照字节进行读写，没有缓冲区就不方便，使用装饰器设计模式，可以对基本的流增加功能。方便使用，在实际使用时，经常会叠加多个装饰类，为流的传输添加更多的功能，比如对流缓冲、对八种基本类型和字符串进行读写、对流压缩和解压缩等

按照字节处理对文本文件不够友好，能够方便按照字符处理文本数据的类是Reader和Writer

关于文件路径，文件源数据、文件目录、访问权限等等文件属性，Java使用File这个类进行管理

### NIO

NIO （new input and output）是一种看待输入输出不同的方式，他有缓冲区和通道的概念，更接近操作系统。

StandardIO是对字节流的读写，在进行IO之前，首先创建一个流对象，流对象进行读写操作都是按照字节，一个字节一个字节的来读或这些，而NIO把IO抽象成块，类似于磁盘的读写，每次IO操作的单维都是一个块，块被读入内存之后就是一个byte数组，NIO一次可以读或者写多个字节。

#### 流和块

IO和NIO最重要的区别是数据打包和传输的方式，IO以流的方式处理数据，而NIO以块的方式处理数据，面向流的IO一次处理一个字节数据，一个输入流产生一个字节数据，一个输出流消费一个字节数据。为流式数据创建过滤器很容易，连接几个过滤器，以便每个过滤器只负责复杂处理机制的一部分，但是面向流的IO通常非常的慢。

面向块的IO一次处理一个数据块，按块处理数据比按流处理数据要快得多，但是面向块的io缺少一些面向流的简单性。

#### 通道和缓冲区

**通道**是对原IO包中流的模拟，可以通过它读取和写入数据。通道与流的不同之处在于，流只能在一个方向上移动，而通道是双向的，可以用于读写或者同时读写。通道有以下几种类型：

* FileChannel： 从文件中读写数据
* DatagramChannel： 通过UDP读写网络中的数据
* SocketChannel：通过TCP读写网络中的数据
* ServerSocketChannel：可以监听新进来的TCP连接，对每一个新来的连接都会创建一个SocketChannel

#### 缓冲区

发送给一个通道中的所有数据都必须首先放到缓冲区中，同样地，从通道中读取的任何数据都要先读到缓冲区中。也就是说，不会直接对通道进行读写数据而是要先经过缓冲区。

缓冲区实质上是一个数组，但是它不仅仅是一个数据，缓冲区提供了对数据的结构化访问，而且还可以跟踪系统的读写进程

缓冲区包括：

* ByteBuffer
* CharBuffer
* ShortBuffer
* IntBuffer
* LongBuffer
* FloatBuffer
* DoubleBuffer

#### 缓冲区状态变量

在缓冲区中，这几个变量描述了缓冲区当前的状态：capacity最大容量。position当前已经读写的字节数limit还可以读写的字节数

状态变量改变：新建一个大小为8字节的缓冲区，此时位置position为0，而limit=capacity=8，从输入通道中读取5个字节数据写入缓冲区，此时position移动设置为5，limit保持不变。在将缓冲区数据写到输出通道之前，需要先调用flip方法，这个方法将limit设置为当前position并且将position设置为0。然后从缓冲区取4个字节。最后调用clear方法清空缓冲区，此时position和limit都被设置为初始位置。

#### 选择器

NIO常常被叫做非阻塞IO，主要是因为NIO在网络通信中的非阻塞性质而被广泛使用。NIO实现了IO多路复用中的Reactor模型，一个线程Thread使用一个选择器，通过轮询的方式去监听多个通道CHannel上的事件，从而让一个线程就可以处理多个事件。通过配置监听的通道为非阻塞，那么当通道上的IO事件还没到达时，就不会进入阻塞状态一直等待，而是继续轮询其他Channel，找到IO事件已经到达的通道执行。因为创建和切换线程的开销很大，因此，使用一个线程处理多个事件而不是使用一个线程处理一个事件具有更好的性能。

#### IO多路复用——Reactor模型

对于传统的IO模型，其主要是一个Server对接N个客户端，在客户端连接之后，为每个客户端都分配一个执行线程。

传统IO每个客户端连接到达之后，服务端会分配一个线程给该客户端，该线程会处理包括读取数据，解码，业务计算，编码，以及发送数据整个过程。同一时刻，服务器的吞吐量和服务器能提供的线程数是呈线性关系的。

但是这个设计模式有些问题，服务器的并发量对于服务端能够创建的线程数有很大的依赖关系，但是服务器线程却是不能无限增长的。服务端的每个线程不仅需要进行IO读写操作，还需要进行业务计算。服务端在互殴去客户端连接，读取数据以及写入数据的过程都是阻塞类型的，在网络情况不好的时候，会极大降低服务器每个线程的利用率，从而降低服务器吞吐量。

在传统IO模型中，线程在等待连接以及进行IO操作时都会阻塞当前线程，这部分损耗很大，而jdk1.4中提供了非阻塞IO的API。本质上是以事件驱动来处理网络事件的，而Reactor是基于该API提出的一套IO模型。

在Reactor模型中，有4个角色，客户端连接Reactor（响应者）Acceptor（接收者）handler（处理者），Acceptor会不断地接受客户端的连接，然后将接收到的连接交由Reactor响应者进行分发，最后由具体的处理者处理。这种模型如果仅仅使用一个线程池来处理客户端连接的网络读写和业务计算，那么效率上没有什么提升，但是这个模型以事件驱动，能够接受客户端连接和网络读写，以及业务计算进行拆分。从而极大地提升处理效率。因为该模型是异步非阻塞模型，工作线程在没有网络事件时就可以处理其他任务，而不用像传统IO一样阻塞等待

Reactor模型中，由于网络读写和业务操作都在同一个线程中，在高并发情况下，系统瓶颈主要是高频率的网络读写事件处理，大量的业务操作处理。

在多线程进行业务操作的模型下，主要有这样的特点：使用一个线程进行客户端连接的接受以及网络读写事件的处理。在接收到客户端连接后，将该连接交由线程池进行数据的编解码以及业务计算。

这种业务模式相较于前面的模式性能有了很大的提升，主要在于及逆行网络读写的同时，也进行了业务计算从而大大提升了系统的吞吐量，但是，网络读写是一个比较消耗CPU的操作，在高并发的情况下，将会有大量的客户端数据需要进行网络读写，此时，一个线程不足以处理这么多的请求。

对于使用线程池处理业务操作的模型，由于网络读写在高并发的情况下会成为系统的瓶颈。因而提出了一种改进的模型，使用线程池进行网络读写，仅仅使用一个线程专门接受客户端连接。

改进后的Reactor模型将Reactor拆分成为了mainReactor subReactor。第一个及逆行客户端连接的处理，处理完成之后将该连接交由subReactor用来处理客户端的网络读写。这里的subReactor则是使用一个线程池来支撑的，其读写能力将会随着线程数的增多而增加。对于业务操作，也是使用一个线程池，而每个业务请求都只需要进行编解码和业务计算。

#### channel

通道是被建立的一个应用程序和操作系统交互事件、传递内容的渠道。一个通道会有一个专属的文件状态描述符。那么既然是和操作系统进行内容的传递，那么说明应用程序可以通过通道读取数据，也可以通过通道像操作系统写数据。

所有被Selector选择器注册的通道只能是继承了SelectableChannel类的子类。

* ServerSocketChannel：应用服务器程序的监听通道，只有通过这个通道，应用程序才能向操作系统注册支持多路复用IO的端口监听，同时支持UDP和TCP协议。
* SocketChannel：TCP Socket套接字的监听通道，一个Socket套接字对应了一个客户端IP。端口到服务器IP
* DatagramChannel：UDP数据报文的监听通道

#### Buffer

数据缓冲区。

### 序列化和反序列化

序列化就是将内存中的Java对象持久的保存到一个六种，反序列化就是从流中恢复Java对象到内存，序列化和反序列化，主要目的是：一对象状态持久化，网络远程调用传递和返回对象。Java默认的序列化不够通用，现在通常使用json

### File类

**构造方法**

```java
File(String pathname)
File(String parent, String child)
File(File parent, String child)
```

* pathname：表示完整路径，相对路径或绝对路径。路径可以是已存在的，也可以是不存在的
* parent / child：父目录子目录
* 新建一个File对象，并不会实际创建一个文件，只是创建一个表示文件或者目录的对象。

**方法**

| 方法名称                                                     | 说明                                                         |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| String getName()                                             | 获取文件名称                                                 |
| boolean isAbsolute()                                         | 是否是绝对路径                                               |
| String getPath()                                             |                                                              |
| String getAbsolutePath()                                     |                                                              |
| String getCanonicalPath()                                    | 返回标准路径名，去掉冗余                                     |
| String getParent()                                           | 返回父目录路径                                               |
| File getParentFile()                                         | 返回父目录的file对象                                         |
| exists()                                                     | 是否存在                                                     |
| isDirectory()                                                | 是否是目录                                                   |
| isFile()                                                     | 是否为文件                                                   |
| long length()                                                | 文件长度，字节                                               |
| long lasModified()                                           | 最后修改时间                                                 |
| setLastModified(long timr)                                   | 设置最后修改时间                                             |
| isHidden()                                                   | 是否是隐藏                                                   |
| canExecute()                                                 | 是否可以执行                                                 |
| canRead()                                                    | 是否可读                                                     |
| canWrite()                                                   | 是否科协                                                     |
| setReadOnly()                                                | 设置只读                                                     |
| setReadable(boolean readable, boolean ownerOnly)             | 修改读权限                                                   |
| setWriteable(boolean writable, boolean ownerOnly)            | 修改写权限                                                   |
| setExecutable(boolean executable, boolean ownerOnly)         | 修改可执行权限                                               |
| createNewFile()                                              | 创建一个新文件                                               |
| createTempFile(String prefix, String suffix, File directory) | 创建临时文件，可以指定前缀后缀和目录                         |
| delete() / deleteOnExit()                                    | 删除文件、程序退出时删除                                     |
| renameTo(File dest)                                          | 重命名文件                                                   |
| mkdir() / mkdirs()                                           | 创建目录，第二个会创建必须的中间父目录，第一个不会           |
| String[] list() File[] listFiles(FilenameFilter filter)      | 返回直接子目录或者文件，不会返回子目录下的文件。一个返回文件名数组，一个返回File对象数组，可以添加过滤参数 |
|                                                              |                                                              |
|                                                              |                                                              |

### 二进制文件

以二进制方式读写的流主要有：

* InputStream / OutputStream 抽象类
* FileInputStream / FileOutputStream 输入源和输出目标是文件的流
* ByteArrayInputStream / ByteArrayOutputStream 输入源和输出目标是字节数组的流
* DataInputStream / DataOutputStream 装饰类，按照基本类型和字符串而非只是字节读写流
* BufferedInputStream / BufferedOutputStream 装饰类，对输入输出流提供缓冲功能

#### InputStream / OutputStream

从流中读取的基本方法：

```java
public abstract int read() throws IOException;
public int read(byte b[]);
public int read(byte b[], int off, int len);
public void close();
```

read从流中读取下一字节，返回类型是int，但是取值在0-255之间，当读取到流结尾的时候，返回值为-1，如果流中没有数据，方法会阻塞直到数据到来、流关闭、或者异常出现。

一次读取多个字节，读入的字节会放入到参数数组中，以此类推，一次最多读入的字节个数为数组b的长度，但实际可能小于，返回值为实际读入的字节个数。

到最后应该关闭流调用close方法

从流中读取的高级方法：

```java
long skip(long n);
int available();
synchronized void mark(int readlimit);
boolean markSupported();
synchronized void reset();
```

skip跳过输入流的n个字节，因为输入流中剩余字节数可能不到n，所以返回值为实际掠过的字节个数，

available方法返回下一次不需要组测就能读取到的大概字节个数，默认实现是返回0，子类会根据具体情况返回适当的值。

一般的流读取都是一次性的，且只能往前读，不能往后读，但有时可能希望能够先看一下后面的内容，然后根据情况再2重新读取。

后面三个方法就是用于从读过的流中重复读取，先用mark方法将当前位置标记下来，读取一些字节后，希望重新开始读的时候再调用reset方法

是否可以支持需要看markSupported的返回值判断。

写入流中的基本方法：

```java
write(int b);
flush();
close();
```

向流中写入一个字节，flush方法将缓冲而未实际写入的数据进行实际写入，再BufferedOutputStream中，调用会将其缓冲区内容写到其装饰的流中，并调用该流的flush方法。

#### FileInputStream / FileOutputStream

关于文件输入输出流，构造时有两个参数，一类时文件路径，可以是File对象，也可以是文件路径名，如果文件已经存在，需要指定是追加还是覆盖，使用文件输出流会实际打开文件。

文件输入流

### 文本文件和字符流

Java中主要的字符流有这几种：

* Reader / Writer
* InputStreamReader / OutputStreamWriter 适配器类，输入时InputStream 输出是OutputStream，将字节流转化为字符流
* FileReader / FileWriter
* CharArrayReader / CharArrayWriter 输出输出都是字符数组的字符流
* StringReader / StringWriter 输入和输出目标都是字符串的字符流
* BufferedReader / BufferedWriter
* PrintWriter 装饰类，将基本类型和对象转会为其字符串形式输出的

### Unix IO模型

一个输入操作通常包括两个阶段：等待数据准备好，从内核向进程复制数据

对于一个套接字上的输入操作，第一步通常涉及等待数据从网络中到达，当锁等待分组到达时，它被复制到内核中的某个缓冲区，第二步就是把数据从内核缓冲区复制到应用进程缓冲区。

Unix下有五种IO模型。阻塞式IO，非阻塞式IO，IO复用，信号驱动式，异步IO，

#### 阻塞式IO

应用程序被阻塞，直到数据复制到应用进程缓冲区中才返回。应该注意到，在阻塞的过程中，其他程序还可以执行，因此阻塞不意味着整个操作系统都被阻塞。其他程序还可以执行，因此不消耗CPU时间，这种模型的执行效率较高、

#### 非阻塞式IO

应用进程在执行系统调用之后，内核返回一个错误码，应用程序可以继续执行，但是需要不断的执行系统调用来获知IO是否已经完成。这种方式称为轮询。由于CPU要处理更多的系统调用。因此这种模型是比较低效的。

#### IO复用

使用select或者poll等待数据，并且可以等待多个套接字中的任何一个变为可读，这一过程会被阻塞，当某一个套接字可读时返回。之后在使用recvfrom把数据从内核复制到进程中。它可以让单个进程具有处理多个IO时间的能力，又被称为Event Driven IO即事件驱动IO，如果一个WEB服务器没有IO复用，那么每个Socket连接都需要创建一个线程去处理。如果同时有几万个连接，那么就需要创建相同数量的线程。并且相比于多进程和多线程技术，IO复用不需要进程线程创建和切换开销。系统开销更小。

#### 信号驱动IO

应用进程使用sigaction系统调用，内核立即返回，应用程序可以继续执行，也就是说等待数据阶段，应用进程式非阻塞的，内核在数据到达时，向应用进程发送sigio信号。应用进程在收到信号后在信号处理程序中调用recvfrom将数据从内核复制到应用进程中。相比于非阻塞式IO的轮询方式，信号驱动的IOCPU利用效率更高。

#### 异步IO

进行aio_read系统调用会立即返回，应用程序继续执行，不会被阻塞，内核会在所有操作完成之后向应用进程发送信号。异步IO与信号驱动IO的区别在于，异步IO的信号是通知应用进程IO完成，而信号驱动IO的信号是通知应用进程可以开始IO

#### IO模型的比较

同步IO应用程序在调用recvfrom操作时会被阻塞，而异步IO不会。阻塞式IO、非阻塞式IO、IO复用、信号驱动IO都是同步IO，虽然非阻塞式IO和信号驱动IO在等待数据阶段不会被阻塞，但是在之后将数据从内核复制到应用进程这个操作会被阻塞

#### IO多路复用

阻塞式IO和非阻塞式IO：程序级别的概念，主要描述的式程序请求操作系统IO操作后，如果IO资源没有准备好，那么程序该如何处理的问题。阻塞式IO会等待，而非阻塞式IO会继续执行。并且使用线程轮询

同步IO和非同步IO：操作系统级别的，主要描述的式操作系统在收到程序请求IO操作后，如果IO资源没有准备好，应该如何相应程序的问题，前者不响应，直到IO资源准备好以后，后者返回一个标记，当IO资源准备好以后，后者返回一个标记，好让程序和自己知道以后的数据往哪里通知。当IO资源准备好以后，再用事件机制返回给程序。

然而传统的IO通信大多是阻塞模式的。客户端向服务器发送出请求以后，客户端会一直等待，直到服务器端返回结果或者网络出现问题，服务器端同样的，当在处理某个客户端发来的请求时，另一个客户端发来的请求会等待，直到服务器端的这个处理线程完成上一个处理。

那么这样就会存在这样的问题，同一时间，服务器只能接受来自于客户端A的请求信息，虽然客户端A和客户端B的请求时同时进行的，但是客户端B发送的请求信息只能等到服务器接受完A的请求数据后，才能被接受。由于服务器一次只能处理一个客户端请求，当处理完成并返回后，才能进行第二次请求的处理，很显然，这样的处理方式在高并发的情况下，是不能采用的。

#### 多线程伪异步方式

当服务器收到客户端的请求后，读取到所有请求数据后，将这个请求送入一个独立线程进行处理，然后主线程继续接受客户端Y的请求。客户端一侧，也可以使用一个子线程和服务器端进行通信，这样客户端主线程的其他工作就不受影响了，当服务器端有响应信息的时候再由这个子线程通过监听模式、观察模式等通知主线程。

但是使用线程来解决这个问题还是有局限性的。虽然在服务端请求的处理交给了一个独立的线程进行，但是操作系统通知accept的方式还是单个的，也就是说服务器接收到数据报文后的业务处理过程可以多线程，但是数据报文的接受还是需要一个一个来的。

在linux系统中，可以创建的线程是有限的。我们通过cat proc sys等命令可以查看能创建的最大线程数，线程数越多，CPU切换所需的时间也就越长，用来处理真正业务的需求也就越少。

创建一个线程是有较大的资源消耗的，JVM创建一个线程的时候，即使这个线程不做任何的工作，JVM都会分配一个堆栈空间。如果应用程序中大量使用长连接的话，线程是不会关闭的，这样系统资源的消耗更容易失控。如果真的向单纯使用线程解决阻塞问题。自己就可以算出来一个服务器可以一次接受多大的并发了。单纯用线程解决这个问题不是最好的方法。

## 四、Maven

### Maven介绍
Maven是一个Java项目的管理和构建工具。

* Maven使用`pom.xml`定义项目的内容，并且使用预设的目录结构。
* 在Maven中声明一个依赖项就可以自动下载并导入到`classpath`（项目路径中）。
* Maven使用`groupId artifactId version`唯一定位一个依赖（公司名称，项目名称，版本号）。

一个Maven项目的目录结构如下：

```ascii
a-maven-project				 项目名称
├── pom.xml					项目的描述文件
├── src				  		项目源代码存放处
│   ├── main		
│   │   ├── java		 	存放Java源代码
│   │   └── resources		 存放资源文件的目录
│   └── test
│       ├── java		 	 存放测试用的代码
│       └── resources	 	 存放测试用的资源
└── target					所有编译、打包生成的文件
```

**依赖管理**

Maven解决了依赖管理的问题，比如说我们的项目依赖一个文件，而这个文件又依赖另一些文件，那么Maven就可以把这一连串依赖的文件帮我们下载并配置号。

**依赖关系**

Maven定义了几种依赖关系，分别是编译中、测试时、运行中、提供四种依赖关系

| 范围       | 说明                         |
|----------|----------------------------|
| compile  | （默认情况）编译时需要用到              |
| test     | 编译测试文件时需要用到                |
| runtime  | 运行时需要，编译时不需要               |
| provided | 编译时需要用到，但是运行时由JDK或者其他服务器提供 |

### Pom.xml

* pom.xml存放在由maven管理的项目文件根目录中。
* pom.xml中包含当前项目的信息以及用于构建编译项目的各种配置的详细信息。
* pom.xml包含项目执行目标和插件，在执行编译部署等任务时，Maven从当前目录中查找pom.xml，获取所需的配置信息。

#### 父类POM

Super Pom.xml文件，又叫超级Pom，父类pom，是Maven项目的默认pom配置，所得的项目都默认继承自该pom文件。
我们自己配置的pom只包含自己指定的配置，而不包含我们从父类继承的maven配置。
我们使用`mvn help:effective-pom`命令可以查看当前项目包含了默认pom文件的全部配置。

### 构建生命周期

构建生命周期(Build Life Cycle)指的是项目从编译到构建为字节码文件的整个流程。整个构建阶段的流程可以分为以下几个部分

| 阶段                | 描述                        |
|-------------------|---------------------------|
| prepare-resources | 资源复制阶段，在此阶段自定义资源复制到哪个地方   |
| validate          | 验证信息，验证项目是否正确，所有必要的信息是否可用 |
| compile           | 源代码编译阶段                   |
| Test              | 测试代码                      |
| package           | 打包项目的源代码，创建jar/war包       |
| install           | 将程序包安装到本地或者远程的maven仓库     |
| deploy            | 运行部署程序                    |

### 默认生命周期

| 阶段                    | 描述                          | 
|-----------------------|-----------------------------|
| validate              | 验证项目是否正确，检查所有必要的信息          |
| initialize            | 初始化构建状态                     |
| generate-sources      | 生成编译阶段及之后阶段需要的所有源代码         |
| process-sources       | 处理源代码                       |
| generate-resources    | 生成要包含在软件包中的资源文件             |
| process-resources     | 将资源复制到目标目录，为打包阶段做好准备        |
| compile               | 编译源代码                       |
| process-classes       | 对编译后生成的类文件进行处理，例如对字节码增强和优化  |
| generate-test-sources | 生成测试源代码                     |
| process-test-sources  |                             |
| test-compile          |                             |
| process-test-classes  |                             |
| test                  | 使用合适的测试框架对代码进行测试            |
| prepare-package       | 在实际对代码打包之前需要进行的操作           |
| package               | 将代码打包为jar、war等文件包           |
| pre-integration-test  | 执行集成测试之前需要的操作               |
| integration-test      | 讲程序包部署到可运行集成测试的环境中，并且进行集成测试 |
| post-integration-test | 集成测试后需要的操作，如清理环境            |
| verify                | 运行任何检查验证包是否有效并符合质量标准        |
| install               | 将软件安装到本地仓库                  |
| deploy                | 将软件复制到远程仓库                  |

* 当通过maven命令调用整个流程的某一个阶段时，maven会运行该阶段以及之前的所有阶段。
* 根据软件包的类型jar/war，不同的maven构建目标会采用maven流程的不同阶段

### 站点生命周期

site阶段，用于？

### 构建流程的配置

构建（build）项是是一组配置值。制定了项目采用maven构建整个流程的各种配置。
在不同环境下我们可以使用profiles使用不同的构建配置。这样maven就可以在不同环境下采用不同的配置。

### 构建的配置文件

构建配置文件有三种类型
在每个项目的根目录pom.xml中
每个用户中%USER_HOME%/.m2/setting.xml
每台计算机全局配置%M2_HOME%/conf/setting.xml

我们可以通过多种方式激活maven的配置文件，默认情况下，maven配置文件激活的顺序是项目——用户——全局

* 在控制台通过命令激活
* 通过maven设置
* 基于环境变量
* 操作系统的设置

### maven插件

maven本身只是一个容纳插件的容器，每个构建任务都是由插件完成的，
这些插件通常用于编译源代码、单元测试、构建项目文档、创建项目构建成果报告、
将源代码打包为jar/war文件等任务。

插件本身提供了许多构建目标，我们可以通过mvn命令指定某个插件，执行某项构建目标。
通用的语法如下所示：
mvn [plugin-name]:[goal-name]

#### maven插件类型

1. 构建项目的插件
2. 创建报告的插件

### maven外部依赖包

如果任何远程仓库和中央仓库都没有项目对应的依赖包，
那么我们就需要maven为我们提供的外部依赖包功能。
这样的话maven就能为我们管理本地依赖文件。

```xml

<dependency>
  <groupId>ldapjdk</groupId>
  <artifactId>ldapjdk</artifactId>
  <!--  将范围指定为本地系统-->
  <scope>system</scope>
  <version>1.0</version>
  <!--  指定依赖包相对于项目位置的系统路径-->
  <systemPath>${basedir}\src\lib\ldapjdk.jar</systemPath>
</dependency>
```

### maven创建项目文档网站

maven通过site插件能够为项目生成说明文档的静态网页。在项目目录的target/site文件夹下。随后我们可以将这些静态网页部署到服务器上。

### maven项目模板

maven为用户提供了多个项目模板，帮助用户快速创建各种不同类型的Java项目。
maven使用archetype插件来完成这个功能，这个插件就能根据模板来创建项目结构。

### maven自动化构建

假设我们有多个项目，其中一个项目依赖另一个或多个项目，这几个项目之间有复杂的依赖关系，我们如果想要构建这样的项目，就会比较费时费力，在构建时就需要
人工指定顺序，而现在maven能帮助我们自动构建这些项目，而不必每次构建时都需要关心他们之间的依赖关系。
对于这种复杂依赖的多个项目，我们可以采用以下三种方式解决：

1. 在构建项目之前整理清晰每个项目的依赖关系，对于依赖其他项目的项目，添加一个post-build目标。在构建该项目之前先构建其所以来的项目
2. 通过持续集成工具自动管理构建
3. 采用父项目聚合所有子项目并规定构建顺序

### maven依赖管理

## 五、JVM

### 1、类加载机制

类加载器就是加载其他类的类，她负责将字节码文件加载到内存，创建Class对象。

运行Java程序，需要执行Java这个命令，指定包含main方法的完整类名，类路径。Java在运行时，会根据类的完全限定名寻找并加载类，寻找的方式是在系统类和指定的类路径中寻找。负责加载类的就是类加载器，一般程序运行时，有三个：

1. 启动类加载器：是Java虚拟机实现的一部分，负责加载Java的基础类
2. 扩展类加载器：负责加载Java的一些扩展类
3. 应用程序类加载器：负责加载应用程序的类
