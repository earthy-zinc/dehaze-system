# 一、并发

## 并发问题三要素

多线程是一种并发的操作，并发操作会出现一些问题，这些问题有三个特点：

* 可见性：由CPU的缓存引起，在多CPU的情况下，每个CPU都有自己的高速缓存，由于CPU和主存之间速度不匹配的问题，其中一个CPU对共享变量的修改并不会立即写回主存中，此时另一个线程从主存中读取该值并加载到另一个CPU高速缓存中。另一个线程并没有看到该变量已经被修改过了，这就是线程之间共享数据的可见性问题。
* 原子性：一个操作或多个操作，是不能被任何因素打断的执行。否则就不执行。由于操作系统分时复用CPU，就会导致这些操作可能会被打断。
* 有序性：程序执行按照代码的先后顺序执行。编译程序会优化指令执行次序，会导致有序性被破坏。发生指令重新排序
  * 编译器优化的重排序，编译器在不改变单线程程序语义情况下，重新安排语句执行次序
  * 指令级并行重排序，如果不存在数据依赖性，处理器可以改变语句对应的机器指令的执行顺序
  * 内存系统重排序，处理器有高速缓存和读写缓冲区，写回内存时间在微观上不确定，加载和存储操作可能是在乱序进行

## Java解决并发问题方法

### 1）关键字

#### volatile

保证变量的内存可见性。

synchronized负责给线程加锁，一把锁只能同时被一个线程获取，没有获取到锁的线程只能等待。每个实例对对应有自己的一把锁，不同实例之间互不影响。但是锁的如果时类级别的化，锁class、锁static，所有该类的实例对象共用一把锁。用synchronized修饰的方法，无论方法是正常执行完毕还是抛出异常，都会释放锁。

##### 对象锁

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

##### 类锁

synchronized修饰静态的方法或者制定锁对象为class对象

```java
synchronized(SynchronizedObjectLock.class){
    // 代码
}
```

这里锁住的就是类级别的对象

##### 原理分析

###### 加锁和释放锁

我们在加入synchronized关键字后，编译器会给对应的代码加入两个指令，分别是monitorenter 和 monitorexit，这两个指令会让对象在执行时使其锁计数器加减1，每个对象同时只与一个锁相关联。但是一个monitor（监视器）在同一时间只能被一个线程获得，一个对象在尝试获得与这个对象相关练的monitor锁的所有权时，需要进行如下判断：

* 监视器计数器为0，说明目前还没有被获得，那么这个线程就会立刻获得并把锁的计数器+1，其他线程就需要等待锁的释放。
* 如果监视器已经拿到了这个锁的所有权，又重入了这锁，锁的计数器就+1，随着重入次数增多，会一直增加。
* 锁被别的线程获取，需要等待释放。

对于释放监视器锁，计数器-1，再减完后如果计数器变为0，则代表该线程不再拥有锁，释放该锁。

###### 可重入原理

可重入锁：同一个线程在外层方法获取锁的时候，再进入该线程的内层方法会自动获取锁。不会因为之前已经获得还没有释放而阻塞。在同一个锁线程中，每个对象拥有一个monitor计数器，当线程获取该对象锁，计数器加一，释放锁，计数器减一。

##### JVM锁优化

对于monitorenter和monitorexit字节码指令，依赖于底层操作系统的MutexLock实现的，但是由于使用MutexLock需要将当前线程挂起并且从用户态切换到内核态执行，切换代价大，现实中，同步方法时运行在单线程中，调用锁会严重影响程序性能，因此JVM对锁引入了优化。

* 锁粗化：减少不必要的解锁和加锁操作，这些紧连在一起的锁可以扩大成一个大锁。
* 锁消除：通过运行Java即时编译器的逃逸分析来消除一些锁。
* 轻量级锁：假设在真实情况下我们程序中大部分同步代码处于无锁竞争状态，在这种情况下避免调用操作系统的互斥锁，而是依靠一条原子CAS指令完成锁的获取和释放，在存在锁竞争时，执行CAS指令失败的线程调用操作系统的互斥锁进入阻塞状态。
* 偏向锁：是为了在无锁竞争的情况下，谜面在锁获取过程中执行不必要的CAS比较并交换指令
* 适应性自旋锁：当线程获取轻量级锁失败时，在进入重量级锁之前会进入忙等待然后再次尝试，尝试次数一定后如果还没有成功则调用互斥锁进入阻塞状态。

#### synchronized

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

#### final



### 2）Happens-Before (在其之前要发生)规则

* 单一线程原则：在一个线程内，程序前面的操作要先于后面的操作发生
* 管程锁定原则：对于同一个锁，解锁操作要先于加锁操作
* 变量读写原则：对于有volatile关键字的变量，写操作要先于读操作完成。
* 线程启动原则：线程对象的start()方法调用先于此线程的每一个动作
* 线程加入规则：线程对象的结束先于其他线程的加入join()方法
* 线程中断规则：对某个线程调用中断方法先于被中断线程检测到中断时间的发生
* 对象终结原则：对象初始化完成先于它的结束方法finalize()
* 传递性：操作A先于操作B，操作B先于操作C，那么操作A先于操作C

## 线程安全

### 1）概念

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

### 2）解决方法





## Java多线程的实现

### 1）线程状态的分类

![](E:\StudyDoc\6.阅读笔记\图片\java_thread_condition.png)

* New 新建状态：指的是一个线程被创建出来，但没有启动。
* Runnable 可运行的状态：指的是线程可能在运行，也可能在等待时间片轮转。
* Blocked 阻塞状态：等待获取一个排他锁
* Time Waiting 等待状态：在一定时间后会被系统自动唤醒，阻塞和等待的区别是阻塞是被动的，是需要获取一个资源。而等待是主动的，通过调用Object.wait() Thread.sleep()等进入
* Waiting 一直等待状态：等待其他线程唤醒
* Terminated 结束状态：结束了任务或者产生了异常

### 2）执行多线程的方法

在Java中总得来说就是实现接口、继承Thread。实现接口更好一些，因为接口可以多实现，一个类可能要求并不高，单独为其创建一个线程开销过大。具体方法如下：

* 自定义的类继承（extends）Thread类，然后覆写它的run方法。然后定义并启动这个类
* 创建Thread实例时，传入一个Runnable实例。也就是自定义一个类实现Runnable接口，需要实现run方法
* 自定义类实现Callable接口，但它可以有返回值。创建Thread实例时，传入这个类的实例

### 3）线程机制

* Excutor 线程执行器：能够管理多个异步任务的执行，主要有三种：缓冲线程池、固定线程池、单例线程池
* Daemon 守护线程：是程序运行时在后台提供服务的线程，所有非守护线程结束后，程序会终止。
* sleep方法：会休眠当前正在执行的线程
* yield方法：说明当前线程中涉及到线程安全重要的部分已经完成，接下来可以切换给其他线程运行
* interrupt方法：会中断当前线程，之后我们可以调用interrupted() 方法来判断线程是否处于中断状态。如果该线程处于阻塞或等待状态，会抛出中断异常从而提前结束该线程。这种中断不能中断因I/O请求阻塞和同步锁阻塞。
* join方法：可以让调用join方法的线程等待等待主线程结束。

### wait/notify

Java在object类中定义了一些线程协作的基本方法，wait和notify

每个对象都有一把锁和等待对列，一个线程在进入synchronized代码块时，会尝试获取锁，获取不到的话会把当前线程加入等待队列中，除了用于锁的等待队列，每个对象还有另一个等待对列，该队列用于线程间的协作，调用wait就会把当前线程放到条件队列上并阻塞，表示当前线程执行不下去了，他需要等待一个条件，这个条件需要其他线程来改变，当其他线程改变了条件了以后，应该调用notify方法。因此notify做的事情就是从条件队列中选择一个线程，将其从队列中移除并唤醒，notifyAll能够移除条件队列中所有线程并且全部唤醒。

假设现在有两个线程，一个主线程和一个等待线程，协作的条件变量是fire，等待线程等待该变量变成True，在false时调用wait方法等待，主线程会负责设置该变量并且调用notify唤醒等待线程

这样两个线程都需要访问协作的变量fire，所以相关代码都需要被synchronized保护，而wait和notify方法只能在synchronized代码块内调用，如果调用这些方法时，当前线程没有对象锁，就会冒出异常。

wait的具体过程是：

1. 把当前线程放入等待队列，释放对象锁，阻塞等待，线程状态变为waiting
2. 等待时间到或者被其他线程唤醒，这时需要重新竞争对象锁，能够获得锁的话，线程就可继续，并从wait调用中返回

因此调用notify方法只是把线程从条件队列中移除，但是并不会释放对象锁。

### 生产者消费者模式

在生产者消费者模式中协作的共享变量是队列，生产者往队列上放数据，如果满了就等待，而消费者从队列中取数据，队列为空也等待。

Java中提供了专门的阻塞队列实现

* 接口BlockingQueue BlockingDeque
* 基于数组的实现类
* 基于链表的实现类LinkedBlockingQueue LinkedBlockingDeque
* 基于堆的实现类PriorityBlockingQueue

## 并发容器

Java中普通的容器对象在多线程的环境下是不安全的，因此我们可以使用线程安全的容器，他们是给所有容器方法添加上synchronized来实现安全的。这样所有的方法调用就变成了原子操作。但是在调用的时候仍需要注意：

1. 复合操作——多次操作，而这些操作是需要原子化的
2. 伪同步——保护线程安全所做的处理实际上作用在了不同的对象上
3. 迭代——单个操作安全，迭代不安全

Java中有一些专门为并发设计的容器类。

### CopyOnWriteArrayList

* 这个类是线程安全的，可以被多个线程并发访问、
* 迭代器不支持修改操作，但是也不会抛出ConcurrentModificationException
* 以原子方式支持一些复合操作

基于synchronized的同步容器，迭代时，需要对整个列表对象枷锁，否则会抛出异常，而这个类就没有这个问题。因为这个类的迭代器不支持修改，当然也就不能支持一些依赖迭代器修改方法的那些操作。

这个类的内部也是一个数组，但是这个数组是以原子的方式被整体更新的，每次修改操作，都会新建一个数组，赋值原数组的内容到新的数组，在新数组上进行需要的修改，然后以原子的方式设置内部的数组引用，这就是写时拷贝。

所有的读操作，都是先拿到当前引用的数组，然后直接访问该数组，在度的过程中，可能内部的数组引用已经被修改，但是不会影响读操作，仍然能够访问原数组的内容。

也就是说，数组的内容一直都是只读的，写操作都是通过新建数组，然后原子性的修改数组引用来实现的。

### CopyOnWriteArraySet

### ConcurrentHashMap

是HashMap的并发版本，这个有以下几个特点：

* 并发安全
* 直接支持一些原子复合操作
* 支持高并发、读操作完全并行、写操作一定程度的并行
* 与同步容器相比，迭代不需要加锁
* 弱一致性

同步容器使用的是synchronized，所有的方法，竞争同一个锁，但是ConcurrentHashMap采用分段锁技术，将数据分为多个端，而每一个端都有一个独立的锁，每一个段就相当与一个独立的哈希表。分段的依据也是哈希值。无论是保存键值对还是根据键来查找，都需要先根据哈希值映射到端，再在段对应的哈希表上进行操作。

采用分段锁，可以大大提高并发度，多个段之间可以并行读写，默认情况下，段时16个。对于写操作，需要获取锁，不能并行，但是读操作可以，多个读操作可以并行，写的同时也可以读。

## 异步执行任务

异步执行任务，也就是说，将任务的提交和任务的执行相分离。执行服务封装了任务执行的细节，对于任务提交者而言，它可以关注任务本身，如提交任务获取结果、取消任务。不需要关系任务执行的细节，如线程的创建、任务调度、线程关闭。也就是说，有两个角色，一个任务的提交者。一个是任务的执行者

* Runnable和Callable表示要执行的那个异步的任务
* Executor和ExecutorService表示执行任务的执行器
* Future表示异步任务执行的结果

### Runnable Callable

都表示一个需要执行的任务，对于一个方法实现了RUnnable或者Callable接口，就能够变成能够异步执行的任务

### Executor ExecutorService

Executor表示一个最简单的执行服务，可以执行一个实现了Runnable接口的方法，没有返回结果，接口没有限定任务应该如何执行，可能是创建一个新的线程，也可能是复用线程池中的某个线程。也可能是在调用者线程中执行。

ExecutorService 就扩展了Executor，定义了更多的服务，有提交任务的方法submit，返回值为Future，返回后表示任务已经提交，但不代表已经执行。

通过Future可以查询异步任务的状态、获取最终的结果、取消任务等。

### Future

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

## 线程池

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



## 组合式异步编程

一个软件系统的很多功能可能会被切分为小的服务，在对外展示具体页面时，可能会调用多个服务。为了提高性能充分利用系统资源，这些对外部服务的调用一般是异步的，尽量使并发的。

CompleteableFuture是一个具体的类，实现了两个接口，一个使Future，另一个使CompletionStage，Future表示异步任务的结果，而CompletionStage字面意思就是完成任务的阶段。多个阶段可以用用流水线的方式组合起来，对于其中一个阶段，有一个计算任务，但是可能要等待其他一个或者多个阶段完成才能开始，等待它完成后，可能会触发其他阶段开始运行。

## 锁

### 乐观锁和悲观锁

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

### 自旋锁和适应性自旋锁

阻塞或者唤醒一个Java线程需要操作系统切换CPU状态，这种状态转换需要耗费处理器事件，如果同步代码块中的内容很简单，状态转换消耗的事件甚至会比执行代码的时间还要长。在许多场景中，同步资源的锁定时间很短，为了这一小段时间切换线程，线程挂起和回复现场的花费有可能让系统得不偿失。如果物理机器有多个处理器，能够让两个或者以上的线程同时并行执行，我们就可以让后面那个请求锁的线程不放弃CPU执行时间，看看持有锁的线程是否会很快地释放锁。

那么等待锁释放的那个线程，我们就需要让线程自旋，也就是循环询问是否有锁，如果在自旋完成后，前面锁定同步资源的线程已经释放了锁，那么当前线程就可以不必阻塞而是直接获取同步资源，从而避免切换线程的开销，这就是自旋锁。

**缺点**

自旋锁等待虽然避免了线程切换的开销，但是要占用处理器的时间。如果锁被占用的时间过长，那么自旋的线程就只能白白浪费处理器资源。所以自旋等待的时间必须有一定的限度，如果自旋超过了限定次数，就应该挂起线程，释放处理器资源。

自旋锁的实现同样采用CAS对比后交换愿你指令完成。

### 无锁、偏向锁、轻量级锁、重量级锁

### 公平锁和非公平锁

公平锁指的是多个线程按照申请所得顺序来获取锁，线程直接进入队列中排队，队列中的第一个线程才能够获取锁。公平锁的优点是等待锁的线程不会饿死。缺点是整体吞吐效率相对非公平锁要低。等待队列中除了第一个线程以外的所有线程都会阻塞。CPU唤醒阻塞线程的开销比非公平锁要大。

非公平锁是多个线程加锁时直接尝试获取锁，获取不到时才会到队列队尾等待，但是如果此时的锁刚好可用，那么这个线程就可以无需阻塞直接获取到锁，所以非公平锁有可能出现这种情况：后申请锁的线程先获取到锁。非公平锁的有点事可以减少唤起线程的开销，整体吞吐效率高，因为线程有几率不阻塞直接获取到锁，CPU不必唤醒所有线程逐一询问。但是缺点是处于等待队列中的线程可能会饿死。

### 可重入锁和非可重入锁

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

### 独享锁和共享锁

独享锁也叫做排他锁，是指该锁一次只能被一个线程所持有，如果线程对数据加上排他锁后，其他线程则不能再对数据加任何类型的锁。获取排他锁的线程既能够读取数据也能够修改数据。

共享锁指的是该锁可以被多个线程所持有，如果线程对数据加上共享锁后，其他线程只能对数据加上共享锁而不能加排他锁，获得共享锁的线程只能够读数据而不能够写数据。

## CAS

### 概念

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



### 缺点

CAS的方式是乐观锁，而synchronized是悲观锁，乐观锁解决并发问题在通常情况下性能更优。但是CAS也存在一些问题。

#### ABA问题

CAS对比后交换指令，需要在操作值的时候，检查值有没有变化，如果没有发生变化则更新，但是如果一个值原来是A，变成了B，随后又变成了A，那么使用CAS进行检查时就会发现他的值没有发生变化，但是实际上却已经变化了。

ABA问题我们可以采用版本号来解决，在变量前面追加上版本号，每次变量更新的时候，把版本号加1。

#### 循环开销时间大

自旋的CAS如果长时间不能够成功，会给CPU带来很大的开销，如果JVM能够支持处理器提供的pause指令，那么效率则会有一定的提升。pause指令的作用是：1、延迟流水线执行命令，使CPU不会消耗过多的资源。2、避免在退出循环时因为存在内存顺序冲突而引起CPU流水线被清空，从而提高CPU执行效率。

#### 只能保证一个共享变量的原子操作

对于多个共享变量的操作，循环CAS无法保证原子性，需要使用锁。

### 解决方案

#### AtomicStampedReference

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

## Unsafe类

### 概念

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

## AtomicInteger

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

## 原子更新类

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

## LockSupport

## AQS

## ReentrantLock

## ReentrantReadWriteLock

# 二、集合


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

## 1、List

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

### ArrayList

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



## 2、Set



## 3、Map

* `Map<E>`接口主要有三种实现：EnumMap TreeMap HashMap
* Map有键值的概念，一个键映射到一个值，Map按照键存储和访问值，键不能重复，即一个键只会存储一份，给同一个键重复设值会覆盖原来的值，



# 三、IO


IO流以`byte`（字节）为最小单位，因此也称为*字节流*。在Java中，`InputStream`代表输入字节流，`OuputStream`代表输出字节流，这是最基本的两种IO流。字节流传输的最小数据单位是字节byte。

如果我们要读写的是字符，并且字符不全是以单字节表示的ASCII字符，那么按照char读写更方便，这种流称为*字符流*。Java提供了`Reader`和`Writer`表示字符流，字符流传输的最小数据单位是char。使用`Reader`和`Writer`，读取的数据源虽然是字节，但是他们内部对若干字节做了编码和解码，然后将字节转换成了字符。本质上能够自动编码和解码的`InputStream`和`OuputStream`。

因此实际上我们也能够自己编写逻辑，实现字节到字符的编码和解码，究竟使用`InputStream`和`OuputStream`还是`Reader`和`Writer`取决于数据源是文本还是其他文件。

Java提供一个FILE的类实现对文件的操作。FILE对象即可以表示文件，也可以表示目录。只有当我们调用FILE对象的某些方法时，才回真正执行磁盘操作。

所有的文件都是以二进制的形式保存的，但是为了便于理解和处理文件，文件就有了文件类型的概念，文件类型以后辍名来体现，每种文件类型都有一定的格式，代表着文件含义和二进制之间的映射关系。对于一种文件类型往往有一种或者多种应用程序可以解读它，进行查看和编辑，一个应用程序往往可以解读一种或者多种文件类型。在操作系统中，一种后缀名往往关联一个应用程序，打开文件时，操作系统查找相应关联的应用程序。

文本类型可以粗略的分为文本文件和二进制文件，基本上文本文件每个二进制字节都是某个可打印字符的一部分，但是二进制文件中，每个字节就不一定表示字符，可能表示的是颜色、字体、声音大小等等。

## 文本文件

文本文件包含的基本都是可打印字符，字符到二进制的映射称为编码，编码有多种方式，应用程序应该如何识别编码方式呢，对于utf-8文件，在文件最开头有三个特殊字节，称为BOM字节序标记。

## 文件系统

* 绝对路径：从根目录到当前文件的完整路径
* 相对路径：相对于当前目录而言

## 文件读写

硬盘访问延时，相比内存是比较慢的。一般读写文件需要两次数据拷贝，从用户态和内核态相互切换，然而这种谢欢是由开销的。

为了提高文件操作效率，应用程序使用缓冲区。读取文件时，即使目前只需要少量内容，在预知还会接着读取时，就会异地读取较多内容，放在读缓冲区，下次读取时，就可以直接从缓冲区读取，减少访问操作系统和硬盘。写文件时，先写到缓冲区，满了以后再一次性调用操作系统写入硬盘。

## Java中的文件概念

在Java中文件不是单独处理的，而是为输入输出设备的一种，Java使用同一的概念处理输入输出操作。这个概念叫做流。输入流可以获取数据，输出流可以到显示终端、文件、网络等。

Java中对流的操作有加密、压缩、计算信息摘要、计算检验和等，接收抽象的流，返回流。

而基本的流按照字节进行读写，没有缓冲区就不方便，使用装饰器设计模式，可以对基本的流增加功能。方便使用，在实际使用时，经常会叠加多个装饰类，为流的传输添加更多的功能，比如对流缓冲、对八种基本类型和字符串进行读写、对流压缩和解压缩等

按照字节处理对文本文件不够友好，能够方便按照字符处理文本数据的类是Reader和Writer

关于文件路径，文件源数据、文件目录、访问权限等等文件属性，Java使用File这个类进行管理

## NIO

NIO （new input and output）是一种看待输入输出不同的方式，他有缓冲区和通道的概念，更接近操作系统。

## 序列化和反序列化

序列化就是将内存中的Java对象持久的保存到一个六种，反序列化就是从流中恢复Java对象到内存，序列化和反序列化，主要目的是：一对象状态持久化，网络远程调用传递和返回对象。Java默认的序列化不够通用，现在通常使用json

## File类

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

## 二进制文件

以二进制方式读写的流主要有：

* InputStream / OutputStream 抽象类
* FileInputStream / FileOutputStream 输入源和输出目标是文件的流
* ByteArrayInputStream / ByteArrayOutputStream 输入源和输出目标是字节数组的流
* DataInputStream / DataOutputStream 装饰类，按照基本类型和字符串而非只是字节读写流
* BufferedInputStream / BufferedOutputStream 装饰类，对输入输出流提供缓冲功能

### InputStream / OutputStream

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

### FileInputStream / FileOutputStream

关于文件输入输出流，构造时有两个参数，一类时文件路径，可以是File对象，也可以是文件路径名，如果文件已经存在，需要指定是追加还是覆盖，使用文件输出流会实际打开文件。

文件输入流

## 文本文件和字符流

Java中主要的字符流有这几种：

* Reader / Writer
* InputStreamReader / OutputStreamWriter 适配器类，输入时InputStream 输出是OutputStream，将字节流转化为字符流
* FileReader / FileWriter
* CharArrayReader / CharArrayWriter 输出输出都是字符数组的字符流
* StringReader / StringWriter 输入和输出目标都是字符串的字符流
* BufferedReader / BufferedWriter
* PrintWriter 装饰类，将基本类型和对象转会为其字符串形式输出的

# 四、Maven

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

| 范围     | 说明                                              |
| -------- | ------------------------------------------------- |
| compile  | （默认情况）编译时需要用到                        |
| test     | 编译测试文件时需要用到                            |
| runtime  | 运行时需要，编译时不需要                          |
| provided | 编译时需要用到，但是运行时由JDK或者其他服务器提供 |



# 五、JVM——Java虚拟机

## 1、类加载机制

类加载器就是加载其他类的类，她负责将字节码文件加载到内存，创建Class对象。

运行Java程序，需要执行Java这个命令，指定包含main方法的完整类名，类路径。Java在运行时，会根据类的完全限定名寻找并加载类，寻找的方式是在系统类和指定的类路径中寻找。负责加载类的就是类加载器，一般程序运行时，有三个：

1. 启动类加载器：是Java虚拟机实现的一部分，负责加载Java的基础类
2. 扩展类加载器：负责加载Java的一些扩展类
3. 应用程序类加载器：负责加载应用程序的类
