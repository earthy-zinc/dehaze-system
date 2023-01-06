# 锁

## 乐观锁和悲观锁

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

## 自旋锁和适应性自旋锁

阻塞或者唤醒一个Java线程需要操作系统切换CPU状态，这种状态转换需要耗费处理器事件，如果同步代码块中的内容很简单，状态转换消耗的事件甚至会比执行代码的时间还要长。在许多场景中，同步资源的锁定时间很短，为了这一小段时间切换线程，线程挂起和回复现场的花费有可能让系统得不偿失。如果物理机器有多个处理器，能够让两个或者以上的线程同时并行执行，我们就可以让后面那个请求锁的线程不放弃CPU执行时间，看看持有锁的线程是否会很快地释放锁。

那么等待锁释放的那个线程，我们就需要让线程自旋，也就是循环询问是否有锁，如果在自旋完成后，前面锁定同步资源的线程已经释放了锁，那么当前线程就可以不必阻塞而是直接获取同步资源，从而避免切换线程的开销，这就是自旋锁。

**缺点**

自旋锁等待虽然避免了线程切换的开销，但是要占用处理器的时间。如果锁被占用的时间过长，那么自旋的线程就只能白白浪费处理器资源。所以自旋等待的时间必须有一定的限度，如果自旋超过了限定次数，就应该挂起线程，释放处理器资源。

自旋锁的实现同样采用CAS对比后交换愿你指令完成。

## 无锁、偏向锁、轻量级锁、重量级锁

## 公平锁和非公平锁

公平锁指的是多个线程按照申请所得顺序来获取锁，线程直接进入队列中排队，队列中的第一个线程才能够获取锁。公平锁的优点是等待锁的线程不会饿死。缺点是整体吞吐效率相对非公平锁要低。等待队列中除了第一个线程以外的所有线程都会阻塞。CPU唤醒阻塞线程的开销比非公平锁要大。

非公平锁是多个线程加锁时直接尝试获取锁，获取不到时才会到队列队尾等待，但是如果此时的锁刚好可用，那么这个线程就可以无需阻塞直接获取到锁，所以非公平锁有可能出现这种情况：后申请锁的线程先获取到锁。非公平锁的有点事可以减少唤起线程的开销，整体吞吐效率高，因为线程有几率不阻塞直接获取到锁，CPU不必唤醒所有线程逐一询问。但是缺点是处于等待队列中的线程可能会饿死。

## 可重入锁和非可重入锁

可重入锁又叫做递归锁，是指同一个线程在外层方法获取锁的时候，在进入该线程内层方法时会自动获取锁。不会因为之气那已经获取过还没有释放就阻塞。ReentrantLock和synchronized都是可重入锁。可重入锁在一定程度上可以避免死锁。

比如说同一个类中的两个方法都是被内置的锁synchronized修饰的，在其中一个方法内部调用另一个被synchronized加锁的方法，进入过另一个方法是可以直接获得当前对象的锁，而不必等待刚刚的方法释放锁。而非可重入锁在调用另一个方法是，进入方法内部则无法获取到锁，因为该锁被外部方法占用，且无法释放，此时就会造成死锁。

重入锁和非重入锁中记录了一个同步状态status来计算重入次数，当线程尝试获取锁是，可重入锁先尝试并更新这个值，如果值不为0，则会判断当前线程是否是获取到了这个锁的线程，如果是的status+1.且当前线程可以尝试再次获取锁。而非重入锁在获取更新status值时，如果值不为0，则会导致其获取锁失败，当前线程阻塞。

## 独享锁和共享锁

独享锁也叫做排他锁，是指该锁一次只能被一个线程所持有，如果线程对数据加上排他锁后，其他线程则不能再对数据加任何类型的锁。获取排他锁的线程既能够读取数据也能够修改数据。

共享锁指的是该锁可以被多个线程所持有，如果线程对数据加上共享锁后，其他线程只能对数据加上共享锁而不能加排他锁，获得共享锁的线程只能够读数据而不能够写数据。

# synchronized

synchronized负责给线程加锁，一把锁只能同时被一个线程获取，没有获取到锁的线程只能等待。每个实例对对应有自己的一把锁，不同实例之间互不影响。但是锁的如果时类级别的化，锁class、锁static，所有该类的实例对象共用一把锁。用synchronized修饰的方法，无论方法是正常执行完毕还是抛出异常，都会释放锁。

## 对象锁

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

## 类锁

synchronized修饰静态的方法或者制定锁对象为class对象

```java
synchronized(SynchronizedObjectLock.class){
    // 代码
}
```

这里锁住的就是类级别的对象

## 原理分析

### 加锁和释放锁

我们在加入synchronized关键字后，编译器会给对应的代码加入两个指令，分别是monitorenter 和 monitorexit，这两个指令会让对象在执行时使其锁计数器加减1，每个对象同时只与一个锁相关联。

### 可重入原理



### 可见性原理



## JVM锁优化



# volatile

# final



# CAS

## 概念

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



## 缺点

CAS的方式是乐观锁，而synchronized是悲观锁，乐观锁解决并发问题在通常情况下性能更优。但是CAS也存在一些问题。

### ABA问题

CAS对比后交换指令，需要在操作值的时候，检查值有没有变化，如果没有发生变化则更新，但是如果一个值原来是A，变成了B，随后又变成了A，那么使用CAS进行检查时就会发现他的值没有发生变化，但是实际上却已经变化了。

ABA问题我们可以采用版本号来解决，在变量前面追加上版本号，每次变量更新的时候，把版本号加1。

### 循环开销时间大

自旋的CAS如果长时间不能够成功，会给CPU带来很大的开销，如果JVM能够支持处理器提供的pause指令，那么效率则会有一定的提升。pause指令的作用是：1、延迟流水线执行命令，使CPU不会消耗过多的资源。2、避免在退出循环时因为存在内存顺序冲突而引起CPU流水线被清空，从而提高CPU执行效率。

### 只能保证一个共享变量的原子操作

对于多个共享变量的操作，循环CAS无法保证原子性，需要使用锁。

## 解决方案

### AtomicStampedReference

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

# Unsafe类

## 概念

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

# AtomicInteger

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

# 原子更新类

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

# LockSupport

# AQS

# ReentrantLock

# ReentrantReadWriteLock