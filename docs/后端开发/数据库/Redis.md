# Redis 练习

## 一、概念和基础

Redis(Remote DIctionary Server)远程数据服务，使用C语言编写，是一款内存高速缓存数据库，支持键值对等多种数据结构的存储，可以用于缓存、事件发布、订阅、高速队列这些场景。

### 1、Redis的使用场景

#### 热点数据的缓存

#### 限时业务的应用

#### 计数器相关

#### 分布式锁

#### 延时操作

### 2、Redis的数据类型

redis中所有的key（键）都是字符串，因此数据类型实际上通常指的是值的类型，值的类型通常有5种。String List Set Zset Hash

| 结构类型      | 存储的值                           | 读写能力                                                     |
| ------------- | ---------------------------------- | ------------------------------------------------------------ |
| String 字符串 | 可以是字符串、整数、浮点数         | 能够对字符串或者字符串的一部分进行操作，对整数或者浮点数进行自增自减操作 |
| List          | 一个链表，每个节点都包含一个字符串 | 对链表的两端push pop读取单个或多个元素，根据值查找元素。     |
| Set           | 包含零个或多个字符串的无序集合     | 是一个字符串的集合，可以添加、获取、删除其中的字符串，也可以计算字符串的交集、并集差集等 |
| Hash          | 包含键值对的散列表                 | 添加、获取、删除某个键值对                                   |
| Zset          | 键值对，有序                       | 字符串成员和浮点数之间的有序映射，元素的排列由浮点数的大小决定。 |

#### String

是二进制安全的，可以包含任何数据，如数字、字符串，jpg图片或者是序列化的对象

#### List

可以实现消息排队功能，利用push操作将任务存放在List中，然后再用pop将任务取出执行。

## 一、初步命令

### 1、打开redis命令行

```shell
redis-cli -h host -p port -a password
```

### 2、操作key

redis中 键的概念类似于关系型数据库的表的概念，键内部的值就相当于表的内容。每个键相当于不同的数据表。我们可以通过键唯一标识一条数据。

| 命令                                         | 说明                                       |
|--------------------------------------------| ------------------------------------------ |
| del key                                    | 当键存在时删除键                           |
| dump key                                   | 序列化给定的键，并返回                     |
| exists key                                 | 检查键是否存在                             |
| expire key seconds                         | 为键设置过期时间，以秒计算                 |
| expireat key timestamp                     | 为键设置过期时间，时间戳                   |
| pexpire key milliseconds                   | 为键设置过期时间，以毫秒计算               |
| pexpireat key milliseconds-timestamp       | 为键设置过期时间，毫秒时间戳               |
| keys pattern                               | 查找符合给定模式的键                       |
| move key db                                | 将当前数据库中的键移动到另一个数据库中     |
| presist key                                | 取消键的过期时间                           |
| pttl                                       | 以毫秒为单位返回剩余过期时间               |
| ttl key                                    | 以秒为单位返回剩余过期时间 time to live    |
| randomkey                                  | 随机返回一个键                             |
| rename key newkey                          | 重命名键                                   |
| renamenx key new key                       | 当不存在时，重命名键                       |
| `scan cursor [match pattern] [count count]` | 遍历所有的键，可选符合给定的模式，返回数量 |
| type key                                   | 返回键所存储的值的类型                     |

接下来的String Hash List Set Zset都是针对某个key的值而言的，换句话说，就是对某个key的值的数据结构进行操作。值的结构由单纯的字符串、也有进一步的键值对、列表、集合等。这都是针对某一特定key而言的，一个特定的key的值只能是某一种数据结构，而不能同时为多种。

### 3、操作String

| 命令                               | 说明                                                         |
| ---------------------------------- | ------------------------------------------------------------ |
| set key value                      | 设置指定键的值                                               |
| get key                            | 获取指定键的值                                               |
| getrange key start end             | 获取键的值，但是以start为开头end为结束的那个子字符串         |
| getset key value                   | 为给定的键的值设置新值，并返回旧值                           |
| `mget key1 [key2 ..]`                | 获取一个或多个键的值                                         |
| setbit key offset value            |                                                              |
| setex key seconds value            | set key exists time 设置键值对，并设置值的过期时间           |
| setnx key value                    | set if not exists 只有在键不存在的时候才设置键的值           |
| setrange key offset value          | 从一个偏移量开始覆写所给定的键所存储的值                     |
| strlen key                         | 返回对应键所存储的值的长度                                   |
| `mset key value [key value] [...]`   | 同时设置一个或者多个键值对                                   |
| `msetnx key value [key value] [...]` | 当所有给定的键都不存在的时候，才设置一个或者多个键值对       |
| psetex key milliseconds value      | 以毫秒为单位设置键对应的值的生存时间                         |
| incr key                           | 键的值对应的数字加一                                         |
| incrby key increment               | 键的值对应的数字增加一个给定的值                             |
| incrbyfloat key increment          | 键的值对应的数字增加一个给定的浮点数值                       |
| decr key                           | 键的值对应的数字减一                                         |
| decrby key decrement               | 键的值对应的数字减少一个给定的值                             |
| append key value                   | 如果键已经存在并且是个字符串，那么将指定的值追加到原来值的末尾 |

### 4、操作Hash

哈希映射表是一个字符串类型的键值对。适合存放对象。每个键对应的值都是都是一个子键值对，键所对应的其中的一个值是由字段和其对应的值（field value）组成的。而一个键可以有多个值。每一个值都是由上述的字段-值映射表构成的。

| 命令                             | 说明                                                         |
| -------------------------------- | ------------------------------------------------------------ |
| hdel key field                   | 删除哈希表中的一个或多个字段                                 |
| hexists key field                | 查看名为key的哈希表中的field字段是否存在                     |
| hget key field                   | 获取哈希表中指定字段的值                                     |
| hgetall key                      | 获取哈希表中的所有字段和字段的值                             |
| hincrby key field increment      | 为哈希表指定字段的整数值加上一个增量                         |
| hincrbyfloat key field increment | 为哈希表指定字段的浮点数值加上一个浮点增量                   |
| hkeys key                        | 获取哈希表的所有字段                                         |
| hlen key                         | 获取哈希表中字段的数量                                       |
| `hmget key field1 [field2] [...]`  | get the values associated with multiple fields in the hash stored at the key获取哈希表中的所有给定字段的值 |
| `hmset key field1 value1 [...]`    | 为哈希表同时设置多个字段和它对应的值                         |
| hset key field value             | 设置哈希表中对应的字段的值                                   |
| hsetnx key field value           | 只有当字段不存在的时候才会设置哈希表中对应字段的值           |
| hvals key                        | 获取哈希表中所有的值                                         |
|                                  |                                                              |

### 5、操作列表

列表的底层是用链表实现的，我们可以在一个链表的头部或尾部插入元素。每一个键值都代表了一个列表。left表示的是左边，是列表的表头，right表示的是右边，是列表的表尾。block意为阻塞，在命令的最前方表示该命令会阻塞列表。

| 命令                                  | 说明                                                         |
| ------------------------------------- | ------------------------------------------------------------ |
| `blpop key1 [key2] timeout`             | 移除并获取列表的第一个元素，如果没有元素的话会阻塞列表直到等待超时或者发现可弹出的元素为止 |
| `brpop key1 [key2] timeout`             | 移除并获取列表的最后一个元素，如果列表中没有元素就会阻塞列表直到等待超时或者发现可以弹出的元素为止 |
| brpoplpush source destination timeout | 从列表中弹出一个值，将弹出的元素插入到另外一个列表中并返回它，如果列表中没有元素就会阻塞列表直到等待超时或者发现了可弹出的元素为止 |
| lindex key index                      | 通过索引来获取列表中的元素                                   |
| linsert key before\|after pivot value | 在列表的索引为pivot的元素的前面或者是后面插入值为value的元素 |
| llen key                              | 获取列表的长度                                               |
| lpop key                              | 移除并获取列表的第一个元素                                   |
| `lpush key value1 [value2]`             | 将一个或多个值插入到列表的头部                               |
| lrange key start stop                 | 获取列表指定范围内的所有元素                                 |
| lrem key count value                  | 根据参数count移除列表中的元素。count所代表的含义：如果count>0，从表头开始搜索移除与value相等的元素，数量为count。count<0，从表尾向表头搜索，移除与value相等的元素，数量为count的绝对值。count=0，移除表中所有与value相等的值 |
| lset key index value                  | 通过索引来设置列表里元素的值，如果索引超出范围则会返回错误   |
| ltrim key start stop                  | 让列表只保留指定区间内的元素，不再指定区间的元素都会被删除   |
| rpop key                              | 移除表尾的最后一个元素，并返回它                             |
| rpoplpush source destination          | 移除表尾的最后一个元素并将该元素添加到另一个列表的表头，然后返回该元素值 |
| `rpush key value1 [value2]`             | 向列表中添加一个或多个值                                     |
| rpushx key value                      | 向已存在的列表的表尾添加值，right push element if exists。   |
|                                       |                                                              |

### 6、操作集合

命令中的S意为Set，Set是字符串类型的无序集合，集合的成员是唯一的，集合内部不会出现重复的数据，区分每一个集合是通过key。

| 命令                                            | 说明                                                         |
| ----------------------------------------------- | ------------------------------------------------------------ |
| `sadd key member1 [...]`                          | 向集合中添加一个或多个成员                                   |
| scard key                                       | 获取集合中的成员数                                           |
| `sdiff key1 [key2]`                               | 获取key1这个集合和其他集合之间的差异                         |
| `sdiffstore destination key1 [key2]`              | 返回给定所有集合的差集并存储在destination中                  |
| `sinter key1 [key2]`                              | 返回给定所有集合的交集                                       |
| `sinterstore destination key1 [key2]`             | 返回给定的所有集合的交集并存储在destination中                |
| sismember key member                            | 判断元素是否是集合的成员                                     |
| smembers key                                    | 返回集合中的所有成员                                         |
| smove source destination member                 | 将一个集合中的元素移动到另一个集合                           |
| spop key                                        | 移除并返回集合的一个随机的元素                               |
| `srandmember key [count]`                         | 返回集合中的一个随机的元素，count为正数那么会返回count个元素的数组，如果大于集合元素个数，则返回值整个集合。 |
| `srem key member1 [member2]`                      | 移除集合中一个或多个成员                                     |
| `sunion key1 [key2]`                              | 返回所有给定集合的并集                                       |
| `sunionstore destination key1 [key2]`             | 返回所有给定集合的并集并且存储在destination中                |
| `sscan key cursor [match parttern] [count count]` | 迭代集合中的元素                                             |

### 7、操作有序集合

有序集合也是字符串类型的集合，并且不允许重复的成员，但是集合内部的每一个元素都会关联一个double类型的数值，这个数值决定了该元素的顺序，redis根据这个数值score来为集合中的成员从大到小进行排序。

| 命令                                           | 说明                                                         |
| ---------------------------------------------- | ------------------------------------------------------------ |
| `zadd key score1 member1 [score2 member2 ...]`   | 向有序集合中添加一个或者多个成员，或者更新已存在的成员的分数 |
| `zcard key`                                      | 获取有序集合的成员数量                                       |
| `zcount key min  max`                            | 计算有序集合在指定分数区间内的成员数量                       |
| `zincrby key increment member`                   | 有序集合中对指定成员的分数加上一个增量                       |
| `zinterstore destination numkeys key [key2...]`  | 计算给定的一个或多个有序集合的交集并且将结果存在新的有序集合中 |
| `zrange key start stop [withscores]`             | 返回有序集合在索引区间内部的成员们                           |
| `zrangebyscore key min max [withscores] [limit]` | 返回有序集合在分数区间内的成员们                             |
| `zrank key member`                               | 返回有序集合指定成员的排名，排名按照分数从小到大顺序         |
| `zrem key member [member ...]`                   | 移除有序集合中一个或者多个成员                               |
| `zremrangebyrank key start stop`                 | 移除有序集合在指定排名区间的所有成员                         |
| `zremrangebyscore key min max`                   | 移除有序集合在指定分数区间内的所有成员                       |
| `zrevrank key member`                            | 返回有序集合指定成员的排名                                   |
| `zscore key member`                              | 返回有序集合中指定成员的分数值                               |
| `zunionstore destination numkeys key [key ...]`  | 计算给定的一个或者多个有序集合的并集，并且存储在新的有序集合中 |
| `zscan key cursor [match pattern] [count count]` |                                                              |
|                                                |                                                              |



