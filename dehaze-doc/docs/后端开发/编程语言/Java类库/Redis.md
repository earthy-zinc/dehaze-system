# Redis

Redis 是一个开源的内存数据结构存储系统，可用作数据库、缓存和消息中间件。

## Spring 集成

Spring 通过模板方式提供了对 Redis 的数据查询和操作功能。

### RedisTemplate

RedisTemplate 是 Spring Data Redis 提供的核心类，它在方法中定义了算法骨架，但将具体步骤延迟到子类实现。这种模板方法使得子类可以在不改变算法结构的情况下，重新定义算法的某些步骤。

RedisTemplate 对 Redis 中的五种基础数据类型，分别提供了五个操作接口：

```java
ValueOperations valueOperations = redisTemplate.opsForValue();
HashOperations valueOperations = redisTemplate.opsForHash();
ListOperations valueOperations = redisTemplate.opsForList();
SetOperations valueOperations = redisTemplate.opsForSet();
ZSetOperations valueOperations = redisTemplate.opsForZSet();
```

这些操作接口分别对应 Redis 的字符串(String)、哈希(Hash)、列表(List)、集合(Set)和有序集合(ZSet)五种数据类型。