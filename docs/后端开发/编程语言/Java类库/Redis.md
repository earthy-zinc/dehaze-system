
# Redis

Spring 通过模板方式提供了对 Redis 的数据查询和操作功能。

#### RedisTemplate 就是在一个方法中定义了一个算法的骨架，但是把进一步的步骤延迟到子类去实现，模板方法使得子类可以在不改变算法结构的情况下，重新定义算法的某些步骤。

RedisTemplate 对 Redis 中的物种基础类型，分别提供了五个子类进行操作。

```java
ValueOperations valueOperations = redisTemplate.opsForValue();
HashOperations valueOperations = redisTemplate.opsForHash();
ListOperations valueOperations = redisTemplate.opsForList();
SetOperations valueOperations = redisTemplate.opsForSet();
ZsetOperations valueOperations = redisTemplate.opsForZset();

```
