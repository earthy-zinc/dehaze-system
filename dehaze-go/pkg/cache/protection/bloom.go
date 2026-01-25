package protection

import (
	"hash/fnv"
	"math"
	"sync"
)

// BloomFilter 布隆过滤器实现
type BloomFilter struct {
	mu        sync.RWMutex
	bitset    []uint64
	size      uint // 位数组大小
	hashCount uint // 哈希函数数量
}

// NewBloomFilter 创建布隆过滤器
// expectedItems: 预期元素数量
// falsePositiveRate: 期望的误判率（如0.01表示1%）
func NewBloomFilter(expectedItems uint, falsePositiveRate float64) *BloomFilter {
	if expectedItems == 0 {
		expectedItems = 10000
	}
	if falsePositiveRate <= 0 || falsePositiveRate >= 1 {
		falsePositiveRate = 0.01
	}

	// 计算最优的位数组大小: m = -n*ln(p) / (ln2)^2
	m := uint(math.Ceil(-float64(expectedItems) * math.Log(falsePositiveRate) / math.Pow(math.Ln2, 2)))

	// 计算最优的哈希函数数量: k = (m/n) * ln2
	k := uint(math.Ceil(float64(m) / float64(expectedItems) * math.Ln2))

	// 确保至少有1个哈希函数
	if k < 1 {
		k = 1
	}

	// 位数组按64位对齐
	arraySize := (m + 63) / 64

	return &BloomFilter{
		bitset:    make([]uint64, arraySize),
		size:      m,
		hashCount: k,
	}
}

// Add 添加元素到布隆过滤器
func (bf *BloomFilter) Add(key string) {
	bf.mu.Lock()
	defer bf.mu.Unlock()

	h1, h2 := bf.hash(key)
	for i := uint(0); i < bf.hashCount; i++ {
		pos := (h1 + i*h2) % bf.size
		bf.setBit(pos)
	}
}

// MayExist 判断元素是否可能存在
func (bf *BloomFilter) MayExist(key string) bool {
	bf.mu.RLock()
	defer bf.mu.RUnlock()

	h1, h2 := bf.hash(key)
	for i := uint(0); i < bf.hashCount; i++ {
		pos := (h1 + i*h2) % bf.size
		if !bf.getBit(pos) {
			return false
		}
	}
	return true
}

// Reset 重置布隆过滤器
func (bf *BloomFilter) Reset() {
	bf.mu.Lock()
	defer bf.mu.Unlock()

	for i := range bf.bitset {
		bf.bitset[i] = 0
	}
}

// hash 使用双哈希技术生成多个哈希值
func (bf *BloomFilter) hash(key string) (uint, uint) {
	h1 := fnv.New64a()
	h1.Write([]byte(key))
	hash1 := h1.Sum64()

	h2 := fnv.New64()
	h2.Write([]byte(key))
	hash2 := h2.Sum64()

	return uint(hash1), uint(hash2)
}

func (bf *BloomFilter) setBit(pos uint) {
	idx := pos / 64
	bit := pos % 64
	bf.bitset[idx] |= 1 << bit
}

func (bf *BloomFilter) getBit(pos uint) bool {
	idx := pos / 64
	bit := pos % 64
	return (bf.bitset[idx] & (1 << bit)) != 0
}
