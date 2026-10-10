SYSTEM_USER_ID = 0

# ---- 跨进程/跨端共享的键名与运行时参数 ----
# 不可配置化（曾放在 Settings 内、可由 env 覆盖）：这些值要么被多端共用（env 覆盖会
# 造成端间不一致，如 cache:invalidation 广播频道、captcha_code: 验证码键由 Java/Go 读取），
# 要么是进程内实现参数（L1 缓存容量/TTL、心跳与在线 TTL），随部署漂移只会带来排查困难。
CAPTCHA_KEY_PREFIX = "captcha_code:"
WS_REDIS_CHANNEL = "dehaze:ws:broadcast"
WS_ONLINE_KEY = "dehaze:ws:online_users"
WS_HEARTBEAT_INTERVAL = 30
WS_ONLINE_TTL = 90  # 应 >= 3 倍心跳间隔
TASK_REDIS_KEY_PREFIX = "task:running"
TASK_HEARTBEAT_INTERVAL = 30
TASK_REDIS_TTL = 3600
CACHE_INVALIDATION_CHANNEL = "cache:invalidation"
CACHE_L1_ENABLED = True
CACHE_L1_MAXSIZE = 1000
CACHE_L1_TTL = 300
CACHE_SINGLEFLIGHT_ENABLED = True
CACHE_NULL_ENABLED = True
CACHE_NULL_TTL = 60

# ---- 知识库检索/索引的实现常量 ----
# Top-K 业务默认值以 sys_knowledge_base.top_k 为准（需求规格：1-50，默认 5），此处仅保留契约上限；
# RRF 融合常量取算法标准默认（rank_constant=60），索引批量与重试属实现细节。
KB_SEARCH_MAX_TOP_K = 50
KB_SEARCH_HYBRID_RANK_CONSTANT = 60
KB_SEARCH_HYBRID_RANK_WINDOW = 20
KB_INDEX_BATCH_SIZE = 100
KB_ASYNC_MAX_RETRY = 3

TASK_CACHE_PREFIX = "task:running:cache:"
TASK_PROGRESS_PREFIX = "task:running:progress:"
TASK_CANCEL_PREFIX = "task:cancel:"
IDEMPOTENCY_KEY_PREFIX = "idempotency:task:"

TASK_EXPIRE_HOURS = 24
CANCEL_FLAG_TTL_SECONDS = 5 * 60
RESULT_FILE_EXPIRE_DAYS = 7

MAX_ROWS = 100_000
MAX_IMPORT_FILE_SIZE = 20 * 1024 * 1024
SYNC_THRESHOLD = 1000
BATCH_SIZE = 1000
IMPORT_BATCH_TX_SIZE = 500
MAX_CONCURRENT_PER_USER = 5

PROGRESS_MIN_INTERVAL_SECONDS = 2.0
PROGRESS_MIN_PERCENT_DELTA = 5

CATEGORY_IMPORT = "import"
CATEGORY_EXPORT = "export"
