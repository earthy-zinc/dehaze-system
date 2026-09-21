/**
 * MongoDB 审计日志键名迁移：camelCase → snake_case
 *
 * 背景
 *   `login_log` / `audit_log` 两个集合由 dehaze-java 与 dehaze-python 共写共读，python 一律按
 *   snake_case 写键；java 在修复前用实体字段名直接落库（camelCase），导致同名集合内两套键并存、
 *   两端数据互不可见（java 只读到自己写的、python 读不到 java 写的）。java 侧已改为 `@Field`
 *   snake_case 映射（同时条件/排序/索引键全部对齐），本脚本负责把**修复前 java 写入的历史文档**迁到
 *   snake_case，并清掉按 camelCase 建的旧索引。
 *
 * 影响面
 *   · 只做字段重命名（`$rename`）与索引删除，不删除任何文档。
 *   · `ai_api_call_log` 无需迁移：java 只读不写，且该集合键名本就是 snake_case（有 30 天 TTL）。
 *
 * 幂等性
 *   迁移条件天然幂等：只命中「仍带 camelCase 键」且「尚无对应 snake_case 键」的文档，例如
 *   `{$and: [{$or: [{userId: {$exists: true}}, ...]}, {$or: [{user_id: {$exists: false}}, ...]}]}`。
 *   已迁移文档不再命中，重复执行只会 matched=0；`$rename` 对不存在的源字段本身也是空操作。
 *   索引删除同理：按实际 `getIndexes()` 结果匹配，已删除的索引不会再次出现。
 *
 * 前置检查
 *   1) 先确认 java 服务已升级到带 `@Field` 映射的版本（否则迁移后旧版 java 反而读不到数据）。
 *   2) 脚本首步即统计并在最后校验「待迁移/遗留文档数」（迁移前 >0、迁移后应为 0），
 *      可单独用 `countDocuments` 预检，不必先跑迁移。
 *   3) 建议先备份：`mongodump --host 127.0.0.1 --port 27017 -u root -p "$MONGODB_PASSWORD" \
 *      --authenticationDatabase admin --db dehaze --collection login_log --collection audit_log \
 *      --out /tmp/mongo-backup-$(date +%F)`
 *   4) 记录当前索引：`db.login_log.getIndexes()` / `db.audit_log.getIndexes()`（脚本也会打印）。
 *
 * 执行（连接串从根 `.env` 的 MONGODB_* 取值，密码勿写进命令历史可用环境变量）
 *   mongosh "mongodb://$MONGODB_USERNAME:$MONGODB_PASSWORD@$MONGODB_HOST:$MONGODB_PORT/$MONGODB_DATABASE" \
 *     --file dehaze-test/scripts/mongo_key_migrate.js
 *
 * 回滚（仅当需要退回旧 java 版本时执行；回滚不恢复被删索引，需按下方注释重建）
 *   反向重命名：把每处 `$rename` 的源/目标互换（见文件末尾 ROLLBACK 段）。
 */

const LEGACY_KEYS = {
  login_log: ["userId", "deviceType", "createTime"],
  audit_log: [
    "operatorId",
    "targetType",
    "targetId",
    "beforeValue",
    "afterValue",
    "userAgent",
    "createTime",
  ],
};

const RENAMES = {
  login_log: { userId: "user_id", deviceType: "device_type", createTime: "create_time" },
  audit_log: {
    operatorId: "operator_id",
    targetType: "target_type",
    targetId: "target_id",
    beforeValue: "before_value",
    afterValue: "after_value",
    userAgent: "user_agent",
    createTime: "create_time",
  },
};

function legacyFilter(renames) {
  // 只匹配「仍带 camelCase 键且尚无 snake_case 键」的文档，保证幂等且不覆盖已迁移数据
  const clauses = Object.keys(renames).map((legacy) => ({ [legacy]: { $exists: true } }));
  const migrated = Object.values(renames).map((current) => ({ [current]: { $exists: false } }));
  return { $and: [{ $or: clauses }, { $or: migrated }] };
}

function dropLegacyIndexes(collectionName) {
  const coll = db.getCollection(collectionName);
  const legacy = LEGACY_KEYS[collectionName];
  const dropped = [];
  for (const index of coll.getIndexes()) {
    if (index.name === "_id_") {
      continue;
    }
    const keys = Object.keys(index.key || {});
    if (keys.some((key) => legacy.includes(key))) {
      coll.dropIndex(index.name);
      dropped.push(index.name);
    }
  }
  return dropped;
}

function migrateCollection(collectionName) {
  const renames = RENAMES[collectionName];
  const coll = db.getCollection(collectionName);
  print(`\n=== ${collectionName} ===`);
  print(`迁移前：遗留(camelCase) 文档 ${coll.countDocuments(legacyFilter(renames))} 条，`
    + `文档总数 ${coll.estimatedDocumentCount()} 条`);

  const result = coll.updateMany(legacyFilter(renames), { $rename: renames });
  print(`迁移结果：matched=${result.matchedCount} modified=${result.modifiedCount}`);

  const droppedIndexes = dropLegacyIndexes(collectionName);
  print(`已删除旧 camelCase 索引：${droppedIndexes.length ? droppedIndexes.join(", ") : "无"}`);

  const remaining = coll.countDocuments(legacyFilter(renames));
  print(`迁移后校验：遗留(camelCase) 文档 ${remaining} 条 ${remaining === 0 ? "✓" : "✗ 请复查"}`);
  print("当前索引：" + coll.getIndexes().map((index) => index.name).join(", "));
}

print(`目标库：${db.getName()}`);
migrateCollection("login_log");
migrateCollection("audit_log");

print("\n提示：snake_case 索引由 java 启动时 MongoConfig 自动 ensure，预期出现"
  + " user_id_1_create_time_-1 / create_time_-1 / status_1（login_log）与"
  + " operator_id_1_create_time_-1 / target_type_1_target_id_1_create_time_-1 /"
  + " module_1_create_time_-1（audit_log）；确认后重启一次 java 服务即可，无需手工建索引。");
print("如需手工重建（例如无法立刻重启 java），可执行：");
print('  db.login_log.createIndex({user_id: 1, create_time: -1}, {name: "user_id_1_create_time_-1"})');
print('  db.login_log.createIndex({create_time: -1}, {name: "create_time_-1"})');
print('  db.login_log.createIndex({status: 1}, {name: "status_1"})');
print('  db.audit_log.createIndex({operator_id: 1, create_time: -1}, {name: "operator_id_1_create_time_-1"})');
print('  db.audit_log.createIndex({target_type: 1, target_id: 1, create_time: -1}, {name: "target_type_1_target_id_1_create_time_-1"})');
print('  db.audit_log.createIndex({module: 1, create_time: -1}, {name: "module_1_create_time_-1"})');

/* ROLLBACK（退回旧 java 版本时使用；先停写再执行，执行后旧 camelCase 索引需按上方语句类比重建）
db.login_log.updateMany({user_id: {$exists: true}, userId: {$exists: false}},
  {$rename: {user_id: "userId", device_type: "deviceType", create_time: "createTime"}});
db.audit_log.updateMany({operator_id: {$exists: true}, operatorId: {$exists: false}},
  {$rename: {operator_id: "operatorId", target_type: "targetType", target_id: "targetId",
            before_value: "beforeValue", after_value: "afterValue",
            user_agent: "userAgent", create_time: "createTime"}});
*/
