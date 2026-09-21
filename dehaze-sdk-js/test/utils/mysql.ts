import mysql from "mysql2/promise";
import {
  MYSQL_HOST,
  MYSQL_PORT,
  MYSQL_USERNAME,
  MYSQL_PASSWORD,
  MYSQL_DATABASE,
} from "#/config/constant";

let pool: mysql.Pool | null = null;

function getPool(): mysql.Pool {
  if (!pool) {
    pool = mysql.createPool({
      host: MYSQL_HOST,
      port: MYSQL_PORT,
      user: MYSQL_USERNAME,
      password: MYSQL_PASSWORD,
      database: MYSQL_DATABASE,
      waitForConnections: true,
      connectionLimit: 5,
    });
  }
  return pool;
}

export async function resetMemberQuota(userIds: number[]): Promise<void> {
  const pool = getPool();
  for (const userId of userIds) {
    await pool.execute(
      "UPDATE sys_member SET monthly_dehaze_used = 0, monthly_evaluate_used = 0 WHERE user_id = ?",
      [userId]
    );
  }
}

/**
 * 确保用户余额账户存在且可用余额不低于 minAmount（测试套件自愈）。
 *
 * 余额支付真实扣减 sys_balance，多文件/多轮运行会耗尽 USER/ADMIN 余额，
 * 导致 pay 拒绝 A053B 级联失败。充值无公开入账端点（回调驱动），此处直写库。
 * 金额上限需远小于边界用例的天价套餐（999999900），否则 A053B 用例失效。
 */
export async function ensureBalance(userId: number, minAmount: number): Promise<void> {
  const pool = getPool();
  await pool.execute(
    `INSERT INTO sys_balance (user_id, balance, frozen_balance, version, deleted)
     VALUES (?, ?, 0, 0, 0)
     ON DUPLICATE KEY UPDATE user_id = user_id`,
    [userId, minAmount]
  );
  await pool.execute(
    "UPDATE sys_balance SET balance = ? WHERE user_id = ? AND balance < ? AND deleted = 0",
    [minAmount, userId, minAmount]
  );
}

export async function createCompletedPredLog(
  userId: number,
  algorithmId: number = 13
): Promise<number> {
  const pool = getPool();
  const [result] = await pool.execute(
    `INSERT INTO sys_pred_log (algorithm_id, status, time, create_by, update_by, create_time, update_time)
     VALUES (?, 2, 100, ?, ?, NOW(), NOW())`,
    [algorithmId, userId, userId]
  );
  return (result as any).insertId;
}

export async function disconnectMysql(): Promise<void> {
  if (pool) {
    await pool.end();
    pool = null;
  }
}
