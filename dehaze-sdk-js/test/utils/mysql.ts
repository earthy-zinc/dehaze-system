import mysql from "mysql2/promise";

let pool: mysql.Pool | null = null;

function getPool(): mysql.Pool {
  if (!pool) {
    pool = mysql.createPool({
      host: process.env.MYSQL_HOST || "127.0.0.1",
      port: Number(process.env.MYSQL_PORT) || 3306,
      user: process.env.MYSQL_USER || "root",
      password: process.env.MYSQL_PASSWORD || process.env.REDIS_PASSWORD || "12345678",
      database: process.env.MYSQL_DATABASE || "dehaze",
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
