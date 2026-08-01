import mysql from "mysql2/promise";
import { DEHAZE_HOST, DEHAZE_PASSWORD } from "#/config/constant";

let pool: mysql.Pool | null = null;

function getPool(): mysql.Pool {
  if (!pool) {
    pool = mysql.createPool({
      host: DEHAZE_HOST,
      port: 3306,
      user: "root",
      password: DEHAZE_PASSWORD,
      database: "dehaze",
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
