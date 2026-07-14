import asyncio
import aiomysql


async def main():
    conn = await aiomysql.connect(
        host="127.0.0.1", port=3306, user="root",
        password="12345678", db="dehaze"
    )
    cur = await conn.cursor()
    await cur.execute("SELECT id, name, type FROM sys_file WHERE type LIKE '.%'")
    rows = await cur.fetchall()
    print(f"Files with dotted type: {len(rows)}")
    for r in rows:
        print(f"  id={r[0]}, name={r[1]}, type={r[2]}")
    await cur.close()
    conn.close()


asyncio.run(main())
