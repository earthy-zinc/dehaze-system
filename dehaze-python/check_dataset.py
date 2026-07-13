import asyncio
from app.database import engine
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import text


async def main():
    async with AsyncSession(engine) as s:
        r = await s.execute(text("SELECT id, name, type, status FROM sys_dataset WHERE name = '文件测试数据集'"))
        rows = r.fetchall()
        print(f"Datasets named '文件测试数据集': {len(rows)}")
        for row in rows:
            print(row)


asyncio.run(main())
