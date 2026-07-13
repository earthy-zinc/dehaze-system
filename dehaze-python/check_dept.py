import pymysql

conn = pymysql.connect(host='127.0.0.1', port=3306, user='root', password='12345678', database='dehaze', charset='utf8mb4')
cursor = conn.cursor()

# Delete ALL residual dept records with SQL injection or XSS names
cursor.execute(
    "DELETE FROM sys_dept WHERE "
    "name LIKE \"%' OR '%\" OR name LIKE '%admin%--%' "
    "OR name LIKE '%DROP TABLE%' OR name LIKE '%UNION SELECT%' "
    "OR name LIKE '%SELECT * FROM%' "
    "OR name LIKE '%<script>%' OR name LIKE '%<img%' "
    "OR name LIKE '%javascript:%' OR name LIKE '%<svg%' "
    "OR name LIKE '%onerror%'"
)
deleted = cursor.rowcount
conn.commit()
print(f"Deleted {deleted} residual dept records")

# Verify cleanup
cursor.execute(
    "SELECT id, name FROM sys_dept WHERE "
    "name LIKE \"%' OR '%\" OR name LIKE '%admin%--%' "
    "OR name LIKE '%DROP TABLE%' OR name LIKE '%UNION SELECT%' "
    "OR name LIKE '%SELECT * FROM%' "
    "OR name LIKE '%<script>%' OR name LIKE '%<img%' "
    "OR name LIKE '%javascript:%' OR name LIKE '%<svg%'"
)
remaining = cursor.fetchall()
print(f"Remaining residual records: {len(remaining)}")
for row in remaining:
    print(f"  id={row[0]}, name={repr(row[1])}")

cursor.close()
conn.close()
