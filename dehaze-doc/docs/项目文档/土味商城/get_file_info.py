import os
import re
import pymysql
import tqdm
from git import Repo
from datetime import datetime

conn = pymysql.connect(host='192.168.31.2', user='root', password='142536aA', db='blog')
cursor = conn.cursor()


def traverse_dirs(root_dir, exclude_dirs=[]):
    """
    遍历文件夹root_dir，并将符合条件的文件夹名称添加到字符串数组中返回。
    条件：当前文件夹内不再含有文件夹，或者当前文件夹内的文件夹名称不再包含中文，
    且当前文件夹名称不为数组exclude_dir中的任意元素的值。
    """
    result_dirs = []

    # 遍历root_dir下的所有文件和文件夹
    for dirpath, dirnames, filenames in os.walk(root_dir):
        if dirpath == root_dir:
            continue
        # 检查当前文件夹名称是否包含exclude_dir列表中的任何一个子串
        if not any(exclude_dir in dirpath for exclude_dir in exclude_dirs):
            # 检查当前文件夹内是否还有子文件夹
            if not dirnames:
                # 当前文件夹内没有子文件夹，直接添加到结果数组
                result_dirs.append(dirpath.replace(root_dir, ''))
            else:
                # 当前文件夹内有子文件夹，判断当前文件夹中是否存在md文件
                # 如果有，将当前文件夹添加到结果数组
                if any(file.endswith('.md') for file in filenames):
                    result_dirs.append(dirpath.replace(root_dir, ''))

    result_dirs = sorted(set(result_dirs))
    result_dirs = [s for s in result_dirs if not any(sub in s for sub in exclude_dirs)]
    return result_dirs


def get_article_info_v2(root_path: str, exclude_dirs: list):
    result = []
    repo = Repo(root_path)
    count = 0
    docs_path = os.path.join(root_path, 'docs')

    def read_file_content(file_path: str):
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()

    def get_commit_time(commit_command: list):
        try:
            commit = repo.git.log(*commit_command).split('\n')[0]
            return datetime.strptime(commit, '%Y-%m-%d %H:%M:%S %z')
        except (IndexError, ValueError):
            return None

    articles = [
        (os.path.join(root, file), os.path.relpath(root, docs_path), os.path.splitext(file)[0])
        for root, _, files in os.walk(docs_path)
        if root != docs_path and not any(exclude_dir in root for exclude_dir in exclude_dirs)
        for file in files if file.endswith('.md')
    ]

    for file_path, folder_name, file_name_without_ext in articles:

        file_content = read_file_content(file_path)
        create_time = get_commit_time(['--all', '--reverse', '--format=%ai', '--', file_path])
        last_modified_time = get_commit_time(['--all', '--format=%ai', '--', file_path])

        article_dict = {
            'user_id': 1,
            'category_name': folder_name,
            'article_cover': '',
            'article_title': file_name_without_ext,
            'article_content': file_content,
            'create_time': create_time,
            'update_time': last_modified_time
        }
        count = count + 1
        print(f"读取第 {count} 条笔记——{file_name_without_ext}")
        execute_article_sql_one(article_dict)
        result.append(article_dict)

    return result


def get_category_id(category_name: str):
    sql = "SELECT id FROM t_category WHERE category_name=%s"
    # 执行查询操作
    cursor.execute(sql, (category_name,))
    # 获取查询结果
    result = cursor.fetchone()
    # 输出查询结果
    if result:
        return result[0]
    else:
        return None


def execute_artile_category_sql(result_dirs: list):
    for idx, category in enumerate(result_dirs):
        # 检查category_name是否已存在
        cursor.execute("SELECT COUNT(*) FROM `t_category` WHERE `category_name` = %s", (category,))
        if cursor.fetchone()[0] == 0:
            # 如果不存在，则插入新记录
            insert_sql = """
            INSERT INTO `t_category` (id, category_name, create_time) 
            VALUES (%s, %s, %s)
            """
            values = (idx + 1, category, datetime.now().strftime('%Y-%m-%d %H:%M:%S'),)
            print(cursor.mogrify(insert_sql, values))
            cursor.execute(insert_sql, values)
            conn.commit()
        else:
            # 如果存在，撤销操作
            conn.rollback()
            print(f"分类 '{category}' 已经存在")


def execute_article_sql(article_info: list):
    insert_sql = """INSERT INTO `t_article` (
        user_id, category_id, article_cover, 
        article_title, article_content, 
        create_time, update_time
    )
    VALUES (
        %s, %s, %s,
        %s, %s, 
        %s, %s
    )
    """
    for article in article_info:
        category_id = get_category_id(article['category_name'])
        values = (article['user_id'], category_id, article['article_cover'],
                  article['article_title'], article['article_content'],
                  article['create_time'].strftime('%Y-%m-%d %H:%M:%S'),
                  article['update_time'].strftime('%Y-%m-%d %H:%M:%S'),)
        print(cursor.mogrify(insert_sql, values))
        cursor.execute(insert_sql, values)
    # 提交事务
    conn.commit()


def execute_article_sql_one(article: dict):
    insert_sql = """INSERT INTO `t_article` (
        user_id, category_id, article_cover, 
        article_title, article_content, 
        create_time, update_time
    )
    VALUES (
        %s, %s, %s,
        %s, %s, 
        %s, %s
    )
    ON DUPLICATE KEY UPDATE 
        update_time = IF(update_time < VALUES(update_time), VALUES(update_time), update_time);
    """
    category_id = get_category_id(article['category_name'])
    values = (article['user_id'], category_id, article['article_cover'],
              article['article_title'], article['article_content'],
              article['create_time'].strftime('%Y-%m-%d %H:%M:%S'),
              article['update_time'].strftime('%Y-%m-%d %H:%M:%S'),)

    try:
        cursor.execute(insert_sql, values)
        # 提交事务
        conn.commit()
    except pymysql.err.IntegrityError as e:
        print(cursor.mogrify(insert_sql, values))
        print(e)
        conn.rollback()


def update_tag():
    # 从t_article表中获取所有的id
    sql = "SELECT id FROM t_article"
    cursor.execute(sql)
    article_ids = [row[0] for row in cursor.fetchall()]

    # 对于每一个article_id，更新或插入到t_article_tag表中
    tag_id = 15
    for article_id in article_ids:
        # 检查记录是否已存在，如果存在则更新，否则插入
        check_sql = "SELECT * FROM t_article_tag WHERE article_id=%s AND tag_id=%s"
        cursor.execute(check_sql, (article_id, tag_id))
        if cursor.fetchone() is None:
            # 如果记录不存在，则插入新记录
            insert_sql = "INSERT INTO t_article_tag (article_id, tag_id) VALUES (%s, %s)"
            cursor.execute(insert_sql, (article_id, tag_id))
            print(cursor.mogrify(insert_sql, (article_id, tag_id)))
        else:
            # 如果记录已存在，则更新记录（这里假设你不需要更新，只是作为一个示例）
            # update_sql = "UPDATE t_article_tag SET ... WHERE article_id=%s AND tag_id=%s"
            # cursor.execute(update_sql, (article_id, tag_id))
            continue
    # 提交事务
    conn.commit()

if __name__ == '__main__':
    docs_path = '/mnt/e/ProgramProject/reading-note/docs/'
    root_path = '/mnt/e/ProgramProject/reading-note'
    exclude_dirs = ['.vuepress', 'assets', '.assets', 'attachments']
    # result_dir = traverse_dirs(docs_path, exclude_dirs)
    # execute_artile_category_sql(result_dir)
    # article_info = get_article_info_v2(root_path, exclude_dirs)
    # execute_article_sql(article_info)
    update_tag()
    # 关闭连接
    cursor.close()
    conn.close()
