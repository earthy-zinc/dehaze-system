import os.path
from app.utils.file import calculate_file_md5

root_path = "/mnt/d/DeepLearning/dataset"

clean_flag = "clean"
hazy_flag = "hazy"

def generate_sql(dataset_name):
    dataset_path = os.path.join(root_path, dataset_name)
    clean_path = os.path.join(dataset_path, clean_flag)
    hazy_path = os.path.join(dataset_path, hazy_flag)
    new_dataset_path = os.path.join(root_path, "WPX", dataset_name)
    new_clean_path = os.path.join(new_dataset_path, clean_flag)
    new_hazy_path = os.path.join(new_dataset_path, hazy_flag)

    def get_sql(old_dataset_path, new_dataset_path, output_file):
        old_list = sorted(os.listdir(old_dataset_path))
        new_list = sorted(os.listdir(new_dataset_path))
        with open(output_file, 'a') as f:
            for i in range(len(old_list)):
                old_abs_path = os.path.join(old_dataset_path, old_list[i])
                origin_path = os.path.relpath(old_abs_path, root_path)
                origin_md5 = calculate_file_md5(old_abs_path)

                new_abs_path = os.path.join(new_dataset_path, new_list[i])
                new_path = os.path.relpath(new_abs_path, root_path)
                new_md5 = calculate_file_md5(new_abs_path)
                f.write(
                    f"INSERT INTO sys_wpx_file (origin_path, origin_md5, new_path, new_md5) VALUES ('{origin_path}', '{origin_md5}', '{new_path}', '{new_md5}');\n")

    # 读取文件转为BytesIO并计算md5值
    output_sql_file = dataset_name + ".sql"
    get_sql(clean_path, new_clean_path, output_sql_file)
    get_sql(hazy_path, new_hazy_path, output_sql_file)

if __name__ == '__main__':
    generate_sql("O-HAZE")
    generate_sql("I-HAZE")
    generate_sql("Dense-Haze")
    generate_sql("NH-HAZE-2020")
    generate_sql("NH-HAZE-2021")
    generate_sql("NH-HAZE-2023")
