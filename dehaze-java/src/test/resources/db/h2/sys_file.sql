create table dehaze_test.sys_file
(
    id          int auto_increment comment '文件id'
        primary key,
    type        varchar(100)             null comment '文件类型',
    url         text                     null comment '文件url',
    name        varchar(100)             not null comment '文件原始名',
    object_name varchar(100)             not null comment '文件存储名',
    size        varchar(100) default '0' not null comment '文件大小（格式化显示）',
    size_bytes  bigint                   null comment '文件大小（原始字节数）',
    path        varchar(255)             not null comment '文件路径',
    md5         char(32)                 not null comment '文件的MD5值，用于比对文件是否相同',
    create_time datetime                 not null comment '创建时间',
    update_time datetime                 null comment '更新时间',
    constraint md5
        unique (md5),
    constraint md5_key
        unique (md5)
);

