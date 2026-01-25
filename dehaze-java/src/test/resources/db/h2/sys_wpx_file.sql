create table dehaze_test.sys_wpx_file
(
    id             bigint auto_increment comment 'id'
        primary key,
    origin_file_id bigint       null comment '旧文件id',
    origin_md5     char(32)     not null comment '旧文件的MD5值',
    origin_path    varchar(255) not null comment '旧文件路径',
    new_file_id    bigint       null comment '新文件id',
    new_path       varchar(255) not null comment '新文件路径',
    new_md5        char(32)     not null comment '新文件的MD5值',
    constraint new_md5
        unique (new_md5),
    constraint origin_md5
        unique (origin_md5)
);

create index idx_origin_md5
    on dehaze_test.sys_wpx_file (origin_md5);

