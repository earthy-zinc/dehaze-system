create table dehaze_test.sys_pred_log
(
    id             bigint auto_increment comment 'id'
        primary key,
    algorithm_id   bigint                             not null comment '算法id',
    origin_file_id bigint                             null comment '原始图像文件id（有雾图像）',
    origin_md5     char(32)                           not null comment '原始图像md5值',
    origin_url     text                               not null comment '原始图像url',
    pred_file_id   bigint                             null comment '预测图像文件id',
    pred_md5       char(32)                           not null comment '预测图像md5值',
    pred_url       text                               not null comment '预测图像url',
    time           int      default 0                 null comment '推理时间（秒）',
    create_time    datetime default CURRENT_TIMESTAMP not null comment '创建时间',
    update_time    datetime default CURRENT_TIMESTAMP not null on update CURRENT_TIMESTAMP comment '更新时间',
    create_by      bigint                             null comment '创建人ID',
    update_by      bigint                             null comment '修改人ID'
);

create index idx_algorithm_id
    on dehaze_test.sys_pred_log (algorithm_id);

create index idx_origin_md5
    on dehaze_test.sys_pred_log (origin_md5);

create index idx_pred_md5
    on dehaze_test.sys_pred_log (pred_md5);

