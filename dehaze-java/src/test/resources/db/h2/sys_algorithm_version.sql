create table dehaze_test.sys_algorithm_version
(
    id            bigint auto_increment comment '主键'
        primary key,
    algorithm_id  bigint       not null comment '关联算法ID',
    version       varchar(50)  not null comment '版本号',
    change_log    text         null comment '变更日志',
    status        int          null comment '该版本时的状态',
    config_json   text         null comment '该版本时的配置JSON',
    model_file_id bigint       null comment '模型文件ID',
    is_active     tinyint(1)   default 0 null comment '是否当前活跃版本',
    create_time   datetime     default CURRENT_TIMESTAMP null comment '创建时间',
    update_time   datetime     default CURRENT_TIMESTAMP null comment '更新时间',
    create_by     bigint       null comment '创建人ID',
    update_by     bigint       null comment '修改人ID',
    constraint uk_algo_version
        unique (algorithm_id, version)
);
