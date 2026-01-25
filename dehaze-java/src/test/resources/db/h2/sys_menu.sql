create table dehaze_test.sys_menu
(
    id          bigint auto_increment
        primary key,
    parent_id   bigint                  not null comment '父菜单ID',
    tree_path   varchar(255)            null comment '父节点ID路径',
    name        varchar(64)  default '' not null comment '菜单名称',
    type        tinyint                 not null comment '菜单类型(1:菜单 2:目录 3:外链 4:按钮)',
    path        varchar(128) default '' null comment '路由路径(浏览器地址栏路径)',
    component   varchar(128)            null comment '组件路径(vue页面完整路径，省略.vue后缀)',
    perm        varchar(128)            null comment '权限标识',
    visible     tinyint(1)   default 1  not null comment '显示状态(1-显示;0-隐藏)',
    sort        int          default 0  null comment '排序',
    icon        varchar(64)  default '' null comment '菜单图标',
    redirect    varchar(128)            null comment '跳转路径',
    create_time datetime                null comment '创建时间',
    update_time datetime                null comment '更新时间',
    always_show tinyint                 null comment '【目录】只有一个子路由是否始终显示(1:是 0:否)',
    keep_alive  tinyint                 null comment '【菜单】是否开启页面缓存(1:是 0:否)'
);

