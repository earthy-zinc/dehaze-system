package com.pei.dehaze.sdk.model.menu;

/**
 * 菜单类型枚举
 * 对齐后端 Menu.type：CATALOG=目录，MENU=菜单，BUTTON=按钮，EXTLINK=外链
 */
public enum MenuType {
    CATALOG("CATALOG", "目录"),
    MENU("MENU", "菜单"),
    BUTTON("BUTTON", "按钮"),
    EXTLINK("EXTLINK", "外链");

    private final String value;
    private final String label;

    MenuType(String value, String label) {
        this.value = value;
        this.label = label;
    }

    public String getValue() {
        return value;
    }

    public String getLabel() {
        return label;
    }

    public static MenuType fromValue(String value) {
        if (value == null) return null;
        for (MenuType type : values()) {
            if (type.value.equalsIgnoreCase(value)) {
                return type;
            }
        }
        return null;
    }
}
