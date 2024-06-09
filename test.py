from importlib import import_module

if __name__ == "__main__":
    try:
        common = import_module("common")
        common.ok()
    except ImportError as e:
        print("无法导入" + "comon")
    except AttributeError as e:
        print("common" + "模块未定义ok函数")