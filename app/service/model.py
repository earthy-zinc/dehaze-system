from app.models import SysAlgorithm


def get_root_algorithm(sys_algorithm: SysAlgorithm) -> SysAlgorithm:
    if (sys_algorithm.parent_id is None
        or sys_algorithm.parent_id == 0):
        return sys_algorithm
    return get_root_algorithm(sys_algorithm)

def get_flag(sys_algorithm: SysAlgorithm) -> bool:
    return get_root_algorithm(sys_algorithm).name == "WPXNet"
