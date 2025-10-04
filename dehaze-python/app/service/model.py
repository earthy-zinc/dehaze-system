from app.models import SysAlgorithm


def get_root_algorithm(sys_algorithm: SysAlgorithm) -> SysAlgorithm:
    if (sys_algorithm.parent_id is None
        or sys_algorithm.parent_id == 0):
        return sys_algorithm
    else:
        parent_id = sys_algorithm.parent_id
        parent_algorithm = SysAlgorithm.query.filter_by(id=parent_id).first()
        return get_root_algorithm(parent_algorithm)

def get_flag(sys_algorithm: SysAlgorithm) -> bool:
    return get_root_algorithm(sys_algorithm).name == "WPXNet"
