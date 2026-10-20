package utils

// TreeDataTree 树形结构节点接口
type TreeDataNode interface {
	IDGetter
	ParentIDGetter
}

// IDGetter 获取节点ID的接口
type IDGetter interface {
	GetID() int64
}

// ParentIDGetter 获取父节点ID的接口
type ParentIDGetter interface {
	GetParentID() int64
}

// TreeDataUtils 树形数据工具类
type TreeDataUtils struct{}

// NewTreeDataUtils 创建树形数据工具实例
func NewTreeDataUtils() *TreeDataUtils {
	return &TreeDataUtils{}
}

// FindLeafNodesBFS 使用BFS查找指定父节点下的所有叶子节点ID
// nodes: 节点列表
// rootID: 根节点ID
// idMapper: 从节点获取ID的函数
// parentIdMapper: 从节点获取父节点ID的函数
func (u *TreeDataUtils) FindLeafNodesBFS(nodes []TreeDataNode, rootID int64) []int64 {
	if len(nodes) == 0 {
		return []int64{}
	}

	// 构建父节点到子节点的映射
	parentToChildren := make(map[int64][]TreeDataNode)
	childrenIds := make(map[int64]bool)
	allIds := make(map[int64]bool)

	for _, node := range nodes {
		id := node.GetID()
		parentId := node.GetParentID()
		allIds[id] = true

		if parentToChildren[parentId] == nil {
			parentToChildren[parentId] = []TreeDataNode{}
		}
		parentToChildren[parentId] = append(parentToChildren[parentId], node)
	}

	// 找出所有是父节点的ID
	for _, node := range nodes {
		childrenIds[node.GetParentID()] = true
	}

	// 叶子节点：是节点但不作为任何节点的父ID
	leafNodes := []int64{}
	for id := range allIds {
		if !childrenIds[id] {
			leafNodes = append(leafNodes, id)
		}
	}

	// 如果指定了rootID，则只返回指定根节点下的叶子节点
	if rootID != 0 {
		return u.findLeafNodesUnder(rootID, parentToChildren)
	}

	return leafNodes
}

// findLeafNodesUnder 递归查找指定节点下的所有叶子节点
func (u *TreeDataUtils) findLeafNodesUnder(parentID int64, parentToChildren map[int64][]TreeDataNode) []int64 {
	children := parentToChildren[parentID]
	if len(children) == 0 {
		// 没有子节点，自己是叶子节点
		return []int64{parentID}
	}

	leafNodes := []int64{}
	for _, child := range children {
		leafNodes = append(leafNodes, u.findLeafNodesUnder(child.GetID(), parentToChildren)...)
	}

	return leafNodes
}

// GetChild IDs 获取指定节点的所有子孙节点ID
func (u *TreeDataUtils) GetDescendantIDs(nodes []TreeDataNode, rootID int64) []int64 {
	if len(nodes) == 0 {
		return []int64{}
	}

	// 构建父节点到子节点的映射
	parentToChildren := make(map[int64][]TreeDataNode)
	for _, node := range nodes {
		parentId := node.GetParentID()
		if parentToChildren[parentId] == nil {
			parentToChildren[parentId] = []TreeDataNode{}
		}
		parentToChildren[parentId] = append(parentToChildren[parentId], node)
	}

	// BFS遍历获取所有子孙节点
	var result []int64
	queue := []int64{rootID}
	visited := make(map[int64]bool)

	for len(queue) > 0 {
		currentID := queue[0]
		queue = queue[1:]

		if visited[currentID] {
			continue
		}
		visited[currentID] = true

		children := parentToChildren[currentID]
		for _, child := range children {
			childID := child.GetID()
			if !visited[childID] {
				result = append(result, childID)
				queue = append(queue, childID)
			}
		}
	}

	return result
}

// FindRootIDs 获取所有根节点ID（不在任何节点的ParentID中出现的ID）
func (u *TreeDataUtils) FindRootIDs(nodes []TreeDataNode) []int64 {
	if len(nodes) == 0 {
		return []int64{}
	}

	nodeIDs := make(map[int64]bool)
	parentIDs := make(map[int64]bool)

	for _, node := range nodes {
		nodeIDs[node.GetID()] = true
		parentIDs[node.GetParentID()] = true
	}

	// 根节点ID：在nodeIDs中但不在parentIDs中的ID（除了0）
	var rootIDs []int64
	for id := range nodeIDs {
		if !parentIDs[id] {
			rootIDs = append(rootIDs, id)
		}
	}

	return rootIDs
}
