import os, re

base = '/data/workspace/dehaze-system/dehaze-react-native'

# Files with "perms" pattern: need to move perms inside useCallback
# Fix: replace `const hasPerm = useCallback((p: string) => perms.includes(p), [perms]);`
# with `const hasPerm = useCallback((p: string) => (useAuthStore.getState().userInfo?.perms ?? []).includes(p), []);`
# and remove `const perms = userInfo?.perms ?? [];` if userInfo/permm not used elsewhere

perms_files = [
    ('src/pages/dashboard/index.tsx', True),   # userInfo used for hero subtitle
    ('src/pages/system/algorithm/index.tsx', False),  # userInfo only for perms
    ('src/pages/system/dataset/index.tsx', False),
    ('src/pages/system/dept/index.tsx', False),
    ('src/pages/system/dict/index.tsx', False),
    ('src/pages/system/dict/items.tsx', False),
    ('src/pages/system/menu/index.tsx', False),
    ('src/pages/system/package/index.tsx', False),
    ('src/pages/system/role/index.tsx', False),
    ('src/pages/system/user/index.tsx', False),
]

for fp, keep_userInfo in perms_files:
    path = os.path.join(base, fp)
    with open(path) as f:
        content = f.read()
    
    # Replace hasPerm definition
    content = content.replace(
        "const hasPerm = useCallback((p: string) => perms.includes(p), [perms]);",
        "const hasPerm = useCallback((p: string) => (useAuthStore.getState().userInfo?.perms ?? []).includes(p), []);"
    )
    
    # Remove perms line if userInfo not needed elsewhere
    if not keep_userInfo:
        content = re.sub(r"\s*const perms = userInfo\?\.perms \?\? \[\];\s*\n", "\n", content)
        # Also remove userInfo if it was only for perms
        if "userInfo" not in re.sub(r"const userInfo = useAuthStore\(s => s\.userInfo\);\s*\n", "", content):
            content = content.replace("const userInfo = useAuthStore(s => s.userInfo);\n", "")
    
    with open(path, 'w') as f:
        f.write(content)
    print(f'FIXED: {fp}')

print('\nDone!')
