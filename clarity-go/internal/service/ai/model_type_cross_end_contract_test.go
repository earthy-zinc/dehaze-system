package ai

import (
	"os"
	"path/filepath"
	"regexp"
	"sort"
	"strings"
	"testing"

	"github.com/earthyzinc/clarity-go/pkg/common"
	"github.com/earthyzinc/clarity-go/pkg/config"
	"github.com/stretchr/testify/require"
)

// 跨端 `model_type` 值域契约守卫。
//
// 背景：三端各自的值域信息源是**三份独立声明**（python `AiModelType` Literal / go
// `ValidateModelType` switch / java `MODEL_TYPES` Set.of），各端单测只覆盖本端行为，
// "只扩一端"（如只给 java 加 `music_gen`）三端全绿却在运行期产生语义分裂：
// 某端能建的模型类型，另端筛选/派生 billing_unit 时落到 default 分支（静默错计量）。
//
// 本守卫读三端**各自的声明文本**并断言三端集合 == python 事实源，且 go 声明与 go
// 运行期行为一致（`ValidateModelType` 接受集合 == 解析集合）。
var (
	// python：`AiModelType = Literal[ ... ]`
	pyModelTypeLiteralRe = regexp.MustCompile(`(?s)AiModelType\s*=\s*Literal\[(.*?)\]`)
	// go：`ValidateModelType` 内的 `case "a", "b":` 分支
	goModelTypeCaseRe = regexp.MustCompile(`(?s)func ValidateModelType\([^)]*\)\s*error\s*\{.*?case\s+((?:"[^"]+"\s*,\s*)*"[^"]+")\s*:`)
	// java：`MODEL_TYPES = Set.of( ... );`
	javaModelTypeSetRe = regexp.MustCompile(`(?s)MODEL_TYPES\s*=\s*Set\.of\((.*?)\)`)
	quotedValueRe      = regexp.MustCompile(`"([^"]+)"`)
)

const (
	pyModelTypeSource   = "clarity-python/app/models/schema/ai_conversation.py"
	goModelTypeSource   = "clarity-go/internal/service/ai/model_service.go"
	javaModelTypeSource = "clarity-java/src/main/java/com/pei/clarity/service/impl/AiModelServiceImpl.java"
)

func quotedSet(section string) map[string]bool {
	out := map[string]bool{}
	for _, m := range quotedValueRe.FindAllStringSubmatch(section, -1) {
		out[m[1]] = true
	}
	return out
}

func readRepoFile(t *testing.T, rel string) string {
	t.Helper()
	raw, err := os.ReadFile(filepath.Join(config.RepoRoot(), filepath.FromSlash(rel)))
	require.NoErrorf(t, err, "读取跨端声明文件失败：%s", rel)
	return string(raw)
}

func extractedSet(t *testing.T, source, file string, re *regexp.Regexp) map[string]bool {
	t.Helper()
	text := readRepoFile(t, file)
	match := re.FindStringSubmatch(text)
	require.NotNilf(t, match, "%s：未能在 %s 中定位值域声明（提取式失效，护栏会假绿）", source, file)
	set := quotedSet(match[1])
	require.NotEmptyf(t, set, "%s：声明解析为空", source)
	return set
}

func sortedKeys(set map[string]bool) []string {
	keys := make([]string, 0, len(set))
	for k := range set {
		keys = append(keys, k)
	}
	sort.Strings(keys)
	return keys
}

// TestModelTypeSetEqualAcrossEnds 三端 model_type 值域集合必须逐值相等（python 为事实源）。
//
// 会红场景：任一端新增/删除/改名一个取值（"只扩一端"），或某一端把声明搬走导致提取式失效。
func TestModelTypeSetEqualAcrossEnds(t *testing.T) {
	want := extractedSet(t, "python", pyModelTypeSource, pyModelTypeLiteralRe)
	goSet := extractedSet(t, "go", goModelTypeSource, goModelTypeCaseRe)
	javaSet := extractedSet(t, "java", javaModelTypeSource, javaModelTypeSetRe)

	// 值域明显缩水 ⇒ 解析口径或声明结构已变，先失败而不是拿残缺集合去比
	require.GreaterOrEqualf(t, len(want), 10, "python 值域骤减（%d 项）：%v", len(want), sortedKeys(want))

	require.Equalf(t, sortedKeys(want), sortedKeys(goSet),
		"go `ValidateModelType` 值域与 python `AiModelType` 不等（缺/多：%v / %v）",
		missing(want, goSet), missing(goSet, want))
	require.Equalf(t, sortedKeys(want), sortedKeys(javaSet),
		"java `MODEL_TYPES` 值域与 python `AiModelType` 不等（缺/多：%v / %v）",
		missing(want, javaSet), missing(javaSet, want))
}

// TestModelTypeGoDeclarationMatchesRuntimeBehavior 声明 == 运行期行为：
// 解析出的每个取值都必须被 `ValidateModelType` 放行，非成员必须 A0400。
// （否则"声明集合相等"可被一份与 switch 无关的注释/冗余常量骗过）
func TestModelTypeGoDeclarationMatchesRuntimeBehavior(t *testing.T) {
	declared := extractedSet(t, "go", goModelTypeSource, goModelTypeCaseRe)

	for _, modelType := range sortedKeys(declared) {
		require.NoErrorf(t, ValidateModelType(modelType),
			"声明内含 %q 但运行期被拒（声明与行为不一致）", modelType)
	}

	for _, outsider := range []string{"", "chat ", "CHAT", "llm", "chat_v2", "image", "asr_stream"} {
		if declared[outsider] {
			continue
		}
		err := ValidateModelType(outsider)
		bizErr, ok := common.AsBizError(err)
		require.Truef(t, ok, "%q 应被拒为业务错误，实际 err=%v", outsider, err)
		require.Equal(t, common.PARAM_ERROR, bizErr.Code(), "%q 错误码应为 A0400", outsider)
	}
}

// TestModelTypeContractDetectsMutation 提取器自证：篡改声明文本必须导致解析结果变化。
// 若提取式退化成"取常量"，上两条断言就会被假绿吞掉。
func TestModelTypeContractDetectsMutation(t *testing.T) {
	text := readRepoFile(t, pyModelTypeSource)
	match := pyModelTypeLiteralRe.FindStringSubmatch(text)
	require.NotNil(t, match)

	base := quotedSet(match[1])
	require.Contains(t, base, "music_gen")

	mutated := strings.Replace(match[1], `"music_gen",`, "", 1)
	require.NotEqual(t, match[1], mutated, "变异未生效：声明的书写形态可能已变，本自证用例须同步更新")
	require.NotEqual(t, sortedKeys(base), sortedKeys(quotedSet(mutated)),
		"摘除一个取值后解析结果不变 ⇒ 提取器失效（护栏会假绿）")
}

func missing(want, got map[string]bool) []string {
	var out []string
	for k := range want {
		if !got[k] {
			out = append(out, k)
		}
	}
	sort.Strings(out)
	return out
}
