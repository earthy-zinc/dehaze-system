package message

import "testing"

func TestEscapeLike(t *testing.T) {
	cases := []struct {
		in   string
		want string
	}{
		{"", ""},
		{"plain", "plain"},
		{"100%", `100\%`},
		{"under_score", `under\_score`},
		{`back\slash`, `back\\slash`},
		// 字面反斜杠双写在前，通配符转义在后：与 Python/Java 顺序实现结果一致
		{`%\_`, `\%\\\_`},
		{"你%好_", `你\%好\_`},
	}
	for _, c := range cases {
		if got := EscapeLike(c.in); got != c.want {
			t.Errorf("EscapeLike(%q) = %q, want %q", c.in, got, c.want)
		}
	}
}
