// Command genwolf parses the AstroBWTv3 reference (internal/astrobwt/pow.go)
// and emits a micro-op table for the 256-case wolf loop, so the loop can be
// driven by a small table (a switch over ~13 op ids) instead of 256 literal
// cases. This is the naga-safe representation the WGSL front-half kernel needs
// (no 256-way const-array dispatch). The table is verified bit-exact against
// the CPU oracle by wolftable_test.go before it is trusted.
//
// Each case's per-byte body is a run of assignments to step_3[i] (aliased x),
// optionally referencing step_3[pos2] (aliased p). We classify each into an
// (op, param) pair. Structural specials (case 0's swap, 253's xxhash, 254/255's
// RC4 rekey) are handled in the interpreter, not the table.
package main

import (
	"fmt"
	"go/ast"
	"go/parser"
	"go/token"
	"os"
	"strconv"
	"strings"
)

// micro-op ids — keep in sync with internal/astrobwt/wolftable.go and the WGSL.
const (
	opNone     = 0
	opRotlK    = 1  // x = rotl8(x, param)
	opXorRotlK = 2  // x = x ^ rotl8(x, param)
	opShl      = 3  // x = x << (x & 3)
	opShr      = 4  // x = x >> (x & 3)
	opXorP     = 5  // x = x ^ p
	opAndP     = 6  // x = x & p
	opNot      = 7  // x = ^x
	opXorPop   = 8  // x = x ^ popcount8(x)
	opRev      = 9  // x = reverse8(x)
	opRotlX    = 10 // x = rotl8(x, x)
	opAdd      = 11 // x = x + x
	opMul      = 12 // x = x * x
	opSub97    = 13 // x = x - (x ^ 97)
)

func main() {
	src := "internal/astrobwt/pow.go"
	fset := token.NewFileSet()
	f, err := parser.ParseFile(fset, src, nil, 0)
	if err != nil {
		die("parse %s: %v", src, err)
	}

	// Find the switch op { ... } inside astroBWTv3Stream.
	var sw *ast.SwitchStmt
	ast.Inspect(f, func(n ast.Node) bool {
		if s, ok := n.(*ast.SwitchStmt); ok && s.Tag != nil {
			if id, ok := s.Tag.(*ast.Ident); ok && id.Name == "op" {
				sw = s
				return false
			}
		}
		return true
	})
	if sw == nil {
		die("could not find `switch op` in %s", src)
	}

	// table[case] = ordered op codes (op<<8 | param); count[case] = len.
	table := make([][]int, 256)
	special := make([]string, 256)
	for _, stmt := range sw.Body.List {
		cc, ok := stmt.(*ast.CaseClause)
		if !ok {
			continue
		}
		for _, cv := range cc.List { // one clause may list several case values (254, 255)
			n := intLit(cv)
			ops, spec := classifyCase(cc.Body)
			table[n] = ops
			special[n] = spec
		}
	}

	emit(table, special)
}

// classifyCase walks a case body: find the `for i := pos1; i < pos2; i++` loop
// and classify each step_3[i] assignment. Returns the op list and a special tag.
func classifyCase(body []ast.Stmt) ([]int, string) {
	var ops []int
	spec := ""
	for _, st := range body {
		switch s := st.(type) {
		case *ast.ForStmt:
			for _, inner := range s.Body.List {
				code, isSpecial := classifyStmt(inner)
				if isSpecial {
					spec = "swap" // case 0's reverse-swap of pos1/pos2
					continue
				}
				if code >= 0 {
					ops = append(ops, code)
				}
			}
		case *ast.AssignStmt:
			// case 254/255: rc4 rekey `rc4s = NewCipher(...)` before the loop.
			if callName(s.Rhs) == "NewCipher" {
				spec = "rekey"
			}
		}
	}
	// case 253 appends xxhash inside the loop body (an assign to lhash).
	return ops, spec
}

// classifyStmt maps one statement to an op code, or -1 to skip. Second return
// is true for the case-0 reverse-swap of step_3[pos1]/[pos2].
func classifyStmt(st ast.Stmt) (int, bool) {
	as, ok := st.(*ast.AssignStmt)
	if !ok {
		return -1, false
	}
	// The multi-assign reverse swap (case 0) / lhash updates (case 253).
	if len(as.Lhs) == 2 {
		return -1, true // reverse-swap
	}
	lhs := exprStr(as.Lhs[0])
	if lhs == "lhash" || lhs == "prev_lhash" {
		return -1, false // case 253 per-byte hash — structural, skip
	}
	if !strings.HasPrefix(lhs, "step_3[i]") {
		return -1, false
	}
	rhs := exprStr(as.Rhs[0])
	switch as.Tok {
	case token.ADD_ASSIGN:
		return pack(opAdd, 0), false // step_3[i] += step_3[i]
	case token.MUL_ASSIGN:
		return pack(opMul, 0), false // step_3[i] *= step_3[i]
	case token.SUB_ASSIGN:
		return pack(opSub97, 0), false // step_3[i] -= (step_3[i] ^ 97)
	}
	// plain `=`; classify by RHS shape (normalized: step_3[i]->x, step_3[pos2]->p)
	r := normalize(rhs)
	switch {
	case r == "^x":
		return pack(opNot, 0), false
	case r == "x^p":
		return pack(opXorP, 0), false
	case r == "x&p":
		return pack(opAndP, 0), false
	case r == "x^byte(bits.OnesCount8(x))":
		return pack(opXorPop, 0), false
	case r == "bits.Reverse8(x)":
		return pack(opRev, 0), false
	case r == "bits.RotateLeft8(x,int(x))":
		return pack(opRotlX, 0), false
	case r == "x<<(x&3)":
		return pack(opShl, 0), false
	case r == "x>>(x&3)":
		return pack(opShr, 0), false
	}
	if k, ok := rotlK(r); ok {
		return pack(opRotlK, k), false
	}
	if k, ok := xorRotlK(r); ok {
		return pack(opXorRotlK, k), false
	}
	die("unclassified op RHS: %q (normalized %q)", rhs, r)
	return -1, false
}

func rotlK(r string) (int, bool) {
	// bits.RotateLeft8(x,K)
	if strings.HasPrefix(r, "bits.RotateLeft8(x,") && strings.HasSuffix(r, ")") {
		arg := r[len("bits.RotateLeft8(x,") : len(r)-1]
		if k, err := strconv.Atoi(arg); err == nil {
			return k, true
		}
	}
	return 0, false
}

func xorRotlK(r string) (int, bool) {
	// x^bits.RotateLeft8(x,K)
	const pre = "x^bits.RotateLeft8(x,"
	if strings.HasPrefix(r, pre) && strings.HasSuffix(r, ")") {
		arg := r[len(pre) : len(r)-1]
		if k, err := strconv.Atoi(arg); err == nil {
			return k, true
		}
	}
	return 0, false
}

func pack(op, param int) int { return op<<8 | param }

func normalize(s string) string {
	s = strings.ReplaceAll(s, " ", "")
	s = strings.ReplaceAll(s, "step_3[pos2]", "p")
	s = strings.ReplaceAll(s, "step_3[i]", "x")
	return s
}

func exprStr(e ast.Expr) string {
	switch v := e.(type) {
	case *ast.Ident:
		return v.Name
	case *ast.IndexExpr:
		return exprStr(v.X) + "[" + exprStr(v.Index) + "]"
	case *ast.SelectorExpr:
		return exprStr(v.X) + "." + v.Sel.Name
	case *ast.BasicLit:
		return v.Value
	case *ast.ParenExpr:
		return "(" + exprStr(v.X) + ")"
	case *ast.BinaryExpr:
		return exprStr(v.X) + v.Op.String() + exprStr(v.Y)
	case *ast.UnaryExpr:
		return v.Op.String() + exprStr(v.X)
	case *ast.CallExpr:
		args := make([]string, len(v.Args))
		for i, a := range v.Args {
			args[i] = exprStr(a)
		}
		return exprStr(v.Fun) + "(" + strings.Join(args, ",") + ")"
	}
	return fmt.Sprintf("<%T>", e)
}

func callName(rhs []ast.Expr) string {
	if len(rhs) != 1 {
		return ""
	}
	if c, ok := rhs[0].(*ast.CallExpr); ok {
		if id, ok := c.Fun.(*ast.Ident); ok {
			return id.Name
		}
	}
	return ""
}

func intLit(e ast.Expr) int {
	if bl, ok := e.(*ast.BasicLit); ok {
		n, _ := strconv.Atoi(bl.Value)
		return n
	}
	die("case value not an int literal: %s", exprStr(e))
	return -1
}

func emit(table [][]int, special []string) {
	var b strings.Builder
	b.WriteString("// Code generated by cmd/genwolf; DO NOT EDIT.\n\npackage astrobwt\n\n")
	b.WriteString("// wolfOps[c] is the ordered micro-op sequence for switch case c\n")
	b.WriteString("// (op<<8 | param). wolfSpecial[c] tags structural cases.\n")
	b.WriteString("var wolfOps = [256][]uint16{\n")
	maxOps := 0
	for c := 0; c < 256; c++ {
		if len(table[c]) > maxOps {
			maxOps = len(table[c])
		}
		parts := make([]string, len(table[c]))
		for i, v := range table[c] {
			parts[i] = fmt.Sprintf("0x%04x", v)
		}
		b.WriteString(fmt.Sprintf("\t{%s},\n", strings.Join(parts, ", ")))
	}
	b.WriteString("}\n\n")
	b.WriteString("var wolfSpecial = [256]string{\n")
	for c := 0; c < 256; c++ {
		if special[c] != "" {
			b.WriteString(fmt.Sprintf("\t%d: %q,\n", c, special[c]))
		}
	}
	b.WriteString("}\n\n")
	b.WriteString(fmt.Sprintf("const wolfMaxOps = %d\n", maxOps))

	out := "internal/astrobwt/wolftable_gen.go"
	if err := os.WriteFile(out, []byte(b.String()), 0o644); err != nil {
		die("write %s: %v", out, err)
	}
	fmt.Printf("wrote %s (maxOps=%d)\n", out, maxOps)
}

func die(f string, a ...interface{}) {
	fmt.Fprintf(os.Stderr, "genwolf: "+f+"\n", a...)
	os.Exit(1)
}
