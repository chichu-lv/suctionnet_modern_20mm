"""Repository policy: no active SHA-256 computation or verification.

This is a static regression gate, not a general program-equivalence proof. It
resolves common Python aliases/constants and command/native API forms, without
executing scanned source, opening datasets, or treating legacy fields as code.
"""
from __future__ import annotations

import ast
import os
from pathlib import Path
import re
import shlex

import pytest


ROOT = Path(__file__).resolve().parents[1]
SOURCE_SUFFIXES = {'.py', '.sh', '.c', '.cpp', '.cc', '.cxx', '.h', '.hpp'}
SKIP_DIRS = {
    '.git', '.venv', 'venv', 'env', '.tox', '.pytest_cache', '__pycache__',
    '.mypy_cache', '.ruff_cache', 'build', 'dist', 'deps', 'dependencies',
    'site-packages', 'node_modules', 'tmp', 'temp', 'wheelhouse',
}
UNKNOWN = object()


def source_files(root):
    """Include docs/research helpers and new source; exclude generated/dependency trees."""
    for directory, names, filenames in os.walk(root, followlinks=False):
        names[:] = sorted(name for name in names if name not in SKIP_DIRS
                          and not name.startswith(('pytest-of-', 'tmp'))
                          and not name.endswith('-deps')
                          and not (Path(directory) / name).is_symlink())
        for name in sorted(filenames):
            path = Path(directory) / name
            if path.suffix.lower() in SOURCE_SUFFIXES and not path.is_symlink():
                yield path


def sha256_algorithm(value):
    return isinstance(value, str) and re.sub(r'[^a-z0-9]', '', value.lower()) in {'sha256', 'sha2256'}


class Scope:
    def __init__(self, parent=None):
        self.parent, self.bindings = parent, {}

    def find(self, name):
        if name in self.bindings:
            return self, self.bindings[name]
        return self.parent.find(name) if self.parent else (None, None)


def make_scope(body, parent=None, arguments=None):
    scope = Scope(parent)
    if arguments:
        positional = arguments.posonlyargs + arguments.args
        for arg in positional + arguments.kwonlyargs:
            scope.bindings[arg.arg] = ('unknown', None)
        for arg in (arguments.vararg, arguments.kwarg):
            if arg:
                scope.bindings[arg.arg] = ('unknown', None)
        for arg, default in zip(positional[-len(arguments.defaults):] if arguments.defaults else [], arguments.defaults):
            scope.bindings[arg.arg] = ('outer_expr', default)
        for arg, default in zip(arguments.kwonlyargs, arguments.kw_defaults):
            if default:
                scope.bindings[arg.arg] = ('outer_expr', default)

    class Bindings(ast.NodeVisitor):
        def visit_Import(self, node):
            for alias in node.names:
                scope.bindings[alias.asname or alias.name.split('.')[0]] = (
                    'import', alias.name if alias.asname else alias.name.split('.')[0])

        def visit_ImportFrom(self, node):
            for alias in node.names:
                scope.bindings[alias.asname or alias.name] = ('import', f'{node.module}.{alias.name}')

        def assign(self, target, value):
            if isinstance(target, ast.Name):
                scope.bindings[target.id] = ('expr', value)
            elif isinstance(target, (ast.Tuple, ast.List)):
                for index, child in enumerate(target.elts):
                    item = value.elts[index] if isinstance(value, (ast.Tuple, ast.List)) and index < len(value.elts) else None
                    self.assign(child, item)

        def visit_Assign(self, node):
            for target in node.targets:
                self.assign(target, node.value)

        def visit_AnnAssign(self, node):
            self.assign(node.target, node.value)

        def visit_NamedExpr(self, node):
            self.assign(node.target, node.value)

        def visit_FunctionDef(self, node):
            scope.bindings[node.name] = ('unknown', None)

        visit_AsyncFunctionDef = visit_FunctionDef
        visit_ClassDef = visit_FunctionDef

        def visit_Lambda(self, node):
            pass

    visitor = Bindings()
    for statement in body:
        visitor.visit(statement)
    return scope


def binding(node, scope, seen):
    owner, value = scope.find(node.id)
    key = (id(owner), node.id)
    if owner is None or key in seen:
        return None, None, seen
    return owner, value, seen | {key}


def constant(node, scope, seen=frozenset()):
    """Partial evaluation of literals only; never eval/execute scanned code."""
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.Name):
        owner, value, seen = binding(node, scope, seen)
        if value and value[0] in ('expr', 'outer_expr'):
            return constant(value[1], owner.parent if value[0] == 'outer_expr' else owner, seen)
        return UNKNOWN
    if isinstance(node, (ast.List, ast.Tuple)):
        return [constant(value, scope, seen) for value in node.elts]
    if isinstance(node, ast.Dict):
        keys = [constant(key, scope, seen) for key in node.keys]
        if all(isinstance(key, (str, int)) for key in keys):
            return dict(zip(keys, (constant(value, scope, seen) for value in node.values)))
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        left, right = constant(node.left, scope, seen), constant(node.right, scope, seen)
        if isinstance(left, (str, list)) and isinstance(right, type(left)):
            return left + right
    if isinstance(node, ast.JoinedStr):
        pieces = [constant(value.value if isinstance(value, ast.FormattedValue) else value, scope, seen)
                  for value in node.values]
        return ''.join('__DYNAMIC__' if value is UNKNOWN else str(value) for value in pieces)
    if isinstance(node, ast.Subscript):
        value, index = constant(node.value, scope, seen), constant(node.slice, scope, seen)
        try:
            if isinstance(value, (list, dict)) and isinstance(index, (str, int)):
                return value[index]
        except (KeyError, IndexError, TypeError):
            pass
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
        value = constant(node.func.value, scope, seen)
        args = [constant(arg, scope, seen) for arg in node.args]
        if isinstance(value, str) and not node.keywords:
            if node.func.attr in ('lower', 'upper', 'casefold') and not args:
                return getattr(value, node.func.attr)()
            if node.func.attr == 'replace' and len(args) == 2 and all(isinstance(arg, str) for arg in args):
                return value.replace(*args)
    return UNKNOWN


def qualified(node, scope, seen=frozenset()):
    if isinstance(node, ast.Name):
        owner, value, seen = binding(node, scope, seen)
        if not value:
            return node.id
        if value[0] == 'import':
            return value[1]
        if value[0] in ('expr', 'outer_expr'):
            return qualified(value[1], owner.parent if value[0] == 'outer_expr' else owner, seen)
        return None
    if isinstance(node, ast.Attribute):
        parent = qualified(node.value, scope, seen)
        return f'{parent}.{node.attr}' if parent else None
    if isinstance(node, ast.Call):
        name = qualified(node.func, scope, seen)
        if name in ('getattr', 'builtins.getattr') and len(node.args) >= 2:
            parent, attr = qualified(node.args[0], scope, seen), constant(node.args[1], scope, seen)
            return f'{parent}.{attr}' if parent and isinstance(attr, str) else None
        if name in ('__import__', 'importlib.import_module') and node.args:
            module = constant(node.args[0], scope, seen)
            return module if isinstance(module, str) else None
        if name == 'functools.partial' and node.args:
            return qualified(node.args[0], scope, seen)
    return None


DIRECT_HASH_CALLS = {
    'hashlib.sha256', '_sha256.sha256', '_hashlib.openssl_sha256',
    'Crypto.Hash.SHA256.new', 'Cryptodome.Hash.SHA256.new',
    'Crypto.Hash.SHA256.SHA256Hash', 'Cryptodome.Hash.SHA256.SHA256Hash',
    'cryptography.hazmat.primitives.hashes.SHA256',
}
SELECTED_HASH_CALLS = {
    'hashlib.new': (0, 'name'), '_hashlib.new': (0, 'name'),
    'hashlib.pbkdf2_hmac': (0, 'hash_name'),
    'hashlib.file_digest': (1, 'digest'),
    'hmac.new': (2, 'digestmod'), 'hmac.digest': (2, 'digest'),
}
PROCESS_CALLS = {'subprocess.run', 'subprocess.call', 'subprocess.check_call',
                 'subprocess.check_output', 'subprocess.Popen', 'os.system', 'os.popen'}


def scan_python(source):
    try:
        tree = ast.parse(source)
    except SyntaxError as error:
        return [(error.lineno or 1, 'Python syntax error prevents policy scan')]
    issues = []

    class Visitor(ast.NodeVisitor):
        scope = make_scope(tree.body)

        def nested(self, node, arguments=None):
            old = self.scope
            self.scope = make_scope(node.body, old, arguments)
            for statement in node.body:
                self.visit(statement)
            self.scope = old

        def visit_FunctionDef(self, node):
            for value in node.decorator_list + node.args.defaults + [v for v in node.args.kw_defaults if v]:
                self.visit(value)
            self.nested(node, node.args)

        visit_AsyncFunctionDef = visit_FunctionDef

        def visit_Lambda(self, node):
            old = self.scope
            self.scope = make_scope([], old, node.args)
            self.visit(node.body)
            self.scope = old

        def visit_ClassDef(self, node):
            for value in node.bases + node.decorator_list:
                self.visit(value)
            self.nested(node)

        def visit_Call(self, node):
            name = qualified(node.func, self.scope)
            if name in DIRECT_HASH_CALLS:
                issues.append((node.lineno, f'forbidden computation: {name}'))
            if name in SELECTED_HASH_CALLS:
                index, keyword = SELECTED_HASH_CALLS[name]
                algorithm = node.args[index] if len(node.args) > index else next(
                    (item.value for item in node.keywords if item.arg == keyword), None)
                if sha256_algorithm(constant(algorithm, self.scope)) or qualified(algorithm, self.scope) in DIRECT_HASH_CALLS:
                    issues.append((node.lineno, f'forbidden algorithm: {name}'))
            if name in PROCESS_CALLS:
                command = node.args[0] if node.args else next(
                    (item.value for item in node.keywords if item.arg in ('args', 'command')), None)
                value = constant(command, self.scope)
                nested = scan_shell(value) if isinstance(value, str) else scan_command(value) if isinstance(value, list) else []
                issues.extend((node.lineno, reason) for _, reason in nested)
            if name in ('exec', 'eval', 'compile', 'builtins.exec', 'builtins.eval', 'builtins.compile') and node.args:
                value = constant(node.args[0], self.scope)
                if isinstance(value, str):
                    issues.extend((node.lineno, reason) for _, reason in scan_python(value))
            self.generic_visit(node)

    Visitor().visit(tree)
    return sorted(set(issues))


def scan_command(words):
    if not words:
        return []
    words = ['__DYNAMIC__' if word is UNKNOWN else str(word) for word in words]
    while words and words[0] in ('command', 'exec', 'env', 'sudo', 'if', 'then', 'do', '!', '{'):
        words.pop(0)
        while words and (words[0].startswith('-') or re.match(r'^\w+=', words[0])):
            words.pop(0)
    if not words:
        return []
    executable = Path(words[0]).name
    if executable in ('sha256sum', 'sha256'):
        return [(1, f'forbidden checksum command: {executable}')]
    if executable == 'shasum' and any(
            token in ('-a256', '--algorithm=256') or
            token in ('-a', '--algorithm') and i + 1 < len(words) and words[i + 1] == '256'
            for i, token in enumerate(words)):
        return [(1, 'forbidden checksum command: shasum algorithm 256')]
    if executable == 'openssl' and any(sha256_algorithm(token.lstrip('-')) for token in words[1:]):
        return [(1, 'forbidden checksum command: openssl SHA-256')]
    if executable in ('sh', 'bash', 'dash', 'zsh'):
        for i, token in enumerate(words[1:], 1):
            if token.startswith('-') and 'c' in token and i + 1 < len(words):
                return scan_shell(words[i + 1])
    if re.fullmatch(r'python(?:\d+(?:\.\d+)*)?', executable) and '-c' in words:
        i = words.index('-c')
        if i + 1 < len(words):
            return scan_python(words[i + 1])
    return []


def scan_shell(source):
    issues, variables, pending = [], {}, ''
    # Python supplied directly to an interpreter is executable, not a data string.
    here = re.compile(r'(?m)^\s*(?:exec\s+)?python(?:\d+(?:\.\d+)*)?(?:\s+-)?\s+<<-?\s*[\'"]?(\w+)[\'"]?[^\n]*\n')
    for match in reversed(list(here.finditer(source))):
        end = re.search(r'(?m)^' + re.escape(match.group(1)) + r'\s*$', source[match.end():])
        if end:
            start_line = source[:match.end()].count('\n')
            body_end = match.end() + end.start()
            issues.extend((start_line + line, reason) for line, reason in scan_python(source[match.end():body_end]))
            stop = match.end() + end.end()
            source = source[:match.start()] + '\n' * source[match.start():stop].count('\n') + source[stop:]

    def expand(token):
        return re.sub(r'\$\{(\w+)\}|\$(\w+)',
                      lambda match: variables.get(match.group(1) or match.group(2), '__DYNAMIC__'), token)

    for line, fragment in enumerate(source.splitlines(), 1):
        pending += fragment + '\n'
        if fragment.rstrip().endswith('\\'):
            continue
        try:
            lexer = shlex.shlex(pending, posix=True, punctuation_chars=';&|()')
            lexer.whitespace_split = True
            tokens = list(lexer)
        except ValueError:
            continue  # A quoted multiline argument continues on the next line.
        pending = ''
        chunks, chunk = [], []
        for token in tokens + [';']:
            if token and set(token) <= set(';&|()'):
                chunks.append(chunk)
                chunk = []
            else:
                chunk.append(token)
        for words in chunks:
            words = [expand(token) for token in words]
            if words and words[0] in ('export', 'readonly'):
                words = words[1:]
            while words and re.match(r'^\w+=', words[0]):
                key, value = words.pop(0).split('=', 1)
                variables[key] = value
            issues.extend((line, reason) for _, reason in scan_command(words))
    return sorted(set(issues))


def scan_native(source):
    # Preserve literals for algorithm arguments, remove comments before tokenizing.
    token_pattern = re.compile(r'//[^\n]*|/\*.*?\*/|(?:u8|L|u|U)?"(?:\\.|[^"\\])*"|\'(?:\\.|[^\'\\])*\'|[A-Za-z_]\w*|\S', re.S)
    tokens = [(match.group(), source[:match.start()].count('\n') + 1)
              for match in token_pattern.finditer(source)
              if not match.group().startswith(('//', '/*'))]
    constants, issues = {}, []

    def value(items):
        parts = []
        for token, _ in items:
            if re.fullmatch(r'(?:u8|L|u|U)?"(?:\\.|[^"\\])*"', token):
                parts.append(token[token.index('"') + 1:-1])
            elif token in constants:
                parts.append(constants[token])
            elif token not in ('(', ')', '+'):
                return UNKNOWN
        return ''.join(parts) if parts else UNKNOWN

    for i, (token, line) in enumerate(tokens):
        # #define NAME "sha256" or const char *NAME = "sha256";
        if token == 'define' and i and tokens[i - 1][0] == '#' and i + 2 < len(tokens):
            name = tokens[i + 1][0]
            rest = [item for item in tokens[i + 2:] if item[1] == line]
            resolved = value(rest)
            if isinstance(resolved, str):
                constants[name] = resolved
        if token == '=' and i:
            end = next((j for j in range(i + 1, len(tokens)) if tokens[j][0] == ';'), len(tokens))
            resolved = value(tokens[i + 1:end])
            if isinstance(resolved, str):
                constants[tokens[i - 1][0]] = resolved
        if i + 1 >= len(tokens) or tokens[i + 1][0] != '(':
            continue
        if re.fullmatch(r'(?:EVP_sha256|SHA256(?:_Init|_Update|_Final)?|SHA256Init|SHA256Update|SHA256Final|CC_SHA256(?:_Init|_Update|_Final)?|mbedtls_sha256(?:_ret|_starts|_update|_finish|_starts_ret|_update_ret|_finish_ret)?)', token):
            issues.append((line, f'forbidden native computation: {token}'))
        if token not in ('EVP_get_digestbyname', 'EVP_MD_fetch', 'BCryptOpenAlgorithmProvider', 'system', 'popen'):
            continue
        args, current, depth = [], [], 1
        for item in tokens[i + 2:]:
            text = item[0]
            if text == '(':
                depth += 1
            elif text == ')':
                depth -= 1
                if depth == 0:
                    args.append(current)
                    break
            if text == ',' and depth == 1:
                args.append(current)
                current = []
            else:
                current.append(item)
        index = 1 if token in ('EVP_MD_fetch', 'BCryptOpenAlgorithmProvider') else 0
        algorithm = value(args[index]) if len(args) > index else UNKNOWN
        if token in ('system', 'popen') and isinstance(algorithm, str):
            issues.extend((line, reason) for _, reason in scan_shell(algorithm))
        elif sha256_algorithm(algorithm) or (len(args) > index and any(
                item[0] == 'BCRYPT_SHA256_ALGORITHM' for item in args[index])):
            issues.append((line, f'forbidden native algorithm: {token}'))
    return sorted(set(issues))


def scan_source(source, suffix):
    if suffix == '.py':
        return scan_python(source)
    if suffix == '.sh':
        return scan_shell(source)
    return scan_native(source)


def test_repository_has_no_active_sha256_computation_or_verification():
    files = list(source_files(ROOT))
    assert files, 'Policy gate found no repository source files'
    failures = [f'{path.relative_to(ROOT)}:{line}: {reason}'
                for path in files
                for line, reason in scan_source(path.read_text(encoding='utf-8'), path.suffix.lower())]
    assert not failures, '\n' + '\n'.join(failures)


@pytest.mark.parametrize('source', [
    'import hashlib\nhashlib.sha256(b"x")',
    'import hashlib as h\nh.sha256()',
    'from hashlib import sha256 as fingerprint\nfingerprint(b"x")',
    'import hashlib\nconstructor = hashlib.sha256\nconstructor()',
    'from functools import partial\nimport hashlib\nconstructor = partial(hashlib.sha256)\nconstructor(b"data")',
    'import hashlib\nALG = "sha" + "256"\nhashlib.new(ALG)',
    'from hashlib import new as factory\nMETHOD = "SHA-256".lower().replace("-", "")\nfactory(name=METHOD)',
    'import hashlib\nOPTIONS = {"method": "sha256"}\nhashlib.new(OPTIONS["method"])',
    'import hashlib as h\ngetattr(h, "sha" + "256")()',
    '__import__("hashlib").sha256()',
    'import hashlib\ndef compute():\n return hashlib.new(ALGORITHM)\nALGORITHM = "sha256"',
    'import hashlib\ndef compute(algorithm):\n algorithm = "sha256"\n return hashlib.new(algorithm)',
    'import hashlib\nALGORITHM="sha256"\ndef compute(algorithm=ALGORITHM):\n return hashlib.new(algorithm)',
    'import hashlib\ncompute = lambda algorithm="sha256": hashlib.new(algorithm)',
    'import hashlib\nhashlib.file_digest(file, "sha256")',
    'import hashlib\nhashlib.file_digest(file, hashlib.sha256)',
    'import hashlib\nhashlib.new("SHA2-256")',
    'import hmac\nMETHOD = "sha256"\nhmac.digest(b"key", b"data", METHOD)',
    'import hmac, hashlib\nhmac.new(b"key", b"data", hashlib.sha256)',
    'from cryptography.hazmat.primitives import hashes\nhashes.SHA256()',
    'from Crypto.Hash import SHA256\nSHA256.new(b"data")',
    'import subprocess\nsubprocess.run(["sha256sum", unknown_path])',
    'from subprocess import check_output as run\nALG="sha256"\nrun(["openssl", "dgst", "-" + ALG, path])',
    'import os\nos.system(f"sha256sum {path}")',
    'import subprocess\nsubprocess.run(args=["bash", "-lc", "openssl dgst -sha256 file"])',
    'import subprocess\nsubprocess.run(["python3", "-c", "import hashlib; hashlib.sha256()"] )',
    'exec("import hashlib; hashlib.sha256()")',
])
def test_python_scanner_detects_calls_aliases_and_algorithm_constants(source):
    assert scan_python(source)


@pytest.mark.parametrize('source', [
    'sha256sum file.npz',
    'ALG=sha256\nopenssl dgst -$ALG file.npz',
    'ALG=sha256; openssl dgst -$ALG file.npz',
    'SUM=sha256sum\n"$SUM" file',
    'shasum -a 256 file',
    'shasum --algorithm=256 file',
    'command /usr/bin/sha256sum file',
    'env MODE=test sha256sum file',
    'bash -c "openssl dgst -sha256 file"',
    'python3 - <<\'PY\'\nimport hashlib\nhashlib.sha256()\nPY\n',
    'python3 <<\'PY\'\nimport hashlib\nhashlib.sha256()\nPY\n',
])
def test_shell_scanner_detects_executable_commands(source):
    assert scan_shell(source)


@pytest.mark.parametrize('source', [
    'SHA256(buffer, length, output);',
    'EVP_DigestInit_ex(ctx, EVP_sha256(), NULL);',
    '#define ALGORITHM "sha256"\nEVP_get_digestbyname(ALGORITHM);',
    'const char *alg = "sha" "256"; EVP_MD_fetch(NULL, alg, NULL);',
    'BCryptOpenAlgorithmProvider(&provider, BCRYPT_SHA256_ALGORITHM, NULL, 0);',
    'system("sha256sum file");',
])
def test_native_scanner_detects_api_calls_and_algorithm_constants(source):
    assert scan_native(source)


@pytest.mark.parametrize('suffix,source', [
    ('.py', '# hashlib.sha256(file)\ntext = "sha256sum file"'),
    ('.py', 'record.pop("sha256", None)\nrecord.get("source_sha256")'),
    ('.py', 'legacy = {"sha256": "old-value"}\nfor key in legacy: print(key)'),
    ('.py', 'fixture = "import hashlib; hashlib.sha256(b\'test\')"'),
    ('.py', 'import hashlib\nprint(hashlib.sha256)\nprint(hashlib.algorithms_available)'),
    ('.py', 'import hashlib\nhashlib.sha1(b"blob 3\\0abc")'),
    ('.py', 'import hashlib\nhashlib.blake2b(pose.tobytes(), digest_size=8)'),
    ('.py', 'import hashlib\nALG = "sha256"\ndef other():\n ALG="sha1"\n return hashlib.new(ALG)'),
    ('.py', 'import hashlib\ndef inspect(hashlib):\n return hashlib.sha256'),
    ('.py', 'import subprocess\nsubprocess.run(["echo", "sha256sum file"])'),
    ('.sh', '# sha256sum file\necho "openssl dgst -sha256 file"'),
    ('.sh', 'ALGORITHM=sha256\nprintf "%s" "$ALGORITHM"'),
    ('.cpp', '// SHA256(x, n, out);\nconst char *legacy_field = "SHA256";'),
    ('.cpp', '/* EVP_sha256(); */\nPyModule_AddStringConstant(m, "SOURCE_SHA256", "legacy");'),
])
def test_scanner_allows_legacy_data_comments_git_ids_and_nonverification_seed(suffix, source):
    assert not scan_source(source, suffix)


def test_source_scope_includes_docs_research_and_excludes_generated_dependencies(tmp_path):
    wanted = ['setup.py', 'docs/reproduce.py', 'research/check.sh', 'models/csrc/operator.cpp']
    ignored = ['docs/results.json', 'build/generated.py', 'deps/vendor.py', 'tmp123456/run.py',
               'pytest-of-agent/test.py', 'env/lib/vendor.py', 'node_modules/task.py']
    for name in wanted + ignored:
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('')
    assert {str(path.relative_to(tmp_path)) for path in source_files(tmp_path)} == set(wanted)
