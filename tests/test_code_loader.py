"""Tests for the source code loader."""

import pytest
from doc2graph.loaders.code import load_code, detect_language


class TestDetectLanguage:
    @pytest.mark.parametrize("filename,expected", [
        ("main.py",    "python"),
        ("app.js",     "javascript"),
        ("app.ts",     "typescript"),
        ("App.tsx",    "typescript"),
        ("Main.java",  "java"),
        ("main.go",    "go"),
        ("lib.rs",     "rust"),
        ("main.cpp",   "cpp"),
        ("main.c",     "c"),
        ("App.cs",     "csharp"),
        ("script.sh",  "bash"),
        ("data.sql",   "sql"),
        ("config.yml", "yaml"),
        ("data.json",  "json"),
    ])
    def test_known_extensions(self, filename, expected):
        assert detect_language(filename) == expected

    def test_unknown_extension_returns_none(self):
        assert detect_language("file.xyz") is None

    def test_case_insensitive(self):
        assert detect_language("main.PY") == "python"
        assert detect_language("App.TS") == "typescript"


class TestLoadCode:
    def test_loads_python_file(self, tmp_path):
        f = tmp_path / "main.py"
        f.write_text("def hello():\n    return 'hi'\n")
        text = load_code(str(f))
        assert "def hello" in text

    def test_language_tag_prepended_by_default(self, tmp_path):
        f = tmp_path / "app.py"
        f.write_text("x = 1\n")
        text = load_code(str(f))
        assert text.startswith("# language: python")

    def test_language_tag_suppressed(self, tmp_path):
        f = tmp_path / "app.py"
        f.write_text("x = 1\n")
        text = load_code(str(f), tag_language=False)
        assert "# language:" not in text

    def test_unknown_extension_no_tag(self, tmp_path):
        f = tmp_path / "script.xyz"
        f.write_text("some content\n")
        text = load_code(str(f))
        assert "# language:" not in text
        assert "some content" in text

    def test_file_not_found_raises(self):
        with pytest.raises(FileNotFoundError):
            load_code("/no/such/file.py")

    def test_utf16_code_file(self, tmp_path):
        f = tmp_path / "main.py"
        f.write_bytes("print('hello')".encode("utf-16"))
        text = load_code(str(f))
        assert "print" in text

    def test_javascript_tag(self, tmp_path):
        f = tmp_path / "index.js"
        f.write_text("console.log('hi');\n")
        text = load_code(str(f))
        assert "# language: javascript" in text
        assert "console.log" in text

    def test_sql_file(self, tmp_path):
        f = tmp_path / "query.sql"
        f.write_text("SELECT * FROM users;\n")
        text = load_code(str(f))
        assert "# language: sql" in text
        assert "SELECT" in text
