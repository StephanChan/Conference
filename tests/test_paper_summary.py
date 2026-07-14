import importlib.util
import sys
import types
from pathlib import Path


# Provide lightweight stubs for optional runtime dependencies so the module can
# be imported in a minimal test environment.
httpx = types.ModuleType("httpx")


class _DummyClient:
    def __init__(self, *args, **kwargs):
        pass

    def get(self, *args, **kwargs):
        raise RuntimeError("network disabled")


httpx.Client = _DummyClient
httpx.HTTPStatusError = type("HTTPStatusError", (Exception,), {})
httpx.RequestError = type("RequestError", (Exception,), {})
httpx.TimeoutException = type("TimeoutException", (Exception,), {})
httpx.Limits = lambda *args, **kwargs: None
sys.modules.setdefault("httpx", httpx)

mcp_module = types.ModuleType("mcp")
server_module = types.ModuleType("mcp.server")
models_module = types.ModuleType("mcp.server.models")


class _Server:
    def __init__(self, *args, **kwargs):
        pass

    def list_tools(self):
        return lambda *args, **kwargs: None

    def call_tool(self):
        return lambda *args, **kwargs: None

    def get_capabilities(self, *args, **kwargs):
        return {}


class _NotificationOptions:
    def __init__(self, *args, **kwargs):
        pass


class _InitializationOptions:
    def __init__(self, *args, **kwargs):
        pass


class _Tool:
    def __init__(self, *args, **kwargs):
        pass


class _TextContent:
    def __init__(self, *args, **kwargs):
        pass


class _EmbeddedResource:
    def __init__(self, *args, **kwargs):
        pass


server_module.Server = _Server
server_module.NotificationOptions = _NotificationOptions
mcp_module.server = server_module
models_module.InitializationOptions = _InitializationOptions

mcp_types_module = types.ModuleType("mcp.types")
mcp_types_module.Tool = _Tool
mcp_types_module.TextContent = _TextContent
mcp_types_module.EmbeddedResource = _EmbeddedResource

sys.modules.setdefault("mcp", mcp_module)
sys.modules.setdefault("mcp.server", server_module)
sys.modules.setdefault("mcp.server.models", models_module)
sys.modules.setdefault("mcp.types", mcp_types_module)


MODULE_PATH = Path(__file__).resolve().parents[1] / "field_research_server.py"
SPEC = importlib.util.spec_from_file_location("field_research_server", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_parse_prompt_args_supports_name_affiliation_field_format():
    parsed_slash = MODULE._parse_prompt_args("/research-researcher/Weiyi_Song/Shandong_university/OCT")
    parsed_comma = MODULE._parse_prompt_args("/research-researcher/Weiyi Song, Shandong University, OCT")

    assert parsed_slash == {
        "researcher_name": "Weiyi Song",
        "affiliation": "Shandong university",
        "field": "OCT",
    }
    assert parsed_comma == {
        "researcher_name": "Weiyi Song",
        "affiliation": "Shandong University",
        "field": "OCT",
    }


def test_summary_sections_include_problem_methods_results_and_discussion():
    paper = {
        "title": "Attention is All You Need",
        "year": 2017,
        "venue": "NeurIPS",
        "citationCount": 120000,
        "referenceCount": 100,
        "influentialCitationCount": 8000,
        "fieldsOfStudy": ["Computer Science", "Machine Learning"],
        "abstract": (
            "We propose a new neural architecture for sequence modeling. "
            "Our approach uses self-attention to capture long-range dependencies. "
            "Experiments show that the model achieves strong results on translation tasks. "
            "The results demonstrate that the architecture is both efficient and scalable."
        ),
    }

    sections = MODULE._generate_paper_summary_sections(paper)
    labels = [label for label, _ in sections]

    assert labels == ["Problem Statement", "Methods", "Results", "Discussion"]

    content_by_label = dict(sections)
    for label in ["Problem Statement", "Methods", "Results", "Discussion"]:
        assert content_by_label[label]
        assert isinstance(content_by_label[label], str)

    problem_text = content_by_label["Problem Statement"].lower()
    methods_text = content_by_label["Methods"].lower()
    results_text = content_by_label["Results"].lower()

    assert "sequence" in problem_text or "dependencies" in problem_text
    assert "self-attention" in methods_text or "architecture" in methods_text
    assert "translation" in results_text or "scalable" in results_text
