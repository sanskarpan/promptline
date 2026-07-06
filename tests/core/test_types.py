from promptline.core.types import (
    Candidate,
    Demo,
    Field,
    ModuleState,
    Signature,
)


def test_render_system_includes_instruction():
    sig = Signature(
        instruction="Classify sentiment",
        inputs=[Field("text", "input text")],
        outputs=[Field("label", "sentiment label")],
    )
    rendered = sig.render_system()
    assert "Classify sentiment" in rendered
    assert "text" in rendered
    assert "input text" in rendered
    assert "label" in rendered
    assert "sentiment label" in rendered


def test_render_system_format():
    sig = Signature(
        instruction="Do something",
        inputs=[Field("a", "field a")],
        outputs=[Field("b", "field b")],
    )
    rendered = sig.render_system()
    assert "Inputs:" in rendered
    assert "Outputs:" in rendered
    assert "[[b]]" in rendered


def test_parse_output_two_fields():
    sig = Signature(
        instruction="X",
        inputs=[Field("q", "question")],
        outputs=[Field("answer", "the answer"), Field("confidence", "confidence score")],
    )
    text = "[[answer]]: Paris\n[[confidence]]: high"
    result = sig.parse_output(text)
    assert result == {"answer": "Paris", "confidence": "high"}


def test_parse_output_single_fallback():
    sig = Signature(
        instruction="X",
        inputs=[Field("q", "question")],
        outputs=[Field("answer", "the answer")],
    )
    # No [[field]]: markers → falls back to {field: text}
    result = sig.parse_output("Paris is the capital of France")
    assert result == {"answer": "Paris is the capital of France"}


def test_parse_output_multi_failure_returns_none():
    sig = Signature(
        instruction="X",
        inputs=[Field("q", "question")],
        outputs=[Field("answer", "the answer"), Field("confidence", "confidence score")],
    )
    result = sig.parse_output("no markers here")
    assert result is None


def test_child_has_new_id():
    parent = Candidate.seed({"mod": ModuleState(instruction="hello")})
    child = parent.child({"mod": ModuleState(instruction="world")}, optimizer="test")
    assert child.id != parent.id


def test_child_lineage():
    parent = Candidate.seed({"mod": ModuleState(instruction="hello")})
    child = parent.child({"mod": ModuleState(instruction="world")}, optimizer="test")
    assert child.parent_ids == [parent.id]


def test_seed_creates_valid_candidate():
    demo = Demo(inputs={"q": "hi"}, outputs={"a": "hello"})
    c = Candidate.seed({"mod": ModuleState(instruction="hello", demos=[demo])})
    assert c.id
    assert c.modules["mod"].instruction == "hello"
    assert len(c.modules["mod"].demos) == 1


# Fix 2: parse_output key validation tests
def test_parse_output_unknown_key_multi_returns_none():
    """Unknown [[key]] section with multi-output sig → no declared fields found → None."""
    sig = Signature(
        instruction="X",
        inputs=[Field("q", "question")],
        outputs=[Field("answer", "the answer"), Field("confidence", "confidence score")],
    )
    result = sig.parse_output("[[bogus]]: something")
    assert result is None


def test_parse_output_partial_outputs_returns_none():
    """Only one of two declared output fields present → None."""
    sig = Signature(
        instruction="X",
        inputs=[Field("q", "question")],
        outputs=[Field("answer", "the answer"), Field("confidence", "confidence score")],
    )
    result = sig.parse_output("[[answer]]: Paris")
    assert result is None


def test_parse_output_unknown_extra_section_dropped():
    """Unknown extra [[section]] alongside all declared fields → only declared fields returned."""
    sig = Signature(
        instruction="X",
        inputs=[Field("q", "question")],
        outputs=[Field("answer", "the answer"), Field("confidence", "confidence score")],
    )
    result = sig.parse_output("[[answer]]: Paris\n[[confidence]]: high\n[[bogus]]: extra")
    assert result == {"answer": "Paris", "confidence": "high"}


# Fix 1: [[ inside a value must not truncate the value
def test_parse_output_value_with_double_brackets_preserved():
    """A value legitimately containing [[...]] (no field-marker colon) is kept in full."""
    sig = Signature(
        instruction="X",
        inputs=[Field("q", "question")],
        outputs=[Field("answer", "the answer")],
    )
    result = sig.parse_output("[[answer]]: use [[brackets]] here")
    assert result == {"answer": "use [[brackets]] here"}


def test_parse_output_value_with_double_brackets_multi_field_preserved():
    """[[ inside a value is preserved even in a multi-field parse."""
    sig = Signature(
        instruction="X",
        inputs=[Field("q", "question")],
        outputs=[Field("answer", "the answer"), Field("confidence", "confidence score")],
    )
    text = "[[answer]]: use [[brackets]] here\n[[confidence]]: high"
    result = sig.parse_output(text)
    assert result == {"answer": "use [[brackets]] here", "confidence": "high"}


def test_parse_output_still_splits_on_declared_field_marker():
    """A value that contains another declared field marker ([[name]]:) still splits."""
    sig = Signature(
        instruction="X",
        inputs=[Field("q", "question")],
        outputs=[Field("answer", "the answer"), Field("confidence", "confidence score")],
    )
    text = "[[answer]]: see [[confidence]]: high"
    result = sig.parse_output(text)
    assert result == {"answer": "see", "confidence": "high"}
