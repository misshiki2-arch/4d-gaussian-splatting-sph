"""P2 JSON reading and common types, NOT formal semantic acceptance.

``read_json`` takes an already-open binary stream; ``parse_json_bytes`` takes
bytes. Both return a ParsedJSON whose root still needs the later semantic
resolver (required/allowed fields, field domains, paths, fixed values, and
immutable verified state). No defaults, config merging, runtime or writer are
provided here. JSON numbers remain NumberToken objects until explicitly checked
as Int or Num; parsing alone does not choose a field's numeric domain.
"""

from dataclasses import dataclass
from json.decoder import JSONDecodeError, scanstring
import math
import re
from typing import BinaryIO


MAX_INPUT_BYTES = 262144
MAX_CONTAINER_DEPTH = 16
MAX_ARRAY_ELEMENTS = 4096
MAX_STRING_SCALARS = 4096
MAX_INT = 2147483647
_NUMBER = re.compile(r"-?(?:0|[1-9][0-9]*)(?:\.[0-9]+)?(?:[eE][+-]?[0-9]+)?")
_WHITESPACE = " \t\r\n"


class JSONInputError(ValueError):
    """Bounded code only; input text, keys and paths are never echoed."""

    def __init__(self, code: str):
        self.code = code
        super().__init__(code)


@dataclass(frozen=True)
class NumberToken:
    """Original JSON numeric lexeme, not an accepted field value."""

    raw: str

    def __post_init__(self):
        if (type(self.raw) is not str or len(self.raw) > MAX_INPUT_BYTES
                or _NUMBER.fullmatch(self.raw) is None):
            raise JSONInputError("number_token")

    @property
    def is_integer(self) -> bool:
        return not any(c in self.raw for c in ".eE")


@dataclass(frozen=True)
class ParsedJSON:
    """Syntactically parsed, semantically UNVERIFIED root.

    The nested dicts/lists are ordinary mutable parse data, not the future
    immutable verified-state contract. Even an empty root is not a valid formal
    configuration without the separate semantic resolver's acceptance.
    """

    root: dict


def require_int(value: object) -> int:
    """P2 nonnegative Int up to MAX_INT; field-specific bounds remain separate."""
    if type(value) is not NumberToken or not value.is_integer:
        raise JSONInputError("int_type")
    digits = value.raw.lstrip("-")
    # Bound before int conversion, including tokens beyond Python's digit cap.
    maximum = str(MAX_INT)
    if (len(digits) > len(maximum)
            or (len(digits) == len(maximum) and digits > maximum)
            or (value.raw.startswith("-") and digits != "0")):
        raise JSONInputError("int_range")
    return int(value.raw)


def require_num(value: object) -> float:
    """P2 finite binary64 Num; no Int bound or field-specific domain inferred."""
    if type(value) is not NumberToken:
        raise JSONInputError("num_type")
    result = float(value.raw)
    if not math.isfinite(result):
        raise JSONInputError("num_overflow")
    # Inspect only the significand: 0e999999 is true zero, 1e-999999 is not.
    significand = re.split("[eE]", value.raw, maxsplit=1)[0]
    if result == 0.0 and any(c in "123456789" for c in significand):
        raise JSONInputError("num_underflow")
    return result


def require_bool(value: object) -> bool:
    """Only JSON true/false (not numeric tokens, Python numbers or strings)."""
    if type(value) is not bool:
        raise JSONInputError("bool_type")
    return value


def read_json(stream: BinaryIO) -> ParsedJSON:
    """Read at most the byte limit plus one detection byte, without closing it.

    Ordinary blocking binary streams, including short reads, are supported.
    Filesystem locator/path policy belongs to the caller and later preflight.
    """
    chunks = []
    size = 0
    while size <= MAX_INPUT_BYTES:
        chunk = stream.read(MAX_INPUT_BYTES + 1 - size)
        if type(chunk) is not bytes:
            raise JSONInputError("binary_stream_required")
        if not chunk:
            break
        size += len(chunk)
        if size > MAX_INPUT_BYTES:
            raise JSONInputError("input_bytes")
        chunks.append(chunk)
    return parse_json_bytes(b"".join(chunks))


def parse_json_bytes(data: bytes) -> ParsedJSON:
    """Read P2 syntax/limits only; never imply whole-config verification."""
    if type(data) is not bytes:
        raise JSONInputError("bytes_required")
    if len(data) > MAX_INPUT_BYTES:
        raise JSONInputError("input_bytes")
    if data.startswith(b"\xef\xbb\xbf"):
        raise JSONInputError("bom")
    try:
        text = data.decode("utf-8", errors="strict")
    except UnicodeDecodeError:
        raise JSONInputError("utf8") from None
    parser = _Parser(text)
    parser.whitespace()
    if parser.peek() != "{":
        raise JSONInputError("root_object")
    root = parser.value(0)
    parser.whitespace()
    if parser.pos != len(text):
        raise JSONInputError("trailing_value")
    return ParsedJSON(root)


class _Parser:
    """Bounded descent: check depth before entering each object/array."""

    def __init__(self, text):
        self.text = text
        self.pos = 0

    def peek(self):
        return self.text[self.pos:self.pos + 1]

    def whitespace(self):
        while self.pos < len(self.text) and self.text[self.pos] in _WHITESPACE:
            self.pos += 1

    def value(self, parent_depth):
        self.whitespace()
        c = self.peek()
        if c in ("{", "["):
            depth = parent_depth + 1
            if depth > MAX_CONTAINER_DEPTH:
                raise JSONInputError("container_depth")
            return self.container(c, depth)
        if c == '"':
            return self.string()
        for literal, result in (("true", True), ("false", False), ("null", None)):
            if self.text.startswith(literal, self.pos):
                self.pos += len(literal)
                return result
        number = _NUMBER.match(self.text, self.pos)
        if number is not None:
            self.pos = number.end()
            return NumberToken(number.group())
        raise JSONInputError("json_syntax")

    def string(self):
        # stdlib handles JSON escape syntax and combines valid surrogate pairs.
        # This single token is already bounded by the pre-parse byte limit.
        try:
            value, end = scanstring(self.text, self.pos + 1, True)
        except (JSONDecodeError, ValueError):
            raise JSONInputError("json_string") from None
        if any(0xD800 <= ord(c) <= 0xDFFF for c in value):
            raise JSONInputError("unicode_scalar")
        if len(value) > MAX_STRING_SCALARS:
            raise JSONInputError("string_scalars")
        self.pos = end
        return value

    def container(self, opening, depth):
        is_object = opening == "{"
        closing = "}" if is_object else "]"
        result = {} if is_object else []
        self.pos += 1
        self.whitespace()
        if self.peek() == closing:
            self.pos += 1
            return result
        while True:
            if is_object:
                if self.peek() != '"':
                    raise JSONInputError("json_syntax")
                key = self.string()
                if key in result:
                    raise JSONInputError("duplicate_key")
                self.whitespace()
                if self.peek() != ":":
                    raise JSONInputError("json_syntax")
                self.pos += 1
                result[key] = self.value(depth)
            else:
                if len(result) >= MAX_ARRAY_ELEMENTS:
                    raise JSONInputError("array_elements")
                result.append(self.value(depth))
            self.whitespace()
            c = self.peek()
            if c == closing:
                self.pos += 1
                return result
            if c != ",":
                raise JSONInputError("json_syntax")
            self.pos += 1
            self.whitespace()
            if self.peek() == closing:
                raise JSONInputError("json_syntax")
