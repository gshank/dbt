from dataclasses import fields
from enum import Enum
from typing import Type

from dbt.dataclass_schema import dbtClassMixin, FieldEncoder


class StrEnum(str, Enum):
    def __str__(self) -> str:
        return self.value

    # https://docs.python.org/3.6/library/enum.html#using-automatic-values
    def _generate_next_value_(name, start, count, last_values):
        return name


def StrLiteral(value: str) -> Type[StrEnum]:
    # mypy doesn't think this works, but it does
    return StrEnum(value, value)  # type: ignore


def register_pattern(base_type: Type, pattern: str) -> None:
    """base_type should be a typing.NewType that should always have the given
    regex pattern. That means that its underlying type ('__supertype__') had
    better be a str!
    """

    class PatternEncoder(FieldEncoder):
        @property
        def json_schema(self):
            return {"type": "string", "pattern": pattern}

    dbtClassMixin.register_field_encoders({base_type: PatternEncoder()})


class HyphenateddbtClassMixin(dbtClassMixin):
    @classmethod
    def field_mapping(cls):
        result = {}
        for field in fields(cls):
            skip = field.metadata.get("preserve_underscore")
            if skip:
                continue

            if "_" in field.name:
                result[field.name] = field.name.replace("_", "-")
        return result


class ExtensibleDbtClassMixin(dbtClassMixin):
    ADDITIONAL_PROPERTIES = True
