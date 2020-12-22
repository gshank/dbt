import functools
from typing import (
    Optional, Type, Union, Any, Dict, cast, Tuple, List, TypeVar,
    get_type_hints, Callable, Generic,
)
import re
from datetime import datetime
from dataclasses import (
    fields, is_dataclass, Field, MISSING, dataclass, asdict
)
from uuid import UUID
from enum import Enum
import threading
import warnings

from dateutil.parser import parse
import jsonschema  # type: ignore

from mashumaro import DataClassDictMixin
from mashumaro.types import SerializableType

JSON_ENCODABLE_TYPES = {
    str: {"type": "string"},
    int: {"type": "integer"},
    bool: {"type": "boolean"},
    float: {"type": "number"},
    type(None): {"type": "null"},
}

JsonEncodable = Union[int, float, str, bool, None]
JsonDict = Dict[str, Any]

OPTIONAL_TYPES = ["Union", "Optional"]



class ValidationError(jsonschema.ValidationError):
    pass


class FutureValidationError(ValidationError):
    # a validation error where we haven't called str() on inputs yet.
    def __init__(self, field: str, errors: Dict[str, Exception]):
        self.errors = errors
        self.field = field
        super().__init__("generic validation error")
        self.initialized = False

    def late_initialize(self):
        lines: List[str] = []
        for name, exc in self.errors.items():
            # do not use getattr(exc, 'message', str(exc)), it's slow!
            if hasattr(exc, "message"):
                msg = exc.message
            else:
                msg = str(exc)
            lines.append(f"{name}: {msg}")

        super().__init__(
            "Unable to decode value for '{}: No members matched:\n{}".format(
                self.field, lines
            )
        )
        self.initialized = True

    def __str__(self):
        if not self.initialized:
            self.late_initialize()
        return super().__str__()


def is_enum(field_type: Any) -> bool:
    return issubclass_safe(field_type, Enum)


def issubclass_safe(klass: Any, base: Type) -> bool:
    try:
        return issubclass(klass, base)
    except TypeError:
        return False


def is_optional(field: Any) -> bool:
    if str(field).startswith("typing.Union") or str(field).startswith(
        "typing.Optional"
    ):
        for arg in field.__args__:
            if isinstance(arg, type) and issubclass(arg, type(None)):
                return True

    return False


TV = TypeVar("TV")


class FieldEncoder(Generic[TV]):
    """Base class for encoding fields to and from JSON encodable values"""

    def to_wire(self, value: TV) -> JsonEncodable:
        return value  # type: ignore

    def to_python(self, value: JsonEncodable) -> TV:
        return value  # type: ignore

    @property
    def json_schema(self) -> JsonDict:
        raise NotImplementedError()


T = TypeVar("T", bound="dbtClassMixin")


@functools.lru_cache()
def _to_camel_case(value: str) -> str:
    if "_" in value:
        parts = value.split("_")
        return "".join(
            [parts[0]] + [part[0].upper() + part[1:] for part in parts[1:]]
        )
    else:
        return value


@dataclass
class FieldMeta:
    default: Any = None
    description: Optional[str] = None

    @property
    def as_dict(self) -> Dict:
        return {
            _to_camel_case(k): v
            for k, v in asdict(self).items()
            if v is not None
        }


@functools.lru_cache()
def _validate_schema(schema_cls: Type[T]) -> JsonDict:
    schema = schema_cls.json_schema()
    jsonschema.Draft7Validator.check_schema(schema)
    return schema


# a restriction is a list of Field, str pairs
Restriction = List[Tuple[Field, str]]
# a restricted variant is a pair of an object that has fields with restrictions
# and those restrictions. Only dbtClassMixin subclasses may have restrictied
# fields.
Variant = Tuple[Type[T], Optional[Restriction]]


def _get_restrictions(variant_type: Type) -> Restriction:
    """Return a list of all restrictions on the given variant of a
    union, in the form of a Field, name pair, where `name` is the
    field's name in json and the Field is the dataclass-level
    field name.

    If the variant isn't a dbtClassMixin subclass,
    there are no restrictions.
    """
    if not issubclass_safe(variant_type, dbtClassMixin):
        return []
    restrictions: Restriction = []
    for field, target_name in variant_type._get_fields():
        if field.metadata and "restrict" in field.metadata:
            restrictions.append((field, target_name))
    return restrictions


def get_union_fields(field_type: Union[Any]) -> List[Variant]:
    """
    Unions have a __args__ that is all their variants (after typing's
    type-collapsing magic has run, so caveat emptor...)

    dbtClassMixin dataclasses have `Field`s, returned by the
    `_get_fields` method.

    This method returns list of 2-tuples:
        - the first value is always a type
        - the second value is None if there are no restrictions,
            or a list of restrictions if there are restrictions

    The list will be sorted so that unrestricted variants will always
    be at the end.
    """
    fields: List[Variant] = []
    for variant in field_type.__args__:
        restrictions: Optional[Restriction] = _get_restrictions(variant)
        if not restrictions:
            restrictions = None
        fields.append((variant, restrictions))

    # put unrestricted variants last
    fields.sort(key=lambda f: f[1] is None)
    return fields


def _encode_restrictions_met(
    value: Any, restrict_fields: Optional[List[Tuple[Field, str]]]
) -> bool:
    if restrict_fields is None:
        return True
    return all(
        (
            hasattr(value, f.name) and
            getattr(value, f.name) in f.metadata["restrict"]
        )
        for f, _ in restrict_fields
    )


def _decode_restrictions_met(
    value: Any, restrict_fields: Optional[List[Tuple[Field, str]]]
) -> bool:
    if restrict_fields is None:
        return True
    return all(
        n in value and value[n] in f.metadata["restrict"]
        for f, n in restrict_fields
    )


@dataclass
class CompleteSchema:
    schema: JsonDict
    definitions: JsonDict


_DBTCLASS_LOCK = threading.RLock()


class dbtClassMixin(DataClassDictMixin):
    """Mixin which adds methods to generate a JSON schema and
       convert to and from JSON encodable dicts with validation
       against the schema
    """

    def field_mapping(cls) -> Dict[str, str]:
        """Defines the mapping of python field names to JSON field names.

        The main use-case is to allow JSON field names which are Python keywords
        """
        return {}

    def to_dict(self, omit_none=False, validate=False):
        dct = self._to_dict()
        # this is only for connections and should be removed and
        # handled in connection objects only
        if hasattr(self, '_ALIASES') and self._ALIASES:
            # TODO : Mutating these dicts is a TERRIBLE idea - remove this
            for aliased_name, canonical_name in self._ALIASES.items():
                if aliased_name in dct:
                    dct[canonical_name] = dct.pop(aliased_name)

        return dct

    @classmethod
    def from_dict(cls, data, validate=False):
        # this is only for connections and should be removed and
        # handled in connection objects only
        if hasattr(cls, '_ALIASES') and cls._ALIASES:
            for aliased_name, canonical_name in cls._ALIASES.items():
                if aliased_name in data:
                    data[canonical_name] = data.pop(aliased_name)
        # mashumaro from_dict method has been renamed to _from_dict
        return cls._from_dict(data)

    @staticmethod
    def _is_json_schema_subclass(field_type: Type) -> bool:
        return issubclass_safe(field_type, dbtClassMixin)

    @staticmethod
    def _has_definition(field_type: Type) -> bool:
        return (
            issubclass_safe(field_type, dbtClassMixin) and
            field_type.__name__ != "PatternProperty"
        )

    @classmethod
    def _get_field_meta(cls, field: Field) -> Tuple[FieldMeta, bool]:
        required = True
        field_meta = FieldMeta()
        default_value: Optional[Callable[[], Any]] = None
        if field.default is not MISSING and field.default is not None:
            # In case of default value given
            default_value = field.default
        elif (
            field.default_factory is not MISSING and  # type: ignore
            field.default_factory is not None  # type: ignore
        ):  # type: ignore
            # In case of a default factory given, we call it
            default_value = field.default_factory()  # type: ignore

        if default_value is not None:
            field_meta.default = cls._encode_field(
                field.type, default_value, omit_none=False
            )
            required = False

        if field.metadata is not None:
            if "description" in field.metadata:
                field_meta.description = field.metadata["description"]

        return field_meta, required

    @classmethod
    def _encode_restrictions(
        cls, restrictions: Union[List[Any], Type[Enum]]
    ) -> JsonDict:
        field_schema: JsonDict = {}
        member_types = set()
        values = []
        for member in restrictions:
            if isinstance(member, Enum):
                value = member.value
            else:
                value = member
            member_types.add(type(value))
            values.append(value)
        if len(member_types) == 1:
            member_type = member_types.pop()
            if member_type in JSON_ENCODABLE_TYPES:
                field_schema.update(JSON_ENCODABLE_TYPES[member_type])
            else:
                field_schema.update(
                    cls._field_encoders[member_type].json_schema
                )
        else:
            # this used to silently do nothing here, which seems worse
            raise ValidationError(
                "Invalid schema defined: "
                "Found multiple member types - {!s}".format(member_types)
            )
        field_schema["enum"] = values
        return field_schema

    @classmethod
    def _get_schema_for_type(
        cls,
        target: Type,
        required: bool = True,
        restrictions: Optional[List[Any]] = None,
    ) -> Tuple[JsonDict, bool]:

        field_schema: JsonDict = {"type": "object"}

        type_name = cls._get_field_type_name(target)

        if target in cls._field_encoders:
            field_schema.update(cls._field_encoders[target].json_schema)

        elif restrictions:
            field_schema.update(cls._encode_restrictions(restrictions))

        # if Union[..., None] or Optional[...]
        elif type_name in OPTIONAL_TYPES:
            field_schema = {
                "oneOf": [
                    cls._get_field_schema(variant)[0]
                    for variant in target.__args__
                ]
            }

            if is_optional(target):
                required = False

        elif is_enum(target):
            field_schema.update(cls._encode_restrictions(target))

        elif type_name in ("Dict", "Mapping"):
            field_schema = {"type": "object"}
            if target.__args__[1] is not Any:
                field_schema["additionalProperties"] = cls._get_field_schema(
                    target.__args__[1]
                )[0]
        elif type_name == "PatternProperty":
            field_schema = {"type": "object"}
            field_schema["patternProperties"] = {
                ".*": cls._get_field_schema(target.TARGET_TYPE)[0]
            }

        elif type_name in ("Sequence", "List") or (
            type_name == "Tuple" and ... in target.__args__
        ):
            field_schema = {"type": "array"}
            if target.__args__[0] is not Any:
                field_schema["items"] = cls._get_field_schema(
                    target.__args__[0]
                )[0]

        elif type_name == "Tuple":
            tuple_len = len(target.__args__)
            # TODO: How do we handle Optional type within lists / tuples
            field_schema = {
                "type": "array",
                "minItems": tuple_len,
                "maxItems": tuple_len,
                "items": [
                    cls._get_field_schema(type_arg)[0]
                    for type_arg in target.__args__
                ],
            }

        elif target in JSON_ENCODABLE_TYPES:
            field_schema.update(JSON_ENCODABLE_TYPES[target])

        elif hasattr(target, "__supertype__"):  # NewType fields
            field_schema, _ = cls._get_field_schema(target.__supertype__)

        else:
            raise ValidationError(f"Unable to create schema for '{type_name}'")
        return field_schema, required

    @classmethod
    def _get_field_schema(
        cls, field: Union[Field, Type]
    ) -> Tuple[JsonDict, bool]:
        required = True
        restrictions = None

        if isinstance(field, Field):
            field_type = field.type
            field_meta, required = cls._get_field_meta(field)
            if field.metadata is not None:
                restrictions = field.metadata.get("restrict")
        else:
            field_type = field
            field_meta = FieldMeta()

        field_type_name = cls._get_field_type_name(field_type)

        if cls._has_definition(field_type):
            field_schema: JsonDict = {
                "$ref": "#/definitions/{}".format(field_type_name)
            }
        else:
            field_schema, required = cls._get_schema_for_type(
                field_type, required=required, restrictions=restrictions
            )

        field_schema.update(field_meta.as_dict)

        return field_schema, required

    @classmethod
    def _get_field_definitions(cls, field_type: Any, definitions: JsonDict):
        field_type_name = cls._get_field_type_name(field_type)
        if field_type_name == "Tuple":
            # tuples are either like Tuple[T, ...] or Tuple[T1, T2, T3].
            for member in field_type.__args__:
                if member is not ...:
                    cls._get_field_definitions(member, definitions)
        elif field_type_name in ("Sequence", "List"):
            cls._get_field_definitions(field_type.__args__[0], definitions)
        elif field_type_name in ("Dict", "Mapping"):
            cls._get_field_definitions(field_type.__args__[1], definitions)
        elif field_type_name == "PatternProperty":
            cls._get_field_definitions(field_type.TARGET_TYPE, definitions)
        elif field_type_name in OPTIONAL_TYPES:
            for variant in field_type.__args__:
                cls._get_field_definitions(variant, definitions)
        elif cls._is_json_schema_subclass(field_type):
            # Prevent recursion from forward refs & circular type dependencies
            if field_type.__name__ not in definitions:
                definitions[field_type.__name__] = None
                definitions.update(
                    field_type._json_schema_recursive(
                        embeddable=True, definitions=definitions
                    )
                )

    @classmethod
    def all_json_schemas(cls) -> JsonDict:
        """Returns JSON schemas for all subclasses"""
        definitions = {}
        for subclass in cls.__subclasses__():
            if is_dataclass(subclass):
                definitions.update(subclass.json_schema(embeddable=True))
            else:
                definitions.update(subclass.all_json_schemas())
        return definitions

    @classmethod
    def _collect_json_schema(cls, definitions: JsonDict) -> JsonDict:
        """Return the schema dictionary and update the definitions dictionary
        for this class.
        """
        properties = {}
        required = []
        for field, target_field in cls._get_fields():
            properties[target_field], is_required = cls._get_field_schema(
                field
            )
            # this will call get_json_schema_recursive
            cls._get_field_definitions(field.type, definitions)
            if is_required:
                required.append(target_field)

        schema = {
            "type": "object",
            "required": required,
            "properties": properties,
            "additionalProperties": cls.ADDITIONAL_PROPERTIES,
        }
        if cls.__doc__:
            schema["description"] = cls.__doc__
        return schema

    @classmethod
    def _schema_defs_from_cache(cls, definitions: JsonDict) -> CompleteSchema:
        # this has to be done at the classmethod level because each subclass
        # needs its own dict, and we don't want to use metaclasses here (it
        # makes it hard for users to use metaclasses)
        if cls._schema is None:
            with _DBTDCLASS_LOCK:
                # check again, in case we were waiting for someone else to do
                # this.
                if cls._schema is None:
                    cls._schema = {}

        if cls.__name__ in cls._schema:
            return cls._schema[cls.__name__]

        with _DBTDCLASS_LOCK:
            if cls.__name__ in cls._schema:
                return cls._schema[cls.__name__]

            # ok, no schema found. go build schemas
            schema = cls._collect_json_schema(definitions)
            complete_schema = CompleteSchema(
                schema=schema, definitions=definitions
            )
            # now that we finished, write our schema in.
            # In the worst-case we write over another thread's work.
            cls._schema[cls.__name__] = complete_schema
            return complete_schema

    @classmethod
    def _json_schema_recursive(
        cls, embeddable: bool, definitions: JsonDict
    ) -> JsonDict:
        schema = cls._schema_defs_from_cache(definitions)

        if embeddable:
            return {**schema.definitions, cls.__name__: schema.schema}

        return {
            **schema.schema,
            **{
                "definitions": schema.definitions,
                "$schema": "http://json-schema.org/draft-07/schema#",
            },
        }

    @classmethod
    def json_schema(cls, embeddable: bool = False) -> JsonDict:
        """Returns the JSON schema for the dataclass, along with the schema
           of any nested dataclasses within the 'definitions' field.

        Enable the embeddable flag to generate the schema in a format for
        embedding into other schemas
        or documents supporting JSON schema such as Swagger specs.
        """
        if cls is dbtClassMixin:
            warnings.warn(
                "Calling 'dbtClassMixin.json_schema' is deprecated. "
                "Use 'dbtClassMixin.all_json_schemas' instead",
                DeprecationWarning,
            )
            return cls.all_json_schemas()

        definitions: JsonDict = {}
        return cls._json_schema_recursive(
            embeddable=embeddable, definitions=definitions
        )

    @staticmethod
    def _get_field_type_name(field_type: Any) -> str:
        try:
            return field_type.__name__
        except AttributeError:
            # The types in the 'typing' module lack the __name__ attribute
            match = re.match(r"typing\.([A-Za-z]+)", str(field_type))
            return str(field_type) if match is None else match.group(1)

    @classmethod
    def validate(cls, data: Any):
        schema = _validate_schema(cls)
        validator = jsonschema.Draft7Validator(schema)
        error = next(iter(validator.iter_errors(data)), None)
        if error is not None:
            raise ValidationError.create_from(error) from error


class ValidatedStringMixin(str, SerializableType):
    ValidationRegex = None

    @classmethod
    def _deserialize(cls, value: str) -> 'ValidatedStringMixin':
        cls.validate(value)
        return ValidatedStringMixin(value)

    def _serialize(self) -> str:
        return str(self)

    @classmethod
    def validate(cls, value) -> str:
        res = re.match(cls.ValidationRegex, value)

        if res is None:
            raise ValidationError(f"Invalid value: {value}") # TODO

class StrEnum(str, Enum):
    def __str__(self) -> str:
        return self.value

    # https://docs.python.org/3.6/library/enum.html#using-automatic-values
    def _generate_next_value_(name, start, count, last_values):
        return name


def StrLiteral(value: str) -> Type[StrEnum]:
    # mypy doesn't think this works, but it does
    return StrEnum(value, value)  # type: ignore


class HyphenatedDbtClassMixin(dbtClassMixin):
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
