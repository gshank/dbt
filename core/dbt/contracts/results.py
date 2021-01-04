from dbt.contracts.graph.manifest import CompileResultNode
from dbt.contracts.graph.unparsed import (
    FreshnessStatus, FreshnessThreshold
)
from dbt.contracts.graph.parsed import ParsedSourceDefinition
from dbt.contracts.util import (
    BaseArtifactMetadata,
    ArtifactMixin,
    Writable,
    VersionedSchema,
    Replaceable,
    schema_version,
)
from dbt.exceptions import InternalException
from dbt.logger import (
    TimingProcessor,
    JsonOnly,
    GLOBAL_LOGGER as logger,
)
from dbt.utils import lowercase
from dbt.dataclass_schema import dbtClassMixin, StrEnum

import agate
from mashumaro.types import SerializableType

from dataclasses import dataclass, field
from datetime import datetime
from typing import Union, Dict, List, Optional, Any, NamedTuple, Sequence


@dataclass
class TimingInfo(dbtClassMixin):
    name: str
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None

    def begin(self):
        self.started_at = datetime.utcnow()

    def end(self):
        self.completed_at = datetime.utcnow()


class collect_timing_info:
    def __init__(self, name: str):
        self.timing_info = TimingInfo(name=name)

    def __enter__(self):
        self.timing_info.begin()
        return self.timing_info

    def __exit__(self, exc_type, exc_value, traceback):
        self.timing_info.end()
        with JsonOnly(), TimingProcessor(self.timing_info):
            logger.debug('finished collecting timing info')


@dataclass
class BaseResult(dbtClassMixin):
    node: CompileResultNode
    error: Optional[str] = None
    status: Optional[Union[str, int, bool]] = None
    execution_time: Union[str, int, float] = 0
    thread_id: Optional[str] = None
    timing: List[TimingInfo] = field(default_factory=list)
    fail: Optional[bool] = None
    warn: Optional[bool] = None


@dataclass
class PartialResult(BaseResult, Writable):
    pass

    # if the result got to the point where it could be skipped/failed, we would
    # be returning a real result, not a partial.
    @property
    def skipped(self):
        return False


@dataclass
class WritableRunModelResult(BaseResult, Writable):
    skip: bool = False

    @property
    def skipped(self):
        return self.skip

# TODO : Does this work? no-op agate table serialization in RunModelResult
class SerializableAgateTable(agate.Table, SerializableType):
    def _serialize(self) -> None:
        return None


@dataclass
class RunModelResult(WritableRunModelResult):
    agate_table: Optional[SerializableAgateTable] = None


@dataclass
class ExecutionResult(dbtClassMixin):
    results: Sequence[BaseResult]
    elapsed_time: float

    def __len__(self):
        return len(self.results)

    def __iter__(self):
        return iter(self.results)

    def __getitem__(self, idx):
        return self.results[idx]


# TODO: This replaced a Union of PartialResult and WritableRunModelResult
# It's not clear to my why there were different classes. The only difference
# was in the 'skipped' method and 'skip' attribute
class RunResult(WritableRunModelResult):
    pass


@dataclass
class RunResultsMetadata(BaseArtifactMetadata):
    dbt_schema_version: str = field(
        default_factory=lambda: str(RunResultsArtifact.dbt_schema_version)
    )


@dataclass
@schema_version('run-results', 1)
class RunResultsArtifact(
    ExecutionResult,
    ArtifactMixin,
):
    results: Sequence[RunResult]
    args: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_node_results(
        cls,
        results: Sequence[RunResult],
        elapsed_time: float,
        generated_at: datetime,
        args: Dict,
    ):
        meta = RunResultsMetadata(
            dbt_schema_version=str(cls.dbt_schema_version),
            generated_at=generated_at,
        )
        return cls(
            metadata=meta,
            results=results,
            elapsed_time=elapsed_time,
            args=args
        )


@dataclass
class RunOperationResult(ExecutionResult):
    success: bool


@dataclass
class RunOperationResultMetadata(BaseArtifactMetadata):
    dbt_schema_version: str = field(default_factory=lambda: str(
        RunOperationResultsArtifact.dbt_schema_version
    ))


@dataclass
@schema_version('run-operation-result', 1)
class RunOperationResultsArtifact(RunOperationResult, ArtifactMixin):

    @classmethod
    def from_success(
        cls,
        success: bool,
        elapsed_time: float,
        generated_at: datetime,
    ):
        meta = RunResultsMetadata(
            dbt_schema_version=str(cls.dbt_schema_version),
            generated_at=generated_at,
        )
        return cls(
            metadata=meta,
            results=[],
            elapsed_time=elapsed_time,
            success=success,
        )


@dataclass
class SourceFreshnessResultMixin(dbtClassMixin):
    max_loaded_at: datetime
    snapshotted_at: datetime
    age: float


# due to issues with typing.Union collapsing subclasses, this can't subclass
# PartialResult
@dataclass
class SourceFreshnessResult(BaseResult, Writable, SourceFreshnessResultMixin):
    node: ParsedSourceDefinition
    status: FreshnessStatus = FreshnessStatus.Pass

    def __post_init__(self):
        self.fail = self.status == 'error'

    @property
    def warned(self):
        return self.status == 'warn'

    @property
    def skipped(self):
        return False


def _copykeys(src, keys, **updates):
    return {k: getattr(src, k) for k in keys}


class FreshnessErrorEnum(StrEnum):
    runtime_error = 'runtime error'


@dataclass
class SourceFreshnessRuntimeError(dbtClassMixin):
    unique_id: str
    error: str
    state: FreshnessErrorEnum


@dataclass
class SourceFreshnessOutput(dbtClassMixin):
    unique_id: str
    max_loaded_at: datetime
    snapshotted_at: datetime
    max_loaded_at_time_ago_in_s: float
    state: FreshnessStatus
    criteria: FreshnessThreshold


FreshnessNodeResult = Union[PartialResult, SourceFreshnessResult]
FreshnessNodeOutput = Union[SourceFreshnessRuntimeError, SourceFreshnessOutput]


def process_freshness_result(
    result: FreshnessNodeResult
) -> FreshnessNodeOutput:
    unique_id = result.node.unique_id
    if result.error is not None:
        return SourceFreshnessRuntimeError(
            unique_id=unique_id,
            error=result.error,
            state=FreshnessErrorEnum.runtime_error,
        )

    # we know that this must be a SourceFreshnessResult
    if not isinstance(result, SourceFreshnessResult):
        raise InternalException(
            'Got {} instead of a SourceFreshnessResult for a '
            'non-error result in freshness execution!'
            .format(type(result))
        )
    # if we're here, we must have a non-None freshness threshold
    criteria = result.node.freshness
    if criteria is None:
        raise InternalException(
            'Somehow evaluated a freshness result for a source '
            'that has no freshness criteria!'
        )
    return SourceFreshnessOutput(
        unique_id=unique_id,
        max_loaded_at=result.max_loaded_at,
        snapshotted_at=result.snapshotted_at,
        max_loaded_at_time_ago_in_s=result.age,
        state=result.status,
        criteria=criteria,
    )


@dataclass
class FreshnessMetadata(BaseArtifactMetadata):
    dbt_schema_version: str = field(
        default_factory=lambda: str(
            FreshnessExecutionResultArtifact.dbt_schema_version
        )
    )


@dataclass
class FreshnessResult(ExecutionResult):
    metadata: FreshnessMetadata
    results: Sequence[FreshnessNodeResult]

    @classmethod
    def from_node_results(
        cls,
        results: List[FreshnessNodeResult],
        elapsed_time: float,
        generated_at: datetime,
    ):
        meta = FreshnessMetadata(generated_at=generated_at)
        return cls(metadata=meta, results=results, elapsed_time=elapsed_time)


@dataclass
@schema_version('sources', 1)
class FreshnessExecutionResultArtifact(
    ArtifactMixin,
    VersionedSchema,
):
    metadata: FreshnessMetadata
    results: Sequence[FreshnessNodeOutput]
    elapsed_time: float

    @classmethod
    def from_result(cls, base: FreshnessResult):
        processed = [process_freshness_result(r) for r in base.results]
        return cls(
            metadata=base.metadata,
            results=processed,
            elapsed_time=base.elapsed_time,
        )


Primitive = Union[bool, str, float, None]

CatalogKey = NamedTuple(
    'CatalogKey',
    [('database', Optional[str]), ('schema', str), ('name', str)]
)


@dataclass
class StatsItem(dbtClassMixin):
    id: str
    label: str
    value: Primitive
    description: Optional[str]
    include: bool


StatsDict = Dict[str, StatsItem]


@dataclass
class ColumnMetadata(dbtClassMixin):
    type: str
    comment: Optional[str]
    index: int
    name: str


ColumnMap = Dict[str, ColumnMetadata]


@dataclass
class TableMetadata(dbtClassMixin):
    type: str
    schema: str
    name: str
    database: Optional[str] = None
    comment: Optional[str] = None
    owner: Optional[str] = None


@dataclass
class CatalogTable(dbtClassMixin, Replaceable):
    metadata: TableMetadata
    columns: ColumnMap
    stats: StatsDict
    # the same table with two unique IDs will just be listed two times
    unique_id: Optional[str] = None

    def key(self) -> CatalogKey:
        return CatalogKey(
            lowercase(self.metadata.database),
            self.metadata.schema.lower(),
            self.metadata.name.lower(),
        )


@dataclass
class CatalogMetadata(BaseArtifactMetadata):
    dbt_schema_version: str = field(
        default_factory=lambda: str(CatalogArtifact.dbt_schema_version)
    )


@dataclass
class CatalogResults(dbtClassMixin):
    nodes: Dict[str, CatalogTable]
    sources: Dict[str, CatalogTable]
    errors: Optional[List[str]]
    _compile_results: Optional[Any] = None


@dataclass
@schema_version('catalog', 1)
class CatalogArtifact(CatalogResults, ArtifactMixin):
    metadata: CatalogMetadata

    @classmethod
    def from_results(
        cls,
        generated_at: datetime,
        nodes: Dict[str, CatalogTable],
        sources: Dict[str, CatalogTable],
        compile_results: Optional[Any],
        errors: Optional[List[str]]
    ) -> 'CatalogArtifact':
        meta = CatalogMetadata(generated_at=generated_at)
        return cls(
            metadata=meta,
            nodes=nodes,
            sources=sources,
            errors=errors,
            _compile_results=compile_results,
        )
