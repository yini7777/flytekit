import asyncio
from contextlib import contextmanager
from functools import partial, wraps
from typing import Any, Optional

from flytekit import current_context, Deck
from flytekit.loggers import logger
from flytekit.configuration import DataConfig, PlatformConfig, S3Config
from flytekit.core.base_task import PythonTask
from flytekit.core.context_manager import ExecutionState, FlyteContext, FlyteContextManager
from flytekit.core.task import task
from flytekit.remote import FlyteRemote


class AsyncEntity:
    def __init__(
        self,
        task,
        remote: Optional[FlyteRemote],
        ctx: FlyteContext,
        async_graph: dict,
        n_polls: int = 1000,
        poll_duration: int = 3,
        force_remote: bool = False,
    ):
        self.task = task
        self.ctx = ctx
        self.async_graph = async_graph
        self.execution_state = self.ctx.execution_state.mode
        self.remote = prepare_remote(remote, ctx, force_remote)
        if self.remote is not None:
            logger.info(f"Using remote config: {self.remote.config}")
        else:
            logger.info("Not using remote, executing locally")
        self._n_polls = n_polls
        self._poll_duration = poll_duration
        self._force_remote = force_remote

    async def __call__(self, **kwargs):
        print(f"calling {self.task}: {self.task.name}")

        # ensure async context is lot provided
        if "async_ctx" in kwargs:
            kwargs.pop("async_ctx")

        if self.execution_state == ExecutionState.Mode.LOCAL_WORKFLOW_EXECUTION and not self._force_remote:
            # If running as a local workflow execution, just execute the python function
            return self.task._task_function(**kwargs)

        # this is a hack to handle the case when the task.name doesn't contain the fully
        # qualified module name
        task_name = (
            f"{self.task._instantiated_in}.{self.task.name}"
            if self.task._instantiated_in not in self.task.name
            else self.task.name
        )

        task = self.remote.fetch_task(name=task_name)
        execution = self.remote.execute(task, inputs=kwargs, type_hints=self.task.python_interface.inputs)
        # try:
        #     execution = self.remote.execute(task, inputs=kwargs)
        # except:
        #     execution = self.remote.execute(task, inputs=kwargs, type_hints=self.task.python_interface.inputs)

        url = self.remote.generate_console_url(execution)
        print(url)

        for _ in range(self._n_polls):
            execution = self.remote.sync(execution)
            if execution.is_done:
                break
            await asyncio.sleep(self._poll_duration)

        outputs = {}
        for key, type_ in self.task.python_interface.outputs.items():
            outputs[key] = execution.outputs.get(key, as_type=type_)
            # try:
            #     outputs[key] = execution.outputs.get(key)
            # except ValueError:
            #     outputs[key] = execution.outputs.get(key, as_type=type_)

        node = AsyncNode(task_name, execution.id, url, inputs=kwargs, outputs=outputs)
        self.async_graph.set_node(node)

        if len(outputs) == 1:
            out, *_ = outputs.values()
            return out
        return outputs
    

class AsyncNode:
    def __init__(self, task_id, execution_id=None, url=None, inputs=None, outputs=None):
        self.task_id = task_id
        self.execution_id = execution_id
        self.url = url
        self.inputs = inputs
        self.outputs = outputs

    def __repr__(self):
        ex_id = self.execution_id
        execution = None if self.execution_id is None else f"{ex_id.project}:{ex_id.domain}:{ex_id.name}"
        return f"<async_node | task: {self.task_id} | execution: {execution} | inputs: {self.inputs} | outputs: {self.outputs}"
    

class AsyncGraph:

    def __init__(self, parent_node: str):
        self.parent_node = parent_node
        self.call_stack = []
        self.prev_node = parent_node

    def __repr__(self):
        return f"<parent_node: '{self.parent_node}' call_stack: {self.call_stack}>"
    
    def set_node(self, node: AsyncNode):
        self.call_stack.append(node)
        self.prev_node = node

    def construct(self):
        """Infer and construct a graph based on the node execution order, inputs, and outputs."""
        ...

    def to_dataframe(self):
        """Render callstack as DataFrame."""
        import pandas as pd

        return pd.DataFrame.from_records([
            {
                "task": node.task_id,
                "execution": node.execution_id,
                "url": node.url,
                "inputs": node.inputs,
                "outputs": node.outputs,
            }
            for node in self.call_stack
        ])
        


def async_output(output: Any, task, execution):
    # TODO: inject flyte metadata to trace data being passed from one task to the next.
    output.__flyte_async_metadata__ = {
        "task": task,
        "execution": execution,
    }


@contextmanager
def eager_mode(fn, remote: FlyteRemote, ctx: FlyteContext, async_graph: dict, force_remote: bool):

    _original_cache = {}
    _globals = fn.__globals__

    # override tasks with async version
    for k, v in _globals.items():
        if isinstance(v, PythonTask):
            _original_cache[k] = v
            _globals[k] = AsyncEntity(v, remote, ctx, async_graph, force_remote=force_remote)

    yield

    # restore old tasks
    for k, v in _original_cache.items():
        _globals[k] = v


NODE_MARKDOWN_TEMPLATE = \
"""
<style>
    #flyte-frame-container > div.active {{font-family: Open sans;}}
</style>

<style>
    #flyte-frame-container div.input-output {{
        font-family: monospace;
        background: #f0f0f0;
        padding: 10px 15px;
        margin: 15px 0;
    }}
</style>

### Task: `{task_name}`

<p>
    <strong>Execution:</strong>
    <a target="_blank" href="{url}">{execution_name}</a>
</p>

<details>
<summary>Inputs</summary>
<div class="input-output">{inputs}</div>
</details>

<details>
<summary>Outputs</summary>
<div class="input-output">{outputs}</div>
</details>

<hr>
"""


async def render_deck(out, async_graph):
    from flytekitplugins.deck.renderer import MarkdownRenderer

    output = "# Nodes\n\n<hr>"
    for node in async_graph.call_stack:
        output = (
            f"{output}\n" +
            NODE_MARKDOWN_TEMPLATE.format(
                task_name=node.task_id,
                execution_name=node.execution_id.name,
                url=node.url,
                inputs=node.inputs,
                outputs=node.outputs,
            )
        )

    Deck("eager workflow", MarkdownRenderer().to_html(output))


def eager(
    _fn=None,
    *,
    remote: FlyteRemote,
    force_remote: bool = False,  # this argument is purely for testing!
    **kwargs,
):
    if _fn is None:
        return partial(eager, remote=remote, force_remote=force_remote, **kwargs)

    @wraps(_fn)
    async def wrapper(*args, **kws):
        # grab the "async_ctx" argument injected by PythonFunctionTask.execute
        ctx = kws.pop("async_ctx")
        exec_params = ctx.user_space_params
        parent_node = AsyncNode(exec_params.task_id, exec_params.execution_id, inputs=kws, outputs={})
        async_graph = AsyncGraph(parent_node)
        with eager_mode(_fn, remote, ctx, async_graph, force_remote):
            out = await _fn(*args, **kws)
        # need to await for _fn to complete, then invoke the deck
        await render_deck(out, async_graph)
        return out
    
    wrapper.__is_eager__ = True  #  HACK!
    return task(wrapper, **kwargs)


def prepare_remote(remote: Optional[FlyteRemote], ctx: FlyteContext, force_remote: bool) -> Optional[FlyteRemote]:
    """Prepare FlyteRemote object for accessing Flyte cluster in a task running on the same cluster."""

    if ctx.execution_state.mode == ExecutionState.Mode.LOCAL_WORKFLOW_EXECUTION and not force_remote:
        # if running the "eager workflow" (which is actually task) locally, run the task as a function,
        # which doesn't need a remote object
        return None
    elif ctx.execution_state.mode == ExecutionState.Mode.LOCAL_WORKFLOW_EXECUTION and force_remote:
        # if running locally with `force_remote=True`, just return the remote object
        return remote

    # Handle the case where this the task is running in a Flyte cluster and needs to access the cluster itself
    # via FlyteRemote.
    is_demo_remote = remote.config.platform.endpoint.startswith("localhost")
    if is_demo_remote:
        # replace sandbox endpoints with internal dns, since localhost won't exist within the Flyte cluster
        return internal_demo_remote(remote)
    return internal_remote(remote)


def internal_demo_remote(remote: FlyteRemote) -> FlyteRemote:
    """Derives a FlyteRemote object from a sandbox yaml configuration, modifying parts to make it work internally."""
    # replace sandbox endpoints with internal dns, since localhost won't exist within the Flyte cluster
    return FlyteRemote(
        config=remote.config.with_params(
            platform=PlatformConfig(
                endpoint="flyte-sandbox.flyte:8089",
                insecure=True,
                auth_mode="Pkce",
            ),
            data_config=DataConfig(
                s3=S3Config(
                    endpoint=f"http://flyte-sandbox-minio.flyte:9000",
                    access_key_id=remote.config.data_config.s3.access_key_id,
                    secret_access_key=remote.config.data_config.s3.secret_access_key,
                ),
            ),
        ),
        default_domain=remote.default_domain,
        default_project=remote.default_project,
    )


def internal_remote(remote: FlyteRemote) -> FlyteRemote:
    """Derives a FlyteRemote object from a yaml configuration file, modifying parts to make it work internally."""
    secrets_manager = current_context().secrets
    client_secret = secrets_manager.get("async-client-secret", "client_secret")
    # get the raw output prefix from the context that's set from the pyflyte-execute entrypoint
    # (see flytekit/bin/entrypoint.py)
    ctx = FlyteContextManager.current_context()
    return FlyteRemote(
        config=remote.config.with_params(
            platform=PlatformConfig(
                endpoint=remote.config.platform.endpoint,
                insecure=remote.config.platform.insecure,
                auth_mode="client_credentials",
                client_id=remote.config.platform.client_id,
                client_credentials_secret=client_secret,
            ),
        ),
        default_domain=remote.default_domain,
        default_project=remote.default_project,
        data_upload_location=ctx.file_access.raw_output_prefix,
    )
