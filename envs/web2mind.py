"""Web2Mind环境."""
from dataclasses import dataclass, field
from functools import cached_property
from pathlib import Path
from typing import TYPE_CHECKING

from pandas import DataFrame, Series, read_json

from utils import make_instance

from .basic import Action, ActionSpace, WebEnv

if TYPE_CHECKING:
    from typing_extensions import Self

    from utils import KWARGS


@dataclass
class Web2MindActionSpace(ActionSpace):
    """Web2Mind Action空间."""

    @cached_property
    def click(self: "Self") -> Action:
        """点击."""

        def func(target_element: str) -> str:  # noqa: ARG001
            """点击.

            Args:
                target_element (str): 目标元素

            Returns:
                str: 执行结果
            """
            return "click success!"

        return make_instance(
            cls=Action,
            name="click",
            func=func,
            description="Click the element in web page",
            input_args_schema={"target_element": str},
        )

    @cached_property
    def hover(self: "Self") -> Action:
        """悬停."""

        def func(target_element: str) -> str:  # noqa: ARG001
            """悬停.

            Args:
                target_element (str): 目标元素

            Returns:
                str: 执行结果
            """
            return "hover success!"

        return make_instance(
            cls=Action,
            name="hover",
            func=func,
            description="hover the element in web page",
            input_args_schema={"target_element": str},
        )

    @cached_property
    def type(self: "Self") -> Action:  # noqa: A003
        """输入."""

        def func(target_element: str, text: str) -> str:  # noqa: ARG001
            """输入.

            Args:
                target_element (str): 目标元素
                text (str): 输入文本

            Returns:
                str: 执行结果
            """
            return "type success!"

        return make_instance(
            cls=Action,
            name="type",
            func=func,
            description="type the element in web page",
            input_args_schema={"target_element": str, "text": str},
        )

    @cached_property
    def select(self: "Self") -> Action:
        """选择."""

        def func(target_element: str, option: dict) -> str:  # noqa: ARG001
            """选择.

            Args:
                target_element (str): 目标元素
                option (dict): 选项

            Returns:
                str: 执行结果
            """
            return "select success!"

        return make_instance(
            cls=Action,
            name="select",
            func=func,
            description="select the element in web page",
            input_args_schema={"target_element": str, "option": dict},
        )

    @cached_property
    def actions(self: "Self") -> list[Action]:
        """actions."""
        return [self.click, self.hover, self.type, self.select]

    @cached_property
    def action_names(self: "Self") -> list[str]:
        """Action names."""
        return [action.name for action in self.actions]


@dataclass
class Web2MindEnv(WebEnv):
    """Web2Mind环境."""

    dataset_path: str

    now_situation: Series | None = None
    situations_step_index: int = -1

    action_space: Web2MindActionSpace = field(
        default_factory=Web2MindActionSpace,
        init=False,
    )
    available_situation_ids: list[str] = field(init=False)
    situations: DataFrame | None = field(init=False)

    def __post_init__(self: "Self", **kwargs: "KWARGS") -> None:
        """初始化."""
        if not Path(self.dataset_path).exists():
            msg = f"{self.dataset_path} not found"
            raise FileNotFoundError(msg)
        self.situations = read_json(self.dataset_path)
        self.situations.index = self.situations["annotation_id"]

        self.available_situation_ids = self.situations["annotation_id"].tolist()
        if kwargs is not None:
            (setattr(self, key, value) for key, value in kwargs.items())

    def reset(self: "Self", situation_id: str) -> list[dict]:
        """重置环境.

        Args:
            situation_id (str): 场景id

        Returns:
            list[dict]: 页面
        """
        self.now_situation = self.situations.loc[situation_id]
        self.situations_step_index = 0
        self.env_task = self.now_situation["confirmed_task"]
        return self.read_page()

    def step(self: "Self", _action: Action) -> list[dict]:
        """执行动作.

        Args:
            _action (Action): 动作

        Returns:
            list[dict]: 页面
        """
        if self.situations_step_index == len(self.now_situation) - 1:
            msg = (
                "No more steps in the situation "
                f"{self.now_situation['annotation_id']}",
            )
            raise ValueError(msg)
        self.situations_step_index += 1
        return self.read_page()

    def read_page(self: "Self") -> list[dict]:
        """读取页面."""
        return sorted(
            self.now_situation["actions"][self.situations_step_index][
                "pos_candidates"
            ]
            + self.now_situation["actions"][self.situations_step_index][
                "neg_candidates"
            ],
            key=lambda x: int(x["backend_node_id"]),
        )
