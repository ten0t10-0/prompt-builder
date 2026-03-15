import modules.scripts as scripts               # type: ignore
from modules.processing import process_images   # type: ignore
from modules.shared import cmd_opts             # type: ignore
from modules import ui_loadsave                 # type: ignore

from os import path as os_path
from abc import ABC, abstractmethod
from typing import Generic, TypeVar, NamedTuple, Sequence, Callable, Any, Type
from collections import OrderedDict
from math import isclose
from decimal import Decimal

from gradio.blocks import (
    Block as grBlock
    , BlockContext as grBlockContext
)
from gradio.components.base import (
    Component as grComponent
)
from gradio import (
    __version__ as gr_version
    , Accordion as grAccordion
    , Button as grButton
    , Checkbox as grCheckbox
    , Column as grColumn
    , Dropdown as grDropdown
    , Group as grGroup
    , Markdown as grMarkdown
    , Number as grNumber
    , Row as grRow
    , Slider as grSlider
    , Tab as grTab
    , Textbox as grTextbox
    , HTML as grHTML
)

match gr_version[0]:
    case "3":
        _gr_update = lambda block: block.update
    case _:
        _gr_update = lambda block: type(block)

def gr_update(block: Any, **kwargs: Any) -> Any:
    return _gr_update(block)(**kwargs)

T = TypeVar('T')
T_B_Prompt = TypeVar("T_B_Prompt", bound="B_Prompt")
T_B_UiContainer = TypeVar("T_B_UiContainer", bound="B_UiContainer")

class B:
    title                           = "B Prompt Builder"
    
    class Webui:
        base_path                   = scripts.basedir()
        scripts_folder              = "scripts"
        decimals                    = 2
        break_prompt                = "BREAK"

    class Flag:
        #!ignore_tagged               = False
        use_alt_list_item_name      = True

    class File:
        script_name                 = "b_prompt_builder"
        dedicated_folder            = "b_prompt_builder"
        layout_file                 = "layout.txt"
        comment_indicator           = "#"
        args_indicator              = "--"
        args_separator              = " "
        stop_indicator              = "."

    class File_LineType:
        SINGLE                      = "SINGLE"
        DUAL                        = "DUAL"
        EDIT                        = "EDIT"
        EDIT_LINK                   = "EDIT_LINK"
        SELECT                      = "SELECT"
        PRESET                      = "PRESET"
        SET                         = "SET"
        VALUE                       = "VALUE"
        GROUP                       = "GROUP"
        TAB                         = "TAB"
        ROW                         = "ROW"
        COLUMN                      = "COLUMN"
        ACCORDION                   = "ACCORDION"
        SEPARATOR                   = "SEPARATOR"
        LIST                        = "LIST"
        LIST_ITEM                   = "ENTRY"
        FROM_LIST                   = "FROM_LIST"
        END                         = "END"
    
    class File_Arg:
        name                        = "i"
        is_activated                = "a"
        scale                       = "scale"
        sort                        = "sort"
        open                        = "open"
        prompt                      = "p"
        emphasis                    = "s"
        prompt_pos                  = "pp"
        emphasis_pos                = "sp"
        prompt_neg                  = "pn"
        emphasis_neg                = "sn"
        edit                        = "r"
        edit_prompt_a               = "pa"
        edit_prompt_b               = "pb"
        prefix                      = "prefix"
        postfix                     = "postfix"
        link                        = "link"
        is_negative                 = "n"
        is_additive                 = "add"
        is_reset_visible            = "reset"
        ignore                      = "x"
    
    class Default:
        is_activated: bool          = False
        is_name_visible: bool       = True
        name_link: str              = ""
        prompt: str                 = ""
        emphasis: float             = 1
        edit: float                 = 0.5
        is_negative: bool           = False
        is_additive: bool           = False
        scale: int                  = 1
        sort: bool                  = True
        open: bool                  = True
        open_settings: bool         = False
        use_break: bool             = True
        prepend: bool               = False
        is_reset_visible: bool      = False
    
    class HTML:
        html_separator              = "<hr style=\"margin: 0.5em 0 !important; border-style: dotted; border-color: var(--border-color-primary);\" />"
        css_footer                  = ("b-footer", "margin-top: auto;")
        cls_ui_select_choice        = "b-ui-select-choice"
        id_prefix                   = "b-ui-i"
    
    class Ui:
        final_prompt_label: str     = "Final Prompt"
        final_prompt_neg_label: str = "Final Negative Prompt"
        reset_all_label: str        = "Reset All"
        settings_label: str         = "Settings"
        use_break_label: str        = "Use 'BREAK' to separate the WebUI prompt from the script prompt"
        prepend_label: str          = "Place the script prompt before the WebUI prompt"
        clear_config_label: str     = "Clear config"
        clear_config_desc: str      = "Removes all UI config entries generated by this script"
        prompt_label: str           = "Prompt"
        emphasis_label: str         = "Emphasis"
        prompt_neg_label: str       = "Prompt (N)"
        emphasis_neg_label: str     = "Emphasis (N)"
        edit_label: str             = "Edit"
        is_negative_label: str      = "Negative?"
        prefix_label: str           = "Prefix"
        postfix_label: str          = "Postfix"
        prompt_apply_label: str     = "Apply"
        prompt_remove_label: str    = "Remove"
        reset_label_prefix: str     = "Reset"
        prompt_scale: int           = 4
        emphasis_scale: int         = 1
        affix_scale: int            = 1
        emphasis_min: float         = 0
        emphasis_step: float        = 0.1
        edit_min: float             = 0
        edit_max: float             = 1
        edit_step: float            = 0.1
        
        @staticmethod
        def min_step() -> float:
            return float(Decimal(1) * Decimal(10) ** -B.Webui.decimals)

class B_Fn:
    class Prompt:
        @staticmethod
        def sanitized(prompt: str | None) -> str:
            return prompt.strip() if prompt is not None else ""
        
        @staticmethod
        def added(prompt: str, prompt_to_add: str, use_space: bool = False) -> str:
            if len(prompt_to_add) > 0:
                if len(prompt) > 0:
                    prompt += ((", " if prompt[-1] != "," else " ") if not use_space else " ") + prompt_to_add
                else:
                    prompt = prompt_to_add
            return prompt
        
        @staticmethod
        def decorated(prompt: str, prefix: str = "", postfix: str = "") -> str:
            if len(prompt) > 0:
                if len(prefix) > 0:
                    prompt = f"{prefix} {prompt}"
                if len(postfix) > 0:
                    prompt = f"{prompt} {postfix}"
            return prompt
        
        @staticmethod
        def emphasized(prompt: str, emphasis: float) -> str:
            if len(prompt) == 0 or emphasis == 0:
                return ""
            if emphasis != 1:
                prompt = f"({prompt}:{round(emphasis, B.Webui.decimals)})"
            return prompt

class B_Log:
    @classmethod
    def general(cls, message: str) -> None:
        print(f"* {B.title}: {message}")

    @classmethod
    def warning(cls, obj: Any, name: str, message: str) -> None:
        cls.general(f"WARNING/{(obj if isinstance(obj, type) else type(obj)).__name__}/{name}: {message}")

class PromptPair(NamedTuple):
    pos: str
    neg: str

class B_Args:
    __slots__ = ('source_args')

    def __init__(self, args: dict[str, str]):
        self.source_args = args
    
    @property
    def NO_ARGS(self) -> bool:
        return len(self.source_args) == 0 or (len(self.source_args) == 1 and self.name is not None)
    
    @property
    def name(self) -> str | None:
        return self.source_args.get(B.File_Arg.name)
    
    @property
    def is_activated(self) -> bool | None:
        v = self.source_args.get(B.File_Arg.is_activated)
        return bool(int(v)) if v is not None else None
    
    @property
    def scale(self) -> int | None:
        v = self.source_args.get(B.File_Arg.scale)
        return int(v) if v is not None else None
    
    @property
    def sort(self) -> bool | None:
        v = self.source_args.get(B.File_Arg.sort)
        return bool(int(v)) if v is not None else None
    
    @property
    def open(self) -> bool | None:
        v = self.source_args.get(B.File_Arg.open)
        return bool(int(v)) if v is not None else None
    
    @property
    def prompt(self) -> str | None:
        return self.source_args.get(B.File_Arg.prompt)
    
    @property
    def emphasis(self) -> float | None:
        v = self.source_args.get(B.File_Arg.emphasis)
        return float(v) if v is not None else None
    
    @property
    def prompt_pos(self) -> str | None:
        return self.source_args.get(B.File_Arg.prompt_pos)
    
    @property
    def emphasis_pos(self) -> float | None:
        v = self.source_args.get(B.File_Arg.emphasis_pos)
        return float(v) if v is not None else None
    
    @property
    def prompt_neg(self) -> str | None:
        return self.source_args.get(B.File_Arg.prompt_neg)
    
    @property
    def emphasis_neg(self) -> float | None:
        v = self.source_args.get(B.File_Arg.emphasis_neg)
        return float(v) if v is not None else None
    
    @property
    def edit(self) -> float | None:
        v = self.source_args.get(B.File_Arg.edit)
        return float(v) if v is not None else None
    
    @property
    def edit_prompt_a(self) -> str | None:
        return self.source_args.get(B.File_Arg.edit_prompt_a)
    
    @property
    def edit_prompt_b(self) -> str | None:
        return self.source_args.get(B.File_Arg.edit_prompt_b)
    
    @property
    def prefix(self) -> str | None:
        return self.source_args.get(B.File_Arg.prefix)
    
    @property
    def postfix(self) -> str | None:
        return self.source_args.get(B.File_Arg.postfix)
    
    @property
    def name_link(self) -> str | None:
        return self.source_args.get(B.File_Arg.link)
    
    @property
    def is_negative(self) -> bool | None:
        v = self.source_args.get(B.File_Arg.is_negative)
        return bool(int(v)) if v is not None else None
    
    @property
    def is_additive(self) -> bool | None:
        v = self.source_args.get(B.File_Arg.is_additive)
        return bool(int(v)) if v is not None else None
    
    @property
    def is_reset_visible(self) -> bool | None:
        v = self.source_args.get(B.File_Arg.is_reset_visible)
        return bool(int(v)) if v is not None else None
    
    @property
    def ignore(self) -> bool | None:
        v = self.source_args.get(B.File_Arg.ignore)
        return bool(int(v)) if v is not None else None

class B_Value(Generic[T]):
    __slots__ = ('_default', '_current')

    def __init__(self, default: T):
        self._default = default
        self._current = default
    
    @property
    def current(self) -> T:
        return self._current
    @current.setter
    def current(self, new: T):
        self._current = new
    
    @property
    def default(self) -> T:
        return self._default
    @default.setter
    def default(self, new: T):
        self._default = new
    
    def update(self, new_value: T | None, default_if_none: bool):
        if new_value is not None:
            self.current = new_value
            return True
        elif default_if_none:
            self.reset()
        return False
    
    def reset(self):
        self.current = self.default

class B_PresetMapping:
    __slots__ = ('target', 'args', 'child_mappings')

    def __init__(self, target: str, args: B_Args):
        self.target = target
        self.args = args
        self.child_mappings = OrderedDict[str, B_PresetMapping]()

class B_Preset:
    __slots__ = ('name', 'is_additive', 'map')

    def __init__(self, name: str, is_additive: bool | None = None, _map: OrderedDict[str, B_PresetMapping] | None = None):
        self.name = name
        self.is_additive = is_additive if is_additive is not None else B.Default.is_additive
        self.map = _map if _map is not None else OrderedDict[str, B_PresetMapping]()
    
    def get_targets(self) -> list["B_Ui"]:
        target_b_ui_list: list[B_Ui] = []
        if self.is_additive:
            for target in self.map.keys():
                target_b_ui = B_UiMap.get(target)
                if target_b_ui is None:
                    B_Log.warning(self, "get_b_ui_list()", f"Invalid target '{target}' for preset '{self.name}'")
                    continue
                target_b_ui_list.append(target_b_ui)
        else:
            target_b_ui_list = B_UiMap.get_all()
        return target_b_ui_list
    
    def apply(self) -> None:
        for target, mapping in self.map.items():
            target_b_ui = B_UiMap.get(target)
            if target_b_ui is None:
                B_Log.warning(self, "apply()", f"Invalid target '{target}' for preset '{self.name}'")
                continue
            target_b_ui.apply_preset_mapping(mapping, self.is_additive)

# <PROMPT
class B_Prompt(ABC):
    class UiMeta(Generic[T]):
        __slots__ = ('value', 'visible', 'enabled')
        
        def __init__(self, value: T, visible: bool, enabled: bool):
            self.value = B_Value(value)
            self.visible = visible
            self.enabled = enabled
    
    @classmethod
    def from_args(cls, b_args: B_Args) -> "B_Prompt":
        return cls(
            name=b_args.name if b_args.name is not None else "[PROMPT]"
            , is_activated=b_args.is_activated
        )

    __slots__ = (
        'name', 'is_name_visible', 'is_remove_visible', 'is_activated_value'
        , 'prompt', 'emphasis'
        , 'prompt_neg', 'emphasis_neg'
        , 'edit_prompt_a', 'edit_prompt_b', 'edit'
        , 'prefix', 'postfix'
        , 'is_negative'
    )

    def __init__(
        self
        , name: str
        , register: bool = True
        , is_activated: bool | None = None
        , is_name_visible: bool | None = None
        , is_remove_visible: bool = True

        # PROMPT
        , prompt_value: str | None = None
        , prompt_visible: bool = False
        , prompt_enabled: bool = False

        , emphasis_value: float | None = None
        , emphasis_visible: bool = False
        , emphasis_enabled: bool = False

        # PROMPT NEGATIVE
        , prompt_neg_value: str | None = None
        , prompt_neg_visible: bool = False
        , prompt_neg_enabled: bool = False

        , emphasis_neg_value: float | None = None
        , emphasis_neg_visible: bool = False
        , emphasis_neg_enabled: bool = False

        # EDIT
        , edit_prompt_a_value: str | None = None
        , edit_prompt_a_visible: bool = False
        , edit_prompt_a_enabled: bool = False

        , edit_prompt_b_value: str | None = None
        , edit_prompt_b_visible: bool = False
        , edit_prompt_b_enabled: bool = False

        , edit_value: float | None = None
        , edit_visible: bool = False
        , edit_enabled: bool = False

        # <SHARED>
        , is_negative_value: bool | None = None
        , is_negative_visible: bool = False
        , is_negative_enabled: bool = False

        , prefix_value: str | None = None
        , prefix_visible: bool = False
        , prefix_enabled: bool = False

        , postfix_value: str | None = None
        , postfix_visible: bool = False
        , postfix_enabled: bool = False
    ):
        self.name = name
        self.is_name_visible = is_name_visible if is_name_visible is not None else B.Default.is_name_visible
        self.is_remove_visible = is_remove_visible
        self.is_activated_value = B_Value(is_activated if is_activated is not None else B.Default.is_activated)

        self.prompt = B_Prompt.UiMeta(prompt_value if prompt_value is not None else B.Default.prompt, prompt_visible, prompt_enabled)
        self.emphasis = B_Prompt.UiMeta(emphasis_value if emphasis_value is not None else B.Default.emphasis, emphasis_visible, emphasis_enabled)

        self.prompt_neg = B_Prompt.UiMeta(prompt_neg_value if prompt_neg_value is not None else B.Default.prompt, prompt_neg_visible, prompt_neg_enabled)
        self.emphasis_neg = B_Prompt.UiMeta(emphasis_neg_value if emphasis_neg_value is not None else B.Default.emphasis, emphasis_neg_visible, emphasis_neg_enabled)

        self.edit_prompt_a = B_Prompt.UiMeta(edit_prompt_a_value if edit_prompt_a_value is not None else B.Default.prompt, edit_prompt_a_visible, edit_prompt_a_enabled)
        self.edit_prompt_b = B_Prompt.UiMeta(edit_prompt_b_value if edit_prompt_b_value is not None else B.Default.prompt, edit_prompt_b_visible, edit_prompt_b_enabled)
        self.edit = B_Prompt.UiMeta(edit_value if edit_value is not None else B.Default.edit, edit_visible, edit_enabled)

        self.prefix = B_Prompt.UiMeta(prefix_value if prefix_value is not None else B.Default.prompt, prefix_visible, prefix_enabled)
        self.postfix = B_Prompt.UiMeta(postfix_value if postfix_value is not None else B.Default.prompt, postfix_visible, postfix_enabled)
        
        self.is_negative = B_Prompt.UiMeta(is_negative_value if is_negative_value is not None else B.Default.is_negative, is_negative_visible, is_negative_enabled)

        if register:
            B_PromptMap.add(self)
    
    def reset_values(self) -> None:
        self.is_activated_value.reset()
        
        self.prompt.value.reset()
        self.emphasis.value.reset()
        
        self.prompt_neg.value.reset()
        self.emphasis_neg.value.reset()
        
        self.edit_prompt_a.value.reset()
        self.edit_prompt_b.value.reset()
        self.edit.value.reset()
        
        self.prefix.value.reset()
        self.postfix.value.reset()
        
        self.is_negative.value.reset()
    
    def update(self, b_args: B_Args, default_if_none: bool):
        if not b_args.NO_ARGS:
            self.prompt.value.update(b_args.prompt, default_if_none) or self.prompt.value.update(b_args.prompt_pos, default_if_none)
            self.emphasis.value.update(b_args.emphasis, default_if_none) or self.emphasis.value.update(b_args.emphasis_pos, default_if_none)
            self.prompt_neg.value.update(b_args.prompt_neg, default_if_none)
            self.emphasis_neg.value.update(b_args.emphasis_neg, default_if_none)
            self.edit.value.update(b_args.edit, default_if_none)
            self.is_negative.value.update(b_args.is_negative, default_if_none)
    
    def build_prompt(self) -> PromptPair:
        return PromptPair("", "")

class B_PromptSingle(B_Prompt):
    @classmethod
    def from_args(cls, b_args: B_Args) -> "B_PromptSingle":
        return cls(
            name=b_args.name if b_args.name is not None else "[PROMPT_SINGLE]"
            , is_activated=b_args.is_activated
            , prompt=b_args.prompt
            , emphasis=b_args.emphasis
            , is_negative=b_args.is_negative
            , prefix=b_args.prefix
            , postfix=b_args.postfix
        )
    
    __slots__ = ()
    
    def __init__(
        self
        , name: str
        , is_activated: bool | None = None
        , prompt: str | None = None
        , emphasis: float | None = None
        , is_negative: bool | None = None
        , prefix: str | None = None
        , postfix: str | None = None
    ):
        super().__init__(
            name
            , is_activated=is_activated
            , prompt_value=prompt
            , prompt_enabled=True
            , prompt_visible=True
            , emphasis_value=emphasis
            , emphasis_enabled=True
            , emphasis_visible=True
            , is_negative_value=is_negative
            , is_negative_enabled=True
            , is_negative_visible=True
            , prefix_value=prefix
            , postfix_value=postfix
        )
    
    def build_prompt(self) -> PromptPair:
        prompt = B_Fn.Prompt.emphasized(
            B_Fn.Prompt.decorated(
                B_Fn.Prompt.sanitized(self.prompt.value.current)
                , B_Fn.Prompt.sanitized(self.prefix.value.current)
                , B_Fn.Prompt.sanitized(self.postfix.value.current)
            )
            , self.emphasis.value.current
        )
        if not self.is_negative.value.current:
            return PromptPair(prompt, "")
        else:
            return PromptPair("", prompt)

class B_PromptListItem(B_PromptSingle):
    @classmethod
    def from_args(cls, b_args: B_Args) -> "B_PromptListItem":
        return cls(
            prompt=b_args.prompt if b_args.prompt is not None else B.Default.prompt
            , postfix=b_args.postfix if b_args.postfix is not None else B.Default.prompt
        )
    
    def __init__(
        self
        , prompt: str
        , postfix: str
        , is_activated: bool = False
    ):
        super().__init__(
            f"{postfix.capitalize()} - {prompt.capitalize()}" if B.Flag.use_alt_list_item_name else f"{prompt.capitalize()} {postfix.lower()}"
            , is_activated=is_activated
            , prompt=prompt
            , postfix=postfix)

class B_PromptDual(B_Prompt):
    @classmethod
    def from_args(cls, b_args: B_Args) -> "B_PromptDual":
        return cls(
            name=b_args.name if b_args.name is not None else "[PROMPT_DUAL]"
            , is_activated=b_args.is_activated
            , prompt_pos=b_args.prompt_pos
            , emphasis_pos=b_args.emphasis_pos
            , prompt_neg=b_args.prompt_neg
            , emphasis_neg=b_args.emphasis_neg
        )
    
    __slots__ = ()
    
    def __init__(
        self
        , name: str
        , is_activated: bool | None = None
        , prompt_pos: str | None = None
        , emphasis_pos: float | None = None
        , prompt_neg: str | None = None
        , emphasis_neg: float | None = None
        , prefix: str | None = None
        , postfix: str | None = None
    ):
        super().__init__(
            name
            , is_activated=is_activated
            , prompt_value=prompt_pos
            , prompt_enabled=True
            , prompt_visible=True
            , emphasis_value=emphasis_pos
            , emphasis_enabled=True
            , emphasis_visible=True
            , prompt_neg_value=prompt_neg
            , prompt_neg_enabled=True
            , prompt_neg_visible=True
            , emphasis_neg_value=emphasis_neg
            , emphasis_neg_enabled=True
            , emphasis_neg_visible=True
            , postfix_value=postfix
            , prefix_value=prefix
        )
    
    def build_prompt(self) -> PromptPair:
        prompt = B_Fn.Prompt.emphasized(
            B_Fn.Prompt.decorated(
                B_Fn.Prompt.sanitized(self.prompt.value.current)
                , B_Fn.Prompt.sanitized(self.prefix.value.current)
                , B_Fn.Prompt.sanitized(self.postfix.value.current)
            )
            , self.emphasis.value.current
        )
        prompt_negative = B_Fn.Prompt.emphasized(
            B_Fn.Prompt.decorated(
                B_Fn.Prompt.sanitized(self.prompt_neg.value.current)
                , B_Fn.Prompt.sanitized(self.prefix.value.current)
                , B_Fn.Prompt.sanitized(self.postfix.value.current)
            )
            , self.emphasis_neg.value.current
        )
        return PromptPair(prompt, prompt_negative)

class B_PromptEdit(B_Prompt):
    @classmethod
    def from_args(cls, b_args: B_Args) -> "B_PromptEdit":
        return cls(
            name=b_args.name if b_args.name is not None else "[PROMPT_EDIT]"
            , is_activated=b_args.is_activated
            , prompt_a=b_args.edit_prompt_a if b_args.edit_prompt_a is not None else B.Default.prompt
            , prompt_b=b_args.edit_prompt_b if b_args.edit_prompt_b is not None else B.Default.prompt
            , edit=b_args.edit
            , is_negative=b_args.is_negative
            , prefix=b_args.prefix
            , postfix=b_args.postfix
        )
    
    __slots__ = ()
    
    @staticmethod
    def _build_prompt(
        prompt_a: str
        , prompt_b: str
        , edit: float
        , is_negative: bool
        , prefix: str
        , postfix: str
    ) -> PromptPair:
        prompt_a = B_Fn.Prompt.decorated(
            B_Fn.Prompt.sanitized(prompt_a)
            , B_Fn.Prompt.sanitized(prefix)
            , B_Fn.Prompt.sanitized(postfix)
        )
        prompt_b = B_Fn.Prompt.decorated(
            B_Fn.Prompt.sanitized(prompt_b)
            , B_Fn.Prompt.sanitized(prefix)
            , B_Fn.Prompt.sanitized(postfix)
        )

        prompt: str = ""
        if isclose(edit, 0):
            prompt = prompt_a
        elif isclose(edit, 1):
            prompt = prompt_b
        else:
            prompt = f"[{prompt_a}:{prompt_b}:{round(1 - edit, B.Webui.decimals)}]"
        
        if not is_negative:
            return PromptPair(prompt, "")
        else:
            return PromptPair("", prompt)
    
    def __init__(
        self
        , name: str
        , prompt_a: str
        , prompt_b: str
        , is_activated: bool | None = None
        , edit: float | None = None
        , is_negative: bool | None = None
        , prefix: str | None = None
        , postfix: str | None = None
    ):
        super().__init__(
            name
            , is_activated=is_activated
            , edit_prompt_a_value=prompt_a
            , edit_prompt_b_value=prompt_b
            , edit_value=edit
            , edit_enabled=True
            , edit_visible=True
            , is_negative_value=is_negative
            , is_negative_enabled=True
            , is_negative_visible=True
            , prefix_value=prefix
            , postfix_value=postfix
        )
    
    def build_prompt(self) -> PromptPair:
        return self._build_prompt(
            self.edit_prompt_a.value.current
            , self.edit_prompt_b.value.current
            , self.edit.value.current
            , self.is_negative.value.current
            , self.prefix.value.current
            , self.postfix.value.current
        )

class B_PromptEditLink(B_Prompt):
    @classmethod
    def from_args(cls, b_args: B_Args) -> "B_PromptEditLink":
        return cls(
            name=b_args.name if b_args.name is not None else "[PROMPT_EDIT_LINK]"
            , name_link=b_args.name_link if b_args.name_link is not None else B.Default.name_link
            , prompt_a=b_args.edit_prompt_a if b_args.edit_prompt_a is not None else B.Default.prompt
            , prompt_b=b_args.edit_prompt_b if b_args.edit_prompt_b is not None else B.Default.prompt
            , is_activated=b_args.is_activated
            , is_negative=b_args.is_negative
            , prefix=b_args.prefix
            , postfix=b_args.postfix
        )
    
    __slots__ = ('name_link')

    def __init__(
        self
        , name: str
        , name_link: str
        , prompt_a: str
        , prompt_b: str
        , is_activated: bool | None = None
        , is_negative: bool | None = None
        , prefix: str | None = None
        , postfix: str | None = None
    ):
        super().__init__(
            name
            , is_activated=is_activated
            , edit_prompt_a_value=prompt_a
            , edit_prompt_b_value=prompt_b
            , is_negative_value=is_negative
            , is_negative_enabled=True
            , is_negative_visible=True
            , prefix_value=prefix
            , postfix_value=postfix
        )

        self.name_link = name_link

        if (name_link == B.Default.name_link):
            B_Log.warning(self, "__init__()", "Missing or empty link name")
    
    def build_prompt(self) -> PromptPair:
        b_prompt_link = B_PromptMap.get(self.name_link)
        
        if (b_prompt_link is None):
            B_Log.warning(self, f"{self.name} - build_prompt()", f"Linked prompt not found -> '{self.name_link}'")
            return PromptPair("", "")
        
        return B_PromptEdit._build_prompt(
            self.edit_prompt_a.value.current
            , self.edit_prompt_b.value.current
            , b_prompt_link.edit.value.current
            , self.is_negative.value.current
            , self.prefix.value.current
            , self.postfix.value.current
        )
# PROMPT>

# <PROMPT MAP
class B_PromptMap:
    _map = OrderedDict[str, B_Prompt]()

    sentinel = B_Prompt("[PROMPT_SENTINEL]", register=False) #! `register=False` is the only thing stopping this from breaking

    @classmethod
    def add(cls, b_prompt: B_Prompt) -> B_Prompt:
        if b_prompt.name == cls.sentinel.name:
            B_Log.warning(cls, "add()", f"Returning SENTINEL")
            return cls.sentinel
        if b_prompt.name in cls._map:
            B_Log.warning(cls, "add()", f"Duplicate key -> '{b_prompt.name}'")
        cls._map[b_prompt.name] = b_prompt
        return b_prompt
    
    @classmethod
    def get(cls, key: str) -> B_Prompt | None:
        if key == cls.sentinel.name:
            B_Log.warning(cls, "get()", f"Returning SENTINEL")
            return cls.sentinel
        b_prompt = cls._map.get(key)
        if b_prompt is None:
            B_Log.warning(cls, "get()", f"Key not found -> '{key}'")
        return b_prompt
    
    @classmethod
    def build_prompts(cls) -> PromptPair:
        final_prompt: str = ""
        final_prompt_neg: str = ""

        for b_prompt in cls._map.values():
            if not b_prompt.is_activated_value.current:
                continue
            prompt, prompt_neg = b_prompt.build_prompt()
            final_prompt = B_Fn.Prompt.added(final_prompt, prompt)
            final_prompt_neg = B_Fn.Prompt.added(final_prompt_neg, prompt_neg)

        return PromptPair(final_prompt, final_prompt_neg)
# PROMPT MAP>

# <UI
class B_Ui(ABC):
    @classmethod
    @abstractmethod
    def from_args(cls, b_args: B_Args) -> "B_Ui":
        pass

    __slots__ = ("name")

    def __init__(self, name: str, register: bool):
        self.name = name

        if register:
            B_UiMap.add(self)
    
    @abstractmethod
    def build(self) -> Sequence[grComponent]:
        pass

    def bind(self, gr_target_update: Callable[[], list[Any]], gr_target_output: list[grBlock]) -> None:
        pass

    def gr_input(self) -> list[grComponent]:
        return []
    
    def on_input(self, *args: Any) -> None:
        pass

    def gr_output(self) -> list[grBlock]:
        return []

    def gr_output_update(self) -> list[Any]:
        return []
    
    def reset(self) -> None:
        return
    
    def apply_preset_mapping(self, mapping: B_PresetMapping, is_additive: bool) -> None:
        pass

class B_UiSeparator(B_Ui):
    @classmethod
    def from_args(cls, _: B_Args) -> "B_UiSeparator":
        return cls()

    __slots__ = ("gr")

    def __init__(self):
        super().__init__("[UI_SEPARATOR]", register=False)

        self.gr: grMarkdown
    
    def build(self) -> Sequence[grComponent]:
        self.gr = grMarkdown(B.HTML.html_separator)
        return []

class B_UiPrompt(B_Ui, Generic[T_B_Prompt], ABC):
    identity_seed: int = 1

    __slots__ = (
        "identity"
        , "b_prompt"
        , "gr_container", "gr_name"
        , "gr_apply", "gr_remove"
    )

    def __init__(self, b_prompt: T_B_Prompt, register: bool = True):
        super().__init__(b_prompt.name, register)

        self.identity = B_UiPrompt.identity_seed
        B_UiPrompt.identity_seed += 1

        self.b_prompt = b_prompt

        self.gr_container: grColumn
        self.gr_name: grMarkdown

        self.gr_apply: grButton
        self.gr_remove: grButton
    
    @property
    def html_id(self) -> str:
        return f"{B.HTML.id_prefix}{self.identity}"
    
    @abstractmethod
    def build_prompt_ui(self) -> list[grComponent]:
        pass

    @abstractmethod
    def bind_prompt_ui(self, final_apply_args: dict[str, Any]) -> None:
        pass
    
    def build(self) -> Sequence[grComponent]:
        self.gr_container = grColumn(
            variant = "panel"
            , visible=self.gr_container_visible()
            , elem_id=self.html_id)
        with self.gr_container:
            self.gr_name = grMarkdown(
                value=f"<b>{self.b_prompt.name}</b>"
                , visible=self.b_prompt.is_name_visible
            )

            prompt_ui = self.build_prompt_ui()

            B_UiSeparator().build()
            with grRow(elem_classes=B.HTML.css_footer[0]):
                self.gr_apply = grButton(
                    value=B.Ui.prompt_apply_label
                )
                self.gr_remove = grButton(
                    value=B.Ui.prompt_remove_label
                    , interactive=self.gr_remove_interactive()
                    , visible=self.b_prompt.is_remove_visible
                )
        
        return prompt_ui + [
            self.gr_apply
            , self.gr_remove
        ]

    def bind(self, gr_target_update: Callable[[], list[Any]], gr_target_output: list[grBlock]) -> None:
        def on_apply(*args: Any):
            if (self.b_prompt != B_PromptMap.sentinel):
                self.b_prompt.is_activated_value.current = True
                self.on_input(*args)
            else:
                B_Log.warning(self, "bind().on_apply()", "Apply event caught for sentinel prompt")
            return [
                gr_update(self.gr_remove, interactive=self.gr_remove_interactive())
            ] + gr_target_update()

        # <APPLY
        apply_args = {
            "fn": on_apply
            , "inputs": self.gr_input()
            , "outputs": [
                self.gr_remove
            ] + gr_target_output
        }
        self.gr_apply.click(**apply_args)
        # APPLY>

        # <REMOVE
        def on_remove():
            self.b_prompt.is_activated_value.current = False
            
            return [
                gr_update(self.gr_remove, interactive=self.gr_remove_interactive())
            ] + gr_target_update()

        self.gr_remove.click(
            fn=on_remove
            , outputs=[
                self.gr_remove
            ] + gr_target_output
        )
        # REMOVE>
        
        self.bind_prompt_ui(apply_args)
    
    def gr_output(self) -> list[grBlock]:
        return [
            self.gr_container
            , self.gr_name
            , self.gr_remove
        ]
    
    def gr_output_update(self) -> list[Any]:
        return [
            gr_update(self.gr_container
                , visible=self.gr_container_visible()
                )
            , gr_update(self.gr_name
                , value=f"<b>{self.b_prompt.name}</b>"
                , visible=self.b_prompt.is_name_visible
                )
            , gr_update(self.gr_remove
                , interactive=self.gr_remove_interactive()
                , visible=self.b_prompt.is_remove_visible
                )
        ]
    
    def reset(self) -> None:
        self.b_prompt.reset_values()
    
    def apply_preset_mapping(self, mapping: B_PresetMapping, is_additive: bool) -> None:
        self.b_prompt.is_activated_value.current = not mapping.args.NO_ARGS
    
    def gr_container_visible(self) -> bool:
        return True
    
    def gr_remove_interactive(self) -> bool:
        return self.b_prompt.is_activated_value.current

class B_UiPromptTemplate(B_UiPrompt[B_Prompt]):
    @classmethod
    def from_args(cls, _: B_Args, b_prompt: B_Prompt | None = None) -> "B_UiPromptTemplate":
        return cls(b_prompt)

    __slots__ = (
        "gr_prompt_container", "gr_prompt", "gr_emphasis"
        , "gr_prompt_neg_container", "gr_prompt_neg", "gr_emphasis_neg"
        , "gr_edit"
        , "gr_is_negative"
        , "gr_prefix", "gr_postfix"
    )

    def __init__(self, b_prompt: B_Prompt | None = None, register: bool = False):
        super().__init__(b_prompt if b_prompt is not None else B_PromptMap.sentinel, register)

        self.gr_prompt_container: grRow
        self.gr_prompt: grTextbox
        self.gr_emphasis: grNumber

        self.gr_prompt_neg_container: grRow
        self.gr_prompt_neg: grTextbox
        self.gr_emphasis_neg: grNumber

        self.gr_edit: grSlider

        self.gr_is_negative: grCheckbox

        self.gr_prefix: grTextbox
        self.gr_postfix: grTextbox
    
    def build_prompt_ui(self) -> list[grComponent]:
        self.gr_prompt_container = grRow(visible=self.b_prompt.prompt.visible or self.b_prompt.emphasis.visible)
        with self.gr_prompt_container:
            self.gr_prompt = grTextbox(
                label=B.Ui.prompt_label
                , value=self.b_prompt.prompt.value.current
                , scale=B.Ui.prompt_scale
                , visible=self.b_prompt.prompt.visible
                , interactive=self.b_prompt.prompt.enabled
            )
            self.gr_emphasis = grNumber(
                label=B.Ui.emphasis_label
                , value=self.b_prompt.emphasis.value.current
                , minimum=B.Ui.emphasis_min
                , step=B.Ui.emphasis_step
                , scale=B.Ui.emphasis_scale
                , visible=self.b_prompt.emphasis.visible
                , interactive=self.b_prompt.emphasis.enabled
            )
        
        self.gr_prompt_neg_container = grRow(visible=self.b_prompt.prompt_neg.visible or self.b_prompt.emphasis_neg.visible)
        with self.gr_prompt_neg_container:
            self.gr_prompt_neg = grTextbox(
                label=B.Ui.prompt_neg_label
                , value=self.b_prompt.prompt_neg.value.current
                , scale=B.Ui.prompt_scale
                , visible=self.b_prompt.prompt_neg.visible
                , interactive=self.b_prompt.prompt_neg.enabled
            )
            self.gr_emphasis_neg = grNumber(
                label=B.Ui.emphasis_neg_label
                , value=self.b_prompt.emphasis_neg.value.current
                , minimum=B.Ui.emphasis_min
                , step=B.Ui.emphasis_step
                , scale=B.Ui.emphasis_scale
                , visible=self.b_prompt.emphasis_neg.visible
                , interactive=self.b_prompt.emphasis_neg.enabled
            )
        
        self.gr_edit = grSlider(
            label=B.Ui.edit_label
            , value=self.b_prompt.edit.value.current
            , minimum=B.Ui.edit_min
            , maximum=B.Ui.edit_max
            , step=B.Ui.edit_step
            , visible=self.b_prompt.edit.visible
            , interactive=self.b_prompt.edit.enabled
        )

        self.gr_is_negative = grCheckbox(
            label=B.Ui.is_negative_label
            , value = self.b_prompt.is_negative.value.current
            , visible=self.b_prompt.is_negative.visible
            , interactive=self.b_prompt.is_negative.enabled
        )

        self.gr_prefix = grTextbox(
            label=B.Ui.prefix_label
            , value=self.b_prompt.prefix.value.current
            , scale=B.Ui.affix_scale
            , visible=self.b_prompt.prefix.visible
            , interactive=self.b_prompt.prefix.enabled
        )

        self.gr_postfix = grTextbox(
            label=B.Ui.postfix_label
            , value=self.b_prompt.postfix.value.current
            , scale=B.Ui.affix_scale
            , visible=self.b_prompt.postfix.visible
            , interactive=self.b_prompt.postfix.enabled
        )

        return [
            self.gr_prompt
            , self.gr_emphasis
            , self.gr_prompt_neg
            , self.gr_emphasis_neg
            , self.gr_edit
            , self.gr_is_negative
            , self.gr_prefix
            , self.gr_postfix
        ]
    
    def gr_input(self) -> list[grComponent]:
        return [
            self.gr_prompt
            , self.gr_emphasis
            , self.gr_prompt_neg
            , self.gr_emphasis_neg
            , self.gr_edit
            , self.gr_is_negative
        ]
    
    def on_input(
        self
        , prompt: str
        , emphasis: float
        , prompt_neg: str
        , emphasis_neg: float
        , edit: float
        , is_negative: bool
    ) -> None:
        self.b_prompt.prompt.value.current = prompt
        self.b_prompt.emphasis.value.current = emphasis
        self.b_prompt.prompt_neg.value.current = prompt_neg
        self.b_prompt.emphasis_neg.value.current = emphasis_neg
        self.b_prompt.edit.value.current = edit
        self.b_prompt.is_negative.value.current = is_negative
    
    def bind(self, gr_target_update: Callable[[], list[Any]], gr_target_output: list[grBlock]) -> None:
        if self == B_UiMap.prompt_template:
            def gr_target_update_extended():
                self.update_prompt(None)
                return [
                    gr_update(self.gr_container, visible=False)
                ] + gr_target_update()
            return super().bind(gr_target_update_extended, [self.gr_container] + gr_target_output)
        return super().bind(gr_target_update, gr_target_output)
    
    def bind_prompt_ui(self, final_apply_args: dict[str, Any]) -> None:
        self.gr_prompt.submit(**final_apply_args)
        self.gr_emphasis.submit(**final_apply_args)
        self.gr_prompt_neg.submit(**final_apply_args)
        self.gr_emphasis_neg.submit(**final_apply_args)
    
    def gr_output(self) -> list[grBlock]:
        return super().gr_output() + [
            self.gr_prompt_container
            , self.gr_prompt
            , self.gr_emphasis

            , self.gr_prompt_neg_container
            , self.gr_prompt_neg
            , self.gr_emphasis_neg

            , self.gr_edit

            , self.gr_is_negative
        ]
    
    def gr_output_update(self) -> list[Any]:
        return super().gr_output_update() + [
            gr_update(self.gr_prompt_container
                , visible=self.b_prompt.prompt.visible or self.b_prompt.emphasis.visible
                )
            , gr_update(self.gr_prompt
                , value=self.b_prompt.prompt.value.current
                , visible=self.b_prompt.prompt.visible
                , interactive=self.b_prompt.prompt.enabled
                )
            , gr_update(self.gr_emphasis
                , value=self.b_prompt.emphasis.value.current
                , visible=self.b_prompt.emphasis.visible
                , interactive=self.b_prompt.emphasis.enabled
                , step=B.Ui.emphasis_step
                )
            
            , gr_update(self.gr_prompt_neg_container
                , visible=self.b_prompt.prompt_neg.visible or self.b_prompt.emphasis_neg.visible
                )
            , gr_update(self.gr_prompt_neg
                , value=self.b_prompt.prompt_neg.value.current
                , visible=self.b_prompt.prompt_neg.visible
                , interactive=self.b_prompt.prompt_neg.enabled
                )
            , gr_update(self.gr_emphasis_neg
                , value=self.b_prompt.emphasis_neg.value.current
                , visible=self.b_prompt.emphasis_neg.visible
                , interactive=self.b_prompt.emphasis_neg.enabled
                , step=B.Ui.emphasis_step
                )
                
            , gr_update(self.gr_edit
                , value=self.b_prompt.edit.value.current
                , visible=self.b_prompt.edit.visible
                , interactive=self.b_prompt.edit.enabled
                , step=B.Ui.edit_step
                )
            
            , gr_update(self.gr_is_negative
                , value=self.b_prompt.is_negative.value.current
                , visible=self.b_prompt.is_negative.visible
                , interactive=self.b_prompt.is_negative.enabled
                )
        ]
    
    def apply_preset_mapping(self, mapping: B_PresetMapping, is_additive: bool) -> None:
        super().apply_preset_mapping(mapping, is_additive)
        self.b_prompt.update(mapping.args, not is_additive)
    
    def update_prompt(self, b_prompt: B_Prompt | None) -> None:
        self.b_prompt = b_prompt if b_prompt is not None else B_PromptMap.sentinel
    
    def gr_container_visible(self) -> bool:
        return self.b_prompt != B_PromptMap.sentinel

class B_UiSelectChoice(B_Ui):
    @classmethod
    def from_args(cls, _: B_Args, b_prompt: B_Prompt, b_preset: B_Preset | None = None) -> "B_UiSelectChoice":
        return cls(b_prompt, b_preset)

    __slots__ = ("b_prompt", "b_preset", "gr")
    
    def __init__(
        self
        , b_prompt: B_Prompt
        , b_preset: B_Preset | None = None
    ):
        super().__init__(b_prompt.name, register=False)
        
        self.b_prompt = b_prompt
        self.b_preset = b_preset

        self.gr: grButton

        #! overrides
        self.b_prompt.is_remove_visible = False
    
    def build(self) -> Sequence[grComponent]:
        self.gr = grButton(
            value=self.b_prompt.name
            , variant="primary"
            , size="sm"
            , visible=self.b_prompt.is_activated_value.current
            , elem_classes=B.HTML.cls_ui_select_choice
        )
        return [self.gr]
    
    def bind(self, gr_target_update: Callable[[], list[Any]], gr_target_output: list[grBlock]) -> None:
        def on_click():
            if B_UiMap.prompt_template.b_prompt != self.b_prompt:
                B_UiMap.prompt_template.update_prompt(self.b_prompt)
            else:
                B_UiMap.prompt_template.update_prompt(None)
            return B_UiMap.prompt_template.gr_output_update()
        self.gr.click(
            fn=on_click
            , outputs=B_UiMap.prompt_template.gr_output()
        )
    
    def gr_output(self) -> list[grBlock]:
        return [self.gr]
    
    def gr_output_update(self) -> list[Any]:
        return [gr_update(self.gr, visible=self.b_prompt.is_activated_value.current)]
    
    def reset(self) -> None:
        self.b_prompt.reset_values()

class B_UiSelect(B_Ui):
    @classmethod
    def from_args(cls, b_args: B_Args, choices: list[B_UiSelectChoice]) -> "B_UiSelect":
        return cls(
            name=b_args.name if b_args.name is not None else "[UI_SELECT]"
            , choices=choices
            , sort=b_args.sort
            , scale=b_args.scale
            , prefix=b_args.prefix
            , postfix=b_args.postfix
        )

    __slots__ = (
        "choice_map"
        , "scale"
        , "gr_dropdown", "gr_selections_container"
    )
    
    def __init__(
        self
        , name: str
        , choices: list[B_UiSelectChoice]
        , sort: bool | None = None
        , scale: int | None = None
        , prefix: str | None = None
        , postfix: str | None = None
    ):
        super().__init__(name, register=True)

        self.scale = scale if scale is not None else B.Default.scale

        self.gr_dropdown: grDropdown
        self.gr_selections_container: grRow
        
        self.choice_map = OrderedDict(
            (b_ui_choice.b_prompt.name, b_ui_choice)
            for b_ui_choice in (
                sorted(choices, key=lambda choice: choice.b_prompt.name) if (sort if sort is not None else B.Default.sort)
                else choices
            )
        )

        #! override prefix & postfix:
        prefix = prefix if prefix is not None else B.Default.prompt
        postfix = postfix if postfix is not None else B.Default.prompt
        if (len(prefix) + len(postfix) > 0):
            for b_ui_choice in self.choice_map.values():
                if (len(prefix) > 0):
                    b_ui_choice.b_prompt.prefix.value.default = prefix
                    b_ui_choice.b_prompt.prefix.value.reset()
                if (len(postfix) > 0):
                    b_ui_choice.b_prompt.postfix.value.default = postfix
                    b_ui_choice.b_prompt.postfix.value.reset()
    
    def build(self) -> Sequence[grComponent]:
        with grColumn(scale=self.scale):
            self.gr_dropdown = grDropdown(
                label=self.name
                , choices=[b_ui_choice.b_prompt.name for b_ui_choice in self.choice_map.values()]
                , multiselect=True
                , allow_custom_value=False
                , value=[b_ui_choice.b_prompt.name for b_ui_choice in self.choice_map.values() if b_ui_choice.b_prompt.is_activated_value.current]
            )

            self.gr_selections_container = grRow(variant="panel", visible=self.gr_selections_container_visible())
            with self.gr_selections_container:
                gr_selections = [
                    gr_selection
                    for b_ui_choice in self.choice_map.values()
                    for gr_selection in b_ui_choice.build()
                ]

        return [self.gr_dropdown] + gr_selections
    
    def bind(self, gr_target_update: Callable[[], list[Any]], gr_target_output: list[grBlock]) -> None:
        presets_b_ui: list[B_Ui] = []
        for b_ui_choice in self.choice_map.values():
            if b_ui_choice.b_preset is None:
                continue
            presets_b_ui += [b_ui for b_ui in b_ui_choice.b_preset.get_targets() if b_ui not in presets_b_ui]

        def on_select(selected_choices: str | list[str] | None):
            selected_choices = selected_choices if isinstance(selected_choices, list) else [selected_choices] if isinstance(selected_choices, str) else []

            for b_ui_choice in self.choice_map.values():
                b_ui_choice.b_prompt.is_activated_value.current = b_ui_choice.b_prompt.name in selected_choices
                if b_ui_choice.b_prompt.is_activated_value.current:
                    if b_ui_choice.b_preset is not None:
                        b_ui_choice.b_preset.apply()
            
            B_UiMap.prompt_template.update_prompt(None)
        
            return (
                self.gr_selections_update()
                + B_UiMap.prompt_template.gr_output_update()
                + [
                    x
                    for b_ui in presets_b_ui
                    for x in b_ui.gr_output_update()
                ]
                + gr_target_update()
            )
        self.gr_dropdown.input(
            fn=on_select
            , inputs=self.gr_dropdown
            , outputs=(
                self.gr_selections_output()
                + B_UiMap.prompt_template.gr_output()
                + [
                    gr
                    for b_ui in presets_b_ui
                    for gr in b_ui.gr_output()
                ]
                + gr_target_output
            )
        )

        for b_ui_choice in self.choice_map.values():
            b_ui_choice.bind(gr_target_update, gr_target_output)
    
    def gr_output(self) -> list[grBlock]:
        return [
            self.gr_dropdown
        ] + self.gr_selections_output()
    
    def gr_output_update(self) -> list[Any]:
        return [
            [
                b_ui_choice.b_prompt.name
                for b_ui_choice in self.choice_map.values()
                if b_ui_choice.b_prompt.is_activated_value.current
            ]
        ] + self.gr_selections_update()
    
    def reset(self) -> None:
        for b_ui_choice in self.choice_map.values():
            b_ui_choice.reset()
    
    def apply_preset_mapping(self, mapping: B_PresetMapping, is_additive: bool) -> None:
        if not mapping.args.NO_ARGS:
            pass # placeholder; currently no params applicable to the select itself

        for choice in self.choice_map.values():
            child_mapping = mapping.child_mappings.get(choice.name)
            choice.b_prompt.is_activated_value.current = child_mapping is not None
            if child_mapping is not None:
                choice.b_prompt.update(child_mapping.args, not is_additive)
    
    def gr_selections_container_visible(self) -> bool:
        return any(b_ui_choice.b_prompt.is_activated_value.current for b_ui_choice in self.choice_map.values())
    
    def gr_selections_output(self) -> list[grBlock]:
        return (
            [self.gr_selections_container]
            + [
                gr_choice_output
                for b_ui_choice in self.choice_map.values()
                for gr_choice_output in b_ui_choice.gr_output()
            ]
        )

    def gr_selections_update(self) -> list[Any]:
        return (
            [gr_update(self.gr_selections_container, visible=self.gr_selections_container_visible())]
            + [
                gr_choice_update
                for b_ui_choice in self.choice_map.values()
                for gr_choice_update in b_ui_choice.gr_output_update()
            ]
        )

class B_UiPreset(B_Ui):
    @classmethod
    def from_args(cls, b_args: B_Args) -> "B_UiPreset":
        b_preset = B_Preset(
            b_args.name if b_args.name is not None else "[UI_PRESET]",
            b_args.is_additive
        )
        return cls(b_preset)
    
    __slots__ = ("b_preset", "gr")
    
    def __init__(
        self
        , b_preset: B_Preset
    ):
        super().__init__(b_preset.name, register=False)

        self.b_preset = b_preset

        self.gr: grButton
    
    def build(self) -> Sequence[grComponent]:
        self.gr = grButton(self.name, variant="primary")
        return [self.gr]
    
    def bind(self, gr_target_update: Callable[[], list[Any]], gr_target_output: list[grBlock]) -> None:
        target_b_ui_list = self.b_preset.get_targets()
        def on_apply():
            self.b_preset.apply()
            B_UiMap.prompt_template.update_prompt(None) #! will hide prompt template regardless of where it is...
            return [
                x
                for b_ui in target_b_ui_list
                for x in b_ui.gr_output_update()
            ] + B_UiMap.prompt_template.gr_output_update() + gr_target_update()
        self.gr.click(
            fn=on_apply,
            outputs=[
                gr
                for b_ui in target_b_ui_list
                for gr in b_ui.gr_output()
            ] + B_UiMap.prompt_template.gr_output() + gr_target_output
        )

class B_UiContainer(B_Ui, ABC):
    @classmethod
    @abstractmethod
    def from_args(cls, b_args: B_Args, children: list[B_Ui]) -> "B_UiContainer":
        pass
    
    __slots__ = ("children", "is_reset_visible", "gr_container", "gr_reset")

    def __init__(self, name: str, children: list[B_Ui] | None = None, is_reset_visible: bool | None = None):
        super().__init__(name, register=False)

        self.children = children if children is not None else []

        self.is_reset_visible = is_reset_visible if is_reset_visible is not None else B.Default.is_reset_visible

        self.gr_container: grBlockContext
        self.gr_reset: grButton
    
    def build(self) -> Sequence[grComponent]:
        gr_components: list[grComponent] = []

        self.gr_container = self.build_container(self.name)
        with self.gr_container:
            for b_ui in self.children:
                gr_components += b_ui.build()
            
            show_buttons: bool = self.is_reset_visible
            with grColumn(visible=show_buttons):
                B_UiSeparator().build()
                with grRow():
                    self.gr_reset = grButton(
                        f"{B.Ui.reset_label_prefix} {self.name}"
                        , visible=self.is_reset_visible
                    )
        
        return [self.gr_reset] + gr_components
    
    def bind(self, gr_target_update: Callable[[], list[Any]], gr_target_output: list[grBlock]) -> None:
        # <CHILDREN
        for b_ui in self.children:
            b_ui.bind(gr_target_update, gr_target_output)
        # CHILDREN>

        # <SELF
        def on_reset():
            B_UiMap.prompt_template.update_prompt(None) #! will hide prompt template regardless of where it is...
            self.reset()
            return self.gr_output_update() + B_UiMap.prompt_template.gr_output_update() + gr_target_update()
        self.gr_reset.click(
            fn=on_reset
            , outputs=self.gr_output() + B_UiMap.prompt_template.gr_output() + gr_target_output
        )
        # SELF>
    
    def gr_output(self) -> list[grBlock]:
        return [
            gr
            for b_ui in self.children
            for gr in b_ui.gr_output()
        ]
    
    def gr_output_update(self) -> list[Any]:
        return [
            update
            for b_ui in self.children
            for update in b_ui.gr_output_update()
        ]
    
    def reset(self) -> None:
        for b_ui in self.children:
            b_ui.reset()
    
    @abstractmethod
    def build_container(self, name: str) -> grBlockContext:
        pass

class B_UiContainerTab(B_UiContainer):
    @classmethod
    def from_args(cls, b_args: B_Args, children: list[B_Ui]) -> "B_UiContainerTab":
        return cls(
            b_args.name if b_args.name is not None else "[UI_TAB]"
            , children=children
            , is_reset_visible=b_args.is_reset_visible
        )
    
    __slots__ = ()

    def __init__(self, name: str, children: list[B_Ui] | None = None, is_reset_visible: bool | None = None):
        super().__init__(name, children, is_reset_visible if is_reset_visible is not None else True)
    
    def build_container(self, name: str) -> grBlockContext:
        return grTab(name)
    
class B_UiContainerRow(B_UiContainer):
    @classmethod
    def from_args(cls, b_args: B_Args, children: list[B_Ui]) -> "B_UiContainerRow":
        return cls(
            children=children
            , is_reset_visible=b_args.is_reset_visible
        )

    __slots__ = ()

    def __init__(self, children: list[B_Ui] | None = None, is_reset_visible: bool | None = None):
        super().__init__("[UI_ROW]", children, is_reset_visible)
    
    def build_container(self, _: str) -> grBlockContext:
        return grRow()
    
class B_UiContainerColumn(B_UiContainer):
    @classmethod
    def from_args(cls, b_args: B_Args, children: list[B_Ui]) -> "B_UiContainerColumn":
        return cls(
            children=children
            , scale=b_args.scale
            , is_reset_visible=b_args.is_reset_visible
        )

    __slots__ = ("scale")

    def __init__(self, children: list[B_Ui] | None = None, scale: int | None = None, is_reset_visible: bool | None = None):
        super().__init__("[UI_COLUMN]", children, is_reset_visible)
        
        self.scale = scale if scale is not None else B.Default.scale
    
    def build_container(self, _: str) -> grBlockContext:
        return grColumn(scale=self.scale)

class B_UiContainerAccordion(B_UiContainer):
    @classmethod
    def from_args(cls, b_args: B_Args, children: list[B_Ui]) -> "B_UiContainerAccordion":
        return cls(
            b_args.name if b_args.name is not None else "[UI_ACCORDION]"
            , children=children
            , is_open=b_args.open
            , is_reset_visible=b_args.is_reset_visible
        )

    __slots__ = ("open")

    def __init__(self, name: str, children: list[B_Ui] | None = None, is_open: bool | None = None, is_reset_visible: bool | None = None):
        super().__init__(name, children, is_reset_visible)

        self.open = is_open if is_open is not None else B.Default.open
    
    def build_container(self, name: str) -> grBlockContext:
        return grAccordion(name, open=self.open)
    
class B_UiContainerGroup(B_UiContainer):
    @classmethod
    def from_args(cls, b_args: B_Args, children: list[B_Ui]) -> "B_UiContainerGroup":
        return cls(
            children=children
            , is_reset_visible=b_args.is_reset_visible
        )
    
    __slots__ = ()

    def __init__(self, children: list[B_Ui] | None = None, is_reset_visible: bool | None = None):
        super().__init__("[UI_GROUP]", children, is_reset_visible)
    
    def build_container(self, _: str) -> grBlockContext:
        return grGroup()
# UI>

# <UI MAP
class B_UiMap:
    _map = OrderedDict[str, B_Ui]()

    prompt_template = B_UiPromptTemplate(register=False) #! `register=False` is the only thing stopping this from breaking

    @classmethod
    def add(cls, ui: B_Ui) -> None:
        if ui.name in cls._map:
            B_Log.warning(cls, "add()", f"Duplicate key -> '{ui.name}'")
        cls._map[ui.name] = ui
    
    @classmethod
    def get(cls, key: str) -> B_Ui | None:
        ui = cls._map.get(key)
        if ui is None:
            B_Log.warning(cls, "get()", f"Key not found -> '{key}'")
        return ui
    
    @classmethod
    def get_all(cls) -> list[B_Ui]:
        return list(cls._map.values())
# UI MAP>

# <MASTER
class B_Master:
    path_script_config = os_path.join(B.Webui.base_path, B.Webui.scripts_folder, B.File.dedicated_folder)
    path_layout = os_path.join(path_script_config, B.File.layout_file)

    @classmethod
    def parse_layout(cls) -> list[B_Ui]:
        # <SCOPED
        class Tracked(Generic[T]):
            __slots__ = ('value')

            def __init__(self, initial_value: T):
                self.value = initial_value

        class StagedUiContainer(NamedTuple, Generic[T_B_UiContainer]):
            type: Type[T_B_UiContainer]
            args: B_Args
            children: list[B_Ui]
        
        class StagedSelect(NamedTuple):
            args: B_Args
            choices: list[B_UiSelectChoice]

        class PromptList(NamedTuple):
            name: str
            entries: list[str]
        
        class FileLine(NamedTuple):
            type: str
            args: B_Args
            
            #! improve validation (missing arg names and values, etc):
            @classmethod
            def parse(cls, l: str) -> "FileLine":
                l_type: str = ""
                l_args: dict[str, str] = {}

                l = l.strip()
                if len(l) > 0:
                    idx_args = l.find(B.File.args_indicator)

                    l_type = l[:idx_args] if idx_args > 0 else "" if idx_args == 0 else l
                    
                    if idx_args != -1:
                        for l_arg in l[len(l_type):].split(B.File.args_indicator)[1:]:
                            idx_arg_value = l_arg.index(B.File.args_separator)
                            l_arg_name = l_arg[:idx_arg_value] if idx_arg_value != -1 else l_arg
                            l_arg_value = l_arg[len(l_arg_name) + len(B.File.args_separator):] if idx_arg_value != -1 else ""
                            if (l_arg_value.endswith(B.File.args_separator)):
                                l_arg_value = l_arg_value[:-len(B.File.args_separator)]
                            l_args[l_arg_name.strip()] = l_arg_value

                return cls(
                    l_type.strip().upper()
                    , B_Args(l_args)
                )
        # SCOPED>
        
        b_ui_list: list[B_Ui] = []

        stack_containers: list[StagedUiContainer[B_UiContainer]] = []

        current_select = Tracked[StagedSelect | None](None)
        current_select_choice = Tracked[B_UiSelectChoice | None](None)
        
        current_prompt_list = Tracked[PromptList | None](None)
        prompt_lists: dict[str, PromptList] = {}

        current_preset = Tracked[B_UiPreset | None](None)
        presets: dict[str, B_Preset] = {}

        skip = Tracked[int](0)

        def handle_ui(b_ui: B_Ui):
            if len(stack_containers) > 0:
                stack_containers[-1].children.append(b_ui)
            else:
                b_ui_list.append(b_ui)
        
        def begin_ui_container(cls: Type[T_B_UiContainer], b_args: B_Args):
            stack_containers.append(StagedUiContainer(cls, b_args, []))

        def end_ui_container():
            staged = stack_containers.pop() if len(stack_containers) > 0 else None
            if staged is None:
                return False
            else:
                handle_ui(staged.type.from_args(staged.args, staged.children))
                return True
        
        def begin_ui_select(b_args: B_Args):
            current_select.value = StagedSelect(b_args, [])
        
        def begin_ui_select_choice(choice: B_UiSelectChoice):
            current_select_choice.value = choice
        
        def end_ui_select_choice():
            if current_select_choice.value is None:
                return False
            else:
                current_select_choice.value = None
                return True
        
        def end_ui_select():
            if current_select.value is None:
                return False
            else:
                handle_ui(B_UiSelect.from_args(current_select.value.args, current_select.value.choices))
                current_select.value = None
                return True
        
        def begin_prompt_list(list_name: str):
            current_prompt_list.value = PromptList(list_name, [])
        
        def end_prompt_list():
            if current_prompt_list.value is None:
                return False
            else:
                prompt_lists[current_prompt_list.value.name] = current_prompt_list.value
                current_prompt_list.value = None
                return True
        
        def begin_preset(b_args: B_Args):
            current_preset.value = B_UiPreset.from_args(b_args)
            presets[current_preset.value.b_preset.name] = current_preset.value.b_preset
            handle_ui(current_preset.value)

        def end_preset():
            if current_preset.value is None:
                return False
            else:
                current_preset.value = None
                return True

        def handle_prompt(cls: Type[T_B_Prompt], b_args: B_Args):
            b_prompt = cls.from_args(b_args)
            if current_select.value is not None:
                current_select.value.choices.append(B_UiSelectChoice(b_prompt))
            else:
                handle_ui(B_UiPromptTemplate(b_prompt, register=True)) #! change to new, more specific classes (for performance)?

        with open(cls.path_layout, "r", encoding="utf8") as file_layout:
            line_number: int = 0

            for l in file_layout:
                line_number += 1
                l = l.strip()

                if len(l) == 0:
                    continue

                if l.startswith(B.File.comment_indicator):
                    B_Log.general(f"# LAYOUT - commented out line @{line_number}")
                    continue

                if l.startswith(B.File.stop_indicator):
                    break

                b_l = FileLine.parse(l)
                
                match b_l.type:
                    case B.File_LineType.SINGLE:
                        handle_prompt(B_PromptSingle, b_l.args)
                        
                    case B.File_LineType.DUAL:
                        handle_prompt(B_PromptDual, b_l.args)
                        
                    case B.File_LineType.EDIT:
                        handle_prompt(B_PromptEdit, b_l.args)
                        
                    case B.File_LineType.EDIT_LINK:
                        handle_prompt(B_PromptEditLink, b_l.args)
                    
                    case B.File_LineType.SELECT:
                        begin_ui_select(b_l.args)
                    
                    case B.File_LineType.PRESET:
                        preset_name = b_l.args.name

                        if current_preset.value is not None:
                            B_Log.warning(cls, "parse_layout()", f"Already began preset '{current_preset.value.name}'")
                        elif preset_name is None:
                            B_Log.warning(cls, "parse_layout()", f"No name specified for {B.File_LineType.PRESET}")
                        elif preset_name in presets:
                            B_Log.warning(cls, "parse_layout()", f"Preset '{preset_name}' already defined")
                        else:
                            begin_preset(b_l.args)
                    
                    case B.File_LineType.SET:
                        target_name = b_l.args.name

                        if target_name is None:
                            B_Log.warning(cls, "parse_layout()", f"No target specified for {B.File_LineType.SET}")
                            continue

                        if current_select.value is not None:
                            # SELECT choice context
                            choice: B_UiSelectChoice
                            if current_select_choice.value is None:
                                if len(current_select.value.choices) == 0:
                                    B_Log.warning(cls, "parse_layout()", f"No current {B.File_LineType.LIST} choice for {B.File_LineType.SET}")
                                    continue
                                choice = current_select.value.choices[-1]
                                begin_ui_select_choice(choice)
                            else:
                                choice = current_select_choice.value

                            if choice.b_preset is None:
                                choice.b_preset = B_Preset(choice.b_prompt.name, True)

                            if target_name in choice.b_preset.map:
                                B_Log.warning(cls, "parse_layout()", f"Mapping '{target_name}' for current {B.File_LineType.LIST} choice preset '{choice.b_preset.name}' already exists")
                            else:
                                choice.b_preset.map[target_name] = B_PresetMapping(target_name, b_l.args)
                        elif current_preset.value is not None:
                            # PRESET standalone context
                            if (target_name in current_preset.value.b_preset.map):
                                B_Log.warning(cls, "parse_layout()", f"Mapping '{target_name}' for preset '{current_preset.value.name}' already exists")
                                continue

                            current_preset.value.b_preset.map[target_name] = B_PresetMapping(target_name, b_l.args)
                        else:
                            B_Log.warning(cls, "parse_layout()", f"Unexpected {B.File_LineType.SET}")
                    
                    case B.File_LineType.VALUE:
                        target_name = b_l.args.name

                        if target_name is None:
                            B_Log.warning(cls, "parse_layout()", f"No target specified for {B.File_LineType.VALUE}")
                            continue

                        context_preset: B_Preset | None = None

                        if current_select_choice.value is not None:
                            # SELECT choice preset context
                            if current_select_choice.value.b_preset is None:
                                B_Log.warning(cls, "parse_layout()", f"No preset defined in current {B.File_LineType.LIST} choice '{current_select_choice.value.b_prompt.name}' for {B.File_LineType.VALUE}")
                            elif len(current_select_choice.value.b_preset.map) == 0:
                                B_Log.warning(cls, "parse_layout()", f"No mapping in current {B.File_LineType.LIST} choice preset '{current_select_choice.value.b_preset.name}' for {B.File_LineType.VALUE}")
                            else:
                                context_preset = current_select_choice.value.b_preset
                        elif current_preset.value is not None:
                            # PRESET standalone context
                            if len(current_preset.value.b_preset.map) == 0:
                                B_Log.warning(cls, "parse_layout()", f"No mapping in current preset '{current_preset.value.name}' for {B.File_LineType.VALUE}")
                            else:
                                context_preset = current_preset.value.b_preset
                        else:
                            B_Log.warning(cls, "parse_layout()", f"Unexpected {B.File_LineType.VALUE}")
                        
                        if context_preset is None:
                            continue
                        
                        mapping = next(reversed(context_preset.map.values()))
                        if target_name in mapping.child_mappings:
                            B_Log.warning(cls, "parse_layout()", f"Child mapping '{target_name}' for current preset mapping target '{mapping.target}' already exists")
                        else:
                            mapping.child_mappings[target_name] = B_PresetMapping(target_name, b_l.args)
                    
                    case B.File_LineType.GROUP:
                        begin_ui_container(B_UiContainerGroup, b_l.args)
                    
                    case B.File_LineType.TAB:
                        begin_ui_container(B_UiContainerTab, b_l.args)
                    
                    case B.File_LineType.ROW:
                        begin_ui_container(B_UiContainerRow, b_l.args)
                    
                    case B.File_LineType.COLUMN:
                        begin_ui_container(B_UiContainerColumn, b_l.args)
                    
                    case B.File_LineType.ACCORDION:
                        begin_ui_container(B_UiContainerAccordion, b_l.args)
                    
                    case B.File_LineType.SEPARATOR:
                        handle_ui(B_UiSeparator.from_args(b_l.args))
                    
                    case B.File_LineType.LIST:
                        list_name = b_l.args.name

                        if current_prompt_list.value is not None:
                            B_Log.warning(cls, "parse_layout()", f"Already began list '{current_prompt_list.value.name}'")
                        elif list_name is None:
                            B_Log.warning(cls, "parse_layout()", f"No name specified for {B.File_LineType.LIST}")
                        elif list_name in prompt_lists:
                            B_Log.warning(cls, "parse_layout()", f"List '{list_name}' already defined")
                        else:
                            begin_prompt_list(list_name)
                    
                    case B.File_LineType.LIST_ITEM:
                        prompt = b_l.args.prompt

                        if current_prompt_list.value is None:
                            B_Log.warning(cls, "parse_layout()", f"No current {B.File_LineType.LIST}")
                        elif prompt is None or len(prompt) == 0:
                            B_Log.warning(cls, "parse_layout()", f"Missing prompt in {B.File_LineType.LIST_ITEM}")
                        else:
                            current_prompt_list.value.entries.append(prompt.lower())
                    
                    case B.File_LineType.FROM_LIST:
                        target_list_name = b_l.args.name
                        postfix = b_l.args.postfix

                        if target_list_name is None:
                            B_Log.warning(cls, "parse_layout()", f"No target name specified for {B.File_LineType.FROM_LIST}")
                            continue

                        prompt_list = prompt_lists.get(target_list_name)
                        if prompt_list is None:
                            B_Log.warning(cls, "parse_layout()", f"No '{target_list_name}' list defined")
                        elif postfix is None or len(postfix) == 0:
                            B_Log.warning(cls, "parse_layout()", f"Missing postfix for {B.File_LineType.FROM_LIST}")
                        else:
                            #! maybe improve; avoid having to reconstruct args:
                            for prompt in prompt_list.entries:
                                handle_prompt(
                                    B_PromptListItem
                                    , B_Args({
                                        B.File_Arg.prompt: prompt
                                        , B.File_Arg.postfix: postfix
                                    })
                                )

                    case B.File_LineType.END:
                        if not (
                            end_prompt_list()
                            or end_preset()
                            or end_ui_select_choice()
                            or end_ui_select()
                            or end_ui_container()
                        ):
                            B_Log.warning(cls, "parse_layout()", f"Unexpected {B.File_LineType.END}")

                    case _:
                        B_Log.warning(cls, "parse_layout()", f"Missing/invalid line type @{line_number}")

        return b_ui_list

    __slots__ = (
        "ui_main"
        , "gr_final_prompt", "gr_final_prompt_neg"
        , "gr_reset_all"
        , "gr_use_break", "gr_prepend", "gr_clear_config", "gr_clear_config_status"
    )

    def __init__(self, b_ui_list: list[B_Ui] | None = None):
        self.ui_main = (
            b_ui_list if b_ui_list is not None else []
        ) + self.parse_layout()

        self.gr_final_prompt: grTextbox
        self.gr_final_prompt_neg: grTextbox

        self.gr_reset_all: grButton

        self.gr_use_break: grCheckbox
        self.gr_prepend: grCheckbox
        self.gr_clear_config: grButton
        self.gr_clear_config_status: grMarkdown
    
    def ui(self) -> Sequence[grComponent]:
        self._html()
        grs = self._build()
        self._bind()
        return grs
    
    def _html(self) -> None:
        b_footer_cls, b_footer_style = B.HTML.css_footer

        grHTML(f"""
            <style>
                .{b_footer_cls} {{ {b_footer_style} }}
            </style>
        """)
    
    def _build(self) -> Sequence[grComponent]:
        gr_components: list[grComponent] = []
        sep = B_UiSeparator()

        # <PROMPT TEMPLATE
        gr_components += B_UiMap.prompt_template.build()
        # PROMPT TEMPLATE>

        # <MAIN
        sep.build()
        for b_ui in self.ui_main:
            gr_components += b_ui.build()
        # MAIN>

        # <CONTROLS
        sep.build()
        final_prompt = B_PromptMap.build_prompts()
        self.gr_final_prompt = grTextbox(
            label=B.Ui.final_prompt_label
            , value=final_prompt.pos
        )
        self.gr_final_prompt_neg = grTextbox(
            label=B.Ui.final_prompt_neg_label
            , value=final_prompt.neg
        )
        sep.build()
        with grRow():
            self.gr_reset_all = grButton(B.Ui.reset_all_label)
        # CONTROLS>

        # <SETTINGS
        sep.build()
        with grAccordion(B.Ui.settings_label, open=B.Default.open_settings):
            self.gr_use_break = grCheckbox(label=B.Ui.use_break_label, value=B.Default.use_break)
            self.gr_prepend = grCheckbox(label=B.Ui.prepend_label, value=B.Default.prepend)
            sep.build()
            self.gr_clear_config = grButton(B.Ui.clear_config_label)
            self.gr_clear_config_status = grMarkdown(f"<b>{B.Ui.clear_config_desc}</b>")
        # SETTINGS>

        return [
            self.gr_final_prompt
            , self.gr_final_prompt_neg
            , self.gr_use_break
            , self.gr_prepend
            , self.gr_reset_all
            , self.gr_clear_config
        ] + gr_components
    
    def _bind(self) -> None:
        def gr_target_update() -> list[Any]:
            prompt = B_PromptMap.build_prompts()
            return [prompt.pos, prompt.neg]
        gr_target_output: list[grBlock] = [self.gr_final_prompt, self.gr_final_prompt_neg]

        # <PROMPT TEMPLATE
        B_UiMap.prompt_template.bind(gr_target_update, gr_target_output)
        # PROMPT TEMPLATE>
        
        # <MAIN
        for b_ui in self.ui_main:
            b_ui.bind(gr_target_update, gr_target_output)
        # MAIN>
        
        # <CONTROLS
        def on_reset_all():
            B_UiMap.prompt_template.update_prompt(None)

            gr_output_updates: list[Any] = []
            for b_ui in self.ui_main:
                b_ui.reset()
                gr_output_updates += b_ui.gr_output_update()

            return gr_output_updates + B_UiMap.prompt_template.gr_output_update() + gr_target_update()
        self.gr_reset_all.click(
            fn=on_reset_all
            , outputs=(
                [
                    gr
                    for b_ui in self.ui_main
                    for gr in b_ui.gr_output()
                ]
                + B_UiMap.prompt_template.gr_output()
                + gr_target_output
            )
        )
        # CONTROLS>

        # <SETTINGS
        def on_clear_config() -> Sequence[Any]:
            loadsave = ui_loadsave.UiLoadsave(cmd_opts.ui_config_file)
            ui_settings: dict[str, Any] = loadsave.ui_settings.copy()
            removed_count = 0
            for k in list(ui_settings.keys()):
                if k.find(B.File.script_name) != -1:
                    del ui_settings[k]
                    removed_count += 1
            loadsave.write_to_file(ui_settings)
            return [
                gr_update(self.gr_clear_config, interactive=False)
                , gr_update(self.gr_clear_config_status, value=f"Removed {removed_count} entr{'y' if removed_count == 1 else 'ies'} from {os_path.basename(cmd_opts.ui_config_file)}")
            ]
        self.gr_clear_config.click(
            fn=on_clear_config
            , outputs=[self.gr_clear_config, self.gr_clear_config_status]
        )
        # SETTINGS>
# MASTER>

# <SCRIPT
class Script(scripts.Script):
    b_master = B_Master()
    
    def title(self):
        return B.title
    
    def show(self, is_img2img):
        return not is_img2img #!
    
    def ui(self, is_img2img):
        return self.b_master.ui()

    def run(
        self
        , p
        , b_final_prompt: str
        , b_final_prompt_neg: str
        , use_break: bool
        , prepend: bool
        , reset_all_btn: str
        , clear_config_btn: str
        , *output: Any
    ):
        prompt_a: str = p.prompt
        prompt_b: str = b_final_prompt
        prompt_neg_a: str = p.negative_prompt
        prompt_neg_b: str = b_final_prompt_neg

        if prepend:
            prompt_a, prompt_b, prompt_neg_a, prompt_neg_b = prompt_b, prompt_a, prompt_neg_b, prompt_neg_a

        if use_break:
            if (len(prompt_a) > 0 and len(prompt_b) > 0):
                prompt_a = B_Fn.Prompt.added(prompt_a, B.Webui.break_prompt)
            if (len(prompt_neg_a) > 0 and len(prompt_neg_b) > 0):
                prompt_neg_a = B_Fn.Prompt.added(prompt_neg_a, B.Webui.break_prompt)

        p.prompt = B_Fn.Prompt.added(prompt_a, prompt_b)
        p.negative_prompt = B_Fn.Prompt.added(prompt_neg_a, prompt_neg_b)

        proc = process_images(p)
        return proc
# SCRIPT>