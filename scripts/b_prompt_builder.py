import modules.scripts as scripts
import gradio as gr
import os
import typing
import random
import json

from abc import ABC, abstractmethod
from modules import scripts
from modules.processing import StableDiffusionProcessing, process_images

b_path_base = scripts.basedir()
b_file_name_config = "ui-config.json"
b_folder_name_scripts = "scripts"
b_folder_name_script_config = "b_prompt_builder"
b_file_name_layout = "layout.txt"
b_file_name_presets = "presets.txt"
b_tagged_ignore = False
b_validate_skip = False

def printWarning(type: type, name: str, message: str) -> None:
    print(f"WARNING/{type.__name__}/{name}: {message}")

class B_Value():
    def __init__(self, value_default):
        self.value_default = value_default
        self.value = self.buildDefaultValue()
    
    def buildDefaultValue(self):
        return self.value_default
    
    def reset(self):
        self.value = self.buildDefaultValue()
    
    def reinit(self, value: typing.Any, keep_current: bool = False):
        self.value_default = value
        if not keep_current:
            self.reset()

class B_Prompt(ABC):
    class Meta():
        def __init__(
                self
                , prompt_enable: bool = False
                , negative_enable: bool = False
                , prompt_negative_enable: bool = False
                , prompt_edit_enable: bool = False
            ):
            self.name_visible = True

            self.prompt_visible = prompt_enable
            self.prompt_enable = prompt_enable

            self.prompt_negative_visible = prompt_negative_enable
            self.prompt_negative_enable = prompt_negative_enable
            
            self.negative_visible = negative_enable
            self.prompt_edit_visible = prompt_edit_enable
    
    class Values():
        class Defaults():
            prompt: str = ""

            emphasis: float = 1
            emphasis_min: float = 0
            emphasis_step: float = 0.1

            edit: int = 50
            edit_min: int = 0
            edit_max: int = 100
            edit_step: int = 1

            negative: bool = False
        
        class Keys():
            prompt = "pp"
            prompt_negative = "pn"
            emphasis = "sp"
            emphasis_negative = "sn"
            negative = "n"
            prompt_a = "a"
            prompt_b = "b"
            edit = "r"
            prefix = "prefix"
            postfix = "postfix"
        
        @staticmethod
        def _fromArgs(args: dict[str, str]):
            prompt = args.get(B_Prompt.Values.Keys.prompt)
            emphasis = args.get(B_Prompt.Values.Keys.emphasis)
            prompt_negative = args.get(B_Prompt.Values.Keys.prompt_negative)
            emphasis_negative = args.get(B_Prompt.Values.Keys.emphasis_negative)
            negative = args.get(B_Prompt.Values.Keys.negative)
            edit = args.get(B_Prompt.Values.Keys.edit)
            return B_Prompt.Values(
                prompt = prompt if prompt is not None else None
                , emphasis = float(emphasis) if emphasis is not None else None
                , negative = bool(int(negative)) if negative is not None else None
                , prompt_negative = prompt_negative if prompt_negative is not None else None
                , emphasis_negative = float(emphasis_negative) if emphasis_negative is not None else None
                , edit = int(edit) if edit is not None else None
            )

        def __init__(
                self
                , prompt: str = Defaults.prompt
                , emphasis: float = Defaults.emphasis
                , negative: bool = Defaults.negative
                , prompt_negative: str = Defaults.prompt
                , emphasis_negative: float = Defaults.emphasis
                , prompt_a: str = Defaults.prompt
                , prompt_b: str = Defaults.prompt
                , edit: int = Defaults.edit
                , prefix: str = Defaults.prompt
                , postfix: str = Defaults.prompt
            ):
            self.prompt = B_Value(prompt)
            self.emphasis = B_Value(emphasis)
            self.negative = B_Value(negative)
            self.prompt_negative = B_Value(prompt_negative)
            self.emphasis_negative = B_Value(emphasis_negative)
            self.edit = B_Value(edit)
            self.prompt_a = B_Value(prompt_a)
            self.prompt_b = B_Value(prompt_b)
            self.prefix = B_Value(prefix)
            self.postfix = B_Value(postfix)
        
        def updateFromArgs(self, args: dict[str, str], resetIfNone: bool = False):
            if len(args) == 0:
                return
            
            values_new = self._fromArgs(args)
            if values_new is None:
                return
            
            if values_new.prompt.value is not None:
                self.prompt.value = values_new.prompt.value
            elif resetIfNone:
                self.prompt.reset()
            
            if values_new.emphasis.value is not None:
                self.emphasis.value = values_new.emphasis.value
            elif resetIfNone:
                self.emphasis.reset()
            
            if values_new.prompt_negative.value is not None:
                self.prompt_negative.value = values_new.prompt_negative.value
            elif resetIfNone:
                self.prompt_negative.reset()
            
            if values_new.emphasis_negative.value is not None:
                self.emphasis_negative.value = values_new.emphasis_negative.value
            elif resetIfNone:
                self.emphasis_negative.reset()
            
            if values_new.negative.value is not None:
                self.negative.value = values_new.negative.value
            elif resetIfNone:
                self.negative.reset()
            
            if values_new.edit.value is not None:
                self.edit.value = values_new.edit.value
            elif resetIfNone:
                self.edit.reset()

    class Fn():
        @staticmethod
        def promptSanitized(prompt: str) -> str:
            return prompt.strip() if prompt is not None else ""
        
        @staticmethod
        def promptAdded(promptExisting: str, promptToAdd: str) -> str:
            if len(promptToAdd) > 0:
                if len(promptExisting) > 0:
                    promptExisting += ", " + promptToAdd
                else:
                    promptExisting = promptToAdd
            
            return promptExisting
        
        @staticmethod
        def promptDecorated(prompt: str, prefix: str = "", postfix: str = "") -> str:
            if len(prompt) > 0:
                if len(prefix) > 0:
                    prompt = f"{prefix} {prompt}"
                if len(postfix) > 0:
                    prompt = f"{prompt} {postfix}"
            
            return prompt
        
        @staticmethod
        def promptEmphasized(prompt: str, emphasis: float) -> str:
            if len(prompt) == 0 or emphasis == B_Prompt.Values.Defaults.emphasis_min:
                return ""
            
            if emphasis != 1:
                prompt = f"({prompt}:{emphasis})"
            
            return prompt
    
    @staticmethod
    @abstractmethod
    def _fromArgs(name: str, args: dict[str, str]):
        pass
    
    def __init__(self, name: str, meta: Meta, values: Values):
        self.name = name
        self.meta = meta
        self.values = values

        B_Prompt_Map.add(self)
    
    def reset(self):
        if self.meta.prompt_enable:
            self.values.prompt.reset()
        self.values.emphasis.reset()
        self.values.negative.reset()
        if self.meta.prompt_negative_enable:
            self.values.prompt_negative.reset()
        self.values.emphasis_negative.reset()
        self.values.edit.reset()
        #! not rendered in UI:
        # self.values.prompt_a.reset()
        # self.values.prompt_b.reset()
        # self.values.prefix.reset()
        # self.values.postfix.reset()
    
    def clear(self):
        if self.meta.prompt_enable:
            self.values.prompt.value = B_Prompt.Values.Defaults.prompt
        self.values.emphasis.value = B_Prompt.Values.Defaults.emphasis
        self.values.negative.value = B_Prompt.Values.Defaults.negative
        if self.meta.prompt_negative_enable:
            self.values.prompt_negative.value = B_Prompt.Values.Defaults.prompt
        self.values.emphasis_negative.value = B_Prompt.Values.Defaults.emphasis
        self.values.edit.value = B_Prompt.Values.Defaults.edit
        #! not rendered in UI:
        # self.values.prompt_a.value = B_Prompt.Values.Defaults.prompt
        # self.values.prompt_b.value = B_Prompt.Values.Defaults.prompt
        # self.values.prefix.value = B_Prompt.Values.Defaults.prompt
        # self.values.postfix.value = B_Prompt.Values.Defaults.prompt
    
    @abstractmethod
    def build(self) -> tuple[str, str]:
        pass

class B_Prompt_Single(B_Prompt):
    @staticmethod
    def _fromArgs(name: str, args: dict[str, str]):
        return B_Prompt_Single(
            name
            , args.get(B_Prompt.Values.Keys.prompt, B_Prompt.Values.Defaults.prompt)
            , float(args.get(B_Prompt.Values.Keys.emphasis, B_Prompt.Values.Defaults.emphasis))
            , bool(int(args.get(B_Prompt.Values.Keys.negative, int(B_Prompt.Values.Defaults.negative))))
            , args.get(B_Prompt.Values.Keys.prefix, B_Prompt.Values.Defaults.prompt)
            , args.get(B_Prompt.Values.Keys.postfix, B_Prompt.Values.Defaults.prompt)
        )
    
    def __init__(
            self
            , name: str
            , prompt = B_Prompt.Values.Defaults.prompt
            , emphasis = B_Prompt.Values.Defaults.emphasis
            , negative = B_Prompt.Values.Defaults.negative
            , prefix = B_Prompt.Values.Defaults.prompt
            , postfix = B_Prompt.Values.Defaults.prompt
        ):
        super().__init__(
            name
            , B_Prompt.Meta(
                prompt_enable = True
                , negative_enable = True
            )
            , B_Prompt.Values(
                prompt = prompt
                , emphasis = emphasis
                , negative = negative
                , prefix = prefix
                , postfix = postfix
            )
        )
    
    def build(self) -> tuple[str, str]:
        prompt = B_Prompt.Fn.promptEmphasized(
            B_Prompt.Fn.promptDecorated(
                B_Prompt.Fn.promptSanitized(self.values.prompt.value)
                , B_Prompt.Fn.promptSanitized(self.values.prefix.value)
                , B_Prompt.Fn.promptSanitized(self.values.postfix.value)
            )
            , self.values.emphasis.value
        )
        if not self.values.negative.value:
            return prompt, ""
        else:
            return "", prompt

class B_Prompt_Dual(B_Prompt):
    @staticmethod
    def _fromArgs(name: str, args: dict[str, str]):
        return B_Prompt_Dual(
            name
            , args.get(B_Prompt.Values.Keys.prompt, B_Prompt.Values.Defaults.prompt)
            , float(args.get(B_Prompt.Values.Keys.emphasis, B_Prompt.Values.Defaults.emphasis))
            , args.get(B_Prompt.Values.Keys.prompt_negative, B_Prompt.Values.Defaults.prompt)
            , float(args.get(B_Prompt.Values.Keys.emphasis_negative, B_Prompt.Values.Defaults.emphasis))
        )
    
    def __init__(
            self
            , name: str
            , prompt = B_Prompt.Values.Defaults.prompt
            , emphasis = B_Prompt.Values.Defaults.emphasis
            , prompt_negative = B_Prompt.Values.Defaults.prompt
            , emphasis_negative = B_Prompt.Values.Defaults.emphasis
        ):
        super().__init__(
            name
            , B_Prompt.Meta(
                prompt_enable = True
                , prompt_negative_enable = True
            )
            , B_Prompt.Values(
                prompt = prompt
                , emphasis = emphasis
                , prompt_negative = prompt_negative
                , emphasis_negative = emphasis_negative
            )
        )
    
    def build(self) -> tuple[str, str]:
        prompt = B_Prompt.Fn.promptEmphasized(
            B_Prompt.Fn.promptDecorated(
                B_Prompt.Fn.promptSanitized(self.values.prompt.value)
                , B_Prompt.Fn.promptSanitized(self.values.prefix.value)
                , B_Prompt.Fn.promptSanitized(self.values.postfix.value)
            )
            , self.values.emphasis.value
        )
        prompt_negative = B_Prompt.Fn.promptEmphasized(
            B_Prompt.Fn.promptDecorated(
                B_Prompt.Fn.promptSanitized(self.values.prompt_negative.value)
                , B_Prompt.Fn.promptSanitized(self.values.prefix.value)
                , B_Prompt.Fn.promptSanitized(self.values.postfix.value)
            )
            , self.values.emphasis_negative.value
        )
        return prompt, prompt_negative

class B_Prompt_Edit(B_Prompt):
    @staticmethod
    def _fromArgs(name: str, args: dict[str, str]):
        return B_Prompt_Edit(
            name
            , args.get(B_Prompt.Values.Keys.prompt_a, B_Prompt.Values.Defaults.prompt)
            , args.get(B_Prompt.Values.Keys.prompt_b, B_Prompt.Values.Defaults.prompt)
            , int(args.get(B_Prompt.Values.Keys.edit, B_Prompt.Values.Defaults.edit))
            , bool(int(args.get(B_Prompt.Values.Keys.negative, int(B_Prompt.Values.Defaults.negative))))
        )
    
    @staticmethod
    def _build(
        prompt_a: str
        , prompt_b: str
        , prefix: str
        , postfix: str
        , edit: int
        , negative: bool
    ) -> tuple[str, str]:
        prompt_a = B_Prompt.Fn.promptDecorated(
            B_Prompt.Fn.promptSanitized(prompt_a)
            , B_Prompt.Fn.promptSanitized(prefix)
            , B_Prompt.Fn.promptSanitized(postfix)
        )
        prompt_b = B_Prompt.Fn.promptDecorated(
            B_Prompt.Fn.promptSanitized(prompt_b)
            , B_Prompt.Fn.promptSanitized(prefix)
            , B_Prompt.Fn.promptSanitized(postfix)
        )

        value = float(edit)
        value = value / B_Prompt.Values.Defaults.edit_max
        value = round(1 - value, 2)

        prompt: str = None
        if value == 1:
            prompt = prompt_a
        elif value == 0:
            prompt = prompt_b
        else:
            prompt = f"[{prompt_a}:{prompt_b}:{value}]"
        
        if not negative:
            return prompt, ""
        else:
            return "", prompt
    
    def __init__(
            self
            , name: str
            , prompt_a: str
            , prompt_b: str
            , edit = B_Prompt.Values.Defaults.edit
            , negative = B_Prompt.Values.Defaults.negative
        ):
        super().__init__(
            name
            , B_Prompt.Meta(
                negative_enable = True
                , prompt_edit_enable = True
            )
            , B_Prompt.Values(
                negative = negative
                , prompt_a = prompt_a
                , prompt_b = prompt_b
                , edit = edit
            )
        )
    
    def build(self) -> tuple[str, str]:
        return B_Prompt_Edit._build(
            self.values.prompt_a.value
            , self.values.prompt_b.value
            , self.values.prefix.value
            , self.values.postfix.value
            , self.values.edit.value
            , self.values.negative.value
        )

class B_Prompt_Edit_Link(B_Prompt):
    @staticmethod
    def _fromArgs(name: str, args: dict[str, str]):
        return B_Prompt_Edit_Link(
            name
            , args["link"]
            , args.get(B_Prompt.Values.Keys.prompt_a, B_Prompt.Values.Defaults.prompt)
            , args.get(B_Prompt.Values.Keys.prompt_b, B_Prompt.Values.Defaults.prompt)
            , bool(int(args.get(B_Prompt.Values.Keys.negative, int(B_Prompt.Values.Defaults.negative))))
        )
    
    def __init__(
            self
            , name: str
            , link_name: str
            , prompt_a: str
            , prompt_b: str
            , negative = B_Prompt.Values.Defaults.negative
        ):
        super().__init__(
            name
            , B_Prompt.Meta(
                negative_enable = True
            )
            , B_Prompt.Values(
                negative = negative
                , prompt_a = prompt_a
                , prompt_b = prompt_b
            )
        )

        self.link_name = link_name
    
    def build(self) -> tuple[str, str]:
        b_prompt_link = B_Prompt_Map.get(self.link_name)
        if b_prompt_link is None:
            printWarning(type(self), f"{self.name} - build()", f"Linked prompt not found -> '{self.link_name}'")
        
        return B_Prompt_Edit._build(
            self.values.prompt_a.value
            , self.values.prompt_b.value
            , self.values.prefix.value
            , self.values.postfix.value
            , b_prompt_link.values.edit.value
            , self.values.negative.value
        )
    
    def getLink(self) -> B_Prompt_Edit:
        return B_Prompt_Map.get(self.link_name)

class B_UI(ABC):
    @staticmethod
    @abstractmethod
    def _fromArgs(args: dict[str, str], name: str = None):
        #! hide arg pending
        pass
    
    def __init__(self, name: str):
        self.name = name
    
    @abstractmethod
    def init(self) -> None:
        pass

    @abstractmethod
    def build(self) -> None:
        pass

    @abstractmethod
    def bind(self, gr_prompt: typing.Any, gr_prompt_negative: typing.Any) -> None:
        pass

    @abstractmethod
    def getGrForWebUI(self) -> list[typing.Any]:
        pass

    @abstractmethod
    def reset(self, clear: bool = False) -> None:
        """Reset UI and values to initial values or default values if clear is True"""
        pass

    @abstractmethod
    def update(self, inputValues: tuple) -> int:
        pass
    
    @abstractmethod
    def getInput(self) -> list[typing.Any]:
        pass

    @abstractmethod
    def getOutput(self) -> list[typing.Any]:
        pass

    @abstractmethod
    def getOutputUpdate(self) -> list[typing.Any]:
        pass

    def applyPresetMapping(self, args: dict[str, str], additive: bool):
        pass

class B_UI_Preset(B_UI):
    @staticmethod
    def _fromArgs(args: dict[str, str], name: str = None):
        return B_UI_Preset(
            name
            , bool(int(args.get("is_additive", 0)))
        )
    
    def __init__(self, name: str, additive: bool):
        super().__init__(name)

        self.additive = additive
        
        self.mappings: dict[str, dict[str, str]] = {}

        self.gr_button: typing.Any = None
    
    def init(self) -> None:
        pass

    def build(self) -> None:
        self.gr_button = gr.Button(self.name)
    
    def bind(self, gr_prompt: typing.Any, gr_prompt_negative: typing.Any) -> None:
        def _apply(*inputValues):
            B_UI_Map.consumeInputValues(inputValues)
            self.apply()
            return B_UI_Map.getOutputUpdates() + B_Prompt_Map.buildPromptUpdate()
        self.gr_button.click(
            fn = _apply
            , inputs = B_UI_Map._inputs
            , outputs = B_UI_Map._outputs + [gr_prompt, gr_prompt_negative]
        )

    def getGrForWebUI(self) -> list[typing.Any]:
        return [self.gr_button]
    
    def reset(self, clear: bool = False) -> None:
        pass

    def update(self, inputValues: tuple) -> int:
        return 0
    
    def getInput(self) -> list[typing.Any]:
        return []
    
    def getOutput(self) -> list[typing.Any]:
        return []
    
    def getOutputUpdate(self) -> list[typing.Any]:
        return []
    
    def addMapping(self, name: str, args: dict[str, str]):
        if name in self.mappings:
            printWarning(type(self), f"{self.name} - addMapping()", f"Duplicate name -> '{name}'")
        self.mappings[name] = args
    
    def apply(self):
        if self.additive:
            for k in self.mappings:
                B_UI_Map._map[k].applyPresetMapping(self.mappings[k], True)
        else:
            for k in B_UI_Map._map:
                if k in self.mappings:
                    B_UI_Map._map[k].applyPresetMapping(self.mappings[k], False)
                else:
                    B_UI_Map._map[k].reset() #!

class B_UI_Separator(B_UI):
    #!
    @staticmethod
    def _fromArgs(args: dict[str, str], name: str = "Separator"):
        return B_UI_Separator(name)
    
    @staticmethod
    def _build() -> typing.Any:
        return gr.Markdown(value = "<hr style=\"margin: 0.5em 0; border-style: dotted; border-color: var(--border-color-primary);\" />")
    
    def __init__(self, name: str = "Separator"):
        super().__init__(name)
    
    def init(self) -> None:
        pass
    
    def build(self) -> None:
        B_UI_Separator._build()
    
    def bind(self, gr_prompt: typing.Any, gr_prompt_negative: typing.Any) -> None:
        pass
    
    def getGrForWebUI(self) -> list[typing.Any]:
        return []
    
    def reset(self, clear: bool = False) -> None:
        pass

    def update(self, inputValues: tuple) -> int:
        return 0
    
    def getInput(self) -> list[typing.Any]:
        return []
    
    def getOutput(self) -> list[typing.Any]:
        return []
    
    def getOutputUpdate(self) -> list[typing.Any]:
        return []

class B_UI_Prompt(B_UI):
    _prompt_scale: int = 4
    _emphasis_scale: int = 1

    #!
    @staticmethod
    def _fromArgs(args: dict[str, str], name: str = "Prompt"):
        return B_UI_Prompt(name)
    
    def __init__(self, name: str = "Prompt", b_prompt: B_Prompt = None):
        super().__init__(name)

        self.b_prompt = b_prompt
        self.b_ui_preset: B_UI_Preset = None

        self.gr_container: typing.Any = None

        self.gr_markdown: typing.Any = None

        self.gr_prompt_container: typing.Any = None
        self.gr_prompt: typing.Any = None
        self.gr_emphasis: typing.Any = None

        self.gr_prompt_negative_container: typing.Any = None
        self.gr_prompt_negative: typing.Any = None
        self.gr_emphasis_negative: typing.Any = None

        #! buttons?
        self.gr_slider: typing.Any = None

        self.gr_negative: typing.Any = None

        self.gr_button_apply: typing.Any = None
        self.gr_button_remove: typing.Any = None
        
        self.ui_component_names_apply: list[str] = []
        self.outputs_extra: list[typing.Any] = []
        self.fn_updates_extra: typing.Callable[[], list] = None
    
    def init(self) -> None:
        #! activate on prompt map
        if self.b_prompt is not None:
            B_Prompt_Map.update(self.b_prompt)

    def build(self) -> None:
        meta, values, name, visible, enabled_button_remove = self.getUpdateValues()
        
        self.gr_container = gr.Column(variant = "panel", visible = visible)
        with self.gr_container:
            self.gr_markdown = gr.Markdown(value = name, visible = meta.name_visible)

            self.gr_prompt_container = gr.Row(visible = meta.prompt_visible)
            with self.gr_prompt_container:
                self.gr_prompt = gr.Textbox(
                    label = "Prompt"
                    , value = values.prompt.value
                    , scale = B_UI_Prompt._prompt_scale
                    , interactive = meta.prompt_enable
                )
                self.gr_emphasis = gr.Number(
                    label = "Emphasis"
                    , value = values.emphasis.value
                    , minimum = B_Prompt.Values.Defaults.emphasis_min
                    , step = B_Prompt.Values.Defaults.emphasis_step
                    , scale = B_UI_Prompt._emphasis_scale
                )
            
            self.gr_prompt_negative_container = gr.Row(visible = meta.prompt_negative_visible)
            with self.gr_prompt_negative_container:
                self.gr_prompt_negative = gr.Textbox(
                    label = "Prompt (N)"
                    , value = values.prompt_negative.value
                    , scale = B_UI_Prompt._prompt_scale
                    , interactive = meta.prompt_negative_enable
                )
                self.gr_emphasis_negative = gr.Number(
                    label = "Emphasis (N)"
                    , value = values.emphasis_negative.value
                    , minimum = B_Prompt.Values.Defaults.emphasis_min
                    , step = B_Prompt.Values.Defaults.emphasis_step
                    , scale = B_UI_Prompt._emphasis_scale
                )
            
            self.gr_slider = gr.Slider(
                label = "Edit"
                , value = values.edit.value
                , minimum = B_Prompt.Values.Defaults.edit_min
                , maximum = B_Prompt.Values.Defaults.edit_max
                , step = B_Prompt.Values.Defaults.edit_step
                , visible = meta.prompt_edit_visible
            )
            
            self.gr_negative = gr.Checkbox(
                label = "Negative?"
                , value = values.negative.value
                , visible = meta.negative_visible
            )

            B_UI_Separator._build()

            with gr.Row():
                self.gr_button_apply = gr.Button(
                    value = "Apply"
                )
                self.gr_button_remove = gr.Button(
                    value = "Remove"
                    , interactive = enabled_button_remove
                )
        
        #! register on ui map
        if self.b_prompt is not None:
            B_UI_Map.add(self)
    
    def bind(self, gr_prompt: typing.Any, gr_prompt_negative: typing.Any) -> None:
        if len(self.ui_component_names_apply) == 0 and self.b_ui_preset is not None:
            if not self.b_ui_preset.additive:
                self.ui_component_names_apply = list(B_UI_Map._map.keys())
            else:
                self.ui_component_names_apply = list(self.b_ui_preset.mappings.keys())
        self.ui_component_names_apply = list(filter(lambda k: k != self.name, self.ui_component_names_apply)) #!
        
        inputs: list[typing.Any] = self.getInput()
        outputs: list[typing.Any] = self.getOutput()
        for k in self.ui_component_names_apply:
            inputs += B_UI_Map._map[k].getInput()
            outputs += B_UI_Map._map[k].getOutput()
        outputs += self.outputs_extra + [gr_prompt, gr_prompt_negative]
        
        def _fnApply(*inputValues):
            offset = self.update(inputValues)
            for k in self.ui_component_names_apply:
                offset += B_UI_Map._map[k].update(inputValues[offset:])
            
            self.apply()

            updates: list = self.getOutputUpdate()
            for k in self.ui_component_names_apply:
                updates += B_UI_Map._map[k].getOutputUpdate()
            if self.fn_updates_extra is not None:
                updates += self.fn_updates_extra()
            updates += B_Prompt_Map.buildPromptUpdate()
            
            return updates
        applyArgs = {
            "fn": _fnApply
            , "inputs": inputs
            , "outputs": outputs
        }
        self.gr_prompt.submit(**applyArgs)
        self.gr_emphasis.submit(**applyArgs)
        self.gr_prompt_negative.submit(**applyArgs)
        self.gr_emphasis_negative.submit(**applyArgs)
        self.gr_button_apply.click(**applyArgs)

        # Remove
        def _fnRemove():
            B_Prompt_Map.update(self.b_prompt, True)
            updates = [self.gr_button_remove.update(interactive = False)] + B_Prompt_Map.buildPromptUpdate()
            if self.fn_updates_extra is not None:
                updates += self.fn_updates_extra()
            return updates
        self.gr_button_remove.click(
            fn = _fnRemove
            , outputs = [self.gr_button_remove, gr_prompt, gr_prompt_negative] + self.outputs_extra
        )
    
    def getGrForWebUI(self) -> list[typing.Any]:
        return [
            self.gr_prompt
            , self.gr_emphasis
            , self.gr_negative
            , self.gr_prompt_negative
            , self.gr_emphasis_negative
            , self.gr_slider
            , self.gr_button_apply
            , self.gr_button_remove
        ]
    
    def reset(self, clear: bool = False) -> None:
        if not clear:
            self.b_prompt.reset()
            B_Prompt_Map.update(self.b_prompt)
        else:
            self.b_prompt.clear()
            B_Prompt_Map.update(self.b_prompt, True)
    
    def update(self, inputValues: tuple) -> int:
        offset: int = 0
        
        prompt = str(inputValues[offset])
        offset += 1

        emphasis = float(inputValues[offset])
        offset += 1

        prompt_negative = str(inputValues[offset])
        offset += 1

        emphasis_negative = float(inputValues[offset])
        offset += 1

        edit = int(inputValues[offset])
        offset += 1

        negative = bool(inputValues[offset])
        offset += 1

        if self.b_prompt is not None:
            self.b_prompt.values.prompt.value = prompt
            self.b_prompt.values.emphasis.value = emphasis
            self.b_prompt.values.negative.value = negative
            self.b_prompt.values.prompt_negative.value = prompt_negative
            self.b_prompt.values.emphasis_negative.value = emphasis_negative
            self.b_prompt.values.edit.value = edit

        return offset
    
    def getInput(self) -> list[typing.Any]:
        return [
            self.gr_prompt
            , self.gr_emphasis
            , self.gr_prompt_negative
            , self.gr_emphasis_negative
            , self.gr_slider
            , self.gr_negative
        ]
    
    def getOutput(self) -> list[typing.Any]:
        return [
            self.gr_container
            , self.gr_markdown
            , self.gr_prompt_container
            , self.gr_prompt
            , self.gr_emphasis
            , self.gr_prompt_negative_container
            , self.gr_prompt_negative
            , self.gr_emphasis_negative
            , self.gr_slider
            , self.gr_negative
            , self.gr_button_remove
        ]
    
    def getOutputUpdate(self) -> list[typing.Any]:
        meta, values, name, visible, enabled_button_remove = self.getUpdateValues()

        return [
            self.gr_container.update(visible = visible)
            , self.gr_markdown.update(value = name, visible = meta.name_visible)
            , self.gr_prompt_container.update(visible = meta.prompt_visible)
            , self.gr_prompt.update(value = values.prompt.value, interactive = meta.prompt_enable)
            , values.emphasis.value
            , self.gr_prompt_negative_container.update(visible = meta.prompt_negative_visible)
            , self.gr_prompt_negative.update(value = values.prompt_negative.value, interactive = meta.prompt_negative_enable)
            , values.emphasis_negative.value
            , self.gr_slider.update(visible = meta.prompt_edit_visible, value = values.edit.value, step = B_Prompt.Values.Defaults.edit_step)
            , self.gr_negative.update(visible = meta.negative_visible, value = values.negative.value)
            , self.gr_button_remove.update(interactive = enabled_button_remove)
        ]
    
    def getUpdateValues(self) -> tuple[B_Prompt.Meta, B_Prompt.Values, str, bool, bool]:
        if self.b_prompt is not None:
            return (
                self.b_prompt.meta
                , self.b_prompt.values
                , f"**{self.b_prompt.name}**"
                , True
                , B_Prompt_Map.isSelected(self.b_prompt)
            )
        else:
            return (
                B_Prompt.Meta()
                , B_Prompt.Values()
                , f"**{self.name}**"
                , False
                , False
            )
    
    def apply(self):
        if self.b_prompt is not None:
            B_Prompt_Map.update(self.b_prompt)
        
        if self.b_ui_preset is not None:
            self.b_ui_preset.apply()
    
    def applyPresetMapping(self, args: dict[str, str], additive: bool):
        if self.b_prompt is None:
            return
        self.b_prompt.values.updateFromArgs(args, not additive)

class B_UI_Dropdown(B_UI):
    _choice_empty: str = "-"
    _ui_dropdown_scale: int = 30
    _ui_remove_scale: int = 1
    _ui_remove_width: int = 60

    @staticmethod
    def _fromArgs(args: dict[str, str], name: str = "Dropdown"):
        return B_UI_Dropdown(
            name
            , bool(int(args.get("sort", 1)))
            , B_UI_Dropdown._fromArgsValue(args)
            , args.get(B_Prompt.Values.Keys.prefix, B_Prompt.Values.Defaults.prompt)
            , args.get(B_Prompt.Values.Keys.postfix, B_Prompt.Values.Defaults.prompt)
            , int(args.get("scale", 0))
        )
    
    @staticmethod
    def _fromArgsValue(args: dict[str, str]):
        valueMap: dict[str, dict[str, str]] = {}
        if len(args) > 0:
            choicesWithPromptArgs = list(filter(lambda v: len(v) > 0, map(lambda v: v.strip(), args.get("v", "").split(","))))
            for x in choicesWithPromptArgs:
                promptArgsMap: dict[str, str] = {}
                x_split = x.split("::")
                choice = x_split[0]
                if len(x_split) > 1:
                    promptArgs = x_split[1].split("|")
                    for promptArg in promptArgs:
                        arg_name = promptArg[:promptArg.index(" ")]
                        arg_value = promptArg[len(arg_name) + 1:].strip()
                        promptArgsMap[arg_name] = arg_value
                valueMap[choice] = promptArgsMap
        return valueMap
    
    @staticmethod
    def _buildColorChoicesList(postfix: str = "") -> list[B_Prompt_Single]:
        return list(map(
            lambda text: B_Prompt_Single(
                f"{text} {postfix}"
                , text.lower()
                , postfix = postfix
            )
            , [
                "Dark"
                , "Light"
                , "Black"
                , "Grey"
                , "White"
                , "Brown"
                , "Blue"
                , "Green"
                , "Red"
                , "Blonde"
                , "Rainbow"
                , "Pink"
                , "Purple"
                , "Orange"
                , "Yellow"
                , "Multicolored"
                , "Pale"
                , "Silver"
                , "Gold"
                , "Tan"
            ]
        ))
    
    def __init__(
            self
            , name: str = "Dropdown"
            , sort_choices: bool = True
            , b_prompts_applied_default: dict[str, dict[str, str]] = None
            , prefix = B_Prompt.Values.Defaults.prompt
            , postfix = B_Prompt.Values.Defaults.prompt
            , scale: int = 1
            , b_prompts: list[B_Prompt] = None
        ):
        super().__init__(name)

        self.choice = B_Value(B_UI_Dropdown._choice_empty)
        self.sort_choices = sort_choices
        self.b_prompts_applied_default = b_prompts_applied_default if b_prompts_applied_default is not None else {}
        self.prefix = prefix
        self.postfix = postfix
        self.scale = scale if scale > 0 else 1

        self.choice_list: list[B_Prompt] = []
        if b_prompts is not None:
            for b_prompt in b_prompts:
                self.addChoice(b_prompt)
        
        self.choice_map: dict[str, B_Prompt] = {}
        self.choice_preset_map: dict[str, B_UI_Preset] = {}
        
        self.gr_dropdown: typing.Any = None
        self.gr_remove: typing.Any = None

        self.gr_buttons_container: typing.Any = None
        self.gr_buttons: list[typing.Any] = []
        
        self.b_prompt_ui = B_UI_Prompt(f"{self.name} (Prompt)")
    
    def init(self) -> None:
        # Self
        if self.sort_choices:
            self.choice_list = sorted(self.choice_list, key = lambda b_prompt: b_prompt.name)
        
        for b_prompt in self.choice_list:
            self.choice_map[b_prompt.name] = b_prompt
        
        #!
        for k in self.b_prompts_applied_default:
            b_prompt = self.choice_map[k]
            b_prompt.values.updateFromArgs(self.b_prompts_applied_default[k])
            B_Prompt_Map.update(b_prompt)
        
        # Prompt UI
        self.b_prompt_ui.init()
    
    def build(self) -> None:
        # Self
        with gr.Column(scale = self.scale):
            with gr.Row():
                self.gr_dropdown = gr.Dropdown(
                    label = self.name
                    , choices = [self._choice_empty] + list(map(lambda b_prompt: b_prompt.name, self.choice_list))
                    , multiselect = False
                    , value = self.choice.value
                    , allow_custom_value = False
                    , scale = self._ui_dropdown_scale
                )

                self.gr_remove = gr.Button(
                    value = "x"
                    , scale = self._ui_remove_scale
                    , size = "sm"
                    , min_width = self._ui_remove_width
                )

            # Prompt UI
            self.b_prompt_ui.build()

            self.gr_buttons_container = gr.Row(variant = "panel", visible = self.initButtonContainerVisible())
            with self.gr_buttons_container:
                for b_prompt in self.choice_list:
                    self.gr_buttons.append(
                        gr.Button(
                            value = b_prompt.name
                            , size = "sm"
                            , visible = B_Prompt_Map.isSelected(b_prompt)
                        )
                    )

        #! register on map
        B_UI_Map.add(self)
    
    def bind(self, gr_prompt: typing.Any, gr_prompt_negative: typing.Any) -> None:
        # Self
        def _onSelect(choice: str):
            self.choice.value = choice
            self.b_prompt_ui.b_prompt = self.choice_map.get(choice)
            self.b_prompt_ui.b_ui_preset = self.choice_preset_map.get(choice)
            return self.b_prompt_ui.getOutputUpdate()
        self.gr_dropdown.select(
            fn = _onSelect
            , inputs = self.gr_dropdown
            , outputs = self.b_prompt_ui.getOutput()
        )

        # - Remove All #! confirm
        def _removeAll():
            self.reset(True)
            return self.getOutputUpdate() + B_Prompt_Map.buildPromptUpdate()
        self.gr_remove.click(
            fn = _removeAll
            , outputs = self.getOutput() + [gr_prompt, gr_prompt_negative]
        )

        # - Update selected
        if len(self.gr_buttons) > 0:
            def _fnSelect(k: str):
                updates = _onSelect(k)
                return [self.choice.value] + updates
            for gr_button in self.gr_buttons:
                gr_button.click(
                    fn = _fnSelect
                    , inputs = gr_button
                    , outputs = [self.gr_dropdown] + self.b_prompt_ui.getOutput()
                )

        # Prompt UI
        ui_component_names_override: set[str] = set()
        for b_ui_preset in self.choice_preset_map.values():
            for k in b_ui_preset.mappings:
                ui_component_names_override.add(k)
        if len(ui_component_names_override) > 0:
            self.b_prompt_ui.ui_component_names_apply = list(ui_component_names_override)
        
        self.b_prompt_ui.outputs_extra = [self.gr_buttons_container] + self.gr_buttons
        self.b_prompt_ui.fn_updates_extra = self.getPromptButtonUpdates

        self.b_prompt_ui.bind(gr_prompt, gr_prompt_negative)
    
    def getGrForWebUI(self) -> list[typing.Any]:
        return [self.gr_dropdown, self.gr_remove] + self.gr_buttons + self.b_prompt_ui.getGrForWebUI()
    
    def reset(self, clear: bool = False) -> None:
        #!
        if not clear:
            self.choice.reset()
        else:
            self.choice.value = self._choice_empty

        for b_prompt in self.choice_list:
            if not clear:
                b_prompt.reset()
                b_prompt_value_args = self.b_prompts_applied_default.get(b_prompt.name)
                if b_prompt_value_args is not None:
                    b_prompt.values.updateFromArgs(b_prompt_value_args, True)
                B_Prompt_Map.update(b_prompt, b_prompt_value_args is None)
            else:
                b_prompt.clear()
                B_Prompt_Map.update(b_prompt, True)
        
        self.b_prompt_ui.b_prompt = None
        self.b_prompt_ui.b_ui_preset = None
    
    def update(self, inputValues: tuple) -> int:
        offset: int = 0

        self.choice.value = str(inputValues[offset])
        offset += 1

        offset += self.b_prompt_ui.update(inputValues[offset:])

        return offset
    
    def getInput(self) -> list[typing.Any]:
        return [self.gr_dropdown] + self.b_prompt_ui.getInput()
    
    def getOutput(self) -> list[typing.Any]:
        return [self.gr_dropdown, self.gr_buttons_container] + self.gr_buttons + self.b_prompt_ui.getOutput()
    
    def getOutputUpdate(self) -> list[typing.Any]:
        return [self.choice.value] + self.getPromptButtonUpdates() + self.b_prompt_ui.getOutputUpdate()
    
    def getPromptButtonUpdates(self) -> list:
        updates = [self.gr_buttons_container.update(visible = self.initButtonContainerVisible())]
        i: int = 0
        for b_prompt in self.choice_list:
            updates.append(self.gr_buttons[i].update(visible = B_Prompt_Map.isSelected(b_prompt)))
            i += 1
        return updates
    
    def initButtonContainerVisible(self):
        return any(map(lambda b_prompt: B_Prompt_Map.isSelected(b_prompt), self.choice_list)) #! optimize?
    
    def addChoice(self, item: B_Prompt):
        if len(item.values.prefix.value) == 0:
            item.values.prefix.reinit(self.prefix)
        if len(item.values.postfix.value) == 0:
            item.values.postfix.reinit(self.postfix)
        
        item.meta.name_visible = False
        item.meta.prompt_enable = False
        item.meta.prompt_negative_enable = False
        
        self.choice_list.append(item)
    
    def addChoicePresetMapping(self, target_name: str, target_args: dict[str, str]):
        b_prompt = self.choice_list[-1]
        preset = self.choice_preset_map.get(b_prompt.name, None)
        if preset is None:
            preset = B_UI_Preset(f"{self.name}_{b_prompt.name} (PRESET)", True)
            self.choice_preset_map[b_prompt.name] = preset
        preset.addMapping(target_name, target_args)
    
    def addChoices(self, args: dict[str, str]):
        b_prompt_list: list[B_Prompt] = None
        
        special_type = args.get("type", "")
        match special_type:
            case "COLOR":
                postfix = args.get(B_Prompt.Values.Keys.postfix, "")
                b_prompt_list = self._buildColorChoicesList(postfix)
            case "":
                printWarning(type(self), self.name, f"No CHOICES type specified")
            case _:
                printWarning(type(self), self.name, f"Invalid CHOICES type -> '{special_type}'")
        
        if b_prompt_list is not None:
            for b_prompt in b_prompt_list:
                self.addChoice(b_prompt)
    
    def applyPresetMapping(self, args: dict[str, str], additive: bool):
        valueMap = B_UI_Dropdown._fromArgsValue(args)
        for b_prompt in self.choice_map.values():
            b_prompt_value_args = valueMap.get(b_prompt.name)
            if b_prompt_value_args is not None:
                b_prompt.values.updateFromArgs(b_prompt_value_args, not additive)
            B_Prompt_Map.update(b_prompt, b_prompt_value_args is None)

class B_UI_Container(B_UI, ABC):
    def __init__(self, name: str, build_button_reset: bool = False, build_button_random: bool = False, children: list[B_UI] = None):
        super().__init__(name)

        self.build_button_reset = build_button_reset
        self.build_button_random = build_button_random

        self.children = children if children is not None else []

        self.gr_container: typing.Any = None
        self.gr_reset: typing.Any = None
        self.gr_random: typing.Any = None
    
    def init(self) -> None:
        for b_ui in self.children:
            b_ui.init()
    
    def build(self) -> None:
        self.gr_container = self.buildContainer()
        with self.gr_container:
            for b_ui in self.children:
                b_ui.build()
            
            if self.build_button_reset or self.build_button_random:
                def _buildReset():
                    self.gr_reset = gr.Button(f"Reset {self.name}")
                def _buildRandomize():
                    self.gr_random = gr.Button(f"Randomize {self.name}")

                B_UI_Separator._build()
                
                if self.build_button_reset and self.build_button_random:
                    with gr.Row():
                        _buildRandomize()
                        _buildReset()
                elif self.build_button_random:
                    _buildRandomize()
                else:
                    _buildReset()  
    
    def bind(self, gr_prompt: typing.Any, gr_prompt_negative: typing.Any) -> None:
        # Children
        for b_ui in self.children:
            b_ui.bind(gr_prompt, gr_prompt_negative)
        
        # Self
        # - Reset
        if self.build_button_reset:
            def _reset():
                for b_ui in self.children:
                    b_ui.reset()
                
                return B_Prompt_Map.buildPromptUpdate() + self.getOutputUpdate()
            self.gr_reset.click(
                fn = _reset
                , outputs = [gr_prompt, gr_prompt_negative] + self.getOutput()
            )
        
        #! - Randomize
    
    def getGrForWebUI(self) -> list[typing.Any]:
        gr_list: list[typing.Any] = []

        if self.build_button_random:
            gr_list.append(self.gr_random)
        if self.build_button_reset:
            gr_list.append(self.gr_reset)
        
        for b_ui in self.children:
            gr_list += b_ui.getGrForWebUI()
        return gr_list
    
    def reset(self, clear: bool = False) -> None:
        for b_ui in self.children:
            b_ui.reset(clear)
    
    def update(self, inputValues: tuple) -> int:
        offset: int = 0
        for b_ui in self.children:
            offset += b_ui.update(inputValues[offset:])
        return offset
    
    def getInput(self) -> list[typing.Any]:
        gr_inputs: list[typing.Any] = []
        for b_ui in self.children:
            gr_inputs += b_ui.getInput()
        return gr_inputs
    
    def getOutput(self) -> list[typing.Any]:
        gr_outputs: list[typing.Any] = []
        for b_ui in self.children:
            gr_outputs += b_ui.getOutput()
        return gr_outputs
    
    def getOutputUpdate(self) -> list[typing.Any]:
        gr_updates: list[typing.Any] = []
        for b_ui in self.children:
            gr_updates += b_ui.getOutputUpdate()
        return gr_updates
    
    def addChild(self, item: B_UI):
        self.children.append(item)
    
    @abstractmethod
    def buildContainer(self) -> typing.Any:
        pass

class B_UI_Container_Tab(B_UI_Container):
    @staticmethod
    def _fromArgs(args: dict[str, str], name: str = "Tab"):
        return B_UI_Container_Tab(
            name
            , bool(int(args.get("build_button_reset", 1)))
            , bool(int(args.get("build_button_random", 1)))
        )
    
    def __init__(self, name: str = "Tab", build_button_reset: bool = True, build_button_random: bool = True, children: list[B_UI] = None):
        super().__init__(name, build_button_reset, build_button_random, children)
    
    def buildContainer(self) -> typing.Any:
        return gr.Tab(self.name)

class B_UI_Container_Row(B_UI_Container):
    @staticmethod
    def _fromArgs(args: dict[str, str], name: str = "Row"):
        return B_UI_Container_Row(
            name
            , bool(int(args.get("build_button_reset", 0)))
            , bool(int(args.get("build_button_random", 0)))
        )
    
    def __init__(self, name: str = "Row", build_button_reset: bool = False, build_button_random: bool = False, children: list[B_UI] = None):
        super().__init__(name, build_button_reset, build_button_random, children)
    
    def buildContainer(self) -> typing.Any:
        return gr.Row()

class B_UI_Container_Column(B_UI_Container):
    @staticmethod
    def _fromArgs(args: dict[str, str], name: str = "Column"):
        return B_UI_Container_Column(
            name
            , int(args.get("scale", 1))
            , bool(int(args.get("build_button_reset", 0)))
            , bool(int(args.get("build_button_random", 0)))
        )
    
    def __init__(self, name: str = "Column", scale: int = 1, build_button_reset: bool = False, build_button_random: bool = False, children: list[B_UI] = None):
        super().__init__(name, build_button_reset, build_button_random, children)

        self.scale = scale
    
    def buildContainer(self) -> typing.Any:
        return gr.Column(scale = self.scale)

class B_UI_Container_Accordion(B_UI_Container):
    @staticmethod
    def _fromArgs(args: dict[str, str], name: str = "Accordion"):
        return B_UI_Container_Accordion(
            name
            , bool(int(args.get("build_button_reset", 0)))
            , bool(int(args.get("build_button_random", 0)))
        )
    
    def __init__(self, name: str = "Accordion", build_button_reset: bool = False, build_button_random: bool = False, children: list[B_UI] = None):
        super().__init__(name, build_button_reset, build_button_random, children)
    
    def buildContainer(self) -> typing.Any:
        return gr.Accordion(self.name)

class B_UI_Container_Group(B_UI_Container):
    @staticmethod
    def _fromArgs(args: dict[str, str], name: str = "Group"):
        return B_UI_Container_Group(
            name
            , bool(int(args.get("build_button_reset", 0)))
            , bool(int(args.get("build_button_random", 0)))
        )
    
    def __init__(self, name: str = "Group", build_button_reset: bool = False, build_button_random: bool = False, children: list[B_UI] = None):
        super().__init__(name, build_button_reset, build_button_random, children)
    
    def buildContainer(self) -> typing.Any:
        return gr.Group()

class B_Prompt_Map():
    _map: dict[str, tuple[B_Prompt, bool]] = {}
    
    @staticmethod
    def add(b_prompt: B_Prompt):
        if b_prompt.name in B_Prompt_Map._map:
            printWarning(B_Prompt_Map, "add()", f"Duplicate name -> '{b_prompt.name}'")
        B_Prompt_Map._map[b_prompt.name] = b_prompt, False
    
    @staticmethod
    def update(b_prompt: B_Prompt | None, remove: bool = False):
        if b_prompt is None:
            return
        B_Prompt_Map._map[b_prompt.name] = b_prompt, not remove
    
    @staticmethod
    def get(b_prompt_name: str) -> B_Prompt | None:
        mapping = B_Prompt_Map._map.get(b_prompt_name)
        return mapping[0] if mapping is not None else None
    
    @staticmethod
    def isSelected(b_prompt: B_Prompt | None):
        return b_prompt is not None and B_Prompt_Map._map[b_prompt.name][1]
    
    @staticmethod
    def buildPromptUpdate() -> list[str]:
        prompt: str = ""
        prompt_negative: str = ""

        for b_prompt, selected in B_Prompt_Map._map.values():
            if not selected:
                continue
            
            b_prompt_positive, b_prompt_negative = b_prompt.build()
            prompt = B_Prompt.Fn.promptAdded(prompt, b_prompt_positive)
            prompt_negative = B_Prompt.Fn.promptAdded(prompt_negative, b_prompt_negative)
        
        return [prompt, prompt_negative]

class B_UI_Map():
    _map: dict[str, B_UI] = {}
    _inputs: list[typing.Any] = []
    _outputs: list[typing.Any] = []
    
    @staticmethod
    def add(b_ui: B_UI):
        if b_ui.name in B_UI_Map._map:
            printWarning(B_UI_Map, "add()", f"Duplicate name -> '{b_ui.name}'")
        B_UI_Map._map[b_ui.name] = b_ui
        B_UI_Map._inputs += b_ui.getInput()
        B_UI_Map._outputs += b_ui.getOutput()
    
    @staticmethod
    def getInput():
        inputs = []
        for b_ui in B_UI_Map._map.values():
            inputs += b_ui.getInput()
        return inputs
    
    @staticmethod
    def getOutput():
        outputs = []
        for b_ui in B_UI_Map._map.values():
            outputs += b_ui.getOutput()
        return outputs
    
    @staticmethod
    def consumeInputValues(inputValues: tuple):
        offset = 0
        for b_ui in B_UI_Map._map.values():
            offset += b_ui.update(inputValues[offset:])
    
    @staticmethod
    def getOutputUpdates():
        updates = []
        for b_ui in B_UI_Map._map.values():
            updates += b_ui.getOutputUpdate()
        return updates

class B_UI_Master():
    @staticmethod
    def readLine(l: str) -> tuple[str, str, dict[str, str]]:
        #! TODO: Fix empty str l_name
        l = l.strip()
        
        l_type: str = l
        l_name: str = None
        l_args: dict[str, str] = {}
        
        if len(l) > 0:
            index = l.find(" ")
            if index != -1:
                l_type = l[:index]
            
            l = l[len(l_type) + 1:]
            
            l_arg_index = l.find("--")
            if l_arg_index == -1:
                l_name = l
            elif l_arg_index > 0:
                l_name = l[:l_arg_index - 1 if l_arg_index > -1 else len(l)]
                l = l[len(l_name) + 1:]
            
            l_args = {}
            for l_arg in l.split("--")[1:]:
                l_arg_name = l_arg[:l_arg.index(" ")]
                l_arg_value = l_arg[len(l_arg_name) + 1:].strip()
                l_args[l_arg_name] = l_arg_value
            
        return l_type, l_name, l_args

    def __init__(self, layout: list[B_UI] = None):
        self.path_script_config = os.path.join(b_path_base, b_folder_name_scripts, b_folder_name_script_config)
        self.path_layout = os.path.join(self.path_script_config, b_file_name_layout)
        self.path_presets = os.path.join(self.path_script_config, b_file_name_presets)

        self.layout = (layout if layout is not None else []) + self.parseLayout()
        self.presets = self.parsePresets()

        for b_ui in self.layout:
            b_ui.init()
        
        #! validate

        self.gr_prompt: typing.Any = None
        self.gr_prompt_negative: typing.Any = None
        self.gr_apply: typing.Any = None
        self.gr_remove: typing.Any = None
        self.gr_clear: typing.Any = None
        self.gr_reset: typing.Any = None
        self.gr_clear_config: typing.Any
    
    def parseLayout(self) -> list[B_UI]:
        layout: list[B_UI] = []
        
        stack_containers: list[B_UI_Container] = []
        stack_dropdowns: list[B_UI_Dropdown] = []
        dropdown_choice_has_preset: bool = False

        skip = 0

        def _buildPrompt(item: B_Prompt):
            if len(stack_dropdowns) > 0:
                stack_dropdowns[-1].addChoice(item)
                return
            
            _build(B_UI_Prompt(item.name, item))
        
        def _build(item: B_UI):
            if len(stack_containers) > 0:
                stack_containers[-1].addChild(item)
                return
            
            layout.append(item)
        
        with open(self.path_layout) as file_layout:
            line_number: int = 0

            for l in file_layout:
                line_number += 1

                if l.lstrip().startswith("#"):
                    print(f"# LAYOUT - commented out line @{line_number}")
                    continue
                
                l_type, l_name, l_args = self.readLine(l)
                
                if len(l_type) == 0:
                    continue
                    
                if l_type == ".":
                    break
                
                if l_type == "END":
                    if skip == 0:
                        if dropdown_choice_has_preset:
                            dropdown_choice_has_preset = False
                            continue

                        if len(stack_dropdowns) > 0:
                            item_dropdown = stack_dropdowns.pop()
                            _build(item_dropdown)
                            continue
                        
                        if len(stack_containers) > 0:
                            item_container = stack_containers.pop()
                            _build(item_container)
                            continue

                        continue
                    
                    skip -= 1
                    
                    continue
                
                ignore: bool = skip > 0
                if l_args.get("x", "") == "1":
                    if not ignore:
                        ignore = b_tagged_ignore
                
                match l_type:
                    case "SINGLE":
                        if ignore:
                            continue

                        _buildPrompt(B_Prompt_Single._fromArgs(l_name, l_args))
                    
                    case "DUAL":
                        if ignore:
                            continue

                        _buildPrompt(B_Prompt_Dual._fromArgs(l_name, l_args))
                    
                    case "EDIT":
                        if ignore:
                            continue
                        
                        _buildPrompt(B_Prompt_Edit._fromArgs(l_name, l_args))
                    
                    case "EDIT_LINK":
                        if ignore:
                            continue
                        
                        _buildPrompt(B_Prompt_Edit_Link._fromArgs(l_name, l_args))
                    
                    case "SELECT":
                        if ignore:
                            skip += 1
                            continue
                        
                        stack_dropdowns.append(B_UI_Dropdown._fromArgs(l_args, l_name))
                    
                    case "CHOICES":
                        if ignore:
                            continue

                        stack_dropdowns[-1].addChoices(l_args) #!
                    
                    case "SET":
                        dropdown_choice_has_preset = True
                        stack_dropdowns[-1].addChoicePresetMapping(l_name, l_args)
                    
                    case "GROUP":
                        if ignore:
                            skip += 1
                            continue

                        stack_containers.append(B_UI_Container_Group._fromArgs(l_args))
                    
                    case "TAB":
                        if ignore:
                            skip += 1
                            continue
                        
                        stack_containers.append(B_UI_Container_Tab._fromArgs(l_args, l_name))
                    
                    case "ROW":
                        if ignore:
                            skip += 1
                            continue
                        
                        stack_containers.append(B_UI_Container_Row._fromArgs(l_args))
                    
                    case "COLUMN":
                        if ignore:
                            skip += 1
                            continue
                        
                        stack_containers.append(B_UI_Container_Column._fromArgs(l_args))
                    
                    case "ACCORDION":
                        if ignore:
                            skip += 1
                            continue
                        
                        stack_containers.append(B_UI_Container_Accordion._fromArgs(l_args, l_name))
                    
                    case "SEPARATOR":
                        if ignore:
                            continue
                        
                        _build(B_UI_Separator._fromArgs(l_args))

                    case _:
                        printWarning(type(self), "parseLayout()", f"Invalid layout type -> '{l_type}'")
        
        return layout
    
    def parsePresets(self) -> list[B_UI_Preset]:
        presets: list[B_UI_Preset] = []
        
        preset_current: B_UI_Preset = None
        
        with open(self.path_presets) as file_presets:
            line_number: int = 0

            for l in file_presets:
                line_number += 1

                if l.lstrip().startswith("#"):
                    print(f"# PRESETS - commented out line @{line_number}")
                    continue

                l_type, l_name, l_args = self.readLine(l)
                
                if len(l_type) == 0:
                    continue
                    
                if l_type == ".":
                    break
                
                if l_type == "END":
                    presets.append(preset_current)
                    preset_current = None
                    continue
                
                match l_type:
                    case "PRESET":
                        preset_current = B_UI_Preset._fromArgs(l_args, l_name)
                    
                    case "SET":
                        preset_current.addMapping(l_name, l_args)
                    
                    case _:
                        printWarning(type(self), "parsePresets()", f"Invalid preset type -> '{l_type}'")
        
        return presets
    
    def build(self) -> list[typing.Any]:
        """Builds layout and returns Gradio elements for WebUI"""
        # PRESETS
        B_UI_Separator._build()
        with gr.Accordion("Presets", open = False):
            i = 0
            for preset in self.presets:
                preset.build()
                i += 1
                if i < len(self.presets):
                    B_UI_Separator._build()
        
        # LAYOUT
        B_UI_Separator._build()
        for b_ui in self.layout:
            b_ui.build()
        
        # MAIN
        B_UI_Separator._build()
        prompt = B_Prompt_Map.buildPromptUpdate()
        self.gr_prompt = gr.Textbox(label = "Final Prompt", value = prompt[0])
        self.gr_prompt_negative = gr.Textbox(label = "Final Negative Prompt", value = prompt[1])
        B_UI_Separator._build()
        with gr.Row():
            self.gr_apply = gr.Button("Apply All")
            self.gr_remove = gr.Button("Remove All")
        B_UI_Separator._build()
        with gr.Row():
            self.gr_clear = gr.Button("Clear All")
            self.gr_reset = gr.Button("Reset All")
        
        # EXTRAS
        B_UI_Separator._build()
        with gr.Accordion("Settings", open = False):
            self.gr_clear_config = gr.Button("Clear config")
        
        # (Return)
        gr_list: list[typing.Any] = [
            self.gr_prompt
            , self.gr_prompt_negative
            , self.gr_apply
            , self.gr_remove
            , self.gr_clear
            , self.gr_reset
            , self.gr_clear_config
        ]

        for b_ui in self.layout:
            gr_list += b_ui.getGrForWebUI()
        
        for preset in self.presets:
            gr_list += preset.getGrForWebUI()
        
        return gr_list
    
    def bind(self) -> None:
        # - Presets
        for preset in self.presets:
            preset.bind(self.gr_prompt, self.gr_prompt_negative)

        # - Layout
        for b_ui in self.layout:
            b_ui.bind(self.gr_prompt, self.gr_prompt_negative)
        
        # - Self
        inputs = B_UI_Map._inputs
        outputs = B_UI_Map._outputs + [self.gr_prompt, self.gr_prompt_negative]

        #!
        def _fnApply(*inputValues):
            B_UI_Map.consumeInputValues(inputValues)
            for b_ui in B_UI_Map._map.values():
                if type(b_ui) is B_UI_Prompt:
                    b_ui.apply()
            return B_UI_Map.getOutputUpdates() + B_Prompt_Map.buildPromptUpdate()
        self.gr_apply.click(
            fn = _fnApply
            , inputs = inputs
            , outputs = outputs
        )

        #!
        def _fnRemove(*inputValues):
            B_UI_Map.consumeInputValues(inputValues)
            for b_prompt, selected in B_Prompt_Map._map.values():
                B_Prompt_Map.update(b_prompt, True)
            return B_UI_Map.getOutputUpdates()+ B_Prompt_Map.buildPromptUpdate()
        self.gr_remove.click(
            fn = _fnRemove
            , inputs = inputs
            , outputs = outputs
        )

        #!
        def _fnReset():
            for b_ui in B_UI_Map._map.values():
                b_ui.reset()
            return B_UI_Map.getOutputUpdates() + B_Prompt_Map.buildPromptUpdate()
        self.gr_reset.click(
            fn = _fnReset
            , outputs = outputs
        )

        #!
        def _fnClear():
            for b_ui in B_UI_Map._map.values():
                b_ui.reset(True)
            return B_UI_Map.getOutputUpdates() + B_Prompt_Map.buildPromptUpdate()
        self.gr_clear.click(
            fn = _fnClear
            , outputs = outputs
        )
        
        #! Would be better if the original config file dump function is used somehow:
        def _fnClearConfigFile():
            path = os.path.join(b_path_base, b_file_name_config)
            with open(path, "r+", encoding = "utf-8") as file_config:
                config: dict[str, typing.Any] = json.load(file_config)
                
                config_keys = filter(lambda k: k.find(b_folder_name_script_config) == -1, config.keys())

                config_new: dict[str, typing.Any] = {}
                for k in config_keys:
                    config_new[k] = config[k]
                
                file_config.seek(0)
                json.dump(config_new, file_config, indent = 4)
                file_config.truncate()
        self.gr_clear_config.click(fn = _fnClearConfigFile)

#: Webui script
class Script(scripts.Script):
    b_ui_master = B_UI_Master()
    
    def title(self):
        return "B Prompt Builder"

    def show(self, is_img2img):
        return not is_img2img
    
    def ui(self, is_img2img):
        gr_list = self.b_ui_master.build()
        self.b_ui_master.bind()
        return gr_list

    def run(
            self
            , p
            , prompt: str
            , prompt_negative: str
            , *outputValues
        ):

        p.prompt = prompt
        p.negative_prompt = prompt_negative
        
        proc = process_images(p)
        
        return proc
