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

def printWarning(obj: object, name: str, message: str) -> None:
    print(f"VALIDATE/{obj.__class__.__name__}/{name}: {message}")

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

        def __init__(
                self
                , prompt_enable: bool = False
                , negative_enable: bool = False
                , prompt_negative_enable: bool = False
                , prompt_edit_enable: bool = False
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
            self.prompt_enable = prompt_enable
            self.negative_enable = negative_enable
            self.prompt_negative_enable = prompt_negative_enable
            self.prompt_edit_enable = prompt_edit_enable

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
        
        #!
        def clear(self):
            self.prompt.value = B_Prompt.Values.Defaults.prompt
            self.emphasis.value = B_Prompt.Values.Defaults.emphasis
            self.negative.value = B_Prompt.Values.Defaults.negative
            self.prompt_negative.value = B_Prompt.Values.Defaults.prompt
            self.emphasis_negative.value = B_Prompt.Values.Defaults.emphasis
            self.edit.value = B_Prompt.Values.Defaults.edit
            # self.prompt_a.value = B_Prompt.Values.Defaults.prompt
            # self.prompt_b.value = B_Prompt.Values.Defaults.prompt
            # self.prefix.value = B_Prompt.Values.Defaults.prompt
            # self.postfix.value = B_Prompt.Values.Defaults.prompt
        
        def reset(self):
            self.prompt.reset()
            self.emphasis.reset()
            self.negative.reset()
            self.prompt_negative.reset()
            self.emphasis_negative.reset()
            self.edit.reset()
            # self.prompt_a.reset()
            # self.prompt_b.reset()
            # self.prefix.reset()
            # self.postfix.reset()
    
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
    
    def __init__(self, name: str, values: Values):
        self.name = name
        self.values = values

        B_Prompt_Map.add(self)
    
    def reset(self):
        self.values.reset()
    
    def clear(self):
        self.values.clear()
    
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
            , B_Prompt.Values(
                prompt_enable = True
                , negative_enable = True
                , prompt = prompt
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
            , B_Prompt.Values(
                prompt_enable = True
                , prompt_negative_enable = True
                , prompt = prompt
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
            , B_Prompt.Values(
                negative_enable = True
                , prompt_edit_enable = True
                , negative = negative
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
            , B_Prompt.Values(
                negative_enable = True
                , negative = negative
                , prompt_a = prompt_a
                , prompt_b = prompt_b
            )
        )

        self.link_name = link_name
    
    def build(self) -> tuple[str, str]:
        b_prompt_link = B_Prompt_Map.get(self.link_name)
        if b_prompt_link is None:
            printWarning(self, f"{self.name} - build()", f"Linked prompt not found -> '{self.link_name}'")
        
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
    def reset(self) -> None:
        pass

    @abstractmethod
    def clear(self) -> None:
        pass

    @abstractmethod
    def update(self, *inputValues) -> int:
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

    @staticmethod
    def _decideValue(source: B_Value, target: B_Value):
        if target.value is None:
            source.reset() #!
        else:
            source.value = target.value
    
    @staticmethod
    def _valuesFromArgs(args: dict[str, str]) -> B_Prompt.Values | None:
        if len(args) == 0:
            return None
        
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
    
    @staticmethod
    def _apply(b_prompt: B_Prompt, values: B_Prompt.Values | None, additive: bool):
        if values is None:
            if not additive:
                B_Prompt_Map.update(b_prompt, True)
        else:
            B_UI_Preset._decideValue(b_prompt.values.prompt, values.prompt)
            B_UI_Preset._decideValue(b_prompt.values.emphasis, values.emphasis)
            B_UI_Preset._decideValue(b_prompt.values.prompt_negative, values.prompt_negative)
            B_UI_Preset._decideValue(b_prompt.values.emphasis_negative, values.emphasis_negative)
            B_UI_Preset._decideValue(b_prompt.values.negative, values.negative)
            B_UI_Preset._decideValue(b_prompt.values.edit, values.edit)
            B_Prompt_Map.update(b_prompt)
    
    @staticmethod
    def _applyFromArgs(b_prompt: B_Prompt, args: dict[str, str], additive: bool):
        B_UI_Preset._apply(b_prompt, B_UI_Preset._valuesFromArgs(args), additive)
    
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
        #! dirty?
        def _apply():
            self.apply()
            updates: list = []
            for b_ui in B_UI_Map._map.values():
                updates += b_ui.getOutputUpdate()
            return updates + B_Prompt_Map.buildPromptUpdate()
        outputs: list = []
        for b_ui in B_UI_Map._map.values():
            outputs += b_ui.getOutput()
        self.gr_button.click(
            fn = _apply
            , outputs = outputs + [gr_prompt, gr_prompt_negative]
        )

    def getGrForWebUI(self) -> list[typing.Any]:
        return [self.gr_button]
    
    def reset(self) -> None:
        pass

    def clear(self) -> None:
        pass

    def update(self, *inputValues) -> int:
        return 0
    
    def getInput(self) -> list[typing.Any]:
        return []
    
    def getOutput(self) -> list[typing.Any]:
        return []
    
    def getOutputUpdate(self) -> list[typing.Any]:
        return []
    
    def addMapping(self, name: str, args: dict[str, str]):
        if name in self.mappings:
            printWarning(self, f"{self.name} - addMapping()", f"Duplicate name -> '{name}'")
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
                    B_UI_Map._map[k].clear() #!

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
    
    def reset(self) -> None:
        pass

    def clear(self) -> None:
        pass

    def update(self, *inputValues) -> int:
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

        #!
        if self.b_prompt is not None:
            B_UI_Map.add(self)
    
    def init(self) -> None:
        pass

    def build(self) -> None:
        values, name, visible, visible_button_remove = self.getUpdateValues()
        
        self.gr_container = gr.Column(variant = "panel", visible = visible)
        with self.gr_container:
            self.gr_markdown = gr.Markdown(value = name)

            self.gr_prompt_container = gr.Row(visible = values.prompt_enable)
            with self.gr_prompt_container:
                self.gr_prompt = gr.Textbox(
                    label = "Prompt"
                    , value = values.prompt.value
                    , scale = B_UI_Prompt._prompt_scale
                )
                self.gr_emphasis = gr.Number(
                    label = "Emphasis"
                    , value = values.emphasis.value
                    , minimum = B_Prompt.Values.Defaults.emphasis_min
                    , step = B_Prompt.Values.Defaults.emphasis_step
                    , scale = B_UI_Prompt._emphasis_scale
                )
            
            self.gr_prompt_negative_container = gr.Row(visible = values.prompt_negative_enable)
            with self.gr_prompt_negative_container:
                self.gr_prompt_negative = gr.Textbox(
                    label = "Prompt (N)"
                    , value = values.prompt_negative.value
                    , scale = B_UI_Prompt._prompt_scale
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
                , visible = values.prompt_edit_enable
            )
            
            self.gr_negative = gr.Checkbox(
                label = "Negative?"
                , value = values.negative.value
                , visible = values.negative_enable
            )

            B_UI_Separator._build()

            with gr.Row():
                self.gr_button_apply = gr.Button(
                    value = "Apply"
                )
                self.gr_button_remove = gr.Button(
                    value = "Remove"
                    , visible = visible_button_remove
                )
    
    def bind(self, gr_prompt: typing.Any, gr_prompt_negative: typing.Any) -> None:
        # Add/update #! presets dirty...
        def _fnApply(*inputValues):
            self.update(*inputValues)
            updates: list = []
            for b_ui in B_UI_Map._map.values():
                updates += b_ui.getOutputUpdate()
            #return [self.gr_button_remove.update(visible = True)] + b_prompt_map.buildPromptUpdate()
            return updates + B_Prompt_Map.buildPromptUpdate()
        outputs: list = []
        for b_ui in B_UI_Map._map.values():
            outputs += b_ui.getOutput()
        applyArgs = {
            "fn": _fnApply
            , "inputs": self.getInput()
            , "outputs": outputs + [gr_prompt, gr_prompt_negative]
        }
        self.gr_prompt.submit(**applyArgs)
        self.gr_emphasis.submit(**applyArgs)
        self.gr_prompt_negative.submit(**applyArgs)
        self.gr_emphasis_negative.submit(**applyArgs)
        self.gr_button_apply.click(**applyArgs)

        # Remove
        def _fnRemove():
            B_Prompt_Map.update(self.b_prompt, True)
            return [self.gr_button_remove.update(visible = False)] + B_Prompt_Map.buildPromptUpdate()
        self.gr_button_remove.click(
            fn = _fnRemove
            , outputs = [self.gr_button_remove, gr_prompt, gr_prompt_negative]
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
    
    def reset(self) -> None:
        self.b_prompt.reset()
        B_Prompt_Map.update(self.b_prompt, True)
    
    def clear(self) -> None:
        self.b_prompt.clear()
        B_Prompt_Map.update(self.b_prompt, True)
    
    def update(self, *inputValues) -> int:
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
            B_Prompt_Map.update(self.b_prompt)
        
        if self.b_ui_preset is not None:
            self.b_ui_preset.apply()

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
        values, name, visible, visible_button_remove = self.getUpdateValues()

        return [
            self.gr_container.update(visible = visible)
            , self.gr_markdown.update(value = name)
            , self.gr_prompt_container.update(visible = values.prompt_enable)
            , values.prompt.value
            , values.emphasis.value
            , self.gr_prompt_negative_container.update(visible = values.prompt_negative_enable)
            , values.prompt_negative.value
            , values.emphasis_negative.value
            , self.gr_slider.update(visible = values.prompt_edit_enable, value = values.edit.value, step = B_Prompt.Values.Defaults.edit_step)
            , self.gr_negative.update(visible = values.negative_enable, value = values.negative.value)
            , self.gr_button_remove.update(visible = visible_button_remove)
        ]
    
    def getUpdateValues(self) -> tuple[B_Prompt.Values, str, bool, bool]:
        values = self.b_prompt.values if self.b_prompt is not None else B_Prompt.Values()
        name = f"**{self.b_prompt.name if self.b_prompt is not None else self.name}**"
        visible = self.b_prompt is not None
        visible_button_remove = B_Prompt_Map.isSelected(self.b_prompt)
        return values, name, visible, visible_button_remove
    
    def applyPresetMapping(self, args: dict[str, str], additive: bool):
        B_UI_Preset._applyFromArgs(self.b_prompt, args, additive)

class B_UI_Dropdown(B_UI):
    _choice_empty: str = "-"

    @staticmethod
    def _fromArgs(args: dict[str, str], name: str = "Dropdown"):
        return B_UI_Dropdown(
            name
            , bool(int(args.get("sort", 1)))
            , B_UI_Dropdown._fromArgsValue(args)
            , args.get(B_Prompt.Values.Keys.prefix, B_Prompt.Values.Defaults.prompt)
            , args.get(B_Prompt.Values.Keys.postfix, B_Prompt.Values.Defaults.prompt)
        )
    
    @staticmethod
    def _fromArgsValue(args: dict[str, str]) -> list[str]:
        return list(filter(lambda v: len(v) > 0, map(lambda v: v.strip(), args.get("v", "").split(","))))
    
    @staticmethod
    def _buildColorChoicesList(postfix: str = "") -> list[B_Prompt_Single]:
        return list(map(
            lambda text: B_Prompt_Single(
                text
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
            , b_prompts_applied_default: list[str] = None
            , prefix = B_Prompt.Values.Defaults.prompt
            , postfix = B_Prompt.Values.Defaults.prompt
            , b_prompts: list[B_Prompt] = None
        ):
        super().__init__(name)

        self.choice = B_UI_Dropdown._choice_empty
        self.sort_choices = sort_choices
        self.b_prompts_applied_default = b_prompts_applied_default if b_prompts_applied_default is not None else []
        self.prefix = prefix
        self.postfix = postfix

        self.choice_list: list[B_Prompt] = []
        if b_prompts is not None:
            for b_prompt in b_prompts:
                self.addChoice(b_prompt)
        
        self.choice_map: dict[str, B_Prompt] = {}
        self.choice_preset_map: dict[str, B_UI_Preset] = {}
        
        self.gr_dropdown: typing.Any = None
        
        self.b_prompt_ui = B_UI_Prompt(f"{self.name} (Prompt)")

        B_UI_Map.add(self)
    
    def init(self) -> None:
        # Self
        if self.sort_choices:
            self.choice_list = sorted(self.choice_list, key = lambda b_prompt: b_prompt.name)
        
        for b_prompt in self.choice_list:
            self.choice_map[b_prompt.name] = b_prompt
        
        #!
        for k in self.b_prompts_applied_default:
            B_Prompt_Map.update(self.choice_map[k])
        
        # Prompt UI
        self.b_prompt_ui.init()
    
    def build(self) -> None:
        # Self
        self.gr_dropdown = gr.Dropdown(
            label = self.name
            , choices = [self._choice_empty] + list(map(lambda b_prompt: b_prompt.name, self.choice_list))
            , multiselect = False
            , value = self.choice
            , allow_custom_value = False
        )

        # Prompt UI
        self.b_prompt_ui.build()
    
    def bind(self, gr_prompt: typing.Any, gr_prompt_negative: typing.Any) -> None:
        # Self
        def _onSelect(choice: str):
            self.b_prompt_ui.b_prompt = self.choice_map.get(choice)
            self.b_prompt_ui.b_ui_preset = self.choice_preset_map.get(choice)
            return self.b_prompt_ui.getOutputUpdate()
        self.gr_dropdown.select(
            fn = _onSelect
            , inputs = self.gr_dropdown
            , outputs = self.b_prompt_ui.getOutput()
        )

        # Prompt UI
        self.b_prompt_ui.bind(gr_prompt, gr_prompt_negative)
    
    def getGrForWebUI(self) -> list[typing.Any]:
        return [self.gr_dropdown] + self.b_prompt_ui.getGrForWebUI()
    
    def reset(self) -> None:
        self.choice = B_UI_Dropdown._choice_empty

        for b_prompt in self.choice_list:
            b_prompt.reset()
            B_Prompt_Map.update(b_prompt, True)
        
        self.b_prompt_ui.b_prompt = None
        self.b_prompt_ui.b_ui_preset = None

        #!
        for k in self.b_prompts_applied_default:
            B_Prompt_Map.update(self.choice_map[k])
    
    def clear(self) -> None:
        self.choice = B_UI_Dropdown._choice_empty

        for b_prompt in self.choice_list:
            b_prompt.clear()
            B_Prompt_Map.update(b_prompt, True)
        
        self.b_prompt_ui.b_prompt = None
        self.b_prompt_ui.b_ui_preset = None
    
    def update(self, *inputValues) -> int:
        offset: int = 0

        self.choice = str(inputValues[offset])
        offset += 1

        offset += self.b_prompt_ui.update(*inputValues[offset:])

        return offset
    
    def getInput(self) -> list[typing.Any]:
        return [self.gr_dropdown] + self.b_prompt_ui.getInput()
    
    def getOutput(self) -> list[typing.Any]:
        return [self.gr_dropdown] + self.b_prompt_ui.getOutput()
    
    def getOutputUpdate(self) -> list[typing.Any]:
        return [self.choice] + self.b_prompt_ui.getOutputUpdate()
    
    def addChoice(self, item: B_Prompt):
        if len(item.values.prefix.value) == 0:
            item.values.prefix.reinit(self.prefix)
        if len(item.values.postfix.value) == 0:
            item.values.postfix.reinit(self.postfix)
        
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
            case _:
                print(f"WARNING: Invalid CHOICES type in {self.name} -> {special_type}")
        
        if b_prompt_list is not None:
            for b_prompt in b_prompt_list:
                self.addChoice(b_prompt)
    
    def applyPresetMapping(self, args: dict[str, str], additive: bool):
        choices = B_UI_Dropdown._fromArgsValue(args)
        if additive:
            for k in choices:
                B_UI_Preset._apply(self.choice_map[k], self.choice_map[k].values, additive)
        else:
            for b_prompt in self.choice_map.values():
                B_UI_Preset._apply(b_prompt, b_prompt.values if b_prompt.name in choices else None, additive)

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
    
    def reset(self) -> None:
        for b_ui in self.children:
            b_ui.reset()
    
    def clear(self) -> None:
        for b_ui in self.children:
            b_ui.clear()
    
    def update(self, *inputValues) -> int:
        offset: int = 0
        for b_ui in self.children:
            offset += b_ui.update(*inputValues[offset:])
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
            printWarning("B_Prompt_Map", "add()", f"Duplicate name -> '{b_prompt.name}'")
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
    
    @staticmethod
    def add(b_ui: B_UI):
        if b_ui.name in B_UI_Map._map:
            printWarning("B_UI_Map", "add()", f"Duplicate name -> '{b_ui.name}'")
        B_UI_Map._map[b_ui.name] = b_ui

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
                        printWarning(self, "parseLayout()", f"Invalid layout type -> '{l_type}'")
        
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
                        printWarning(self, "parsePresets()", f"Invalid preset type -> '{l_type}'")
        
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
        prompt = B_Prompt_Map.buildPromptUpdate()
        B_UI_Separator._build()
        self.gr_prompt = gr.Textbox(label = "Final Prompt", value = prompt[0])
        self.gr_prompt_negative = gr.Textbox(label = "Final Negative Prompt", value = prompt[1])
        B_UI_Separator._build()
        with gr.Row():
            self.gr_apply = gr.Button("Apply All")
            self.gr_reset = gr.Button("Reset All")
        
        # EXTRAS
        B_UI_Separator._build()
        with gr.Accordion("Settings", open = False):
            self.gr_clear_config = gr.Button("Clear config")
        
        # Bind
        # - Presets
        for preset in self.presets:
            preset.bind(self.gr_prompt, self.gr_prompt_negative)

        # - Layout
        for b_ui in self.layout:
            b_ui.bind(self.gr_prompt, self.gr_prompt_negative)
        
        # - Self
        gr_inputs: list[typing.Any] = []
        gr_outputs: list[typing.Any] = []
        for b_ui in self.layout:
            gr_inputs += b_ui.getInput()
            gr_outputs += b_ui.getOutput()
        
        self.gr_apply.click(
            fn = self.apply
            , inputs = gr_inputs
            , outputs = [self.gr_prompt, self.gr_prompt_negative] + gr_outputs
        )

        self.gr_reset.click(
            fn = self.reset
            , outputs = [self.gr_prompt, self.gr_prompt_negative] + gr_outputs
        )
        
        self.gr_clear_config.click(fn = self.clearConfigFile)
        
        # (Return)
        gr_list: list[typing.Any] = [
            self.gr_prompt
            , self.gr_prompt_negative
            , self.gr_apply
            , self.gr_reset
            , self.gr_clear_config
        ]

        for b_ui in self.layout:
            gr_list += b_ui.getGrForWebUI()
        
        for preset in self.presets:
            gr_list += preset.getGrForWebUI()
        
        return gr_list
    
    def apply(self, *inputValues) -> list[typing.Any]:
        offset: int = 0
        for b_ui in self.layout:
            offset += b_ui.update(*inputValues[offset:])
        
        #! repeating
        gr_updates: list[typing.Any] = B_Prompt_Map.buildPromptUpdate()
        for b_ui in self.layout:
            gr_updates += b_ui.getOutputUpdate()
        
        return gr_updates
    
    def reset(self) -> list[typing.Any]:
        for b_ui in self.layout:
            b_ui.reset()
        
        #! repeating ^
        gr_updates: list[typing.Any] = B_Prompt_Map.buildPromptUpdate()
        for b_ui in self.layout:
            gr_updates += b_ui.getOutputUpdate()
        
        return gr_updates
    
    #! Would be better if the original config file dump function is used somehow:
    def clearConfigFile(self):
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

#: Webui script
class Script(scripts.Script):
    b_ui_master = B_UI_Master()
    
    def title(self):
        return "B Prompt Builder"

    def show(self, is_img2img):
        return not is_img2img
    
    def ui(self, is_img2img):
        return self.b_ui_master.build()

    def run(
            self
            , p
            , prompt: str
            , prompt_negative: str
            , *outputValues
        ):
        # i = 0
        
        # for preset in self.bUiMap.presets:
        #     i += preset.consumeOutputs(*outputValues[i:])
        
        # for x in self.bUiMap.layout:
        #     i += x.consumeOutputs(*outputValues[i:])
        #     x.handlePrompt(p, self.bUiMap.map)

        p.prompt = prompt
        p.negative_prompt = prompt_negative
        
        proc = process_images(p)
        
        return proc
