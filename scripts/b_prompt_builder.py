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

def printWarning(component: str, name: str, message: str) -> None:
    print(f"VALIDATE/{component}/{name} -> {message}")

class B_Value():
    def __init__(self, value_default):
        self.value_default = value_default
        self.value = self.buildDefaultValue()
    
    def buildDefaultValue(self):
        return self.value_default
    
    def reset(self):
        self.value = self.buildDefaultValue()

class B_Prompt(ABC):
    class Values():
        class Defaults():
            prompt_value: str = ""
            strength_value: float = 1
            negative_value: bool = False

            strength_min: float = 0
            strength_step: float = 0.1

            range_min: int = 0
            range_max: int = 100

        def __init__(
                self
                , prompt_enable: bool = False
                , negative_enable: bool = False
                , prompt_negative_enable: bool = False
                , prompt_range_enable: bool = False
                , prompt_value: str = Defaults.prompt_value
                , strength_value: float = Defaults.strength_value
                , negative_value: bool = Defaults.negative_value
                , prompt_negative_value: str = Defaults.prompt_value
                , strength_negative_value: float = Defaults.strength_value
                , prompt_range_a: str = Defaults.prompt_value
                , prompt_range_b: str = Defaults.prompt_value
                , prefix: str = Defaults.prompt_value
                , postfix: str = Defaults.prompt_value
            ):
            self.prompt_enable = prompt_enable
            self.negative_enable = negative_enable
            self.prompt_negative_enable = prompt_negative_enable
            self.prompt_range_enable = prompt_range_enable

            self.prompt = B_Value(prompt_value)
            self.strength = B_Value(strength_value)
            self.negative = B_Value(negative_value)
            self.prompt_negative = B_Value(prompt_negative_value)
            self.strength_negative = B_Value(strength_negative_value)
            self.prompt_range_a = B_Value(prompt_range_a) #!
            self.prompt_range_b = B_Value(prompt_range_b) #!

            self.prefix = prefix
            self.postfix = postfix
        
        def reset(self):
            self.prompt.reset()
            self.strength.reset()
            self.negative.reset()
            self.prompt_negative.reset()
            self.strength_negative.reset()
            self.prompt_range_a.reset()
            self.prompt_range_b.reset()
    

    class Fn():
        @staticmethod
        def _promptSanitized(prompt: str) -> str:
            return prompt.strip() if prompt is not None else ""
        
        @staticmethod
        def _promptAdded(promptExisting: str, promptToAdd: str) -> str:
            if len(promptToAdd) > 0:
                if len(promptExisting) > 0:
                    promptExisting += ", " + promptToAdd
                else:
                    promptExisting = promptToAdd
            
            return promptExisting
        
        @staticmethod
        def _promptDecorated(prompt: str, prefix: str = "", postfix: str = "") -> str:
            if len(prompt) > 0:
                if len(prefix) > 0:
                    prompt = f"{prefix} {prompt}"
                if len(postfix) > 0:
                    prompt = f"{prompt} {postfix}"
            
            return prompt
        
        @staticmethod
        def _promptStrengthened(prompt: str, strength: float) -> str:
            if len(prompt) == 0 or strength == B_Prompt.Values.Defaults.strength_min:
                return ""
            
            if strength != 1:
                prompt = f"({prompt}:{strength})"
            
            return prompt
    
    def __init__(self, name: str, values: Values, is_standalone: bool = True):
        self.name = name
        self.values = values
        self.is_standalone = is_standalone
    
    def reset(self):
        self.values.reset()
    
    @abstractmethod
    def build(self) -> tuple[str, str]:
        pass

class B_Prompt_Single(B_Prompt):
    def __init__(
            self
            , name: str
            , prompt_value = B_Prompt.Values.Defaults.prompt_value
            , strength_value = B_Prompt.Values.Defaults.strength_value
            , negative_value = B_Prompt.Values.Defaults.negative_value
        ):
        super().__init__(
            name
            , B_Prompt.Values(
                prompt_enable = True
                , negative_enable = True
                , prompt_value = prompt_value
                , strength_value = strength_value
                , negative_value = negative_value
            )
        )
    
    def build(self) -> tuple[str, str]:
        prompt = B_Prompt.Fn._promptStrengthened(
            B_Prompt.Fn._promptDecorated(
                B_Prompt.Fn._promptSanitized(self.values.prompt.value)
                , self.values.prefix
                , self.values.postfix
            )
            , self.values.strength.value
        )
        if not self.values.negative.value:
            return prompt, ""
        else:
            return "", prompt

class B_Prompt_Dual(B_Prompt):
    def __init__(
            self
            , name: str
            , prompt_value = B_Prompt.Values.Defaults.prompt_value
            , strength_value = B_Prompt.Values.Defaults.strength_value
            , prompt_negative_value = B_Prompt.Values.Defaults.prompt_value
            , strength_negative_value = B_Prompt.Values.Defaults.strength_value
        ):
        super().__init__(
            name
            , B_Prompt.Values(
                prompt_enable = True
                , prompt_negative_enable = True
                , prompt_value = prompt_value
                , strength_value = strength_value
                , prompt_negative_value = prompt_negative_value
                , strength_negative_value = strength_negative_value
            )
        )
    
    def build(self) -> tuple[str, str]:
        prompt = B_Prompt.Fn._promptStrengthened(
            B_Prompt.Fn._promptDecorated(
                B_Prompt.Fn._promptSanitized(self.values.prompt.value)
                , self.values.prefix
                , self.values.postfix
            )
            , self.values.strength.value
        )
        prompt_negative = B_Prompt.Fn._promptStrengthened(
            B_Prompt.Fn._promptDecorated(
                B_Prompt.Fn._promptSanitized(self.values.prompt_negative.value)
                , self.values.prefix
                , self.values.postfix
            )
            , self.values.strength_negative.value
        )
        return prompt, prompt_negative

class B_Prompt_Map():
    def __init__(self, b_prompts: list[B_Prompt]):
        self.map: dict[str, B_Prompt] = {}
        for b_prompt in b_prompts:
            self.map[b_prompt.name] = b_prompt if b_prompt.is_standalone else None
    
    def update(self, b_prompt: B_Prompt, remove: bool = False):
        self.map[b_prompt.name] = b_prompt if not remove else None
    
    def isSelected(self, b_prompt: B_Prompt | None):
        return b_prompt is not None and self.map[b_prompt.name] is not None
    
    def buildPromptUpdate(self) -> list[str]:
        prompt: str = ""
        prompt_negative: str = ""

        for b_prompt in self.map.values():
            if b_prompt is None:
                continue
            
            b_prompt_positive, b_prompt_negative = b_prompt.build()
            prompt = B_Prompt.Fn._promptAdded(prompt, b_prompt_positive)
            prompt_negative = B_Prompt.Fn._promptAdded(prompt_negative, b_prompt_negative)
        
        return [prompt, prompt_negative]

class B_UI(ABC):
    @staticmethod
    def _buildSeparator() -> typing.Any:
        return gr.Markdown(value = "<hr style=\"margin: 0.5em 0; border-style: dotted; border-color: var(--border-color-primary);\" />")
    
    def __init__(self, name: str):
        self.name = name
    
    @abstractmethod
    def getBPrompts(self) -> list[B_Prompt]:
        pass

    @abstractmethod
    def build(self, b_prompt_map: B_Prompt_Map) -> None:
        pass

    @abstractmethod
    def bind(self, b_prompt_map: B_Prompt_Map, gr_prompt: typing.Any, gr_prompt_negative: typing.Any) -> None:
        pass

    @abstractmethod
    def getGrForWebUI(self) -> list[typing.Any]:
        pass

    @abstractmethod
    def reset(self, b_prompt_map: B_Prompt_Map) -> None:
        pass

    @abstractmethod
    def update(self, b_prompt_map: B_Prompt_Map, *inputValues) -> int:
        pass
    
    @abstractmethod
    def getInput(self) -> list[typing.Any]:
        pass

    @abstractmethod
    def getOutput(self) -> list[typing.Any]:
        pass

    @abstractmethod
    def getOutputUpdate(self, b_prompt_map: B_Prompt_Map) -> list[typing.Any]:
        pass

class B_UI_Prompt(B_UI):
    def __init__(self, name: str = "Prompt", b_prompt: B_Prompt = None, show_name: bool = False):
        super().__init__(name)

        self.b_prompt = b_prompt
        self.show_name = show_name

        self.gr_container: typing.Any = None

        self.gr_prompt_container: typing.Any = None
        self.gr_prompt: typing.Any = None
        self.gr_strength: typing.Any = None

        self.gr_prompt_negative_container: typing.Any = None
        self.gr_prompt_negative: typing.Any = None
        self.gr_strength_negative: typing.Any = None

        self.gr_negative: typing.Any = None

        self.gr_button_apply: typing.Any = None
        self.gr_button_remove: typing.Any = None
    
    def getBPrompts(self) -> list[B_Prompt]:
        if self.b_prompt is not None:
            return [self.b_prompt]
        else:
            return []

    def build(self, b_prompt_map: B_Prompt_Map) -> None:
        values, visible, visible_button_remove = self.getUpdateValues(b_prompt_map)
        
        self.gr_container = (
            gr.Column(variant = "panel", visible = visible) if not self.show_name else 
            gr.Accordion(self.name, open = False, visible = visible)
        )
        with self.gr_container:
            with gr.Row():
                self.gr_prompt_container = gr.Column(visible = values.prompt_enable)
                with self.gr_prompt_container:
                    self.gr_prompt = gr.Textbox(
                        label = "Prompt"
                        , value = values.prompt.value
                    )
                    self.gr_strength = gr.Number(
                        label = "Strength"
                        , value = values.strength.value
                        , minimum = B_Prompt.Values.Defaults.strength_min
                        , step = B_Prompt.Values.Defaults.strength_step
                    )
                
                self.gr_prompt_negative_container = gr.Column(visible = values.prompt_negative_enable)
                with self.gr_prompt_negative_container:
                    self.gr_prompt_negative = gr.Textbox(
                        label = "Prompt (N)"
                        , value = values.prompt_negative.value
                    )
                    self.gr_strength_negative = gr.Number(
                        label = "Strength (N)"
                        , value = values.strength_negative.value
                        , minimum = B_Prompt.Values.Defaults.strength_min
                        , step = B_Prompt.Values.Defaults.strength_step
                    )
            
            self.gr_negative = gr.Checkbox(
                label = "Negative?"
                , value = values.negative.value
                , visible = values.negative_enable
            )

            B_UI._buildSeparator()

            with gr.Row():
                self.gr_button_apply = gr.Button(
                    value = "Apply"
                )
                self.gr_button_remove = gr.Button(
                    value = "Remove"
                    , visible = visible_button_remove
                )
    
    def bind(self, b_prompt_map: B_Prompt_Map, gr_prompt: typing.Any, gr_prompt_negative: typing.Any) -> None:
        # Add/update
        def _fnApply(*inputValues):
            self.update(b_prompt_map, *inputValues)
            return [self.gr_button_remove.update(visible = True)] + b_prompt_map.buildPromptUpdate()
        self.gr_button_apply.click(
            fn = _fnApply
            , inputs = self.getInput()
            , outputs = [self.gr_button_remove, gr_prompt, gr_prompt_negative]
        )

        # Remove
        def _fnRemove():
            b_prompt_map.update(self.b_prompt, remove = True)
            return [self.gr_button_remove.update(visible = False)] + b_prompt_map.buildPromptUpdate()
        self.gr_button_remove.click(
            fn = _fnRemove
            , outputs = [self.gr_button_remove, gr_prompt, gr_prompt_negative]
        )
    
    def getGrForWebUI(self) -> list[typing.Any]:
        return [
            self.gr_prompt
            , self.gr_strength
            , self.gr_negative
            , self.gr_prompt_negative
            , self.gr_strength_negative
            , self.gr_button_apply
            , self.gr_button_remove
        ]
    
    def reset(self, b_prompt_map: B_Prompt_Map) -> None:
        if self.b_prompt is not None:
            if self.b_prompt.is_standalone:
                self.b_prompt.reset()
                b_prompt_map.update(self.b_prompt)
            else:
                b_prompt_map.update(self.b_prompt, remove = True)
                self.b_prompt = None
    
    def update(self, b_prompt_map: B_Prompt_Map, *inputValues) -> int:
        offset: int = 0
        
        prompt = str(inputValues[offset])
        offset += 1

        strength = float(inputValues[offset])
        offset += 1

        prompt_negative = str(inputValues[offset])
        offset += 1

        strength_negative = float(inputValues[offset])
        offset += 1

        negative = bool(inputValues[offset])
        offset += 1

        if self.b_prompt is not None:
            self.b_prompt.values.prompt.value = prompt
            self.b_prompt.values.strength.value = strength
            self.b_prompt.values.negative.value = negative
            self.b_prompt.values.prompt_negative.value = prompt_negative
            self.b_prompt.values.strength_negative.value = strength_negative
            b_prompt_map.update(self.b_prompt)

        return offset
    
    def getInput(self) -> list[typing.Any]:
        return [
            self.gr_prompt
            , self.gr_strength
            , self.gr_prompt_negative
            , self.gr_strength_negative
            , self.gr_negative
        ]
    
    def getOutput(self) -> list[typing.Any]:
        return [
            self.gr_container
            , self.gr_prompt_container
            , self.gr_prompt
            , self.gr_strength
            , self.gr_prompt_negative_container
            , self.gr_prompt_negative
            , self.gr_strength_negative
            , self.gr_negative
            , self.gr_button_remove
        ]
    
    def getOutputUpdate(self, b_prompt_map: B_Prompt_Map) -> list[typing.Any]:
        values, visible, visible_button_remove = self.getUpdateValues(b_prompt_map)

        return [
            self.gr_container.update(visible = visible)
            , self.gr_prompt_container.update(visible = values.prompt_enable)
            , values.prompt.value
            , values.strength.value
            , self.gr_prompt_negative_container.update(visible = values.prompt_negative_enable)
            , values.prompt_negative.value
            , values.strength_negative.value
            , self.gr_negative.update(visible = values.negative_enable, value = values.negative.value)
            , self.gr_button_remove.update(visible = visible_button_remove)
        ]
    
    def getUpdateValues(self, b_prompt_map: B_Prompt_Map) -> tuple[B_Prompt.Values, bool, bool]:
        values = self.b_prompt.values if self.b_prompt is not None else B_Prompt.Values()
        visible = self.b_prompt is not None
        visible_button_remove = b_prompt_map.isSelected(self.b_prompt)
        return values, visible, visible_button_remove
    
    def getUpdateInit(self, b_prompt: B_Prompt, b_prompt_map: B_Prompt_Map) -> list:
        self.b_prompt = b_prompt
        return self.getOutputUpdate(b_prompt_map)

class B_UI_Dropdown(B_UI):
    _choice_empty: str = "-"

    def __init__(
            self
            , name: str = "Dropdown"
            , b_prompts: list[B_Prompt] = None
            , choice_default: str = _choice_empty
        ):
        super().__init__(name)

        self.choice = B_Value(choice_default)

        self.choice_map: dict[str, B_Prompt] = {}
        if b_prompts is not None and len(b_prompts) > 0:
            for b_prompt in b_prompts:
                b_prompt.is_standalone = False #!
                self.choice_map[b_prompt.name] = b_prompt
        
        self.gr_dropdown: typing.Any = None
        
        self.b_prompt_ui = B_UI_Prompt(f"{self.name} (Prompt)")
    
    def getBPrompts(self) -> list[B_Prompt]:
        return list(self.choice_map.values())
    
    def build(self, b_prompt_map: B_Prompt_Map) -> None:
        # Self
        self.gr_dropdown = gr.Dropdown(
            label = self.name
            , choices = [self._choice_empty] + list(self.choice_map.keys())
            , multiselect = False
            , value = self.choice.value
            , allow_custom_value = False
        )

        # Prompt UI
        self.b_prompt_ui.build(b_prompt_map)
    
    def bind(self, b_prompt_map: B_Prompt_Map, gr_prompt: typing.Any, gr_prompt_negative: typing.Any) -> None:
        # Self
        def _onSelect(choice: str):
            b_prompt = self.choice_map.get(choice)
            return self.b_prompt_ui.getUpdateInit(b_prompt, b_prompt_map)
        self.gr_dropdown.select(
            fn = _onSelect
            , inputs = self.gr_dropdown
            , outputs = self.b_prompt_ui.getOutput()
        )

        # Prompt UI
        self.b_prompt_ui.bind(b_prompt_map, gr_prompt, gr_prompt_negative)
    
    def getGrForWebUI(self) -> list[typing.Any]:
        return [self.gr_dropdown] + self.b_prompt_ui.getGrForWebUI()
    
    def reset(self, b_prompt_map: B_Prompt_Map) -> None:
        self.choice.reset()

        for b_prompt in self.choice_map.values():
            b_prompt.reset()
        
        self.b_prompt_ui.reset(b_prompt_map)
    
    def update(self, b_prompt_map: B_Prompt_Map, *inputValues) -> int:
        offset: int = 0

        self.choice.value = str(inputValues[offset])
        offset += 1

        offset += self.b_prompt_ui.update(b_prompt_map, *inputValues[offset:])

        return offset
    
    def getInput(self) -> list[typing.Any]:
        return [self.gr_dropdown] + self.b_prompt_ui.getInput()
    
    def getOutput(self) -> list[typing.Any]:
        return [self.gr_dropdown] + self.b_prompt_ui.getOutput()
    
    def getOutputUpdate(self, b_prompt_map: B_Prompt_Map) -> list[typing.Any]:
        return [self.choice.value] + self.b_prompt_ui.getOutputUpdate(b_prompt_map)

class B_UI_Container(B_UI, ABC):
    def __init__(self, name: str, children: list[B_UI] = None):
        super().__init__(name)

        self.children = children if children is not None else []

        self.gr_container: typing.Any = None
    
    def getBPrompts(self) -> list[B_Prompt]:
        b_prompt_list: list[B_Prompt] = []
        for b_ui in self.children:
            b_prompt_list += b_ui.getBPrompts()
        return b_prompt_list
    
    def build(self, b_prompt_map: B_Prompt_Map) -> None:
        self.gr_container = self.buildContainer()
        with self.gr_container:
            for b_ui in self.children:
                b_ui.build(b_prompt_map)
    
    def bind(self, b_prompt_map: B_Prompt_Map, gr_prompt: typing.Any, gr_prompt_negative: typing.Any) -> None:
        for b_ui in self.children:
            b_ui.bind(b_prompt_map, gr_prompt, gr_prompt_negative)
    
    def getGrForWebUI(self) -> list[typing.Any]:
        gr_list: list[typing.Any] = []
        for b_ui in self.children:
            gr_list += b_ui.getGrForWebUI()
        return gr_list
    
    def reset(self, b_prompt_map: B_Prompt_Map) -> None:
        for b_ui in self.children:
            b_ui.reset(b_prompt_map)
    
    def update(self, b_prompt_map: B_Prompt_Map, *inputValues) -> int:
        offset: int = 0
        for b_ui in self.children:
            offset += b_ui.update(b_prompt_map, *inputValues[offset:])
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
    
    def getOutputUpdate(self, b_prompt_map: B_Prompt_Map) -> list[typing.Any]:
        gr_updates: list[typing.Any] = []
        for b_ui in self.children:
            gr_updates += b_ui.getOutputUpdate(b_prompt_map)
        return gr_updates
    
    @abstractmethod
    def buildContainer(self) -> typing.Any:
        pass

class B_UI_Container_Tab(B_UI_Container):
    def __init__(self, name: str = "Tab", children: list[B_UI] = None):
        super().__init__(name, children)
    
    def buildContainer(self) -> typing.Any:
        return gr.Tab(self.name)

class B_UI_Master():
    def __init__(self, layout: list[B_UI]):
        self.layout = layout
        self.b_prompt_map = self.buildBPromptMap(layout)

        self.gr_prompt: typing.Any = None
        self.gr_prompt_negative: typing.Any = None
        self.gr_apply: typing.Any = None
        self.gr_reset: typing.Any = None
        self.gr_clear_config: typing.Any
    
    def buildBPromptMap(self, layout: list[B_UI]):
        b_prompts: list[B_Prompt] = []
        for b_ui in layout:
            b_prompts += b_ui.getBPrompts()
        return B_Prompt_Map(b_prompts)
    
    def build(self) -> list[typing.Any]:
        """Builds layout and returns Gradio elements for WebUI"""
        # LAYOUT
        B_UI._buildSeparator()
        for b_ui in self.layout:
            b_ui.build(self.b_prompt_map)
        
        # MAIN
        prompt = self.b_prompt_map.buildPromptUpdate()
        B_UI._buildSeparator()
        self.gr_prompt = gr.Textbox(label = "Final Prompt", value = prompt[0])
        self.gr_prompt_negative = gr.Textbox(label = "Final Negative Prompt", value = prompt[1])
        B_UI._buildSeparator()
        with gr.Row():
            self.gr_apply = gr.Button("Apply All")
            self.gr_reset = gr.Button("Reset All")

        # EXTRAS
        B_UI._buildSeparator()
        with gr.Accordion("Settings", open = False):
            self.gr_clear_config = gr.Button("Clear config")
        
        # Bind
        # - Layout
        for b_ui in self.layout:
            b_ui.bind(self.b_prompt_map, self.gr_prompt, self.gr_prompt_negative)
        
        # - Self
        gr_inputs: list[typing.Any] = []
        gr_outputs: list[typing.Any] = [self.gr_prompt, self.gr_prompt_negative]
        for b_ui in self.layout:
            gr_inputs += b_ui.getInput()
            gr_outputs += b_ui.getOutput()
        
        self.gr_apply.click(
            fn = self.apply
            , inputs = gr_inputs
            , outputs = gr_outputs
        )

        self.gr_reset.click(
            fn = self.reset
            , outputs = gr_outputs
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
        
        return gr_list
    
    def apply(self, *inputValues) -> list[typing.Any]:
        offset: int = 0
        for b_ui in self.layout:
            b_ui.update(self.b_prompt_map, *inputValues[offset:])
        
        #! repeating
        prompt = self.b_prompt_map.buildPromptUpdate()
        gr_updates: list[typing.Any] = [
            prompt[0]
            , prompt[1]
        ]
        for b_ui in self.layout:
            gr_updates += b_ui.getOutputUpdate(self.b_prompt_map)
        
        return gr_updates
    
    def reset(self) -> list[typing.Any]:
        for b_ui in self.layout:
            b_ui.reset(self.b_prompt_map)
        
        #! repeating ^
        prompt = self.b_prompt_map.buildPromptUpdate()
        gr_updates: list[typing.Any] = [
            prompt[0]
            , prompt[1]
        ]
        for b_ui in self.layout:
            gr_updates += b_ui.getOutputUpdate(self.b_prompt_map)
        
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

# #: Gradio Wrappers
# class Gr_Wrapper(ABC):
#     def __init__(self, name: str, is_labeled: bool) -> None:
#         self.name = name

#         self.is_labeled = is_labeled

#         self.gr: typing.Any = None
    
#     def initGr(self) -> None:
#         self.gr = self.buildGr()
    
#     def isGrBuilt(self) -> bool:
#         return self.gr is not None
    
#     def printWarning(self, message: str) -> None:
#         printWarning(self.__class__.__name__, self.name, message)
    
#     def validate(self) -> bool:
#         """VIRTUAL: Base -> True"""
#         return True
    
#     def getOutputUpdate(self, updates_output: list, reset: bool, *value_inputs) -> int:
#         """VIRTUAL: Adds updates to updates_output and returns number of values consumed, Base -> Nothing (0)"""
#         return 0

#     @abstractmethod
#     def buildGr(self) -> typing.Any:
#         pass

# class Gr_Markdown(Gr_Wrapper):
#     def __init__(self, value: str, name: str = "Markdown") -> None:
#         super().__init__(name, False)

#         self.value = value
    
#     def buildGr(self) -> typing.Any:
#         return gr.Markdown(value = self.value)

# class Gr_Output(Gr_Wrapper, ABC):
#     def __init__(self, name: str, is_labeled: bool, is_input: bool = False) -> None:
#         super().__init__(name, is_labeled)

#         self.is_input = is_input

# class Gr_Input(Gr_Output, ABC):
#     def __init__(self, label: str, value_default: typing.Any = None) -> None:
#         super().__init__(label, True, True)

#         self.value_default = value_default

#         self.value = self.buildDefaultValue()
    
#     def validate(self) -> bool:
#         valid = super().validate()

#         valid_value, valid_message = self.validateValue(self.value_default, "Default value")
#         if not valid_value:
#             valid = False
#             self.printWarning(valid_message)
        
#         return valid
    
#     def syncInput(self, value: typing.Any):
#         self.value = value
    
#     def getOutputUpdate(self, updates_output: list, reset: bool, *value_inputs) -> int:
#         value_new = value_inputs[0] if len(value_inputs) > 0 else None

#         if reset and value_new is None:
#             self.syncInput(self.buildDefaultValue())
#         elif value_new is not None:
#             self.syncInput(value_new)
        
#         updates_output.append(self.getUpdate(self.value))
#         return 1
    
#     def buildDefaultValue(self) -> typing.Any:
#         """VIRTUAL: Base -> self.value_default"""
#         return self.value_default
    
#     def validateValue(self, value: typing.Any, value_name: str = "Value") -> tuple[bool, str]:
#         """VIRTUAL: Returns valid, message, Base -> True, None"""
#         return True, None
    
#     def getUpdate(self, value: typing.Any) -> typing.Any:
#         """VIRTUAL: Base -> value"""
#         return value

# class Gr_Textbox(Gr_Input):
#     def __init__(self, label: str = "Textbox", value_default: str = "") -> None:
#         super().__init__(label, value_default)
    
#     def buildGr(self) -> typing.Any:
#         return gr.Textbox(
#             label = self.name
#             , value = self.value
#         )

# class Gr_Number(Gr_Input):
#     def __init__(self, label: str = "Number", value_default: float = 0, value_min: float = 0, value_step: float = 0.1) -> None:
#         super().__init__(label, value_default)

#         self.value_min = value_min
#         self.value_step = value_step
    
#     def validateValue(self, value: typing.Any, value_name: str = "Value") -> tuple[bool, str]:
#         if type(value) is not float and type(value) is not int:
#             return False, f"{value_name} ({value}) is not a float | int"
        
#         value_f: float = value

#         if value_f < self.value_min:
#             return False, f"{value_name} ({value_f} is under minimum ({self.value_min}))"
        
#         return super().validateValue(value, value_name)
    
#     def buildGr(self) -> typing.Any:
#         return gr.Number(
#             label = self.name
#             , value = self.value
#             , minimum = self.value_min
#             , step = self.value_step
#         )

# class Gr_Checkbox(Gr_Input):
#     def __init__(self, label: str = "Checkbox", value_default: bool = False) -> None:
#         super().__init__(label, value_default)
    
#     def buildGr(self) -> typing.Any:
#         return gr.Checkbox(
#             label = self.name
#             , value = self.value
#         )

# class Gr_Slider(Gr_Input):
#     def __init__(self, label: str = "Slider", value_default: int = 0, value_min: int = 0, value_max: int = 100, value_step: int = 1) -> None:
#         super().__init__(label, value_default)

#         self.value_min = value_min
#         self.value_max = value_max
#         self.value_step = value_step
    
#     def validateValue(self, value: typing.Any, value_name: str = "Value") -> tuple[bool, str]:
#         if type(value) is not int:
#             return False, f"{value_name} ({value}) is not an int"
        
#         value_int: int = value

#         if value_int < self.value_min:
#             return False, f"{value_name} ({self.value_default}) is under minimum ({self.value_min})"
        
#         if value_int > self.value_max:
#             return False, f"{value_name} ({self.value_default}) is over maximum ({self.value_max})"
        
#         return super().validateValue(value, value_name)
    
#     def buildGr(self) -> typing.Any:
#         return gr.Slider(
#             label = self.name
#             , value = self.value
#             , minimum = self.value_min
#             , maximum = self.value_max
#             , step = self.value_step
#         )

# class Gr_Dropdown(Gr_Input):
#     @staticmethod
#     def valueSanitized(choices: str | list[str] | None, multiselect: bool, allow_none: bool = False) -> str | list[str] | None:
#         if choices is not None:
#             if type(choices) is list:
#                 if not multiselect:
#                     if len(choices) > 0:
#                         choices = choices[0]
#                     else:
#                         choices = None
#             elif multiselect:
#                 choices = [choices]
#         elif not allow_none:
#             choices = []

#         return choices
    
#     def __init__(self, label: str = "Dropdown", choices: list[str] = None, value_default: str | list[str] = None, multiselect: bool = False) -> None:
#         value_default = self.valueSanitized(value_default, multiselect, True)

#         super().__init__(label, value_default)

#         self.choices = choices if choices is not None else []
#         self.multiselect = multiselect
    
#     def syncInput(self, value: str | list[str] | None):
#         return super().syncInput(self.valueSanitized(value, self.multiselect, not self.multiselect))
    
#     def validateValue(self, value: typing.Any, value_name: str = "Choice(s)") -> tuple[bool, str]:
#         if value is not None and type(value) is not str and type(value) is not list:
#             return False, f"{value_name} ({value}) is not a str | list | None"
        
#         if len(self.choices) > 0:
#             if (
#                 value is not None
#                 and (
#                     (
#                         type(value) is not list
#                         and value not in self.choices
#                     ) or (
#                         type(value) is list
#                         and any(map(lambda c: c not in self.choices, value))
#                     )
#                 )
#             ):
#                 return False, f"Invalid {value_name} -> {value}"
#         elif value is not None:
#             return False, f"No choices set but {value_name} set -> {value}"
        
#         return super().validateValue(value, value_name)
    
#     def buildDefaultValue(self) -> typing.Any:
#         value_default = super().buildDefaultValue()

#         if value_default is None or type(value_default) is not list:
#             return value_default
        
#         return value_default[:]
    
#     def buildGr(self) -> typing.Any:
#         return gr.Dropdown(
#             label = self.name
#             , choices = self.choices
#             , multiselect = self.multiselect
#             , value = self.value
#             , allow_custom_value = False #!
#         )

# class Gr_Button(Gr_Output):
#     def __init__(self, text: str = "Button") -> None:
#         super().__init__(text, True)
    
#     def buildGr(self) -> typing.Any:
#         return gr.Button(value = self.name)

# class Gr_Container(Gr_Output, ABC):
#     def __init__(self, name: str, visible: bool = True) -> None:
#         super().__init__(name, False)

#         self.visible = visible
    
#     def buildGr(self) -> typing.Any:
#         return self.buildGrContainer(self.visible)
    
#     def getOutputUpdate(self, updates_output: list, reset: bool, *value_inputs) -> int:
#         visible = value_inputs[0]

#         if visible is not None:
#             self.visible = bool(visible)
        
#         updates_output.append(self.gr.update(visible = visible))
#         return 1
    
#     def getUpdateVisible(self, updates_output: list, visible: bool) -> None:
#         self.getOutputUpdate(updates_output, False, visible)
    
#     @abstractmethod
#     def buildGrContainer(self, visible: bool) -> typing.Any:
#         pass

# class Gr_Row(Gr_Container):
#     def __init__(self, variant: str = "default", visible: bool = True, name: str = "Row") -> None:
#         super().__init__(name, visible)

#         self.variant = variant
    
#     def buildGrContainer(self, visible: bool) -> typing.Any:
#         return gr.Row(
#             variant = self.variant
#             , visible = visible
#         )

# class Gr_Column(Gr_Container):
#     def __init__(self, scale: int = 1, variant: str = "default", min_width: int = 320, visible: bool = True, name: str = "Column") -> None:
#         super().__init__(name, visible)

#         self.scale = scale
#         self.variant = variant
#         self.min_width = min_width
    
#     def buildGrContainer(self, visible: bool) -> typing.Any:
#         return gr.Column(
#             scale = self.scale
#             , variant = self.variant
#             , min_width = self.min_width
#             , visible = visible
#         )

# class Gr_Group(Gr_Container):
#     def __init__(self, visible: bool = True, name: str = "Group") -> None:
#         super().__init__(name, visible)
    
#     def buildGrContainer(self, visible: bool) -> typing.Any:
#         return gr.Group(visible = visible)

# #!!! Tab does not have visible prop..
# class Gr_Tab(Gr_Container):
#     def __init__(self, visible: bool = True, name: str = "Group (Tab)") -> None:
#         super().__init__(name, visible)

#         self.gr_tab: typing.Any = None
    
#     def buildGrContainer(self, visible: bool) -> typing.Any:
#         self.gr_tab = gr.Tab(self.name)
#         with self.gr_tab:
#             gr_tab_group = gr.Group(visible = visible)
#         return gr_tab_group

# class Gr_Accordion(Gr_Container):
#     def __init__(self, visible: bool = True, name: str = "Accordion") -> None:
#         super().__init__(name, visible)
    
#     def buildGrContainer(self, visible: bool) -> typing.Any:
#         return gr.Accordion(self.name, visible = visible)

# #: UI Wrappers
# class B_Ui(ABC):
#     @staticmethod
#     @abstractmethod
#     def _paramsFromArgs(args: dict[str, str]) -> tuple:
#         pass

#     @staticmethod
#     @abstractmethod
#     def _fromArgs(args: dict[str, str], name: str = None):
#         pass

#     @staticmethod
#     @abstractmethod
#     def _getDefaultName() -> str:
#         pass
    
#     def __init__(self, name: str = None, hidden: bool = False) -> None:
#         self.name = name if name is not None and len(name.strip()) > 0 else self._getDefaultName()
#         self.hidden = hidden

#         self.gr_container: Gr_Container = None
#         self.gr_outputs: list[Gr_Output] = []
#         self.gr_outputs_extras: list[Gr_Output] = []
    
#     def init_main(self, bMap: dict) -> None:
#         self.gr_container = self.initContainer(not self.hidden)

#         self.init(self.gr_outputs, self.gr_outputs_extras, bMap)
    
#     def initUI(self) -> None:
#         self.gr_container.initGr()
#         with self.gr_container.gr:
#             self.buildUI()
    
#     def getInput(self, include_unbuilt: bool = False, base_only: bool = False) -> list[Gr_Input]:
#         gr_list: list[Gr_Input] = []
        
#         for gr_output in self.gr_outputs:
#             if gr_output.is_input and (include_unbuilt or gr_output.isGrBuilt()):
#                 gr_list.append(gr_output)

#         return gr_list
    
#     def getOutput(self, labeled_only: bool = False, exclude_labeled_outputs: bool = False, base_only: bool = False) -> list[Gr_Output]:
#         gr_list: list[Gr_Output] = []

#         for gr_output in self.gr_outputs + self.gr_outputs_extras:
#             if not gr_output.isGrBuilt():
#                 continue

#             if labeled_only and not gr_output.is_labeled:
#                 continue

#             if not gr_output.is_input and gr_output.is_labeled and exclude_labeled_outputs:
#                 continue

#             gr_list.append(gr_output)
        
#         return gr_list
    
#     def getOutputUpdates(self, updates: list, reset: bool, base_only: bool, *inputValues) -> int:
#         """VIRTUAL: Base -> Populates updates with own Gr_Input updates, returns number of values consumed"""
#         offset: int = 0

#         for gr_output in self.gr_outputs:
#             if gr_output.is_input and gr_output.isGrBuilt():
#                 offset += gr_output.getOutputUpdate(updates, reset, *inputValues[offset:])
        
#         self.getOutputUpdatesExtra(updates, self.gr_outputs_extras)
        
#         return offset
    
#     def getOutputUpdatesExtra(self, updates: list, gr_outputs_extras: list[Gr_Output]) -> None:
#         """VIRTUAL: Base -> Nothing"""
#         pass

#     def getOutputUpdatesFromArgs(self, updates: list, reset: bool, *args) -> int:
#         """VIRTUAL: Base -> getOutputUpdates (base only)"""
#         return self.getOutputUpdates(updates, reset, True)
    
#     def initContainer(self, visible: bool) -> Gr_Container:
#         """VIRTUAL: Base -> Gr_Group"""
#         return Gr_Group(visible, f"{self.name} (Group)")
    
#     def validate(self, bMap: dict) -> bool:
#         """VIRTUAL: Base -> validate own Gr_Outputs"""
#         valid = True
        
#         for x in self.gr_outputs:
#             if not x.validate():
#                 valid = False
        
#         return valid
    
#     def consumeOutputs(self, *outputValues) -> int:
#         """VIRTUAL: Base -> sync values on own Gr_Inputs, returns number of values consumed"""
#         offset: int = 0

#         for x in self.gr_outputs + self.gr_outputs_extras:
#             if not x.isGrBuilt():
#                 continue
            
#             if x.is_input:
#                 x_input: Gr_Input = x
#                 x_input.syncInput(outputValues[offset])
#                 offset += 1
#             elif x.is_labeled:
#                 offset += 1
        
#         return offset
    
#     def finalizeUI(self, bMap: dict) -> None:
#         """VIRTUAL: Bindings, etc, Base -> Nothing"""
#         pass

#     def validateArgs(self, *args) -> list[tuple[bool, str]]:
#         """VIRTUAL: Base -> []"""
#         return []
    
#     @abstractmethod
#     def init(self, gr_outputs: list[Gr_Output], gr_outputs_extras: list[Gr_Output], bMap: dict) -> None:
#         """Instantiates Gradio wrappers"""
#         pass

#     @abstractmethod
#     def buildUI(self) -> None:
#         """Builds Gradio layout and components from Gradio wrappers"""

#     @abstractmethod
#     def handlePrompt(self, p: StableDiffusionProcessing, bMap: dict) -> None:
#         pass

# class B_Ui_Separator(B_Ui):
#     _html_separator: str = "<hr style=\"margin: 0.5em 0; border-style: dotted; border-color: var(--border-color-primary);\" />"

#     @staticmethod
#     def _paramsFromArgs(args: dict[str, str]) -> tuple[bool]:
#         hidden = bool(int(args.get("hide", 0)))
#         return hidden,
    
#     @staticmethod
#     def _fromArgs(args: dict[str, str], name: str = None):
#         hidden, = B_Ui_Separator._paramsFromArgs(args)
#         return B_Ui_Separator(
#             hidden = hidden
#         )
    
#     @staticmethod
#     def _getDefaultName() -> str:
#         return "Separator"
    
#     @staticmethod
#     def _build() -> None:
#         Gr_Markdown(B_Ui_Separator._html_separator).initGr()
    
#     def __init__(self, name: str = None, hidden: bool = False) -> None:
#         super().__init__(name, hidden)

#         self.ui: Gr_Markdown = None
    
#     def init(self, gr_outputs: list[Gr_Output], gr_outputs_extras: list[Gr_Output], bMap: dict[str, B_Ui]) -> None:
#         self.ui = Gr_Markdown(self._html_separator, self.name)
#         #gr_outputs_extras.append(self.ui) #!
    
#     def buildUI(self) -> None:
#         self.ui.initGr()
    
#     def handlePrompt(self, p: StableDiffusionProcessing, bMap: dict[str, B_Ui]) -> None:
#         pass

# class B_Ui_Collection(B_Ui, ABC):
#     def __init__(
#             self
#             , name: str = None
#             , items: list[B_Ui] = None
#             , items_sort: bool = False
#             , hidden: bool = False
#         ) -> None:
#         super().__init__(name, hidden)

#         self.items = items if items is not None else []
#         self.items_sort = items_sort
    
#     def validate(self, bMap: dict[str, B_Ui]) -> bool:
#         valid = super().validate(bMap)

#         for x in self.items:
#             if not x.validate(bMap):
#                 valid = False
        
#         return valid
    
#     def finalizeUI(self, bMap: dict[str, B_Ui]) -> None:
#         for x in self.items:
#             x.finalizeUI(bMap)
    
#     def consumeOutputs(self, *outputValues) -> int:
#         offset = super().consumeOutputs(*outputValues)

#         for x in self.items:
#             offset += x.consumeOutputs(*outputValues[offset:])
        
#         return offset
    
#     def getInput(self, include_unbuilt: bool = False, base_only: bool = False) -> list[Gr_Input]:
#         gr_inputs = super().getInput(include_unbuilt, base_only)

#         if not base_only:
#             for x in self.items:
#                 gr_inputs += x.getInput(include_unbuilt, False)
        
#         return gr_inputs
    
#     def getOutput(self, labeled_only: bool = False, exclude_labeled_outputs: bool = False, base_only: bool = False) -> list[Gr_Output]:
#         gr_outputs = super().getOutput(labeled_only, exclude_labeled_outputs, base_only)

#         if not base_only:
#             for x in self.items:
#                 gr_outputs += x.getOutput(labeled_only, exclude_labeled_outputs, False)
        
#         return gr_outputs
    
#     def getOutputUpdates(self, updates: list, reset: bool, base_only: bool, *inputValues) -> int:
#         offset = super().getOutputUpdates(updates, reset, base_only, *inputValues)

#         if not base_only:
#             for x in self.items:
#                 offset += x.getOutputUpdates(updates, reset, False, *inputValues[offset:])
        
#         return offset
    
#     def buildUI(self) -> None:
#         """VIRTUAL: Build items' Gradio elements"""
#         for x in self.items:
#             self.initItemUI(x)
    
#     def handlePrompt(self, p: StableDiffusionProcessing, bMap: dict[str, B_Ui]) -> None:
#         for x in self.items:
#             x.handlePrompt(p, bMap)
    
#     def addItem(self, item: B_Ui) -> None:
#         self.items.append(item)
    
#     def init(self, gr_outputs: list[Gr_Output], gr_outputs_extras: list[Gr_Output], bMap: dict[str, B_Ui]) -> None:
#         """VIRTUAL: Base -> init children and sorts children (if items_sort)"""
#         for x in self.items:
#             x.init_main(bMap)
        
#         #! move to select only
#         if self.items_sort:
#             self.items = sorted(self.items, key = lambda x: x.name)

#     def initItemUI(self, item: B_Ui) -> None:
#         """Virtual: Base -> item.initUI"""
#         item.initUI()

# class B_Ui_Container(B_Ui_Collection, ABC):
#     def __init__(
#             self
#             , name: str
#             , items: list[B_Ui] = None
#             , reset_ui_build: bool = False
#             , random_ui_build: bool = False
#             , hidden: bool = False) -> None:
#         super().__init__(name, items, False, hidden)

#         self.ui_reset_build = reset_ui_build
#         self.ui_random_build = random_ui_build

#         self.ui_reset: Gr_Button = None
#         self.ui_random: Gr_Button = None
    
#     def init(self, gr_outputs: list[Gr_Output], gr_outputs_extras: list[Gr_Output], bMap: dict[str, B_Ui]) -> None:
#         super().init(gr_outputs, gr_outputs_extras, bMap)
        
#         self.ui_reset = Gr_Button(f"Reset {self.name}")
#         gr_outputs_extras.append(self.ui_reset)

#         self.ui_random = Gr_Button(f"Randomize {self.name}")
#         gr_outputs_extras.append(self.ui_random)
    
#     def buildUI(self) -> None:
#         super().buildUI()

#         if self.ui_reset_build or self.ui_random_build:
#             B_Ui_Separator._build()
#             if self.ui_reset_build and self.ui_random_build:
#                 with gr.Row():
#                     with gr.Column():
#                         self.ui_random.initGr()
#                     with gr.Column():
#                         self.ui_reset.initGr()
#             elif self.ui_random_build:
#                 self.ui_random.initGr()
#             elif self.ui_reset_build:
#                 self.ui_reset.initGr()
    
#     def finalizeUI(self, bMap: dict[str, B_Ui]) -> None:
#         super().finalizeUI(bMap)

#         # if self.ui_random_build:
#         #     self.ui_random.gr.click() #!
        
#         if self.ui_reset_build:
#             def _fnReset():
#                 updates: list = []
#                 self.getOutputUpdates(updates, True, False)
#                 return updates
            
#             self.ui_reset.gr.click(
#                 fn = _fnReset
#                 , outputs = list(map(lambda gr_output: gr_output.gr, self.getOutput(exclude_labeled_outputs = True)))
#             )

# class B_Ui_Container_Tab(B_Ui_Container):
#     @staticmethod
#     def _paramsFromArgs(args: dict[str, str]) -> tuple[bool, bool, bool]:
#         reset_ui_build = bool(int(args.get("build_reset_button", 1)))
#         random_ui_build = bool(int(args.get("build_random_button", 1)))
#         hidden = bool(int(args.get("hide", 0)))
#         return reset_ui_build, random_ui_build, hidden
    
#     @staticmethod
#     def _fromArgs(args: dict[str, str], name: str = None):
#         reset_ui_build, random_ui_build, hidden = B_Ui_Container_Tab._paramsFromArgs(args)
#         return B_Ui_Container_Tab(
#             name = name
#             , reset_ui_build = reset_ui_build
#             , random_ui_build = random_ui_build
#             , hidden = hidden
#         )
    
#     @staticmethod
#     def _getDefaultName() -> str:
#         return "Tab"
    
#     def __init__(
#             self
#             , name: str = None
#             , items: list[B_Ui] = None
#             , reset_ui_build: bool = True
#             , random_ui_build: bool = True
#             , hidden: bool = False
#         ) -> None:
#         super().__init__(name, items, reset_ui_build, random_ui_build, hidden)
    
#     def initContainer(self, visible: bool) -> Gr_Container:
#         return Gr_Tab(name = self.name, visible = visible)

# class B_Ui_Container_Row(B_Ui_Container):
#     @staticmethod
#     def _paramsFromArgs(args: dict[str, str]) -> tuple[bool, bool, bool]:
#         reset_ui_build = bool(int(args.get("build_reset_button", 0)))
#         random_ui_build = bool(int(args.get("build_random_button", 0)))
#         hidden = bool(int(args.get("hide", 0)))
#         return reset_ui_build, random_ui_build, hidden
    
#     @staticmethod
#     def _fromArgs(args: dict[str, str], name: str = None):
#         reset_ui_build, random_ui_build, hidden = B_Ui_Container_Row._paramsFromArgs(args)
#         return B_Ui_Container_Row(
#             name = name
#             , reset_ui_build = reset_ui_build
#             , random_ui_build = random_ui_build
#             , hidden = hidden
#         )
    
#     @staticmethod
#     def _getDefaultName() -> str:
#         return "Row"
    
#     def __init__(
#             self
#             , name: str = None
#             , items: list[B_Ui] = None
#             , reset_ui_build: bool = False
#             , random_ui_build: bool = False
#             , hidden: bool = False
#         ) -> None:
#         super().__init__(name, items, reset_ui_build, random_ui_build, hidden)
    
#     def initContainer(self, visible: bool) -> Gr_Container:
#         return Gr_Row(visible = visible)

# class B_Ui_Container_Column(B_Ui_Container):
#     @staticmethod
#     def _paramsFromArgs(args: dict[str, str]) -> tuple[int, bool, bool, bool]:
#         scale = int(args.get("scale", 1))
#         reset_ui_build = bool(int(args.get("build_reset_button", 0)))
#         random_ui_build = bool(int(args.get("build_random_button", 0)))
#         hidden = bool(int(args.get("hide", 0)))
#         return scale, reset_ui_build, random_ui_build, hidden
    
#     @staticmethod
#     def _fromArgs(args: dict[str, str], name: str = None):
#         scale, reset_ui_build, random_ui_build, hidden = B_Ui_Container_Column._paramsFromArgs(args)
#         return B_Ui_Container_Column(
#             name = name
#             , scale = scale
#             , reset_ui_build = reset_ui_build
#             , random_ui_build = random_ui_build
#             , hidden = hidden
#         )
    
#     @staticmethod
#     def _getDefaultName() -> str:
#         return "Column"
    
#     def __init__(
#             self
#             , name: str = None
#             , items: list[B_Ui] = None
#             , scale: int = 1
#             , reset_ui_build: bool = False
#             , random_ui_build: bool = False
#             , hidden: bool = False
#         ) -> None:
#         super().__init__(name, items, reset_ui_build, random_ui_build, hidden)

#         self.scale = scale
    
#     def initContainer(self, visible: bool) -> Gr_Container:
#         return Gr_Column(scale = self.scale, visible = visible)

# class B_Ui_Container_Group(B_Ui_Container):
#     @staticmethod
#     def _paramsFromArgs(args: dict[str, str]) -> tuple[bool, bool, bool]:
#         reset_ui_build = bool(int(args.get("build_reset_button", 0)))
#         random_ui_build = bool(int(args.get("build_random_button", 0)))
#         hidden = bool(int(args.get("hide", 0)))
#         return reset_ui_build, random_ui_build, hidden
    
#     @staticmethod
#     def _fromArgs(args: dict[str, str], name: str = None):
#         reset_ui_build, random_ui_build, hidden = B_Ui_Container_Group._paramsFromArgs(args)
#         return B_Ui_Container_Group(
#             name = name
#             , reset_ui_build = reset_ui_build
#             , random_ui_build = random_ui_build
#             , hidden = hidden
#         )
    
#     @staticmethod
#     def _getDefaultName() -> str:
#         return "Group"
    
#     def __init__(
#             self
#             , name: str = None
#             , items: list[B_Ui] = None
#             , reset_ui_build: bool = False
#             , random_ui_build: bool = False
#             , hidden: bool = False
#         ) -> None:
#         super().__init__(name, items, reset_ui_build, random_ui_build, hidden)
    
#     def initContainer(self, visible: bool) -> Gr_Container:
#         return Gr_Group(visible = visible)

# class B_Ui_Container_Accordion(B_Ui_Container):
#     @staticmethod
#     def _paramsFromArgs(args: dict[str, str]) -> tuple[bool, bool, bool]:
#         reset_ui_build = bool(int(args.get("build_reset_button", 0)))
#         random_ui_build = bool(int(args.get("build_random_button", 0)))
#         hidden = bool(int(args.get("hide", 0)))
#         return reset_ui_build, random_ui_build, hidden
    
#     @staticmethod
#     def _fromArgs(args: dict[str, str], name: str = None):
#         reset_ui_build, random_ui_build, hidden = B_Ui_Container_Accordion._paramsFromArgs(args)
#         return B_Ui_Container_Accordion(
#             name = name
#             , reset_ui_build = reset_ui_build
#             , random_ui_build = random_ui_build
#             , hidden = hidden
#         )
    
#     @staticmethod
#     def _getDefaultName() -> str:
#         return "Accordion"
    
#     def __init__(
#             self
#             , name: str = None
#             , items: list[B_Ui] = None
#             , reset_ui_build: bool = False
#             , random_ui_build: bool = False
#             , hidden: bool = False
#         ) -> None:
#         super().__init__(name, items, reset_ui_build, random_ui_build, hidden)
    
#     def initContainer(self, visible: bool) -> Gr_Container:
#         return Gr_Accordion(name = self.name, visible = visible)

# class B_Ui_Prompt_Single(B_Ui):
#     @staticmethod
#     def _paramsFromArgs(args: dict[str, str]) -> tuple[str, str, str, float, bool, bool]:
#         prefix = args.get("prefix", "")
#         postfix = args.get("postfix", "")
#         prompt = args.get("v", "")
#         strength = float(args.get("s", 1))
#         negative = bool(int(args.get("n", 0)))
#         hidden = bool(int(args.get("hide", 0)))
#         return prefix, postfix, prompt, strength, negative, hidden
    
#     @staticmethod
#     def _fromArgs(args: dict[str, str], name: str = None):
#         prefix, postfix, prompt, strength, negative, hidden = B_Ui_Prompt_Single._paramsFromArgs(args)
#         return B_Ui_Prompt_Single(
#             name = name
#             , prefix = prefix
#             , postfix = postfix
#             , prompt = prompt
#             , strength = strength
#             , negative = negative
#             , hidden = hidden
#         )
    
#     @staticmethod
#     def _getDefaultName() -> str:
#         return "Single Prompt"
    
#     def __init__(
#             self
#             , name: str = None
#             , prompt_ui_build: bool = True
#             , strength_ui_build: bool = True
#             , negative_ui_build: bool = True
#             , prefix: str = ""
#             , postfix: str = ""
#             , prompt: str = ""
#             , strength: float = 1
#             , negative: bool = False
#             , hidden: bool = False
#         ) -> None:
#         super().__init__(name, hidden)

#         self.ui_prompt_build = prompt_ui_build
#         self.ui_strength_build = strength_ui_build
#         self.ui_negative_build = negative_ui_build

#         self.prefix = prefix
#         self.postfix = postfix

#         self.prompt = prompt
#         self.strength = strength
#         self.negative = negative

#         self.ui_prompt: Gr_Textbox = None
#         self.ui_strength: Gr_Number = None
#         self.ui_negative: Gr_Checkbox = None
    
#     def init(self, gr_outputs: list[Gr_Output], gr_outputs_extras: list[Gr_Output], bMap: dict[str, B_Ui]) -> None:
#         self.ui_prompt = Gr_Textbox(self.name, self.prompt)
#         gr_outputs.append(self.ui_prompt)

#         self.ui_strength = Gr_Number(f"{self.name} (S)", self.strength, b_prompt_strength_min, b_prompt_strength_step)
#         gr_outputs.append(self.ui_strength)

#         self.ui_negative = Gr_Checkbox(f"{self.name} (N)", self.negative)
#         gr_outputs.append(self.ui_negative)
    
#     def validateArgs(
#             self
#             , prefix: str
#             , postfix: str
#             , prompt: str
#             , strength: float
#             , negative: bool
#             , hidden: bool
#         ) -> list[tuple[bool, str]]:
#         return [
#             self.ui_prompt.validateValue(prompt)
#             , self.ui_strength.validateValue(strength)
#             , self.ui_negative.validateValue(negative)
#         ]
    
#     def getOutputUpdatesFromArgs(
#             self
#             , updates: list
#             , reset: bool
#             , prefix: str
#             , postfix: str
#             , prompt: str
#             , strength: float
#             , negative: bool
#             , hidden: bool
#             , *args
#         ) -> int:
#         if self.ui_prompt.isGrBuilt():
#             self.ui_prompt.syncInput(prompt)
#         if self.ui_strength.isGrBuilt():
#             self.ui_strength.syncInput(strength)
#         if self.ui_negative.isGrBuilt():
#             self.ui_negative.syncInput(negative)
        
#         return super().getOutputUpdatesFromArgs(updates, reset)
    
#     def buildUI(self) -> None:
#         if self.ui_prompt_build or self.ui_strength_build:
#             with gr.Row():
#                 if self.ui_prompt_build:
#                     self.ui_prompt.initGr()
                
#                 if self.ui_strength_build:
#                     self.ui_strength.initGr()
        
#         if self.ui_negative_build:
#             self.ui_negative.initGr()
    
#     def handlePrompt(self, p: StableDiffusionProcessing, bMap: dict[str, B_Ui]) -> None:
#         prompt = str(self.ui_prompt.value)
#         negative = bool(self.ui_negative.value)
#         strength = float(self.ui_strength.value)

#         prompt = promptSanitized(prompt)

#         if len(prompt) == 0:
#             return
        
#         prompt = promptDecorated(prompt, self.prefix, self.postfix)
        
#         if strength > 0 and strength != 1:
#             prompt = f"({prompt}:{strength})"
        
#         if not negative:
#             p.prompt = promptAdded(p.prompt, prompt)
#         else:
#             p.negative_prompt = promptAdded(p.negative_prompt, prompt)

# class B_Ui_Prompt_Dual(B_Ui):
#     @staticmethod
#     def _paramsFromArgs(args: dict[str, str]) -> tuple[str, str, float, bool]:
#         prompt_positive = args.get("vp", "")
#         prompt_negative = args.get("vn", "")
#         strength = float(args.get("s", 1))
#         hidden = bool(int(args.get("hide", 0)))
#         return prompt_positive, prompt_negative, strength, hidden
    
#     @staticmethod
#     def _fromArgs(args: dict[str, str], name: str = None):
#         prompt_positive, prompt_negative, strength, hidden = B_Ui_Prompt_Dual._paramsFromArgs(args)
#         return B_Ui_Prompt_Dual(
#             name = name
#             , prompt_positive = prompt_positive
#             , prompt_negative = prompt_negative
#             , strength = strength
#             , hidden = hidden
#         )
    
#     @staticmethod
#     def _getDefaultName() -> str:
#         return "Dual Prompt"
    
#     def __init__(
#             self
#             , name: str = None
#             , ui_prompts_build: bool = True
#             , ui_strength_build: bool = True
#             , prompt_positive: str = ""
#             , prompt_negative: str = ""
#             , strength: float = 1
#             , hidden: bool = False
#         ) -> None:
#         super().__init__(name, hidden)

#         self.ui_prompts_build = ui_prompts_build
#         self.ui_strength_build = ui_strength_build
        
#         self.prompt_positive = prompt_positive
#         self.prompt_negative = prompt_negative
#         self.strength = strength

#         self.ui_prompt_positive: Gr_Textbox = None
#         self.ui_prompt_negative: Gr_Textbox = None
#         self.ui_strength: Gr_Number = None
    
#     def init(self, gr_outputs: list[Gr_Output], gr_outputs_extras: list[Gr_Output], bMap: dict[str, B_Ui]) -> None:
#         self.ui_prompt_positive = Gr_Textbox(f"{self.name} (+)", self.prompt_positive)
#         gr_outputs.append(self.ui_prompt_positive)

#         self.ui_prompt_negative = Gr_Textbox(f"{self.name} (-)", self.prompt_negative)
#         gr_outputs.append(self.ui_prompt_negative)

#         self.ui_strength = Gr_Number(f"{self.name} (S)", self.strength, b_prompt_strength_min, b_prompt_strength_step)
#         gr_outputs.append(self.ui_strength)
    
#     def validateArgs(
#             self
#             , prompt_positive: str
#             , prompt_negative: str
#             , strength: float
#             , hidden: bool
#         ) -> list[tuple[bool, str]]:
#         return [
#             self.ui_prompt_positive.validateValue(prompt_positive)
#             , self.ui_prompt_negative.validateValue(prompt_negative)
#             , self.ui_strength.validateValue(strength)
#         ]
    
#     def getOutputUpdatesFromArgs(
#             self
#             , updates: list
#             , reset: bool
#             , prompt_positive: str
#             , prompt_negative: str
#             , strength: float
#             , hidden: bool
#         ) -> int:
#         if self.ui_prompt_positive.isGrBuilt():
#             self.ui_prompt_positive.syncInput(prompt_positive)
#         if self.ui_prompt_negative.isGrBuilt():
#             self.ui_prompt_negative.syncInput(prompt_negative)
#         if self.ui_strength.isGrBuilt():
#             self.ui_strength.syncInput(strength)
        
#         return super().getOutputUpdatesFromArgs(updates, reset)
    
#     def buildUI(self) -> None:
#         if self.ui_prompts_build:
#             self.ui_prompt_positive.initGr()
#             self.ui_prompt_negative.initGr()
        
#         if self.ui_strength_build:
#             self.ui_strength.initGr()
    
#     def handlePrompt(self, p: StableDiffusionProcessing, bMap: dict[str, B_Ui]) -> None:
#         prompt_positive = str(self.ui_prompt_positive.value)
#         prompt_negative = str(self.ui_prompt_negative.value)
#         strength = float(self.ui_strength.value)

#         prompt_positive = promptSanitized(prompt_positive)
#         prompt_negative = promptSanitized(prompt_negative)

#         if strength > 0 and strength != 1:
#             if len(prompt_positive) > 0:
#                 prompt_positive = f"({prompt_positive}:{strength})"
#             if len(prompt_negative) > 0:
#                 prompt_negative = f"({prompt_negative}:{strength})"
        
#         p.prompt = promptAdded(p.prompt, prompt_positive)
#         p.negative_prompt = promptAdded(p.negative_prompt, prompt_negative)

# class B_Ui_Prompt_Range(B_Ui):
#     _value_min: int = -1
#     _value_max: int = 100
#     _value_step: int = 1

#     @staticmethod
#     def _paramsFromArgs(args: dict[str, str]) -> tuple[str, str, bool, bool, int, bool, str, str, bool]:
#         prompt_a = args.get("a", "")
#         prompt_b = args.get("b", "")
#         required = bool(int(args.get("is_required", 0)))
#         negative = bool(int(args.get("n", 0)))
#         value = int(args.get("v", 0)) #!
#         ui_buttons_build = bool(int(args.get("build_buttons", 1)))
#         prompt_a_button_text = args.get("a_button", None)
#         prompt_b_button_text = args.get("b_button", None)
#         hidden = bool(int(args.get("hide", 0)))
#         return prompt_a, prompt_b, required, negative, value, ui_buttons_build, prompt_a_button_text, prompt_b_button_text, hidden
    
#     @staticmethod
#     def _fromArgs(args: dict[str, str], name: str = None):
#         prompt_a, prompt_b, required, negative, value, ui_buttons_build, prompt_a_button_text, prompt_b_button_text, hidden = B_Ui_Prompt_Range._paramsFromArgs(args)
#         return B_Ui_Prompt_Range(
#             name = name
#             , prompt_a = prompt_a
#             , prompt_b = prompt_b
#             , required = required
#             , negative = negative
#             , value = value
#             , ui_buttons_build = ui_buttons_build
#             , prompt_a_button_text = prompt_a_button_text
#             , prompt_b_button_text = prompt_b_button_text
#             , hidden = hidden
#         )
    
#     @staticmethod
#     def _getDefaultName() -> str:
#         return "Range Prompt"
    
#     def __init__(
#             self
#             , prompt_a: str
#             , prompt_b: str
#             , name: str = None
#             , required: bool = False
#             , negative: bool = False
#             , value: int = None
#             , ui_buttons_build: bool = True
#             , ui_negative_build: bool = True
#             , prompt_a_button_text: str = None
#             , prompt_b_button_text: str = None
#             , hidden: bool = False
#         ) -> None:
#         super().__init__(name, hidden)

#         self.prompt_a = prompt_a
#         self.prompt_b = prompt_b
#         self.required = required
#         self.negative = negative
#         self.value = value

#         self.value_min = self._value_min if not required else 0

#         self.ui_buttons_build = ui_buttons_build
#         self.ui_negative_build = ui_negative_build

#         self.ui_button_a_text = prompt_a_button_text
#         self.ui_button_b_text = prompt_b_button_text

#         self.ui_range: Gr_Slider = None
#         self.ui_negative: Gr_Checkbox = None
#         self.ui_button_a: Gr_Button = None
#         self.ui_button_b: Gr_Button = None
    
#     def init(self, gr_outputs: list[Gr_Output], gr_outputs_extras: list[Gr_Output], bMap: dict[str, B_Ui]) -> None:
#         self.ui_range = Gr_Slider(self.name, self.value, self.value_min, self._value_max, self._value_step)
#         gr_outputs.append(self.ui_range)

#         self.ui_negative = Gr_Checkbox(f"{self.name} (N)", self.negative)
#         gr_outputs.append(self.ui_negative)

#         self.ui_button_a = Gr_Button(self.ui_button_a_text)
#         gr_outputs_extras.append(self.ui_button_a)

#         self.ui_button_b = Gr_Button(self.ui_button_b_text)
#         gr_outputs_extras.append(self.ui_button_b)
    
#     def validateArgs(
#             self
#             , prompt_a: str
#             , prompt_b: str
#             , required: bool
#             , negative: bool
#             , value: int
#             , ui_buttons_build: bool
#             , prompt_a_button_text: str
#             , prompt_b_button_text: str
#             , hidden: bool
#         ) -> list[tuple[bool, str]]:
#         return [
#             self.ui_range.validateValue(value)
#             , self.ui_negative.validateValue(negative)
#         ]
    
#     def getOutputUpdatesFromArgs(
#             self
#             , updates: list
#             , reset: bool
#             , prompt_a: str
#             , prompt_b: str
#             , required: bool
#             , negative: bool
#             , value: int
#             , ui_buttons_build: bool
#             , prompt_a_button_text: str
#             , prompt_b_button_text: str
#             , hidden: bool
#         ) -> int:
#         if self.ui_range.isGrBuilt():
#             self.ui_range.syncInput(value)
#         if self.ui_negative.isGrBuilt():
#             self.ui_negative.syncInput(negative)
        
#         return super().getOutputUpdatesFromArgs(updates, reset)
    
#     def buildUI(self) -> None:
#         self.ui_range.initGr()

#         if self.ui_buttons_build:
#             with gr.Row():
#                 self.ui_button_a.initGr()
#                 self.ui_button_b.initGr()
        
#         if self.ui_negative_build:
#             self.ui_negative.initGr()
    
#     def finalizeUI(self, bMap: dict[str, B_Ui]) -> None:
#         if self.ui_buttons_build:
#             self.ui_button_a.gr.click(
#                 fn = lambda: 0
#                 , outputs = self.ui_range.gr
#             )

#             self.ui_button_b.gr.click(
#                 fn = lambda: self.ui_range.value_max
#                 , outputs = self.ui_range.gr
#             )
    
#     def handlePrompt(self, p: StableDiffusionProcessing, bMap: dict[str, B_Ui]) -> None:
#         prompt_range = int(self.ui_range.value)
#         negative = bool(self.ui_negative.value)

#         if prompt_range < 0:
#             return
        
#         prompt_a = promptSanitized(self.prompt_a)
#         prompt_b = promptSanitized(self.prompt_b)
        
#         value = float(prompt_range)
#         value = round(value / 100, 2)
#         value = 1 - value

#         prompt: str = None
#         if value == 1:
#             prompt = prompt_a
#         elif value == 0:
#             prompt = prompt_b
#         else:
#             prompt = f"[{prompt_a}:{prompt_b}:{value}]"

#         if not negative:
#             p.prompt = promptAdded(p.prompt, prompt)
#         else:
#             p.negative_prompt = promptAdded(p.negative_prompt, prompt)

# class B_Ui_Prompt_Select(B_Ui_Collection):
#     _choice_empty: str = "-"
#     _random_choices_max: int = 5
#     _choice_ui_min_width: int = 160

#     @staticmethod
#     def _paramsFromArgs(args: dict[str, str]) -> tuple[bool, str | list[str], bool, bool, bool, int, str, str, bool]:
#         choices_default = args.get("v", None)
#         if choices_default is not None:
#             choices_default = choices_default.strip()
#             if len(choices_default) > 0:
#                 choices_default = list(map(lambda v: v.strip(), choices_default.split(",")))
#             else:
#                 choices_default = None
        
#         choices_sort = bool(int(args.get("sort", 1)))
#         multiselect = bool(int(args.get("multi_select", 0)))
#         custom = bool(int(args.get("allow_custom", 0)))
#         simple = bool(int(args.get("simple", 0)))
#         scale = int(args.get("scale", 1))
#         prefix = args.get("prefix", None)
#         postfix = args.get("postfix", None)
#         hidden = bool(int(args.get("hide", 0)))

#         return choices_sort, choices_default, multiselect, custom, simple, scale, prefix, postfix, hidden

#     @staticmethod
#     def _fromArgs(args: dict[str, str], name: str = None):
#         choices_sort, choices_default, multiselect, custom, simple, scale, prefix, postfix, hidden = B_Ui_Prompt_Select._paramsFromArgs(args)
#         return B_Ui_Prompt_Select(
#             name = name
#             , choices_sort = choices_sort
#             , choices_default = choices_default
#             , multiselect = multiselect
#             , custom = custom
#             , simple = simple
#             , scale = scale
#             , prefix = prefix
#             , postfix = postfix
#             , hidden = hidden
#         )
    
#     @staticmethod
#     def _getDefaultName() -> str:
#         return "Select Prompt"
    
#     #! conflicting names...
#     @staticmethod
#     def _buildColorChoicesList(postfix: str = "") -> list[B_Ui_Prompt_Single]:
#         return list(map(
#             lambda text: B_Ui_Prompt_Single(
#                 name = f"{text} {postfix}"
#                 , prompt_ui_build = False
#                 , postfix = postfix
#                 , prompt = text.lower())
#             , [
#                 "Dark"
#                 , "Light"
#                 , "Black"
#                 , "Grey"
#                 , "White"
#                 , "Brown"
#                 , "Blue"
#                 , "Green"
#                 , "Red"
#                 , "Blonde"
#                 , "Rainbow"
#                 , "Pink"
#                 , "Purple"
#                 , "Orange"
#                 , "Yellow"
#                 , "Multicolored"
#                 , "Pale"
#                 , "Silver"
#                 , "Gold"
#                 , "Tan"
#             ]
#         ))

#     def __init__(
#             self
#             , name: str = None
#             , choices: list[B_Ui] = None
#             , choices_sort: bool = True
#             , choices_default: str | list[str] = None
#             , multiselect: bool = False
#             , custom: bool = False #!
#             , simple: bool = False
#             , scale: int = 1 #!
#             , prefix: str = None
#             , postfix: str = None
#             , hidden: bool = False
#         ) -> None:
#         super().__init__(name, choices, choices_sort, hidden)

#         self.choices_default = choices_default if choices_default is not None or not multiselect else []
#         self.multiselect = multiselect
#         self.prefix = prefix
#         self.postfix = postfix
#         self.simple = simple
#         self.scale = scale

#         self.choicesMap: dict[str, B_Ui] = {}
#         self.choicesContainerMap: dict[str, Gr_Column] = {}
#         self.choicesPresetMap: dict[str, B_Ui_Preset] = {}

#         self.ui_dropdown: Gr_Dropdown = None
#         self.ui_container_contents: Gr_Row = None
    
#     def init(self, gr_outputs: list[Gr_Output], gr_outputs_extras: list[Gr_Output], bMap: dict[str, B_Ui]) -> None:
#         super().init(gr_outputs, gr_outputs_extras, bMap)

#         # INIT choicesMap + choices_list + extras
#         choices_list: list[str] = []
        
#         if not self.multiselect:
#             choices_list.append(self._choice_empty)

#         for x in self.items:
#             choices_list.append(x.name)

#             self.choicesMap[x.name] = x
            
#             if type(x) is B_Ui_Prompt_Single:
#                 if self.prefix is not None:
#                     x.prefix = self.prefix
#                 if self.postfix is not None:
#                     x.postfix = self.postfix
                
#                 x.ui_prompt_build = False
#                 x.ui_negative_build = not self.simple
#                 x.ui_strength_build = not self.simple
#             elif type(x) is B_Ui_Prompt_Dual:
#                 x.ui_prompts_build = False
#                 x.ui_strength_build = not self.simple
#             elif type(x) is B_Ui_Prompt_Range:
#                 x.ui_buttons_build = False
#                 x.ui_negative_build = False

#         # INIT choicesPresetMap
#         for preset in self.choicesPresetMap.values():
#             preset.buildMappings(bMap)

#         self.ui_dropdown = Gr_Dropdown(self.name, choices_list, self.choices_default, self.multiselect)
#         gr_outputs.append(self.ui_dropdown)

#         self.ui_container_contents = Gr_Row("panel", self.getShowContainer(), f"{self.name} (Contents)")
#         gr_outputs_extras.append(self.ui_container_contents)
#         for x in self.items:
#             x_container = Gr_Column(1, "panel", self._choice_ui_min_width, self.getShowChoiceContainer(x), f"{self.name}_{x.name} (Container)")
#             gr_outputs_extras.append(x_container)
#             self.choicesContainerMap[x.name] = x_container
    
#     def validate(self, bMap: dict[str, B_Ui]) -> bool:
#         valid = super().validate(bMap)

#         for preset in self.choicesPresetMap.values():
#             if not preset.validate(bMap):
#                 valid = False
        
#         return valid
    
#     def validateArgs(
#             self
#             , choices_sort: bool
#             , choices_selected: str | list[str]
#             , multiselect: bool
#             , custom: bool
#             , simple: bool
#             , scale: int
#             , prefix: str
#             , postfix: str
#             , hidden: bool
#         ) -> list[tuple[bool, str]]:
#         return [
#             self.ui_dropdown.validateValue(choices_selected)
#         ]
    
#     def getOutputUpdatesFromArgs(
#             self
#             , updates: list
#             , reset: bool
#             , choices_sort: bool
#             , choices_selected: str | list[str]
#             , multiselect: bool
#             , custom: bool
#             , simple: bool
#             , scale: int
#             , prefix: str
#             , postfix: str
#             , hidden: bool
#         ) -> int:
#         if self.ui_dropdown.isGrBuilt():
#             self.ui_dropdown.syncInput(choices_selected)
        
#         return super().getOutputUpdatesFromArgs(updates, reset)
    
#     def initContainer(self, visible: bool) -> Gr_Container:
#         return Gr_Column(self.scale, visible = visible, name = f"{self.name} (Container)")
    
#     def buildUI(self) -> None:
#         self.ui_dropdown.initGr()

#         self.ui_container_contents.initGr()
#         with self.ui_container_contents.gr:
#             super().buildUI()
    
#     def initItemUI(self, item: B_Ui) -> None:
#         item_container = self.choicesContainerMap[item.name]
#         item_container.initGr()
#         with item_container.gr:
#             super().initItemUI(item)
    
#     def finalizeUI(self, bMap: dict[str, B_Ui]) -> None:
#         super().finalizeUI(bMap)

#         # Show/Hide selected
#         def _fnShowHide(choices) -> list:
#             updates: list = []
#             self.ui_dropdown.syncInput(choices)
#             self.getOutputUpdatesExtra(updates, self.gr_outputs_extras)
#             return updates
        
#         self.ui_dropdown.gr.input(
#             fn = lambda choices: _fnShowHide(choices)
#             , inputs = self.ui_dropdown.gr
#             , outputs = [self.ui_container_contents.gr] + list(map(lambda c: c.gr, self.choicesContainerMap.values()))
#         )

#         #! Presets - potentially repeating logic from B_Ui_Preset, encapsulate some of this?
#         if len(self.choicesPresetMap) > 0:
#             bList: list[B_Ui] = []
#             for c in self.choicesPresetMap:
#                 for k in self.choicesPresetMap[c].mappings:
#                     b = bMap[k]
#                     if b not in bList:
#                         bList.append(b)
            
#             inputs: list = []
#             outputs: list = []
#             for b in bList:
#                 for gr_input in b.getInput(base_only = True):
#                     inputs.append(gr_input.gr)
#                 for gr_output in b.getOutput(exclude_labeled_outputs = True, base_only = True):
#                     outputs.append(gr_output.gr)
            
#             def _apply(choices: str | list[str], *inputValues):
#                 self.ui_dropdown.syncInput(choices)

#                 if type(choices) is not list:
#                     choices: list = [choices]
                
#                 presets: list[B_Ui_Preset] = []
#                 for c in choices:
#                     preset = self.choicesPresetMap.get(c, None)
#                     if preset is not None:
#                         presets.append(preset)
                
#                 updates: list = []
#                 offset: int = 0
#                 preset_mapping: tuple = None
#                 for b in bList:
#                     for preset in presets:
#                         preset_mapping = preset.mappings.get(b.name, None)
#                         if preset_mapping is not None:
#                             break #! if 2 choices affect same element, first one prioritized
                    
#                     if preset_mapping is not None:
#                         offset += b.getOutputUpdatesFromArgs(updates, False, *preset_mapping)
#                     else:
#                         offset += b.getOutputUpdates(updates, False, True, *inputValues[offset:])
                
#                 return updates
            
#             self.ui_dropdown.gr.select(
#                 fn = _apply
#                 , inputs = [self.ui_dropdown.gr] + inputs
#                 , outputs = outputs
#             )
    
#     def getOutputUpdatesExtra(self, updates: list, gr_outputs_extras: list[Gr_Output]) -> None:
#         self.ui_container_contents.getUpdateVisible(updates, self.getShowContainer())
#         for x in self.items:
#             self.choicesContainerMap[x.name].getUpdateVisible(updates, self.getShowChoiceContainer(x))
    
#     def handlePrompt(self, p: StableDiffusionProcessing, bMap: dict[str, B_Ui]) -> None:
#         if self.ui_dropdown.value is None or len(self.ui_dropdown.value) == 0:
#             return
        
#         items_selected: list[B_Ui] = []
#         if type(self.ui_dropdown.value) is not list:
#             items_selected.append(self.choicesMap.get(self.ui_dropdown.value, None)) #! could maybe allow custom values here
#         else:
#             for v in self.ui_dropdown.value:
#                 items_selected.append(self.choicesMap[v])
        
#         if len(items_selected) == 0 or items_selected[0] is None:
#             return
        
#         for x in items_selected:
#             x.handlePrompt(p, bMap)
    
#     def addChoicePresetMapping(self, name: str, target_name: str, target_args: dict[str, str]):
#         preset = self.choicesPresetMap.get(name, None)
        
#         if preset is None:
#             preset = B_Ui_Preset(f"{self.name}_{name} (PRESET)", additive = True, hidden = True)
#             self.choicesPresetMap[name] = preset
        
#         preset.addMapping(target_name, target_args)
    
#     def addChoices(self, args: dict[str, str]):
#         choicesList: list[B_Ui] = []
        
#         special_type = args.get("type", "")
#         match special_type:
#             case "COLOR":
#                 postfix = args.get("postfix", "")
#                 choicesList += self._buildColorChoicesList(postfix)
#             case _:
#                 print(f"WARNING: Invalid CHOICES type in {self.name} -> {special_type}")
        
#         self.items += choicesList
    
#     def getShowContainer(self) -> bool:
#         if self.simple:
#             return False
        
#         choice_current = self.ui_dropdown.value
#         return choice_current is not None and len(choice_current) > 0 and choice_current != self._choice_empty
    
#     def getShowChoiceContainer(self, x: B_Ui) -> bool:
#         if self.simple:
#             return False
        
#         choice_current = self.ui_dropdown.value
#         choice = x.name
#         return choice_current is not None and (choice == choice_current or choice in choice_current)

# class B_Ui_Prompt_Range_Link(B_Ui):
#     @staticmethod
#     def _paramsFromArgs(args: dict[str, str]) -> tuple[str, str, str, bool]:
#         name_link = args.get("link", None)
#         prompt_a = args.get("a", "")
#         prompt_b = args.get("b", "")
#         hidden = bool(int(args.get("hide", 0)))
#         return name_link, prompt_a, prompt_b, hidden
    
#     @staticmethod
#     def _fromArgs(args: dict[str, str], name: str = None):
#         name_link, prompt_a, prompt_b, hidden = B_Ui_Prompt_Range_Link._paramsFromArgs(args)
#         return B_Ui_Prompt_Range_Link(
#             name = name
#             , name_link = name_link
#             , prompt_a = prompt_a
#             , prompt_b = prompt_b
#             , hidden = hidden
#         )
    
#     @staticmethod
#     def _getDefaultName() -> str:
#         return "Range Prompt {LINK}"
    
#     def __init__(
#             self
#             , name: str
#             , name_link: str
#             , prompt_a: str
#             , prompt_b: str
#             , hidden: bool = False
#         ) -> None:
#         super().__init__(name, hidden)

#         self.name_link = name_link
#         self.prompt_a = prompt_a
#         self.prompt_b = prompt_b

#         self.ui: Gr_Markdown = None
    
#     def init(self, gr_outputs: list[Gr_Output], gr_outputs_extras: list[Gr_Output], bMap: dict[str, B_Ui]) -> None:
#         self.ui = Gr_Markdown(f"**{self.name}**")
    
#     def validate(self, bMap: dict[str, B_Ui]) -> bool:
#         valid = super().validate(bMap)

#         if self.name_link is None or len(self.name_link) == 0:
#             printWarning(self.__class__.__name__, self.name, f"Invalid link name -> {self.name_link}")
#             return False

#         b_link = bMap.get(self.name_link, None)

#         if b_link is None:
#             printWarning(self.__class__.__name__, self.name, f"No component found with linked name -> '{self.name_link}'")
#             return False
        
#         if type(b_link) is not B_Ui_Prompt_Range:
#             valid = False
#             printWarning(self.__class__.__name__, self.name, f"Linked component type is invalid -> {b_link.__class__.__name__}")
        
#         return valid
    
#     def buildUI(self) -> None:
#         self.ui.initGr()
    
#     def handlePrompt(self, p: StableDiffusionProcessing, bMap: dict[str, B_Ui]) -> None:
#         b_link: B_Ui_Prompt_Range = bMap[self.name_link]

#         prompt_a = promptSanitized(self.prompt_a)
#         prompt_b = promptSanitized(self.prompt_b)

#         negative = bool(b_link.ui_negative.value)

#         value = float(b_link.ui_range.value)
#         value = round(value / 100, 2)
#         value = 1 - value

#         prompt: str = None
#         if value == 1:
#             prompt = prompt_a
#         elif value == 0:
#             prompt = prompt_b
#         else:
#             prompt = f"[{prompt_a}:{prompt_b}:{value}]"
        
#         if not negative:
#             p.prompt = promptAdded(p.prompt, prompt)
#         else:
#             p.negative_prompt = promptAdded(p.negative_prompt, prompt)

# class B_Ui_Preset(B_Ui):
#     @staticmethod
#     def _paramsFromArgs(args: dict[str, str]) -> tuple[bool, bool]:
#         additive = bool(int(args.get("is_additive", 0)))
#         hidden = bool(int(args.get("hide", 0)))
#         return additive, hidden
    
#     @staticmethod
#     def _fromArgs(args: dict[str, str], name: str = None):
#         additive, hidden = B_Ui_Preset._paramsFromArgs(args)
#         return B_Ui_Preset(
#             name = name
#             , additive = additive
#             , hidden = hidden
#         )
    
#     @staticmethod
#     def _getDefaultName() -> str:
#         return "Preset"
    
#     def __init__(
#             self
#             , name: str = None
#             , mappings: dict[str, tuple] = None
#             , additive: bool = False
#             , hidden: bool = False
#         ) -> None:
#         super().__init__(name, hidden)

#         self.mappings = mappings if mappings is not None else {}
#         self.additive = additive
        
#         self.mappings_temp: dict[str, dict[str, str]] = {}

#         self.ui: Gr_Button = None
    
#     def init(self, gr_outputs: list[Gr_Output], gr_outputs_extras: list[Gr_Output], bMap: dict[str, B_Ui]) -> None:
#         self.ui = Gr_Button(self.name)
#         gr_outputs_extras.append(self.ui)

#         self.buildMappings(bMap)
    
#     def validate(self, bMap: dict[str, B_Ui]) -> bool:
#         valid = super().validate(bMap)

#         if len(self.mappings) == 0:
#             printWarning(self.__class__.__name__, self.name, "No entries set")
#             return False
        
#         for k in self.mappings:
#             if k not in bMap:
#                 valid = False
#                 printWarning(self.__class__.__name__, self.name, f"Entry not found -> '{k}'")
#             else:
#                 b = bMap[k]
#                 for v_valid, v_message in b.validateArgs(*self.mappings[k]):
#                     if not v_valid:
#                         valid = False
#                         printWarning(self.__class__.__name__, self.name, v_message)
        
#         return valid
    
#     def buildUI(self) -> None:
#         self.ui.initGr()
    
#     def finalizeUI(self, bMap: dict[str, B_Ui]) -> None:
#         inputs: list = []
#         outputs: list = []

#         bList: list[B_Ui] = None
#         if self.additive:
#             bList = []
#             for b in bMap.values():
#                 if b.name in self.mappings:
#                     bList.append(b)
#         else:
#             bList = list(bMap.values())
        
#         for b in bList:
#             for gr_input in b.getInput(base_only = True):
#                 inputs.append(gr_input.gr)
#             for gr_output in b.getOutput(exclude_labeled_outputs = True, base_only = True):
#                 outputs.append(gr_output.gr)
        
#         def _apply(*inputValues) -> list:
#             updates: list = []
#             offset: int = 0

#             for x in bList:
#                 if x.name in self.mappings:
#                     offset += x.getOutputUpdatesFromArgs(updates, False, *self.mappings[x.name])
#                 elif self.additive:
#                     offset += x.getOutputUpdates(updates, False, True, *inputValues[offset:])
#                 else:
#                     offset += x.getOutputUpdates(updates, True, True)
            
#             return updates

#         self.ui.gr.click(
#             fn = _apply
#             , inputs = inputs
#             , outputs = outputs
#         )
    
#     def handlePrompt(self, p: StableDiffusionProcessing, bMap: dict[str, B_Ui]) -> None:
#         pass

#     def buildMappings(self, bMap: dict[str, B_Ui]) -> None:
#         if len(self.mappings_temp) > 0:
#             for k in self.mappings_temp:
#                 self.mappings[k] = bMap[k]._paramsFromArgs(self.mappings_temp[k])
            
#             del self.mappings_temp #!

#     def addMapping(self, name: str, args: dict[str, str]):
#         if name in self.mappings_temp:
#             printWarning(self.__class__.__name__, self.name, f"Duplicate entry ({name})")
        
#         self.mappings_temp[name] = args

# #: Main UI Wrapper
# class B_Ui_Map():
#     @staticmethod
#     def readLine(l: str) -> tuple[str, str, dict[str, str]]:
#         #! TODO: Fix empty str l_name
#         l = l.strip()
        
#         l_type: str = l
#         l_name: str = None
#         l_args: dict[str, str] = {}
        
#         if len(l) > 0:
#             index = l.find(" ")
#             if index != -1:
#                 l_type = l[:index]
            
#             l = l[len(l_type) + 1:]
            
#             l_arg_index = l.find("--")
#             if l_arg_index == -1:
#                 l_name = l
#             elif l_arg_index > 0:
#                 l_name = l[:l_arg_index - 1 if l_arg_index > -1 else len(l)]
#                 l = l[len(l_name) + 1:]
            
#             l_args = {}
#             for l_arg in l.split("--")[1:]:
#                 l_arg_name = l_arg[:l_arg.index(" ")]
#                 l_arg_value = l_arg[len(l_arg_name) + 1:].strip()
#                 l_args[l_arg_name] = l_arg_value
            
#         return l_type, l_name, l_args

#     def __init__(self) -> None:
#         self.path_script_config = os.path.join(b_path_base, b_folder_name_scripts, b_folder_name_script_config)
#         self.path_layout = os.path.join(self.path_script_config, b_file_name_layout)
#         self.path_presets = os.path.join(self.path_script_config, b_file_name_presets)

#         # PARSE
#         self.layout = self.parseLayout()
#         self.presets = self.parsePresets()

#         # INIT
#         self.map: dict[str, B_Ui] = {}
#         self.buildMapRecursive(self.map, self.layout)

#         for x in self.presets:
#             x.init_main(self.map)

#         for x in self.layout:
#             x.init_main(self.map)
        
#         # VALIDATE
#         if not b_validate_skip:
#             valid: bool = True

#             for preset in self.presets:
#                 if not preset.validate(self.map):
#                     valid = False

#             for x in self.layout:
#                 if not x.validate(self.map):
#                     valid = False
            
#             if not valid:
#                 printWarning("B_Ui_Map", "validate()", "Invalid layout or presets")
    
#     def parseLayout(self) -> list[B_Ui]:
#         layout: list[B_Ui] = []
        
#         stack_containers: list[B_Ui_Container] = []
#         stack_selects: list[B_Ui_Prompt_Select] = []
#         select_choice_has_preset: bool = False

#         skip = 0
        
#         def _build(item: B_Ui) -> None:
#             if len(stack_selects) > 0:
#                 stack_selects[-1].addItem(item)
#                 return
            
#             if len(stack_containers) > 0:
#                 stack_containers[-1].addItem(item)
#                 return
            
#             layout.append(item)
        
#         with open(self.path_layout) as file_layout:
#             line_number: int = 0

#             for l in file_layout:
#                 line_number += 1

#                 if l.lstrip().startswith("#"):
#                     print(f"# LAYOUT - commented out line @{line_number}")
#                     continue
                
#                 l_type, l_name, l_args = self.readLine(l)
                
#                 if len(l_type) == 0:
#                     continue
                    
#                 if l_type == ".":
#                     break
                
#                 if l_type == "END":
#                     if skip == 0:
#                         if select_choice_has_preset:
#                             select_choice_has_preset = False
#                             continue

#                         if len(stack_selects) > 0:
#                             item_select = stack_selects.pop()
#                             _build(item_select)
#                             continue
                        
#                         if len(stack_containers) > 0:
#                             item_container = stack_containers.pop()
#                             _build(item_container)
#                             continue

#                         continue
                    
#                     skip -= 1
                    
#                     continue
                
#                 ignore: bool = skip > 0
#                 if l_args.get("x", "") == "1":
#                     if not ignore:
#                         ignore = b_tagged_ignore
                
#                 match l_type:
#                     case "SINGLE":
#                         if ignore:
#                             continue

#                         _build(B_Ui_Prompt_Single._fromArgs(l_args, l_name))
                    
#                     case "DUAL":
#                         if ignore:
#                             continue

#                         _build(B_Ui_Prompt_Dual._fromArgs(l_args, l_name))
                    
#                     case "RANGE":
#                         if ignore:
#                             continue
                        
#                         _build(B_Ui_Prompt_Range._fromArgs(l_args, l_name))
                    
#                     case "RANGE_LINK":
#                         if ignore:
#                             continue
                        
#                         _build(B_Ui_Prompt_Range_Link._fromArgs(l_args, l_name))
                    
#                     case "SELECT":
#                         if ignore:
#                             skip += 1
#                             continue
                        
#                         stack_selects.append(B_Ui_Prompt_Select._fromArgs(l_args, l_name))
                    
#                     case "CHOICES":
#                         if ignore:
#                             continue

#                         stack_selects[-1].addChoices(l_args)
                    
#                     case "SET":
#                         select_choice_has_preset = True
#                         stack_selects[-1].addChoicePresetMapping(stack_selects[-1].items[-1].name, l_name, l_args)
                    
#                     case "GROUP":
#                         if ignore:
#                             skip += 1
#                             continue

#                         stack_containers.append(B_Ui_Container_Group._fromArgs(l_args, l_name))
                    
#                     case "TAB":
#                         if ignore:
#                             skip += 1
#                             continue
                        
#                         stack_containers.append(B_Ui_Container_Tab._fromArgs(l_args, l_name))
                    
#                     case "ROW":
#                         if ignore:
#                             skip += 1
#                             continue
                        
#                         stack_containers.append(B_Ui_Container_Row._fromArgs(l_args, l_name))
                    
#                     case "COLUMN":
#                         if ignore:
#                             skip += 1
#                             continue
                        
#                         stack_containers.append(B_Ui_Container_Column._fromArgs(l_args, l_name))
                    
#                     case "ACCORDION":
#                         if ignore:
#                             skip += 1
#                             continue
                        
#                         stack_containers.append(B_Ui_Container_Accordion._fromArgs(l_args, l_name))
                    
#                     case "SEPARATOR":
#                         if ignore:
#                             continue
                        
#                         _build(B_Ui_Separator._fromArgs(l_args))

#                     case _:
#                         print(f"WARNING: Invalid layout type -> {l_type}")
        
#         return layout
    
#     def parsePresets(self) -> list[B_Ui_Preset]:
#         presets: list[B_Ui_Preset] = []
        
#         preset_current: B_Ui_Preset = None
        
#         with open(self.path_presets) as file_presets:
#             line_number: int = 0

#             for l in file_presets:
#                 line_number += 1

#                 if l.lstrip().startswith("#"):
#                     print(f"# PRESETS - commented out line @{line_number}")
#                     continue

#                 l_type, l_name, l_args = self.readLine(l)
                
#                 if len(l_type) == 0:
#                     continue
                    
#                 if l_type == ".":
#                     break
                
#                 if l_type == "END":
#                     presets.append(preset_current)
#                     preset_current = None
#                     continue
                
#                 match l_type:
#                     case "PRESET":
#                         preset_current = B_Ui_Preset._fromArgs(l_args, l_name)
                    
#                     case "SET":
#                         preset_current.addMapping(l_name, l_args)
                    
#                     case _:
#                         print(f"WARNING: Invalid preset type -> {l_type}")
        
#         return presets

#     def buildMapRecursive(self, target: dict[str, B_Ui], layout: list[B_Ui]):
#         for x in layout:
#             x_type = type(x)

#             if not issubclass(x_type, B_Ui_Collection) or x_type is B_Ui_Prompt_Select:
#                 if x.name in target:
#                     printWarning("B_Ui_Map", "buildMapRecursive()", f"Duplicate B_Ui name -> '{x.name}'")
                
#                 target[x.name] = x
            
#             if issubclass(x_type, B_Ui_Collection):
#                 x_collection: B_Ui_Collection = x
#                 self.buildMapRecursive(target, x_collection.items)
    
#     def initUI(self) -> list[typing.Any]:
#         gr_list: list[typing.Any] = []

#         # PRESETS
#         B_Ui_Separator._build()

#         with gr.Accordion("Presets", open = False):
#             i = 0
#             for preset in self.presets:
#                 preset.initUI()
#                 gr_list += list(map(lambda gr_output: gr_output.gr, preset.getOutput(True)))

#                 i += 1
#                 if i < len(self.presets) and not preset.hidden:
#                     B_Ui_Separator._build()

#         # LAYOUT
#         B_Ui_Separator._build()
        
#         for x in self.layout:
#             x.initUI()
#             gr_list += list(map(lambda gr_output: gr_output.gr, x.getOutput(True)))
        
#         # SETTINGS
#         B_Ui_Separator._build()

#         with gr.Accordion("Settings", open = False):
#             btnClearConfig = gr.Button("Clear config")
#             btnClearConfig.click(fn = self.clearConfigFile)
            
#             gr_list.append(btnClearConfig)
        
#         # (FINALIZE)
#         for preset in self.presets:
#             preset.finalizeUI(self.map)

#         for x in self.layout:
#             x.finalizeUI(self.map)
        
#         # - DONE -
#         return gr_list
    
#     #! Would be better if the original config file dump function is used somehow:
#     def clearConfigFile(self):
#         path = os.path.join(b_path_base, b_file_name_config)
#         with open(path, "r+", encoding = "utf-8") as file_config:
#             config: dict[str, typing.Any] = json.load(file_config)
            
#             config_keys = filter(lambda k: k.find(b_folder_name_script_config) == -1, config.keys())

#             config_new: dict[str, typing.Any] = {}
#             for k in config_keys:
#                 config_new[k] = config[k]
            
#             file_config.seek(0)
#             json.dump(config_new, file_config, indent = 4)
#             file_config.truncate()

#: Webui script
class Script(scripts.Script):
    #bUiMap = B_Ui_Map()
    b_ui_master = B_UI_Master([
        B_UI_Container_Tab("Tab 1", [
            B_UI_Prompt("Base Prompt", B_Prompt_Dual("Base Prompt", prompt_negative_value = "low res, lowres, blurry, low quality, bad anatomy, bad body anatomy, unusual anatomy, letterbox, deformity, mutilated, malformed, amputee, sketch, monochrome"), True)
            , B_UI_Dropdown("Dropdown 1.1", [
                B_Prompt_Single("Single Prompt 1.1.1")
                , B_Prompt_Dual("Dual Prompt 1.1.2")
            ])
            , B_UI_Dropdown("Dropdown 1.2", [
                B_Prompt_Single("Single Prompt 1.2.1")
                , B_Prompt_Dual("Dual Prompt 1.2.2")
            ])
        ])
    ])
    
    def title(self):
        return "B Prompt Builder"

    def show(self, is_img2img):
        return not is_img2img
    
    def ui(self, is_img2img):
        #return self.bUiMap.initUI()
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
