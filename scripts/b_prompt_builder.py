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

b_prompt_strength_min: float = 0
b_prompt_strength_step: float = 0.1

def printWarning(component: str, name: str, message: str):
    print(f"VALIDATE/{component}/{name} -> {message}")

def promptAdded(promptExisting: str, promptToAdd: str) -> str:
    if len(promptToAdd) > 0:
        if len(promptExisting) > 0:
            promptExisting += ", " + promptToAdd
        else:
            promptExisting = promptToAdd
    
    return promptExisting

def promptSanitized(prompt: str):
    return prompt.strip() if prompt is not None else ""

class Gr_Wrapper(ABC):
    _identifier: int = 0

    @staticmethod
    def _getNextIdentifier() -> int:
        Gr_Wrapper._identifier += 1
        return Gr_Wrapper._identifier

    def __init__(self, name: str, is_labeled: bool) -> None:
        self.name = name

        self.is_labeled = is_labeled

        self.identifier = self._getNextIdentifier()
        self.key = self.buildKey()

        self.gr: typing.Any = None
    
    def buildKey(self) -> str:
        return f"{self.identifier}_{self.name}"
    
    def initGr(self) -> None:
        self.gr = self.buildGr()
    
    def printWarning(self, message: str) -> None:
        printWarning(self.__class__.__name__, self.name, message)
    
    def validate(self) -> bool:
        """VIRTUAL: Base -> True"""
        return True

    @abstractmethod
    def buildGr(self) -> typing.Any:
        pass

    @abstractmethod
    def getOutputUpdate(self, reset: bool, *value_inputs) -> typing.Any:
        pass

#!!!
class Gr_Markdown(Gr_Wrapper):
    def __init__(self, value: str, name: str = "Markdown") -> None:
        super().__init__(name, False)

        self.value = value
    
    def buildGr(self) -> typing.Any:
        return gr.Markdown(value = self.value)
    
    def getOutputUpdate(self, reset: bool, *value_inputs) -> typing.Any:
        self.value = str(value_inputs[0])

        return self.gr.update(value = self.value)

class Gr_Output(Gr_Wrapper, ABC):
    def __init__(self, name: str, is_labeled: bool, is_input: bool = False) -> None:
        super().__init__(name, is_labeled)

        self.is_input = is_input

class Gr_Input(Gr_Output, ABC):
    def __init__(self, label: str, value_default: typing.Any = None) -> None:
        super().__init__(label, True, True)

        self.value_default = value_default

        self.value = self.buildDefaultValue()
    
    def syncInput(self, value: typing.Any):
        self.value = value
    
    def getOutputUpdate(self, reset: bool, *value_inputs) -> typing.Any:
        self.syncInput(value_inputs[0] if len(value_inputs) > 0 else None)

        if reset and self.value is None:
            self.syncInput(self.buildDefaultValue())
        
        return self.getUpdate(self.value)
    
    def buildDefaultValue(self) -> typing.Any:
        """VIRTUAL: Base -> self.value_default"""
        return self.value_default
    
    def getUpdate(self, value: typing.Any) -> typing.Any:
        """VIRTUAL"""
        return self.gr.update(value = value)

class Gr_Textbox(Gr_Input):
    def __init__(self, label: str = "Textbox", value_default: str = "") -> None:
        super().__init__(label, value_default)
    
    def buildGr(self) -> typing.Any:
        return gr.Textbox(
            label = self.name
            , value = self.value
        )

class Gr_Number(Gr_Input):
    def __init__(self, label: str = "Number", value_default: float = 0, value_min: float = 0, value_step: float = 0.1) -> None:
        super().__init__(label, value_default)

        self.value_min = value_min
        self.value_step = value_step
    
    def validate(self) -> bool:
        valid = super().validate()

        if self.value_default < self.value_min:
            valid = False
            self.printWarning(f"Default value is under minimum ({self.value_min})")
        
        return valid
    
    def buildGr(self) -> typing.Any:
        return gr.Number(
            label = self.name
            , value = self.value
            , minimum = self.value_min
            , step = self.value_step
        )
    
    def getUpdate(self, value: typing.Any) -> typing.Any:
        return self.gr.update(value = value, step = self.value_step)

class Gr_Checkbox(Gr_Input):
    def __init__(self, label: str = "Checkbox", value_default: bool = False) -> None:
        super().__init__(label, value_default)
    
    def buildGr(self) -> typing.Any:
        return gr.Checkbox(
            label = self.name
            , value = self.value
        )

class Gr_Slider(Gr_Input):
    def __init__(self, label: str = "Slider", value_default: int = 0, value_min: int = 0, value_max: int = 100, value_step: int = 1) -> None:
        super().__init__(label, value_default)

        self.value_min = value_min
        self.value_max = value_max
        self.value_step = value_step
    
    def validate(self) -> bool:
        valid = super().validate()

        if self.value_default < self.value_min:
            valid = False
            self.printWarning(f"Default value is under minimum ({self.value_min})")
        
        if self.value_default > self.value_max:
            valid = False
            self.printWarning(f"Default value is over maximum ({self.value_max})")
        
        return valid
    
    def buildGr(self) -> typing.Any:
        return gr.Slider(
            label = self.name
            , value = self.value
            , minimum = self.value_min
            , maximum = self.value_max
            , step = self.value_step
        )

class Gr_Dropdown(Gr_Input):
    def __init__(self, label: str = "Dropdown", choices: list[str] = None, value_default: str | list[str] = None, multiselect: bool = False) -> None:
        super().__init__(label, value_default)

        self.choices = choices if choices is not None else []
        self.multiselect = multiselect
    
    def validate(self) -> bool:
        valid = super().validate()

        if len(self.choices) > 0:
            if (
                self.value_default is not None
                and (
                    (
                        type(self.value_default) is not list
                        and self.value_default not in self.choices
                    ) or (
                        type(self.value_default) is list
                        and any(map(lambda c: c not in self.choices, self.value_default))
                    )
                )
            ):
                valid = False
                self.printWarning(f"Invalid default choice(s) -> {self.value_default}")
        elif self.value_default is not None:
            valid = False
            self.printWarning(f"No choices set but default choice(s) set -> {self.value_default}")
        
        return valid
    
    def buildDefaultValue(self) -> typing.Any:
        value_default = super().buildDefaultValue()

        if value_default is None or type(value_default) is not list:
            return value_default
        
        return value_default[:]
    
    def buildGr(self) -> typing.Any:
        return gr.Dropdown(
            label = self.name
            , choices = self.choices
            , multiselect = self.multiselect
            , value = self.value
            , allow_custom_value = False #!
        )

class Gr_Button(Gr_Output):
    def __init__(self, text: str = "Button") -> None:
        super().__init__(text, True)
    
    def buildGr(self) -> typing.Any:
        return gr.Button(value = self.name)
    
    def getOutputUpdate(self, reset: bool, *value_inputs) -> typing.Any:
        return self.gr.update(value = self.name)

class Gr_Container(Gr_Output, ABC):
    def __init__(self, name: str, is_labeled: bool, visible: bool = True) -> None:
        super().__init__(name, is_labeled)

        self.visible = visible
    
    def buildGr(self) -> typing.Any:
        return self.buildGrContainer(self.visible)
    
    def getOutputUpdate(self, reset: bool, *value_inputs) -> typing.Any:
        visible = value_inputs[0]

        if visible is not None:
            self.visible = bool(visible)
        
        return self.getOutputUpdateContainer(self.visible, reset, *value_inputs[1:])
    
    def getUpdateVisible(self, visible: bool) -> typing.Any:
        return self.getOutputUpdate(False, visible)
    
    @abstractmethod
    def buildGrContainer(self, visible: bool) -> typing.Any:
        pass

    @abstractmethod
    def getOutputUpdateContainer(self, visible: bool, reset: bool, *value_inputs) -> typing.Any:
        pass

class Gr_Row(Gr_Container):
    def __init__(self, variant: str = "default", visible: bool = True, name: str = "Row") -> None:
        super().__init__(name, False, visible)

        self.variant = variant
    
    def buildGrContainer(self, visible: bool) -> typing.Any:
        return gr.Row(
            variant = self.variant
            , visible = visible
        )
    
    def getOutputUpdateContainer(self, visible: bool, reset: bool, *value_inputs) -> typing.Any:
        # if len(value_inputs) > 0:
        #     self.variant = str(value_inputs[0])

        return self.gr.update(visible = visible)

class Gr_Column(Gr_Container):
    def __init__(self, scale: int = 1, variant: str = "default", visible: bool = True, name: str = "Column") -> None:
        super().__init__(name, False, visible)

        self.scale = scale
        self.variant = variant
    
    def buildGrContainer(self, visible: bool) -> typing.Any:
        return gr.Column(
            scale = self.scale
            , variant = self.variant
            , visible = visible
        )
    
    def getOutputUpdateContainer(self, visible: bool, reset: bool, *value_inputs) -> typing.Any:
        # if len(value_inputs) > 0:
        #     self.scale = int(value_inputs[0])
        #     self.variant = str(value_inputs[1])

        return self.gr.update(visible = visible)

class Gr_Group(Gr_Container):
    def __init__(self, visible: bool = True, name: str = "Group") -> None:
        super().__init__(name, False, visible)
    
    def buildGrContainer(self, visible: bool) -> typing.Any:
        return gr.Group(visible = visible)
    
    def getOutputUpdateContainer(self, visible: bool, reset: bool, *value_inputs) -> typing.Any:
        return self.gr.update(visible = visible)

#!!! Tab does not have visible prop..
class Gr_Tab(Gr_Container):
    def __init__(self, visible: bool = True, name: str = "Group (Tab)") -> None:
        super().__init__(name, False, visible)
    
    def buildGrContainer(self, visible: bool) -> typing.Any:
        gr_tab = gr.Tab(self.name)
        with gr_tab:
            gr_tab_group = gr.Group(visible = visible)
        return gr_tab_group
    
    def getOutputUpdateContainer(self, visible: bool, reset: bool, *value_inputs) -> typing.Any:
        return self.gr.update(visible = visible)

class Gr_Accordion(Gr_Container):
    def __init__(self, visible: bool = True, name: str = "Accordion") -> None:
        super().__init__(name, False, visible)
    
    def buildGrContainer(self, visible: bool) -> typing.Any:
        return gr.Accordion(self.name, visible = visible)
    
    def getOutputUpdateContainer(self, visible: bool, reset: bool, *value_inputs) -> typing.Any:
        return self.gr.update(visible = visible)

class B_Ui(ABC):
    @staticmethod
    @abstractmethod
    def _fromArgs(name: str = None, **args: str):
        pass
    
    def __init__(self, name: str, hidden: bool = False) -> None:
        self.name = name
        self.hidden = hidden

        self.gr_container: Gr_Container = None
        self.gr_outputs: list[Gr_Output] = []
    
    def init_main(self) -> None:
        self.gr_container = self.initContainer(not self.hidden)

        self.init(self.gr_outputs)
    
    def initUI(self) -> None:
        self.gr_container.initGr()
        with self.gr_container.gr:
            self.buildUI()
    
    def getInput(self) -> list[Gr_Input]:
        gr_list: list[Gr_Input] = []
        
        for gr_output in self.gr_outputs:
            if gr_output.gr is None:
                continue

            if not gr_output.is_input:
                continue
            
            gr_list.append(gr_output)

        return gr_list
    
    def getOutput(self, labeled_only: bool = False) -> list[Gr_Output]:
        gr_list: list[Gr_Output] = []

        for gr_output in self.gr_outputs:
            if gr_output.gr is None:
                continue

            if labeled_only and not gr_output.is_labeled:
                continue

            gr_list.append(gr_output)
        
        return gr_list
    
    #!!!
    def getOutputUpdates(self, reset: bool, *inputValues) -> tuple[list, int]:
        """Returns updates and number of values consumed"""
        updates: list = []
        offset: int = 0

        for x in self.gr_outputs:
            if x.is_input and x.gr is not None:
                updates.append(x.getOutputUpdate(reset, *inputValues[offset:]))
                offset += 1 #!
        
        return updates, offset
    
    def initContainer(self, visible: bool) -> Gr_Container:
        """VIRTUAL: Base -> Gr_Group"""
        return Gr_Group(visible, f"{self.name} (Group)")
    
    def validate(self) -> bool:
        """VIRTUAL: Base -> validate own Gr_Wrappers"""
        valid = True
        
        for x in self.gr_outputs:
            if not x.validate():
                valid = False
        
        return valid
    
    def syncInput(self, *outputValues) -> int:
        """VIRTUAL?: Base -> sync values on own Gr_Wrappers, returns number of values consumed"""
        offset: int = 0

        for x in self.gr_outputs:
            if x.is_input:
                x_input: Gr_Input = x
                x_input.syncInput(outputValues[offset])
            
            offset += 1
        
        offset += len(self.gr_outputs) #!
        
        return offset
    
    def finalizeUI(self, inputMap: dict[str, Gr_Input]) -> None:
        """VIRTUAL: Bindings, etc, Base -> Nothing"""
        pass
    
    @abstractmethod
    def init(self, gr_outputs: list[Gr_Output]) -> None:
        """Instantiates Gradio wrappers"""
        pass

    @abstractmethod
    def buildUI(self) -> None:
        """Builds Gradio layout and components from Gradio wrappers"""

    @abstractmethod
    def handlePrompt(self, p: StableDiffusionProcessing, inputMap: dict[str, Gr_Input]) -> None:
        pass

class B_Ui_Separator(B_Ui):
    _html_separator: str = "<hr style=\"margin: 0.5em 0; border-style: dotted; border-color: var(--border-color-primary);\" />"

    @staticmethod
    def _fromArgs(name: str = None, **args: str):
        return B_Ui_Separator(
            hidden = bool(int(args.get("hide", 0)))
        )
    
    @staticmethod
    def _build() -> None:
        Gr_Markdown(B_Ui_Separator._html_separator).initGr()
    
    def __init__(self, name: str = "Separator", hidden: bool = False) -> None:
        super().__init__(name, hidden)

        self.ui: Gr_Markdown = None
    
    def init(self, gr_outputs: list[Gr_Output]) -> None:
        self.ui = Gr_Markdown(self._html_separator, self.name)
    
    def buildUI(self) -> None:
        self.ui.initGr()
    
    def handlePrompt(self, p: StableDiffusionProcessing, inputMap: dict[str, Gr_Input]) -> None:
        pass

class B_Ui_Collection(B_Ui, ABC):
    def __init__(
            self
            , name: str
            , items: list[B_Ui] = None
            , items_sort: bool = False
            , hidden: bool = False
        ) -> None:
        super().__init__(name, hidden)

        self.items = items if items is not None else []
        self.items_sort = items_sort

        self.ui_container_contents: Gr_Container = None
    
    def validate(self) -> bool:
        valid = super().validate()

        for x in self.items:
            if not x.validate():
                valid = False
        
        return valid
    
    def finalizeUI(self, inputMap: dict[str, Gr_Input]) -> None:
        for x in self.items:
            x.finalizeUI(inputMap)
    
    def syncInput(self, *outputValues) -> int:
        offset = super().syncInput(*outputValues)

        for x in self.items:
            offset += x.syncInput(*outputValues[offset:])
        
        return offset
    
    def getInput(self) -> list[Gr_Input]:
        gr_inputs = super().getInput()

        for x in self.items:
            gr_inputs += x.getInput()
        
        return gr_inputs
    
    def getOutput(self, labeled_only: bool = False) -> list[Gr_Output]:
        gr_outputs = super().getOutput(labeled_only)

        for x in self.items:
            gr_outputs += x.getOutput(labeled_only)
        
        return gr_outputs
    
    def getOutputUpdates(self, reset: bool, *inputValues) -> tuple[list, int]:
        updates, offset = super().getOutputUpdates(reset, *inputValues)

        for x in self.items:
            x_updates, x_offset = x.getOutputUpdates(reset, *inputValues[offset:])
            updates += x_updates
            offset += x_offset
        
        return updates, offset
    
    def buildUI(self) -> None:
        self.buildGrContents_Top()

        self.ui_container_contents.initGr()
        with self.ui_container_contents.gr:
            for x in self.items:
                self.initItemUI(x)
        
        self.buildGrContents_Bottom()
    
    def handlePrompt(self, p: StableDiffusionProcessing, inputMap: dict[str, Gr_Input]) -> None:
        for x in self.items:
            x.handlePrompt(p, inputMap)
    
    def addItem(self, item: B_Ui):
        self.items.append(item)
    
    def init(self, gr_outputs: list[Gr_Output]) -> None:
        """VIRTUAL"""
        self.ui_container_contents = self.initContainerContents(f"{self.name} (Contents)")
        
        for x in self.items:
            x.init_main()
        
        if self.items_sort:
            self.items = sorted(self.items, key = lambda x: x.name)
    
    def initContainerContents(self, name: str) -> Gr_Container:
        """VIRTUAL: Base -> Gr_Group"""
        return Gr_Group(name = name)
    
    def buildGrContents_Top(self) -> None:
        """VIRTUAL: Build Gradio elements at top of container, Base -> nothing"""
        pass

    def buildGrContents_Bottom(self) -> None:
        """VIRTUAL: Build Gradio elements at bottom of container, Base -> nothing"""
        pass

    def initItemUI(self, item: B_Ui) -> None:
        """Virtual: Base -> item.initUI"""
        item.initUI()

class B_Ui_Container(B_Ui_Collection, ABC):
    def __init__(
            self
            , name: str
            , items: list[B_Ui] = None
            , reset_ui_build: bool = False
            , random_ui_build: bool = False
            , hidden: bool = False) -> None:
        super().__init__(name, items, False, hidden)

        self.ui_reset_build = reset_ui_build
        self.ui_random_build = random_ui_build

        self.ui_reset: Gr_Button = None
        self.ui_random: Gr_Button = None
    
    def init(self, gr_outputs: list[Gr_Output]) -> None:
        super().init(gr_outputs)
        
        self.ui_reset = Gr_Button(f"Reset {self.name}")
        gr_outputs.append(self.ui_reset)

        self.ui_random = Gr_Button(f"Randomize {self.name}")
        gr_outputs.append(self.ui_random)
    
    def finalizeUI(self, inputMap: dict[str, Gr_Input]) -> None:
        super().finalizeUI(inputMap)

        # if self.ui_random_build:
        #     self.ui_random.gr.click() #!
        
        if self.ui_reset_build:
            self.ui_reset.gr.click(
                fn = lambda: self.getOutputUpdates(True)[0]
                , outputs = list(map(lambda gr_output: gr_output.gr, self.getOutput()))
            ) #!
    
    def buildGrContents_Bottom(self) -> None:
        if self.ui_reset_build and self.ui_random_build:
            with gr.Row():
                with gr.Column():
                    self.ui_random.initGr()
                with gr.Column():
                    self.ui_reset.initGr()
        elif self.ui_random_build:
            self.ui_random.initGr()
        elif self.ui_reset_build:
            self.ui_reset.initGr()
    
    def getOutputUpdates(self, reset: bool, *inputValues) -> tuple[list, int]:
        updates, offset = super().getOutputUpdates(reset, *inputValues)

        updates_extra: list = []
        if self.ui_reset_build:
            updates_extra.append(self.ui_reset.getOutputUpdate(reset))
        if self.ui_random_build:
            updates_extra.append(self.ui_random.getOutputUpdate(reset))
        updates = updates_extra + updates
        
        return updates, offset

class B_Ui_Container_Tab(B_Ui_Container):
    @staticmethod
    def _fromArgs(name: str = None, **args: str):
        return B_Ui_Container_Tab(
            name = name
            , reset_ui_build = bool(int(args.get("build_reset_button", 1)))
            , random_ui_build = bool(int(args.get("build_random_button", 1)))
            , hidden = bool(int(args.get("hide", 0)))
        )
    
    def __init__(
            self
            , name: str = "Tab"
            , items: list[B_Ui] = None
            , reset_ui_build: bool = True
            , random_ui_build: bool = True
            , hidden: bool = False
        ) -> None:
        super().__init__(name, items, reset_ui_build, random_ui_build, hidden)
    
    def initContainer(self, visible: bool) -> Gr_Container:
        return Gr_Tab(name = self.name, visible = visible)

class B_Ui_Container_Row(B_Ui_Container):
    @staticmethod
    def _fromArgs(name: str = None, **args: str):
        return B_Ui_Container_Row(
            reset_ui_build = bool(int(args.get("build_reset_button", 0)))
            , random_ui_build = bool(int(args.get("build_random_button", 0)))
            , hidden = bool(int(args.get("hide", 0)))
        )
    
    def __init__(
            self
            , name: str = "Row"
            , items: list[B_Ui] = None
            , reset_ui_build: bool = False
            , random_ui_build: bool = False
            , hidden: bool = False
        ) -> None:
        super().__init__(name, items, reset_ui_build, random_ui_build, hidden)
    
    def initContainer(self, visible: bool) -> Gr_Container:
        return Gr_Row(visible = visible)

class B_Ui_Container_Column(B_Ui_Container):
    @staticmethod
    def _fromArgs(name: str = None, **args: str):
        return B_Ui_Container_Column(
            scale = int(args.get("scale", 1))
            , reset_ui_build = bool(int(args.get("build_reset_button", 0)))
            , random_ui_build = bool(int(args.get("build_random_button", 0)))
            , hidden = bool(int(args.get("hide", 0)))
        )
    
    def __init__(
            self
            , name: str = "Column"
            , items: list[B_Ui] = None
            , scale: int = 1
            , reset_ui_build: bool = False
            , random_ui_build: bool = False
            , hidden: bool = False
        ) -> None:
        super().__init__(name, items, reset_ui_build, random_ui_build, hidden)

        self.scale = scale
    
    def initContainer(self, visible: bool) -> Gr_Container:
        return Gr_Column(scale = self.scale, visible = visible)

class B_Ui_Container_Group(B_Ui_Container):
    @staticmethod
    def _fromArgs(name: str = None, **args: str):
        return B_Ui_Container_Group(
            reset_ui_build = bool(int(args.get("build_reset_button", 0)))
            , random_ui_build = bool(int(args.get("build_random_button", 0)))
            , hidden = bool(int(args.get("hide", 0)))
        )
    
    def __init__(
            self
            , name: str = "Group"
            , items: list[B_Ui] = None
            , reset_ui_build: bool = False
            , random_ui_build: bool = False
            , hidden: bool = False
        ) -> None:
        super().__init__(name, items, reset_ui_build, random_ui_build, hidden)
    
    def initContainer(self, visible: bool) -> Gr_Container:
        return Gr_Group(visible = visible)

class B_Ui_Container_Accordion(B_Ui_Container):
    @staticmethod
    def _fromArgs(name: str = None, **args: str):
        return B_Ui_Container_Accordion(
            name = name
            , reset_ui_build = bool(int(args.get("build_reset_button", 0)))
            , random_ui_build = bool(int(args.get("build_random_button", 0)))
            , hidden = bool(int(args.get("hide", 0)))
        )
    
    def __init__(
            self
            , name: str = "Accordion"
            , items: list[B_Ui] = None
            , reset_ui_build: bool = False
            , random_ui_build: bool = False
            , hidden: bool = False
        ) -> None:
        super().__init__(name, items, reset_ui_build, random_ui_build, hidden)
    
    def initContainer(self, visible: bool) -> Gr_Container:
        return Gr_Accordion(name = self.name, visible = visible)

class B_Ui_PromptSingle(B_Ui):
    @staticmethod
    def _fromArgs(name: str = None, **args: str):
        return B_Ui_PromptSingle(
            name = name
            , prefix = args.get("prefix", "")
            , postfix = args.get("postfix", "")
            , prompt = args.get("v", "")
            , strength = float(args.get("s", 1))
            , negative = bool(int(args.get("n", 0)))
            , hidden = bool(int(args.get("hide", 0)))
        )
    
    def __init__(
            self
            , name: str = "Single Prompt"
            , prompt_ui_build: bool = True
            , strength_ui_build: bool = True
            , negative_ui_build: bool = True
            , prefix: str = ""
            , postfix: str = ""
            , prompt: str = ""
            , strength: float = 1
            , negative: bool = False
            , hidden: bool = False
        ) -> None:
        super().__init__(name, hidden)

        self.ui_prompt_build = prompt_ui_build
        self.ui_strength_build = strength_ui_build
        self.ui_negative_build = negative_ui_build

        self.prefix = prefix
        self.postfix = postfix

        self.prompt = prompt
        self.strength = strength
        self.negative = negative

        self.ui_prompt: Gr_Textbox = None
        self.ui_strength: Gr_Number = None
        self.ui_negative: Gr_Checkbox = None
    
    def init(self, gr_outputs: list[Gr_Output]) -> None:
        self.ui_prompt = Gr_Textbox(self.name, self.prompt)
        gr_outputs.append(self.ui_prompt)

        self.ui_strength = Gr_Number(f"{self.name} (S)", self.strength, b_prompt_strength_min, b_prompt_strength_step)
        gr_outputs.append(self.ui_strength)

        self.ui_negative = Gr_Checkbox(f"{self.name} (N)", self.negative)
        gr_outputs.append(self.ui_negative)
    
    def buildUI(self) -> None:
        if self.ui_prompt_build:
            self.ui_prompt.initGr()
        
        if self.ui_strength_build or self.ui_negative_build:
            with gr.Row():
                if self.ui_negative_build:
                    self.ui_negative.initGr()

                if self.ui_strength_build:
                    self.ui_strength.initGr()
    
    def handlePrompt(self, p: StableDiffusionProcessing, inputMap: dict[str, Gr_Input]) -> None:
        prompt = str(self.ui_prompt.value)
        negative = bool(self.ui_negative.value)
        strength = float(self.ui_strength.value)

        prompt = promptSanitized(prompt)
        prefix = promptSanitized(self.prefix)
        postfix = promptSanitized(self.postfix)

        if len(prompt) == 0:
            return
        
        if len(prefix) > 0:
            prompt = f"{prefix} {prompt}"
        if len(postfix) > 0:
            prompt = f"{prompt} {postfix}"
        
        if strength > 0 and strength != 1:
            prompt = f"({prompt}:{strength})"
        
        if not negative:
            p.prompt = promptAdded(p.prompt, prompt)
        else:
            p.negative_prompt = promptAdded(p.negative_prompt, prompt)

class B_Ui_PromptDual(B_Ui):
    @staticmethod
    def _fromArgs(name: str = None, **args: str):
        return B_Ui_PromptDual(
            name = name
            , prompt_positive = args.get("vp", "")
            , prompt_negative = args.get("vn", "")
            , strength = float(args.get("s", 1))
            , hidden = bool(int(args.get("hide", 0)))
        )
    
    def __init__(
            self
            , name: str = "Dual Prompt"
            , ui_prompts_build: bool = True
            , ui_strength_build: bool = True
            , prompt_positive: str = ""
            , prompt_negative: str = ""
            , strength: float = 1
            , hidden: bool = False
        ) -> None:
        super().__init__(name, hidden)

        self.ui_prompts_build = ui_prompts_build
        self.ui_strength_build = ui_strength_build
        
        self.prompt_positive = prompt_positive
        self.prompt_negative = prompt_negative
        self.strength = strength

        self.ui_prompt_positive: Gr_Textbox = None
        self.ui_prompt_negative: Gr_Textbox = None
        self.ui_strength: Gr_Number = None
    
    def init(self, gr_outputs: list[Gr_Output]) -> None:
        self.ui_prompt_positive = Gr_Textbox(f"{self.name} (+)", self.prompt_positive)
        gr_outputs.append(self.ui_prompt_positive)

        self.ui_prompt_negative = Gr_Textbox(f"{self.name} (-)", self.prompt_negative)
        gr_outputs.append(self.ui_prompt_negative)

        self.ui_strength = Gr_Number(f"{self.name} (S)", self.strength, b_prompt_strength_min, b_prompt_strength_step)
        gr_outputs.append(self.ui_strength)
    
    def buildUI(self) -> None:
        if self.ui_prompts_build:
            with gr.Row():
                self.ui_prompt_positive.initGr()
                self.ui_prompt_negative.initGr()
        
        if self.ui_strength_build:
            self.ui_strength.initGr()
    
    def handlePrompt(self, p: StableDiffusionProcessing, inputMap: dict[str, Gr_Input]) -> None:
        prompt_positive = str(self.ui_prompt_positive.value)
        prompt_negative = str(self.ui_prompt_negative.value)
        strength = str(self.ui_strength.value)

        prompt_positive = promptSanitized(prompt_positive)
        prompt_negative = promptSanitized(prompt_negative)

        if strength > 0 and strength != 1:
            if len(prompt_positive) > 0:
                prompt_positive = f"({prompt_positive}:{strength})"
            if len(prompt_negative) > 0:
                prompt_negative = f"({prompt_negative}:{strength})"
        
        p.prompt = promptAdded(p.prompt, prompt_positive)
        p.negative_prompt = promptAdded(p.negative_prompt, prompt_negative)

class B_Ui_PromptRange(B_Ui):
    _value_min: int = -1
    _value_max: int = 100
    _value_step: int = 1
    
    @staticmethod
    def _fromArgs(name: str = None, **args: str):
        return B_Ui_PromptRange(
            name = name
            , prompt_a = args.get("a", "")
            , prompt_b = args.get("b", "")
            , required = bool(int(args.get("is_required", 0)))
            , negative = bool(int(args.get("n", 0)))
            , value = int(args.get("v", None))
            , ui_buttons_build = bool(int(args.get("build_buttons", 1)))
            , prompt_a_button_text = args.get("a_button", None)
            , prompt_b_button_text = args.get("b_button", None)
            , hidden = bool(int(args.get("hide", 0)))
        )
    
    def __init__(
            self
            , prompt_a: str
            , prompt_b: str
            , name: str = "Range Prompt"
            , required: bool = False
            , negative: bool = False
            , value: int = None
            , ui_buttons_build: bool = True
            , ui_negative_build: bool = True
            , prompt_a_button_text: str = None
            , prompt_b_button_text: str = None
            , hidden: bool = False
        ) -> None:
        super().__init__(name, hidden)

        self.prompt_a = prompt_a
        self.prompt_b = prompt_b
        self.required = required
        self.negative = negative
        self.value = value

        self.value_min = self._value_min if not required else 0

        self.ui_buttons_build = ui_buttons_build
        self.ui_negative_build = ui_negative_build

        self.ui_button_a_text = prompt_a_button_text
        self.ui_button_b_text = prompt_b_button_text

        self.ui_range: Gr_Slider = None
        self.ui_negative: Gr_Checkbox = None
        self.ui_button_a: Gr_Button = None
        self.ui_button_b: Gr_Button = None
    
    def init(self, gr_outputs: list[Gr_Output]) -> None:
        self.ui_range = Gr_Slider(self.name, self.value, self.value_min, self._value_max, self._value_step)
        gr_outputs.append(self.ui_range)

        self.ui_negative = Gr_Checkbox(f"{self.name} (N)", self.negative)
        gr_outputs.append(self.ui_negative)

        self.ui_button_a = Gr_Button(self.ui_button_a_text)
        gr_outputs.append(self.ui_button_a)

        self.ui_button_b = Gr_Button(self.ui_button_b_text)
        gr_outputs.append(self.ui_button_b)
    
    def buildUI(self) -> None:
        self.ui_range.initGr()

        if self.ui_buttons_build:
            with gr.Row():
                self.ui_button_a.initGr()
                self.ui_button_b.initGr()
        
        if self.ui_negative_build:
            self.ui_negative.initGr()
    
    def finalizeUI(self, inputMap: dict[str, Gr_Input]) -> None:
        if self.ui_buttons_build:
            self.ui_button_a.gr.click(
                fn = lambda: 0
                , outputs = self.ui_range.gr
            )

            self.ui_button_b.gr.click(
                fn = lambda: 100
                , outputs = self.ui_range.gr
            )
    
    def getOutputUpdates(self, reset: bool, *inputValues) -> tuple[list, int]:
        updates, offset = super().getOutputUpdates(reset, *inputValues)

        if self.ui_buttons_build:
            updates_extra: list = [
                self.ui_button_a.getOutputUpdate(reset)
                , self.ui_button_b.getOutputUpdate(reset)
            ]
            updates = updates_extra + updates
        
        return updates, offset
    
    def handlePrompt(self, p: StableDiffusionProcessing, inputMap: dict[str, Gr_Input]) -> None:
        prompt_range = int(self.ui_range.value)
        negative = bool(self.ui_negative.value)

        if prompt_range < 0:
            return
        
        prompt_a = promptSanitized(self.prompt_a)
        prompt_b = promptSanitized(self.prompt_b)
        
        value = float(prompt_range)
        value = round(value / 100, 2)
        value = 1 - value

        prompt: str = None
        if value == 1:
            prompt = prompt_a
        elif value == 0:
            prompt = prompt_b
        else:
            prompt = f"[{prompt_a}:{prompt_b}:{value}]"

        if not negative:
            p.prompt = promptAdded(p.prompt, prompt)
        else:
            p.negative_prompt = promptAdded(p.negative_prompt, prompt)

class B_Ui_PromptSelect(B_Ui_Collection):
    _choice_empty: str = "-"
    _random_choices_max: int = 5

    @staticmethod
    def _fromArgs(name: str = None, **args: str):
        choices_default = args.get("v", None)
        if choices_default is not None and len(choices_default) > 0:
            choices_default = list(map(lambda v: v.strip(), choices_default.split(","))) #! str | list[str]?
        
        return B_Ui_PromptSelect(
            name = name
            , choices_sort = bool(int(args.get("sort", 1)))
            , choices_default = choices_default
            , multiselect = bool(int(args.get("multi_select", 0)))
            , custom = bool(int(args.get("allow_custom", 0)))
            , simple = bool(int(args.get("simple", 0)))
            , scale = int(args.get("scale", 1))
            , hidden = bool(int(args.get("hide", 0)))
        )
    
    #! conflicting names...
    @staticmethod
    def _buildColorChoicesList(postfix: str = "") -> list[B_Ui_PromptSingle]:
        return list(map(
            lambda text: B_Ui_PromptSingle(
                name = text
                , prompt_ui_build = False
                , postfix = postfix
                , prompt = text.lower())
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
            , name: str = "Select Prompt"
            , choices: list[B_Ui] = None
            , choices_sort: bool = True
            , choices_default: str | list[str] = None
            , multiselect: bool = False
            , custom: bool = False #!
            , simple: bool = False #!
            , scale: int = 1 #!
            , hidden: bool = False
        ) -> None:
        super().__init__(name, choices, choices_sort, hidden)

        self.choices_default = choices_default
        self.multiselect = multiselect

        self.choicesMap: dict[str, B_Ui] = {}
        self.choicesContainerMap: dict[str, Gr_Column] = {}

        self.ui_dropdown: Gr_Dropdown = None
    
    def init(self, gr_outputs: list[Gr_Output]) -> None:
        super().init(gr_outputs)

        self.initChoicesMap()

        self.ui_dropdown = Gr_Dropdown(self.name, self.buildChoicesList(), self.choices_default, self.multiselect)
        self.ui_container_contents.visible = self.getShowContainer()

        for x in self.items:
            x_container = Gr_Column(1, "panel", self.getShowChoiceContainer(x), f"{self.name}_{x.name} (Container)")
            self.choicesContainerMap[x.name] = x_container
            gr_outputs.append(x_container) #!
        
        gr_outputs.append(self.ui_container_contents) #!
        gr_outputs.append(self.ui_dropdown)
    
    def finalizeUI(self, inputMap: dict[str, Gr_Input]) -> None:
        super().finalizeUI(inputMap)

        # Show/Hide selected
        self.ui_dropdown.gr.input(
            fn = lambda *inputValues: self.getOutputUpdates(False, *inputValues)[0]
            , inputs = list(map(lambda gr_input: gr_input.gr, self.getInput()))
            , outputs = list(map(lambda gr_output: gr_output.gr, self.getOutput()))
        )

        #! Presets
    
    def initContainerContents(self, name: str) -> Gr_Container:
        return Gr_Row("panel", name = name)
    
    def buildGrContents_Top(self) -> None:
        self.ui_dropdown.initGr()
    
    def initItemUI(self, item: B_Ui) -> None:
        item_container = self.choicesContainerMap[item.name]
        item_container.initGr()
        with item_container.gr:
            super().initItemUI(item)
    
    def getOutputUpdates(self, reset: bool, *inputValues) -> tuple[list, int]:
        updates, offset = super().getOutputUpdates(reset, *inputValues)

        updates_extra: list = []
        for x in self.items:
            updates_extra.append(self.choicesContainerMap[x.name].getUpdateVisible(self.getShowChoiceContainer(x))) #!!!
        updates_extra.append(self.ui_container_contents.getUpdateVisible(self.getShowContainer()))
        updates = updates_extra + updates
        
        return updates, offset
    
    def handlePrompt(self, p: StableDiffusionProcessing, inputMap: dict[str, Gr_Input]) -> None:
        if self.ui_dropdown.value is None or len(self.ui_dropdown.value) == 0:
            return
        
        items_selected: list[B_Ui] = []
        if type(self.ui_dropdown.value) is not list:
            items_selected.append(self.choicesMap.get(self.ui_dropdown.value, None)) #! could maybe allow custom values here
        else:
            for v in self.ui_dropdown.value:
                items_selected.append(self.choicesMap[v])
        
        if len(items_selected) == 0 or items_selected[0] is None:
            return
        
        for x in items_selected:
            x.handlePrompt(p, inputMap)
    
    def addChoices(self, **args: str):
        choicesList: list[B_Ui] = []
        
        special_type = args.get("type", "")
        match special_type:
            case "COLOR":
                postfix = args.get("postfix", "")
                choicesList += self._buildColorChoicesList(postfix)
            case _:
                print(f"WARNING: Invalid CHOICES type in {self.name} -> {special_type}")
        
        self.items += choicesList
    
    def initChoicesMap(self) -> None:
        for x in self.items:
            self.choicesMap[x.name] = x
    
    def buildChoicesList(self) -> list[str]:
        choicesList: list[str] = []

        if not self.multiselect:
            choicesList.append(self._choice_empty)
        
        for k in self.choicesMap:
            choicesList.append(k)
        
        return choicesList
    
    def getShowContainer(self) -> bool:
        choice_current = self.ui_dropdown.value
        return choice_current is not None and len(choice_current) > 0 and choice_current != self._choice_empty
    
    def getShowChoiceContainer(self, x: B_Ui) -> bool:
        choice_current = self.ui_dropdown.value
        choice = x.name
        return choice_current is not None and (choice == choice_current or choice in choice_current)

# class B_UI_Prompt(B_UI, ABC):
#     #!
#     def setPreset(self, preset: typing.Any):
#         self.preset = preset

class B_Ui_Map():
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

    def __init__(
            self
            , path_base: str
            , folder_name_scripts: str
            , folder_name_script_config: str
            , file_name_layout: str
            , file_name_presets: str
            , tagged_ignore: bool = False
            , validate_skip: bool = False
        ) -> None:
        self.path_base = path_base
        self.path_script_config = os.path.join(path_base, folder_name_scripts, folder_name_script_config)

        self.layout = self.parseLayout(file_name_layout, tagged_ignore)
        #! self.presets = self.parsePresets(file_name_presets)

        for x in self.layout:
            x.init_main()
        
        #! confirm accuracy:
        self.inputMap: dict[str, Gr_Input] = {}
        for x in self.layout:
            for gr_input in x.getInput():
                if gr_input.name in self.inputMap:
                    printWarning("B_Ui_Map", "inputMap", f"Duplicate Gr_Input -> {gr_input.name}")
                
                self.inputMap[gr_input.name] = gr_input

        if not validate_skip:
            self.validate()
    
    def parseLayout(self, file_name_layout: str, tagged_ignore: bool) -> list[B_Ui]:
        layout: list[B_Ui] = []
        
        stack_containers: list[B_Ui_Container] = []
        stack_selects: list[B_Ui_PromptSelect] = []
        #! stack_select_choices: list[B_UI_Prompt] = []

        skip = 0
        
        def _build(item: B_Ui) -> None:
            # if len(stack_select_choices) > 0:
            #     item_select = stack_select_choices.pop()
            
            if len(stack_selects) > 0:
                stack_selects[-1].addItem(item)
                return
            
            if len(stack_containers) > 0:
                stack_containers[-1].addItem(item)
                return
            
            layout.append(item)
        
        # def _buildDropdownChoice(dropdown_choice: tuple[str, B_UI_Preset_Builder, dict[str, str]]) -> bool:
        #     """Returns True if choice is not None and had preset mappings"""
        #     if dropdown_choice is not None:
        #         l_choice_name, l_choice_preset_builder, l_choice_args = dropdown_choice

        #         l_choice_args["prefix"] = builder_current_dropdown.args.get("prefix", "")
        #         l_choice_args["postfix"] = builder_current_dropdown.args.get("postfix", "")

        #         builder_current_dropdown.addChoice(l_choice_name, l_choice_preset_builder, **l_choice_args)
        #         return len(l_choice_preset_builder.mappings) > 0
            
        #     return False
        
        with open(os.path.join(self.path_script_config, file_name_layout)) as file_layout:
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
                        # if dropdown_current_choice is not None:
                        #     had_mappings = _buildDropdownChoice(dropdown_current_choice)
                        #     dropdown_current_choice = None
                        #     if had_mappings:
                        #         continue

                        if len(stack_selects) > 0:
                            item_select = stack_selects.pop()
                            _build(item_select)
                            continue
                        
                        if len(stack_containers) > 0:
                            item_container = stack_containers.pop()
                            _build(item_container)
                            continue

                        continue #!
                    
                    skip -= 1
                    
                    continue
                
                ignore: bool = skip > 0
                if l_args.get("x", "") == "1":
                    if not ignore:
                        ignore = tagged_ignore
                
                match l_type:
                    case "SINGLE":
                        if ignore:
                            continue

                        _build(B_Ui_PromptSingle._fromArgs(l_name, **l_args))
                    
                    case "DUAL":
                        if ignore:
                            continue

                        _build(B_Ui_PromptDual._fromArgs(l_name, **l_args))
                    
                    case "RANGE":
                        if ignore:
                            continue
                        
                        _build(B_Ui_PromptRange._fromArgs(l_name, **l_args))
                    
                    case "SELECT":
                        if ignore:
                            skip += 1
                            continue
                        
                        stack_selects.append(B_Ui_PromptSelect._fromArgs(l_name, **l_args))
                    
                    case "CHOICES":
                        if ignore:
                            continue

                        stack_selects[-1].addChoices(**l_args)
                    
                    # case "CHOICE":
                    #     if ignore:
                    #         continue
                        
                    #     _buildDropdownChoice(dropdown_current_choice)
                        
                    #     dropdown_current_choice = (l_name, B_UI_Preset_Builder(f"{builder_current_dropdown.name}_{l_name}_PRESET", **{ "is_additive": "1", "hide": "1" }), l_args)
                    
                    # case "SET":
                    #     dropdown_current_choice[1].addMapping(l_name, **l_args)
                    
                    case "GROUP":
                        if ignore:
                            skip += 1
                            continue

                        stack_containers.append(B_Ui_Container_Group._fromArgs(l_name, **l_args))
                    
                    case "TAB":
                        if ignore:
                            skip += 1
                            continue
                        
                        stack_containers.append(B_Ui_Container_Tab._fromArgs(l_name, **l_args))
                    
                    case "ROW":
                        if ignore:
                            skip += 1
                            continue
                        
                        stack_containers.append(B_Ui_Container_Row._fromArgs(l_name, **l_args))
                    
                    case "COLUMN":
                        if ignore:
                            skip += 1
                            continue
                        
                        stack_containers.append(B_Ui_Container_Column._fromArgs(l_name, **l_args))
                    
                    case "ACCORDION":
                        if ignore:
                            skip += 1
                            continue
                        
                        stack_containers.append(B_Ui_Container_Accordion._fromArgs(l_name, **l_args))
                    
                    case "SEPARATOR":
                        if ignore:
                            continue
                        
                        _build(B_Ui_Separator._fromArgs(None, **l_args))

                    case _:
                        print(f"WARNING: Invalid layout type -> {l_type}")
        
        return layout
    
    # def parsePresets(self, file_path_presets: str) -> dict[str, B_UI_Preset]:
    #     presets: dict[str, B_UI_Preset] = {}
        
    #     builder_current: B_UI_Preset_Builder = None
        
    #     with open(file_path_presets) as file_presets:
    #         line_number: int = 0

    #         for l in file_presets:
    #             line_number += 1

    #             if l.lstrip().startswith("#"):
    #                 print(f"# PRESETS - commented out line @{line_number}")
    #                 continue

    #             l_type, l_name, l_args = self.readLine(l)
                
    #             if len(l_type) == 0:
    #                 continue
                    
    #             if l_type == ".":
    #                 break
                
    #             if l_type == "END":
    #                 presets[builder_current.name] = builder_current.build()
    #                 builder_current = None
    #                 continue
                
    #             match l_type:
    #                 case "PRESET":
    #                     builder_current = B_UI_Preset_Builder(l_name, **l_args)
                    
    #                 case "SET":
    #                     builder_current.addMapping(l_name, **l_args)
                    
    #                 case _:
    #                     print(f"Invalid preset type: {l_type}")
        
    #     return presets
    
    def validate(self):
        valid: bool = True

        for x in self.layout:
            if not x.validate():
                valid = False
        
        # for preset in self.presets.values():
        #     if not preset.validate(self.inputMap):
        #         valid = False
        
        if not valid:
            printWarning("B_Ui_Map", "valid", "Invalid layout or presets")
    
    def buildUI(self) -> list[typing.Any]:
        gr_list: list[typing.Any] = []

        # PRESETS
        # B_Ui_Separator._build()

        # with gr.Accordion("Presets", open = False):
        #     i = 0
        #     for preset in self.presets.values():
        #         gr += preset.buildUI()

        #         i += 1
        #         if i < len(self.presets) and preset.visible:
        #             B_Ui_Separator._build()

        # LAYOUT
        B_Ui_Separator._build()
        
        for x in self.layout:
            x.initUI()
            gr_list += list(map(lambda gr_output: gr_output.gr, x.getOutput(True)))
        
        # SETTINGS
        B_Ui_Separator._build()

        with gr.Accordion("Settings", open = False):
            btnClearConfig = gr.Button("Clear config")
            btnClearConfig.click(fn = self.clearConfigFile)
            
            gr_list.append(btnClearConfig)
        
        # - DONE -
        return gr_list
    
    def finalizeUI(self):
        for x in self.layout:
            x.finalizeUI(self.inputMap)
        
        # for preset in self.presets.values():
        #     preset.finalizeUI(self.inputMap)
    
    def clearConfigFile(self):
        path = os.path.join(self.path_base, b_file_name_config) #!
        with open(path, "r+", encoding = "utf-8") as file_config:
            config: dict[str, typing.Any] = json.load(file_config)
            
            config_keys = filter(lambda k: k.find(b_folder_name_script_config) == -1, config.keys()) #!

            config_new: dict[str, typing.Any] = {}
            for k in config_keys:
                config_new[k] = config[k]
            
            file_config.seek(0)
            json.dump(config_new, file_config, indent = 4)
            file_config.truncate()

bUiMap = B_Ui_Map(
    path_base = b_path_base
    , folder_name_scripts = b_folder_name_scripts
    , folder_name_script_config = b_folder_name_script_config
    , file_name_layout = b_file_name_layout
    , file_name_presets = b_file_name_presets
    , tagged_ignore = b_tagged_ignore
    , validate_skip = b_validate_skip
)

# class B_UI_Preset(B_UI):
#     @staticmethod
#     def _fromArgs(name: str, mappings: dict[str, list[typing.Any]], **kwargs: str) -> B_UI:
#         return B_UI_Preset(
#             name = name
#             , mappings = mappings
#             , isAdditive = bool(int(kwargs.get("is_additive", 0)))
#             , visible = not bool(int(kwargs.get("hide", 0)))
#         )
    
#     def __init__(self, name: str, mappings: dict[str, list[typing.Any]], isAdditive: bool = False, visible: bool = True):
#         super().__init__(name, visible, True)

#         self.mappings = mappings
#         self.isAdditive = isAdditive
    
#     def getDefaultName(self) -> str:
#         return "Preset"
    
#     def validate(self, componentMap: dict) -> bool:
#         valid: bool = super().validate(componentMap)

#         for k in self.mappings:
#             if k not in componentMap:
#                 valid = False
#                 printWarning("Preset", self.name, "Key is not valid")
#             else:
#                 bComponent: B_UI_Component = componentMap[k]
#                 if not bComponent.validateValue(self.mappings[k]):
#                     valid = False
#                     printWarning("Preset", f"{self.name}: {bComponent.name}", "Value is not valid")

#         return valid
    
#     def finalizeUI(self, componentMap: dict):
#         super().finalizeUI(componentMap)

#         bComponentMap: dict[str, B_UI_Component] = componentMap

#         bComponents: list[B_UI_Component] = []
#         if self.isAdditive:
#             for bComponent in bComponentMap.values():
#                 if bComponent.name in self.mappings:
#                     bComponents.append(bComponent)
#         else:
#             bComponents += list(bComponentMap.values())
        
#         components_inputs: list[typing.Any] = []
#         components_outputs: list[typing.Any] = []
#         for bComponent in bComponents:
#             components_inputs.append(bComponent.ui)
#             components_outputs += [bComponent.ui] + bComponent.ui_extra_outputs
        
#         def _applyPreset(*inputs):
#             updates: list[typing.Any] = []

#             i = 0
#             for bComponent in bComponents:
#                 updates += self.getPresetValue(bComponent, inputs[i])[0]
#                 i += 1
            
#             return updates
        
#         self.ui.click(
#             fn = _applyPreset
#             , inputs = components_inputs
#             , outputs = components_outputs
#         )
    
#     def buildSelf(self) -> typing.Any:
#         return gr.Button(self.name, visible = self.visible)
    
#     def getPresetValue(self, bComponent: B_UI_Component, componentValue) -> tuple[list, int]:
#         """Returns update values and number of inputs consumed"""
#         presetValue = componentValue

#         if bComponent.name in self.mappings:
#             presetValue = self.mappings[bComponent.name]

#             if type(bComponent.ui) is not gr.Dropdown or not bComponent.ui.multiselect:
#                 presetValue = bComponent.defaultValue if len(presetValue) == 0 else presetValue[0]
#         elif not self.isAdditive:
#             presetValue = bComponent.defaultValue
        
#         return bComponent.getUpdate(presetValue)

# class B_Prompt_Link_Slider(B_Prompt):
#     @staticmethod
#     def _fromArgs(name: str, **kwargs: str):
#         return B_Prompt_Link_Slider(
#             name = name
#             , linkedKey = kwargs["link_target"]
#             , promptA = kwargs["a"]
#             , promptB = kwargs["b"]
#         )
    
#     def __init__(self, name: str, linkedKey: str, promptA: str, promptB: str):
#         super().__init__(name)

#         self.linkedKey = linkedKey
#         self.promptA = promptA
#         self.promptB = promptB
        
#         self.isNegativePrompt = False
    
#     def buildPrompt(self, componentMap: dict[str, B_UI_Component]) -> str:
#         component = componentMap.get(self.linkedKey)
        
#         if component is None:
#             print(f"B_Prompt_Link_Slider: Invalid key - '{self.linkedKey}'")
#             return ""
        
#         if type(component) is not B_UI_Component_Slider:
#             print(f"B_Prompt_Link_Slider: Linked entry is not a slider - '{self.linkedKey}'")
#             return ""
        
#         return buildRangePrompt(self.promptA, self.promptB, component.value)
    
#     def getPositive(self, componentMap: dict[str, B_UI_Component]) -> str:
#         if self.isNegativePrompt:
#             return ""
        
#         return self.buildPrompt(componentMap)
    
#     def getNegative(self, componentMap: dict[str, B_UI_Component]) -> str:
#         if not self.isNegativePrompt:
#             return ""
        
#         return self.buildPrompt(componentMap)

# class B_UI_Preset_Builder(B_UI_Builder):
#     def __init__(self, name: str, **kwargs: str):
#         super().__init__(name, **kwargs)
        
#         self.mappings: dict[str, list[str]] = {}
    
#     def addMapping(self, name: str, **kwargs: str):
#         value = kwargs.get("v", "")
        
#         if len(value) > 0:
#             value = list(map(lambda v: v.strip(), value.split(",")))
#         else:
#             value = list[str]([])
        
#         self.mappings[name] = value
    
#     def build(self) -> B_UI:
#         return None if len(self.mappings) == 0 else B_UI_Preset._fromArgs(self.name, self.mappings, **self.args)

# class B_UI_Component_Dropdown_Builder(B_UI_Builder_WithParent):
#     def __init__(self, name: str, parent: B_UI_Container_Builder, **kwargs: str):
#         super().__init__(name, parent, **kwargs)
        
#         self.choicesList: list[B_Prompt] = []
    
#     def addChoice(self, text: str, preset_builder: B_UI_Preset_Builder, **bPromptKwargs: str):
#         bPrompt: B_Prompt = None
        
#         link_type = bPromptKwargs.get("link_type", "")
#         match link_type:
#             case "SLIDER":
#                 bPrompt = B_Prompt_Link_Slider._fromArgs(text, **bPromptKwargs)
#             case _:
#                 preset: B_UI_Preset = preset_builder.build()
#                 bPrompt = B_Prompt_Simple._fromArgs(text, preset, **bPromptKwargs)
        
#         if any(map(lambda bPrompt_existing: bPrompt_existing.name == bPrompt.name, self.choicesList)):
#             print(f"WARNING: Duplicate CHOICE in {self.name} -> {text}")
        
#         self.choicesList.append(bPrompt)
    
#     def addChoices(self, **choicesKwargs: str):
#         choicesList: list[B_Prompt] = []
        
#         special_type = choicesKwargs["type"]
#         match special_type:
#             case "COLOR":
#                 choicesList = B_UI_Component_Dropdown._buildColorChoicesList(choicesKwargs["postfix"])
#             case _:
#                 print(f"Invalid CHOICES type in {self.name} -> {special_type}")
        
#         self.choicesList += choicesList
    
#     def buildExtended(self) -> B_UI:
#         return B_UI_Component_Dropdown._fromArgs(self.name, self.choicesList, **self.args)

class Script(scripts.Script):
    def title(self):
        return "B Prompt Builder"

    def show(self, is_img2img):
        return not is_img2img
    
    def ui(self, is_img2img):
        built = bUiMap.buildUI()
        bUiMap.finalizeUI()
        return built

    def run(self, p, *args):
        i = 0
        
        # for preset in bUiMap.presets.values():
        #     i += preset.setValue(*args[i:])
        
        for x in bUiMap.layout:
            i += x.syncInput(*args[i:])
            x.handlePrompt(p, bUiMap.inputMap)
        
        proc = process_images(p)
        
        return proc
