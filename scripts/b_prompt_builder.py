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
    def __init__(self, name: str, is_labeled: bool) -> None:
        self.name = name

        self.is_labeled = is_labeled

        self.gr: typing.Any = None
    
    def initGr(self) -> None:
        self.gr = self.buildGr()
    
    def isGrBuilt(self) -> bool:
        return self.gr is not None
    
    def printWarning(self, message: str) -> None:
        printWarning(self.__class__.__name__, self.name, message)
    
    def validate(self) -> bool:
        """VIRTUAL: Base -> True"""
        return True
    
    def getOutputUpdate(self, updates_output: list, reset: bool, *value_inputs) -> int:
        """VIRTUAL: Adds updates to updates_output and returns number of values consumed, Base -> Nothing (0)"""
        return 0

    @abstractmethod
    def buildGr(self) -> typing.Any:
        pass

class Gr_Markdown(Gr_Wrapper):
    def __init__(self, value: str, name: str = "Markdown") -> None:
        super().__init__(name, False)

        self.value = value
    
    def buildGr(self) -> typing.Any:
        return gr.Markdown(value = self.value)

class Gr_Output(Gr_Wrapper, ABC):
    def __init__(self, name: str, is_labeled: bool, is_input: bool = False) -> None:
        super().__init__(name, is_labeled)

        self.is_input = is_input

class Gr_Input(Gr_Output, ABC):
    def __init__(self, label: str, value_default: typing.Any = None) -> None:
        super().__init__(label, True, True)

        self.value_default = value_default

        self.value = self.buildDefaultValue()
    
    def validate(self) -> bool:
        valid = super().validate()

        valid_value, valid_message = self.validateValue(self.value_default, "Default value")
        if not valid_value:
            valid = False
            self.printWarning(valid_message)
        
        return valid
    
    def syncInput(self, value: typing.Any):
        self.value = value
    
    def getOutputUpdate(self, updates_output: list, reset: bool, *value_inputs) -> int:
        value_new = value_inputs[0] if len(value_inputs) > 0 else None

        if reset and value_new is None:
            self.syncInput(self.buildDefaultValue())
        elif value_new is not None:
            self.syncInput(value_new)
        
        updates_output.append(self.getUpdate(self.value))
        return 1
    
    def buildDefaultValue(self) -> typing.Any:
        """VIRTUAL: Base -> self.value_default"""
        return self.value_default
    
    def validateValue(self, value: typing.Any, value_name: str = "Value") -> tuple[bool, str]:
        """VIRTUAL: Returns valid, message, Base -> True, None"""
        return True, None
    
    def getUpdate(self, value: typing.Any) -> typing.Any:
        """VIRTUAL: Base -> value"""
        return value

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
    
    def validateValue(self, value: typing.Any, value_name: str = "Value") -> tuple[bool, str]:
        if type(value) is not float and type(value) is not int:
            return False, f"{value_name} ({value}) is not a float | int"
        
        value_f: float = value

        if value_f < self.value_min:
            return False, f"{value_name} ({value_f} is under minimum ({self.value_min}))"
        
        return super().validateValue(value, value_name)
    
    def buildGr(self) -> typing.Any:
        return gr.Number(
            label = self.name
            , value = self.value
            , minimum = self.value_min
            , step = self.value_step
        )

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
    
    def validateValue(self, value: typing.Any, value_name: str = "Value") -> tuple[bool, str]:
        if type(value) is not int:
            return False, f"{value_name} ({value}) is not an int"
        
        value_int: int = value

        if value_int < self.value_min:
            return False, f"{value_name} ({self.value_default}) is under minimum ({self.value_min})"
        
        if value_int > self.value_max:
            return False, f"{value_name} ({self.value_default}) is over maximum ({self.value_max})"
        
        return super().validateValue(value, value_name)
    
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
    
    def validateValue(self, value: typing.Any, value_name: str = "Choice(s)") -> tuple[bool, str]:
        if value is not None and type(value) is not str and type(value) is not list:
            return False, f"{value_name} ({value}) is not a str | list | None"
        
        if len(self.choices) > 0:
            if (
                value is not None
                and (
                    (
                        type(value) is not list
                        and value not in self.choices
                    ) or (
                        type(value) is list
                        and any(map(lambda c: c not in self.choices, value))
                    )
                )
            ):
                return False, f"Invalid {value_name} -> {value}"
        elif value is not None:
            return False, f"No choices set but {value_name} set -> {value}"
        
        return super().validateValue(value, value_name)
    
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

class Gr_Container(Gr_Output, ABC):
    def __init__(self, name: str, visible: bool = True) -> None:
        super().__init__(name, False)

        self.visible = visible
    
    def buildGr(self) -> typing.Any:
        return self.buildGrContainer(self.visible)
    
    def getOutputUpdate(self, updates_output: list, reset: bool, *value_inputs) -> int:
        visible = value_inputs[0]

        if visible is not None:
            self.visible = bool(visible)
        
        updates_output.append(self.gr.update(visible = visible))
        return 1
    
    def getUpdateVisible(self, updates_output: list, visible: bool) -> None:
        self.getOutputUpdate(updates_output, False, visible)
    
    @abstractmethod
    def buildGrContainer(self, visible: bool) -> typing.Any:
        pass

class Gr_Row(Gr_Container):
    def __init__(self, variant: str = "default", visible: bool = True, name: str = "Row") -> None:
        super().__init__(name, visible)

        self.variant = variant
    
    def buildGrContainer(self, visible: bool) -> typing.Any:
        return gr.Row(
            variant = self.variant
            , visible = visible
        )

class Gr_Column(Gr_Container):
    def __init__(self, scale: int = 1, variant: str = "default", visible: bool = True, name: str = "Column") -> None:
        super().__init__(name, visible)

        self.scale = scale
        self.variant = variant
    
    def buildGrContainer(self, visible: bool) -> typing.Any:
        return gr.Column(
            scale = self.scale
            , variant = self.variant
            , visible = visible
        )

class Gr_Group(Gr_Container):
    def __init__(self, visible: bool = True, name: str = "Group") -> None:
        super().__init__(name, visible)
    
    def buildGrContainer(self, visible: bool) -> typing.Any:
        return gr.Group(visible = visible)

#!!! Tab does not have visible prop..
class Gr_Tab(Gr_Container):
    def __init__(self, visible: bool = True, name: str = "Group (Tab)") -> None:
        super().__init__(name, visible)

        self.gr_tab: typing.Any = None
    
    def buildGrContainer(self, visible: bool) -> typing.Any:
        self.gr_tab = gr.Tab(self.name)
        with self.gr_tab:
            gr_tab_group = gr.Group(visible = visible)
        return gr_tab_group

class Gr_Accordion(Gr_Container):
    def __init__(self, visible: bool = True, name: str = "Accordion") -> None:
        super().__init__(name, visible)
    
    def buildGrContainer(self, visible: bool) -> typing.Any:
        return gr.Accordion(self.name, visible = visible)

class B_Ui(ABC):
    @staticmethod
    @abstractmethod
    def _paramsFromArgs(args: dict[str, str]) -> tuple:
        pass

    @staticmethod
    @abstractmethod
    def _fromArgs(args: dict[str, str], name: str = None):
        pass

    @staticmethod
    @abstractmethod
    def _getDefaultName() -> str:
        pass
    
    def __init__(self, name: str = None, hidden: bool = False) -> None:
        self.name = name if name is not None and len(name.strip()) > 0 else self._getDefaultName()
        self.hidden = hidden

        self.gr_container: Gr_Container = None
        self.gr_outputs: list[Gr_Output] = []
        self.gr_outputs_extras: list[Gr_Output] = []
    
    def init_main(self, bMap: dict) -> None:
        self.gr_container = self.initContainer(not self.hidden)

        self.init(self.gr_outputs, self.gr_outputs_extras, bMap)
    
    def initUI(self) -> None:
        self.gr_container.initGr()
        with self.gr_container.gr:
            self.buildUI()
    
    def getInput(self, include_unbuilt: bool = False) -> list[Gr_Input]:
        gr_list: list[Gr_Input] = []
        
        for gr_output in self.gr_outputs:
            if gr_output.is_input and (include_unbuilt or gr_output.isGrBuilt()):
                gr_list.append(gr_output)

        return gr_list
    
    def getOutput(self, labeled_only: bool = False, exclude_labeled_outputs: bool = False) -> list[Gr_Output]:
        gr_list: list[Gr_Output] = []

        for gr_output in self.gr_outputs + self.gr_outputs_extras:
            if not gr_output.isGrBuilt():
                continue

            if labeled_only and not gr_output.is_labeled:
                continue

            if not gr_output.is_input and gr_output.is_labeled and exclude_labeled_outputs:
                continue

            gr_list.append(gr_output)
        
        return gr_list
    
    def getOutputUpdates(self, updates: list, reset: bool, *inputValues) -> int:
        """VIRTUAL: Base -> Populates updates with own Gr_Input updates, returns number of values consumed"""
        offset: int = 0

        for gr_output in self.gr_outputs:
            if gr_output.is_input and gr_output.isGrBuilt():
                offset += gr_output.getOutputUpdate(updates, reset, *inputValues[offset:])
        
        self.getOutputUpdatesExtra(updates, self.gr_outputs_extras)
        
        return offset
    
    def getOutputUpdatesExtra(self, updates: list, gr_outputs_extras: list[Gr_Output]) -> None:
        """VIRTUAL: Base -> Nothing"""
        pass
    
    def initContainer(self, visible: bool) -> Gr_Container:
        """VIRTUAL: Base -> Gr_Group"""
        return Gr_Group(visible, f"{self.name} (Group)")
    
    def validate(self, bMap: dict) -> bool:
        """VIRTUAL: Base -> validate own Gr_Outputs"""
        valid = True
        
        for x in self.gr_outputs:
            if not x.validate():
                valid = False
        
        return valid
    
    def consumeOutputs(self, *outputValues) -> int:
        """VIRTUAL: Base -> sync values on own Gr_Inputs, returns number of values consumed"""
        offset: int = 0

        for x in self.gr_outputs + self.gr_outputs_extras:
            if not x.isGrBuilt():
                continue
            
            if x.is_input:
                x_input: Gr_Input = x
                x_input.syncInput(outputValues[offset])
                offset += 1
            elif x.is_labeled:
                offset += 1
        
        return offset
    
    def finalizeUI(self, bMap: dict) -> None:
        """VIRTUAL: Bindings, etc, Base -> Nothing"""
        pass

    def validateArgs(self, *args) -> list[tuple[bool, str]]:
        """VIRTUAL: Base -> []"""
        return []

    def updateFromArgs(self, *args) -> None:
        """VIRTUAL: Base -> Nothing"""
        pass
    
    @abstractmethod
    def init(self, gr_outputs: list[Gr_Output], gr_outputs_extras: list[Gr_Output], bMap: dict) -> None:
        """Instantiates Gradio wrappers"""
        pass

    @abstractmethod
    def buildUI(self) -> None:
        """Builds Gradio layout and components from Gradio wrappers"""

    @abstractmethod
    def handlePrompt(self, p: StableDiffusionProcessing, bMap: dict) -> None:
        pass

class B_Ui_Separator(B_Ui):
    _html_separator: str = "<hr style=\"margin: 0.5em 0; border-style: dotted; border-color: var(--border-color-primary);\" />"

    @staticmethod
    def _paramsFromArgs(args: dict[str, str]) -> tuple[bool]:
        hidden = bool(int(args.get("hide", 0)))
        return hidden,
    
    @staticmethod
    def _fromArgs(args: dict[str, str], name: str = None):
        hidden, = B_Ui_Separator._paramsFromArgs(args)
        return B_Ui_Separator(
            hidden = hidden
        )
    
    @staticmethod
    def _getDefaultName() -> str:
        return "Separator"
    
    @staticmethod
    def _build() -> None:
        Gr_Markdown(B_Ui_Separator._html_separator).initGr()
    
    def __init__(self, name: str = None, hidden: bool = False) -> None:
        super().__init__(name, hidden)

        self.ui: Gr_Markdown = None
    
    def init(self, gr_outputs: list[Gr_Output], gr_outputs_extras: list[Gr_Output], bMap: dict[str, B_Ui]) -> None:
        self.ui = Gr_Markdown(self._html_separator, self.name)
        gr_outputs_extras.append(self.ui) #!
    
    def buildUI(self) -> None:
        self.ui.initGr()
    
    def handlePrompt(self, p: StableDiffusionProcessing, bMap: dict[str, B_Ui]) -> None:
        pass

class B_Ui_Collection(B_Ui, ABC):
    def __init__(
            self
            , name: str = None
            , items: list[B_Ui] = None
            , items_sort: bool = False
            , hidden: bool = False
        ) -> None:
        super().__init__(name, hidden)

        self.items = items if items is not None else []
        self.items_sort = items_sort

        self.ui_container_contents: Gr_Container = None
    
    def validate(self, bMap: dict[str, B_Ui]) -> bool:
        valid = super().validate(bMap)

        for x in self.items:
            if not x.validate(bMap):
                valid = False
        
        return valid
    
    def finalizeUI(self, bMap: dict[str, B_Ui]) -> None:
        for x in self.items:
            x.finalizeUI(bMap)
    
    def consumeOutputs(self, *outputValues) -> int:
        offset = super().consumeOutputs(*outputValues)

        for x in self.items:
            offset += x.consumeOutputs(*outputValues[offset:])
        
        return offset
    
    def getInput(self, include_unbuilt: bool = False) -> list[Gr_Input]:
        gr_inputs = super().getInput(include_unbuilt)

        for x in self.items:
            gr_inputs += x.getInput(include_unbuilt)
        
        return gr_inputs
    
    def getOutput(self, labeled_only: bool = False, exclude_labeled_outputs: bool = False) -> list[Gr_Output]:
        gr_outputs = super().getOutput(labeled_only, exclude_labeled_outputs)

        for x in self.items:
            gr_outputs += x.getOutput(labeled_only, exclude_labeled_outputs)
        
        return gr_outputs
    
    def getOutputUpdates(self, updates: list, reset: bool, *inputValues) -> int:
        offset = super().getOutputUpdates(updates, reset, *inputValues)

        for x in self.items:
            offset += x.getOutputUpdates(updates, reset, *inputValues[offset:])
        
        return offset
    
    def buildUI(self) -> None:
        self.buildGrContents_Top()

        self.ui_container_contents.initGr()
        with self.ui_container_contents.gr:
            for x in self.items:
                self.initItemUI(x)
        
        self.buildGrContents_Bottom()
    
    def handlePrompt(self, p: StableDiffusionProcessing, bMap: dict[str, B_Ui]) -> None:
        for x in self.items:
            x.handlePrompt(p, bMap)
    
    def addItem(self, item: B_Ui) -> None:
        self.items.append(item)
    
    def init(self, gr_outputs: list[Gr_Output], gr_outputs_extras: list[Gr_Output], bMap: dict[str, B_Ui]) -> None:
        """VIRTUAL: Base -> init contents container + children and sorts children (if items_sort)"""
        self.ui_container_contents = self.initContainerContents(f"{self.name} (Contents)")
        
        for x in self.items:
            x.init_main(bMap)
        
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
    
    def init(self, gr_outputs: list[Gr_Output], gr_outputs_extras: list[Gr_Output], bMap: dict[str, B_Ui]) -> None:
        super().init(gr_outputs, gr_outputs_extras, bMap)
        
        self.ui_reset = Gr_Button(f"Reset {self.name}")
        gr_outputs_extras.append(self.ui_reset)

        self.ui_random = Gr_Button(f"Randomize {self.name}")
        gr_outputs_extras.append(self.ui_random)
    
    def finalizeUI(self, bMap: dict[str, B_Ui]) -> None:
        super().finalizeUI(bMap)

        # if self.ui_random_build:
        #     self.ui_random.gr.click() #!
        
        if self.ui_reset_build:
            def _fnReset():
                updates: list = []
                self.getOutputUpdates(updates, True)
                return updates
            
            self.ui_reset.gr.click(
                fn = _fnReset
                , outputs = list(map(lambda gr_output: gr_output.gr, self.getOutput(exclude_labeled_outputs = True)))
            )
    
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

class B_Ui_Container_Tab(B_Ui_Container):
    @staticmethod
    def _paramsFromArgs(args: dict[str, str]) -> tuple[bool, bool, bool]:
        reset_ui_build = bool(int(args.get("build_reset_button", 1)))
        random_ui_build = bool(int(args.get("build_random_button", 1)))
        hidden = bool(int(args.get("hide", 0)))
        return reset_ui_build, random_ui_build, hidden
    
    @staticmethod
    def _fromArgs(args: dict[str, str], name: str = None):
        reset_ui_build, random_ui_build, hidden = B_Ui_Container_Tab._paramsFromArgs(args)
        return B_Ui_Container_Tab(
            name = name
            , reset_ui_build = reset_ui_build
            , random_ui_build = random_ui_build
            , hidden = hidden
        )
    
    @staticmethod
    def _getDefaultName() -> str:
        return "Tab"
    
    def __init__(
            self
            , name: str = None
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
    def _paramsFromArgs(args: dict[str, str]) -> tuple[bool, bool, bool]:
        reset_ui_build = bool(int(args.get("build_reset_button", 0)))
        random_ui_build = bool(int(args.get("build_random_button", 0)))
        hidden = bool(int(args.get("hide", 0)))
        return reset_ui_build, random_ui_build, hidden
    
    @staticmethod
    def _fromArgs(args: dict[str, str], name: str = None):
        reset_ui_build, random_ui_build, hidden = B_Ui_Container_Row._paramsFromArgs(args)
        return B_Ui_Container_Row(
            reset_ui_build = reset_ui_build
            , random_ui_build = random_ui_build
            , hidden = hidden
        )
    
    @staticmethod
    def _getDefaultName() -> str:
        return "Row"
    
    def __init__(
            self
            , name: str = None
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
    def _paramsFromArgs(args: dict[str, str]) -> tuple[int, bool, bool, bool]:
        scale = int(args.get("scale", 1))
        reset_ui_build = bool(int(args.get("build_reset_button", 0)))
        random_ui_build = bool(int(args.get("build_random_button", 0)))
        hidden = bool(int(args.get("hide", 0)))
        return scale, reset_ui_build, random_ui_build, hidden
    
    @staticmethod
    def _fromArgs(args: dict[str, str], name: str = None):
        scale, reset_ui_build, random_ui_build, hidden = B_Ui_Container_Column._paramsFromArgs(args)
        return B_Ui_Container_Column(
            scale = scale
            , reset_ui_build = reset_ui_build
            , random_ui_build = random_ui_build
            , hidden = hidden
        )
    
    @staticmethod
    def _getDefaultName() -> str:
        return "Column"
    
    def __init__(
            self
            , name: str = None
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
    def _paramsFromArgs(args: dict[str, str]) -> tuple[bool, bool, bool]:
        reset_ui_build = bool(int(args.get("build_reset_button", 0)))
        random_ui_build = bool(int(args.get("build_random_button", 0)))
        hidden = bool(int(args.get("hide", 0)))
        return reset_ui_build, random_ui_build, hidden
    
    @staticmethod
    def _fromArgs(args: dict[str, str], name: str = None):
        reset_ui_build, random_ui_build, hidden = B_Ui_Container_Group._paramsFromArgs(args)
        return B_Ui_Container_Group(
            reset_ui_build = reset_ui_build
            , random_ui_build = random_ui_build
            , hidden = hidden
        )
    
    @staticmethod
    def _getDefaultName() -> str:
        return "Group"
    
    def __init__(
            self
            , name: str = None
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
    def _paramsFromArgs(args: dict[str, str]) -> tuple[bool, bool, bool]:
        reset_ui_build = bool(int(args.get("build_reset_button", 0)))
        random_ui_build = bool(int(args.get("build_random_button", 0)))
        hidden = bool(int(args.get("hide", 0)))
        return reset_ui_build, random_ui_build, hidden
    
    @staticmethod
    def _fromArgs(args: dict[str, str], name: str = None):
        reset_ui_build, random_ui_build, hidden = B_Ui_Container_Accordion._paramsFromArgs(args)
        return B_Ui_Container_Accordion(
            reset_ui_build = reset_ui_build
            , random_ui_build = random_ui_build
            , hidden = hidden
        )
    
    @staticmethod
    def _getDefaultName() -> str:
        return "Accordion"
    
    def __init__(
            self
            , name: str = None
            , items: list[B_Ui] = None
            , reset_ui_build: bool = False
            , random_ui_build: bool = False
            , hidden: bool = False
        ) -> None:
        super().__init__(name, items, reset_ui_build, random_ui_build, hidden)
    
    def initContainer(self, visible: bool) -> Gr_Container:
        return Gr_Accordion(name = self.name, visible = visible)

class B_Ui_Prompt_Single(B_Ui):
    @staticmethod
    def _paramsFromArgs(args: dict[str, str]) -> tuple[str, str, str, float, bool, bool]:
        prefix = args.get("prefix", "")
        postfix = args.get("postfix", "")
        prompt = args.get("v", "")
        strength = float(args.get("s", 1))
        negative = bool(int(args.get("n", 0)))
        hidden = bool(int(args.get("hide", 0)))
        return prefix, postfix, prompt, strength, negative, hidden
    
    @staticmethod
    def _fromArgs(args: dict[str, str], name: str = None):
        prefix, postfix, prompt, strength, negative, hidden = B_Ui_Prompt_Single._paramsFromArgs(args)
        return B_Ui_Prompt_Single(
            name = name
            , prefix = prefix
            , postfix = postfix
            , prompt = prompt
            , strength = strength
            , negative = negative
            , hidden = hidden
        )
    
    @staticmethod
    def _getDefaultName() -> str:
        return "Single Prompt"
    
    def __init__(
            self
            , name: str = None
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
    
    def init(self, gr_outputs: list[Gr_Output], gr_outputs_extras: list[Gr_Output], bMap: dict[str, B_Ui]) -> None:
        self.ui_prompt = Gr_Textbox(self.name, self.prompt)
        gr_outputs.append(self.ui_prompt)

        self.ui_strength = Gr_Number(f"{self.name} (S)", self.strength, b_prompt_strength_min, b_prompt_strength_step)
        gr_outputs.append(self.ui_strength)

        self.ui_negative = Gr_Checkbox(f"{self.name} (N)", self.negative)
        gr_outputs.append(self.ui_negative)
    
    def validateArgs(
            self
            , prefix: str
            , postfix: str
            , prompt: str
            , strength: float
            , negative: bool
            , hidden: bool
        ) -> list[tuple[bool, str]]:
        return [
            self.ui_prompt.validateValue(prompt)
            , self.ui_strength.validateValue(strength)
            , self.ui_negative.validateValue(negative)
        ]
    
    def updateFromArgs(
            self
            , prefix: str
            , postfix: str
            , prompt: str
            , strength: float
            , negative: bool
            , hidden: bool
        ) -> None:
        self.ui_prompt.syncInput(prompt)
        self.ui_strength.syncInput(strength)
        self.ui_negative.syncInput(negative)
    
    def buildUI(self) -> None:
        if self.ui_prompt_build:
            self.ui_prompt.initGr()
        
        if self.ui_strength_build or self.ui_negative_build:
            with gr.Row():
                if self.ui_strength_build:
                    self.ui_strength.initGr()

                if self.ui_negative_build:
                    self.ui_negative.initGr()
    
    def handlePrompt(self, p: StableDiffusionProcessing, bMap: dict[str, B_Ui]) -> None:
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

class B_Ui_Prompt_Dual(B_Ui):
    @staticmethod
    def _paramsFromArgs(args: dict[str, str]) -> tuple[str, str, float, bool]:
        prompt_positive = args.get("vp", "")
        prompt_negative = args.get("vn", "")
        strength = float(args.get("s", 1))
        hidden = bool(int(args.get("hide", 0)))
        return prompt_positive, prompt_negative, strength, hidden
    
    @staticmethod
    def _fromArgs(args: dict[str, str], name: str = None):
        prompt_positive, prompt_negative, strength, hidden = B_Ui_Prompt_Dual._paramsFromArgs(args)
        return B_Ui_Prompt_Dual(
            name = name
            , prompt_positive = prompt_positive
            , prompt_negative = prompt_negative
            , strength = strength
            , hidden = hidden
        )
    
    @staticmethod
    def _getDefaultName() -> str:
        return "Dual Prompt"
    
    def __init__(
            self
            , name: str = None
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
    
    def init(self, gr_outputs: list[Gr_Output], gr_outputs_extras: list[Gr_Output], bMap: dict[str, B_Ui]) -> None:
        self.ui_prompt_positive = Gr_Textbox(f"{self.name} (+)", self.prompt_positive)
        gr_outputs.append(self.ui_prompt_positive)

        self.ui_prompt_negative = Gr_Textbox(f"{self.name} (-)", self.prompt_negative)
        gr_outputs.append(self.ui_prompt_negative)

        self.ui_strength = Gr_Number(f"{self.name} (S)", self.strength, b_prompt_strength_min, b_prompt_strength_step)
        gr_outputs.append(self.ui_strength)
    
    def validateArgs(
            self
            , prompt_positive: str
            , prompt_negative: str
            , strength: float
            , hidden: bool
        ) -> list[tuple[bool, str]]:
        return [
            self.ui_prompt_positive.validateValue(prompt_positive)
            , self.ui_prompt_negative.validateValue(prompt_negative)
            , self.ui_strength.validateValue(strength)
        ]
    
    def updateFromArgs(
            self
            , prompt_positive: str
            , prompt_negative: str
            , strength: float
            , hidden: bool
        ) -> None:
        self.ui_prompt_positive.syncInput(prompt_positive)
        self.ui_prompt_negative.syncInput(prompt_negative)
        self.ui_strength.syncInput(strength)
    
    def buildUI(self) -> None:
        if self.ui_prompts_build:
            with gr.Row():
                self.ui_prompt_positive.initGr()
                self.ui_prompt_negative.initGr()
        
        if self.ui_strength_build:
            self.ui_strength.initGr()
    
    def handlePrompt(self, p: StableDiffusionProcessing, bMap: dict[str, B_Ui]) -> None:
        prompt_positive = str(self.ui_prompt_positive.value)
        prompt_negative = str(self.ui_prompt_negative.value)
        strength = float(self.ui_strength.value)

        prompt_positive = promptSanitized(prompt_positive)
        prompt_negative = promptSanitized(prompt_negative)

        if strength > 0 and strength != 1:
            if len(prompt_positive) > 0:
                prompt_positive = f"({prompt_positive}:{strength})"
            if len(prompt_negative) > 0:
                prompt_negative = f"({prompt_negative}:{strength})"
        
        p.prompt = promptAdded(p.prompt, prompt_positive)
        p.negative_prompt = promptAdded(p.negative_prompt, prompt_negative)

class B_Ui_Prompt_Range(B_Ui):
    _value_min: int = -1
    _value_max: int = 100
    _value_step: int = 1

    @staticmethod
    def _paramsFromArgs(args: dict[str, str]) -> tuple[str, str, bool, bool, int, bool, str, str, bool]:
        prompt_a = args.get("a", "")
        prompt_b = args.get("b", "")
        required = bool(int(args.get("is_required", 0)))
        negative = bool(int(args.get("n", 0)))
        value = int(args.get("v", None))
        ui_buttons_build = bool(int(args.get("build_buttons", 1)))
        prompt_a_button_text = args.get("a_button", None)
        prompt_b_button_text = args.get("b_button", None)
        hidden = bool(int(args.get("hide", 0)))
        return prompt_a, prompt_b, required, negative, value, ui_buttons_build, prompt_a_button_text, prompt_b_button_text, hidden
    
    @staticmethod
    def _fromArgs(args: dict[str, str], name: str = None):
        prompt_a, prompt_b, required, negative, value, ui_buttons_build, prompt_a_button_text, prompt_b_button_text, hidden = B_Ui_Prompt_Range._paramsFromArgs(args)
        return B_Ui_Prompt_Range(
            name = name
            , prompt_a = prompt_a
            , prompt_b = prompt_b
            , required = required
            , negative = negative
            , value = value
            , ui_buttons_build = ui_buttons_build
            , prompt_a_button_text = prompt_a_button_text
            , prompt_b_button_text = prompt_b_button_text
            , hidden = hidden
        )
    
    @staticmethod
    def _getDefaultName() -> str:
        return "Range Prompt"
    
    def __init__(
            self
            , prompt_a: str
            , prompt_b: str
            , name: str = None
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
    
    def init(self, gr_outputs: list[Gr_Output], gr_outputs_extras: list[Gr_Output], bMap: dict[str, B_Ui]) -> None:
        self.ui_range = Gr_Slider(self.name, self.value, self.value_min, self._value_max, self._value_step)
        gr_outputs.append(self.ui_range)

        self.ui_negative = Gr_Checkbox(f"{self.name} (N)", self.negative)
        gr_outputs.append(self.ui_negative)

        self.ui_button_a = Gr_Button(self.ui_button_a_text)
        gr_outputs_extras.append(self.ui_button_a)

        self.ui_button_b = Gr_Button(self.ui_button_b_text)
        gr_outputs_extras.append(self.ui_button_b)
    
    def validateArgs(
            self
            , prompt_a: str
            , prompt_b: str
            , required: bool
            , negative: bool
            , value: int
            , ui_buttons_build: bool
            , prompt_a_button_text: str
            , prompt_b_button_text: str
            , hidden: bool
        ) -> list[tuple[bool, str]]:
        return [
            self.ui_range.validateValue(value)
            , self.ui_negative.validateValue(negative)
        ]
    
    def updateFromArgs(
            self
            , prompt_a: str
            , prompt_b: str
            , required: bool
            , negative: bool
            , value: int
            , ui_buttons_build: bool
            , prompt_a_button_text: str
            , prompt_b_button_text: str
            , hidden: bool
        ) -> None:
        self.ui_range.syncInput(value)
        self.ui_negative.syncInput(negative)
    
    def buildUI(self) -> None:
        self.ui_range.initGr()

        if self.ui_buttons_build:
            with gr.Row():
                self.ui_button_a.initGr()
                self.ui_button_b.initGr()
        
        if self.ui_negative_build:
            self.ui_negative.initGr()
    
    def finalizeUI(self, bMap: dict[str, B_Ui]) -> None:
        if self.ui_buttons_build:
            self.ui_button_a.gr.click(
                fn = lambda: 0
                , outputs = self.ui_range.gr
            )

            self.ui_button_b.gr.click(
                fn = lambda: self.ui_range.value_max
                , outputs = self.ui_range.gr
            )
    
    def handlePrompt(self, p: StableDiffusionProcessing, bMap: dict[str, B_Ui]) -> None:
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

class B_Ui_Prompt_Select(B_Ui_Collection):
    _choice_empty: str = "-"
    _random_choices_max: int = 5

    @staticmethod
    def _paramsFromArgs(args: dict[str, str]) -> tuple[bool, str | list[str], bool, bool, bool, int, bool]:
        choices_default = args.get("v", None)
        if choices_default is not None and len(choices_default) > 0:
            choices_default = list(map(lambda v: v.strip(), choices_default.split(","))) #! str | list[str]?
        
        choices_sort = bool(int(args.get("sort", 1)))
        multiselect = bool(int(args.get("multi_select", 0)))
        custom = bool(int(args.get("allow_custom", 0)))
        simple = bool(int(args.get("simple", 0)))
        scale = int(args.get("scale", 1))
        hidden = bool(int(args.get("hide", 0)))

        return choices_sort, choices_default, multiselect, custom, simple, scale, hidden

    @staticmethod
    def _fromArgs(args: dict[str, str], name: str = None):
        choices_sort, choices_default, multiselect, custom, simple, scale, hidden = B_Ui_Prompt_Select._paramsFromArgs(args)
        return B_Ui_Prompt_Select(
            name = name
            , choices_sort = choices_sort
            , choices_default = choices_default
            , multiselect = multiselect
            , custom = custom
            , simple = simple
            , scale = scale
            , hidden = hidden
        )
    
    @staticmethod
    def _getDefaultName() -> str:
        return "Select Prompt"
    
    #! conflicting names...
    @staticmethod
    def _buildColorChoicesList(postfix: str = "") -> list[B_Ui_Prompt_Single]:
        return list(map(
            lambda text: B_Ui_Prompt_Single(
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
            , name: str = None
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

        self.choices_default = choices_default if choices_default is not None or not multiselect else []
        self.multiselect = multiselect

        self.choicesMap: dict[str, B_Ui] = {}
        self.choicesContainerMap: dict[str, Gr_Column] = {}
        self.choicesPresetMap: dict[str, B_Ui_Preset] = {}

        self.ui_dropdown: Gr_Dropdown = None
    
    def init(self, gr_outputs: list[Gr_Output], gr_outputs_extras: list[Gr_Output], bMap: dict[str, B_Ui]) -> None:
        super().init(gr_outputs, gr_outputs_extras, bMap)

        # INIT choicesMap + choices_list
        choices_list: list[str] = []
        if not self.multiselect:
            choices_list.append(self._choice_empty)

        for x in self.items:
            choices_list.append(x.name)
            self.choicesMap[x.name] = x

        # INIT choicesPresetMap
        for preset in self.choicesPresetMap.values():
            preset.buildMappings(bMap)

        self.ui_dropdown = Gr_Dropdown(self.name, choices_list, self.choices_default, self.multiselect)
        gr_outputs.append(self.ui_dropdown)

        self.ui_container_contents.visible = self.getShowContainer()
        gr_outputs_extras.append(self.ui_container_contents)
        for x in self.items:
            x_container = Gr_Column(1, "panel", self.getShowChoiceContainer(x), f"{self.name}_{x.name} (Container)")
            self.choicesContainerMap[x.name] = x_container
            gr_outputs_extras.append(x_container)
    
    def validateArgs(
            self
            , choices_sort: bool
            , choices_selected: str | list[str]
            , multiselect: bool
            , custom: bool
            , simple: bool
            , scale: int
            , hidden: bool
        ) -> list[tuple[bool, str]]:
        return [
            self.ui_dropdown.validateValue(choices_selected)
        ]
    
    def updateFromArgs(
            self
            , choices_sort: bool
            , choices_selected: str | list[str]
            , multiselect: bool
            , custom: bool
            , simple: bool
            , scale: int
            , hidden: bool
        ) -> None:
        self.ui_dropdown.syncInput(choices_selected)
    
    def finalizeUI(self, bMap: dict[str, B_Ui]) -> None:
        super().finalizeUI(bMap)

        #! Show/Hide selected - find better way to exclude "source" element from event's output?
        def _fnShowHide(*inputValues):
            updates: list = []
            self.getOutputUpdates(updates, False, *inputValues)
            return updates
        
        self.ui_dropdown.gr.input(
            fn = lambda *inputValues: _fnShowHide(*inputValues)[1:]
            , inputs = list(map(lambda gr_input: gr_input.gr, self.getInput()))
            , outputs = list(map(lambda gr_output: gr_output.gr, self.getOutput(exclude_labeled_outputs = True)))[1:]
        )

        #! Presets - potentially repeating logic from B_Ui_Preset, encapsulate some of this?
        if len(self.choicesPresetMap) > 0:
            bList: list[B_Ui] = []
            for c in self.choicesPresetMap:
                for k in self.choicesPresetMap[c].mappings:
                    b = bMap[k]
                    if b not in bList:
                        bList.append(b)
            
            inputs: list = []
            outputs: list = []
            for b in bList:
                for gr_input in b.getInput():
                    inputs.append(gr_input.gr)
                for gr_output in b.getOutput(exclude_labeled_outputs = True):
                    outputs.append(gr_output.gr)
            
            def _apply(choices: str | list[str], *inputValues):
                self.ui_dropdown.syncInput(choices)

                if type(choices) is not list:
                    choices: list = [choices]
                
                presets: list[B_Ui_Preset] = []
                for c in choices:
                    preset = self.choicesPresetMap.get(c, None)
                    if preset is not None:
                        presets.append(preset)
                
                updates: list = []
                offset: int = 0
                preset_mapping: tuple = None
                for b in bList:
                    for preset in presets:
                        preset_mapping = preset.mappings.get(b.name, None)
                        if preset_mapping is not None:
                            break #! if 2 choices affect same element, first one prioritized
                    
                    if preset_mapping is not None:
                        b.updateFromArgs(*preset_mapping)
                        offset += b.getOutputUpdates(updates, False)
                    else:
                        offset += b.getOutputUpdates(updates, False, *inputValues[offset:])
                
                return updates
            
            self.ui_dropdown.gr.select(
                fn = _apply
                , inputs = [self.ui_dropdown.gr] + inputs
                , outputs = outputs
            )
    
    def initContainerContents(self, name: str) -> Gr_Container:
        return Gr_Row("panel", name = name)
    
    def buildGrContents_Top(self) -> None:
        self.ui_dropdown.initGr()
    
    def initItemUI(self, item: B_Ui) -> None:
        item_container = self.choicesContainerMap[item.name]
        item_container.initGr()
        with item_container.gr:
            super().initItemUI(item)
    
    def getOutputUpdatesExtra(self, updates: list, gr_outputs_extras: list[Gr_Output]) -> None:
        self.ui_container_contents.getUpdateVisible(updates, self.getShowContainer())
        for x in self.items:
            self.choicesContainerMap[x.name].getUpdateVisible(updates, self.getShowChoiceContainer(x))
    
    def handlePrompt(self, p: StableDiffusionProcessing, bMap: dict[str, B_Ui]) -> None:
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
            x.handlePrompt(p, bMap)
    
    def addChoicePresetMapping(self, name: str, target_name: str, target_args: dict[str, str]):
        preset = self.choicesPresetMap.get(name, None)
        
        if preset is None:
            preset = B_Ui_Preset(f"{self.name}_{name} (PRESET)", additive = True, hidden = True)
            self.choicesPresetMap[name] = preset
        
        preset.addMapping(target_name, target_args)
    
    def addChoices(self, args: dict[str, str]):
        choicesList: list[B_Ui] = []
        
        special_type = args.get("type", "")
        match special_type:
            case "COLOR":
                postfix = args.get("postfix", "")
                choicesList += self._buildColorChoicesList(postfix)
            case _:
                print(f"WARNING: Invalid CHOICES type in {self.name} -> {special_type}")
        
        self.items += choicesList
    
    def getShowContainer(self) -> bool:
        choice_current = self.ui_dropdown.value
        return choice_current is not None and len(choice_current) > 0 and choice_current != self._choice_empty
    
    def getShowChoiceContainer(self, x: B_Ui) -> bool:
        choice_current = self.ui_dropdown.value
        choice = x.name
        return choice_current is not None and (choice == choice_current or choice in choice_current)

class B_Ui_Prompt_Range_Link(B_Ui):
    @staticmethod
    def _paramsFromArgs(args: dict[str, str]) -> tuple[str, str, str, bool]:
        name_link = args.get("link", None)
        prompt_a = args.get("a", "")
        prompt_b = args.get("b", "")
        hidden = bool(int(args.get("hide", 0)))
        return name_link, prompt_a, prompt_b, hidden
    
    @staticmethod
    def _fromArgs(args: dict[str, str], name: str = None):
        name_link, prompt_a, prompt_b, hidden = B_Ui_Prompt_Range_Link._paramsFromArgs(args)
        return B_Ui_Prompt_Range_Link(
            name = name
            , name_link = name_link
            , prompt_a = prompt_a
            , prompt_b = prompt_b
            , hidden = hidden
        )
    
    @staticmethod
    def _getDefaultName() -> str:
        return "Range Prompt {LINK}"
    
    def __init__(
            self
            , name: str
            , name_link: str
            , prompt_a: str
            , prompt_b: str
            , hidden: bool = False
        ) -> None:
        super().__init__(f"{name} {{LINK: {name_link}}}", hidden)

        self.name_original = name
        self.name_link = name_link
        self.prompt_a = prompt_a
        self.prompt_b = prompt_b

        self.ui: Gr_Markdown = None
    
    def init(self, gr_outputs: list[Gr_Output], gr_outputs_extras: list[Gr_Output], bMap: dict[str, B_Ui]) -> None:
        self.ui = Gr_Markdown(f"**{self.name_original}**")
    
    def validate(self, bMap: dict[str, B_Ui]) -> bool:
        valid = super().validate(bMap)

        if self.name_link is None or len(self.name_link) == 0:
            printWarning(self.__class__.__name__, self.name_original, f"Invalid link name -> {self.name_link}")
            return False

        b_link = bMap.get(self.name_link, None)

        if b_link is None:
            printWarning(self.__class__.__name__, self.name_original, f"No component found with linked name -> '{self.name_link}'")
            return False
        
        if type(b_link) is not B_Ui_Prompt_Range:
            valid = False
            printWarning(self.__class__.__name__, self.name_original, f"Linked component type is invalid -> {b_link.__class__.__name__}")
        
        return valid
    
    def buildUI(self) -> None:
        self.ui.initGr()
    
    def handlePrompt(self, p: StableDiffusionProcessing, bMap: dict[str, B_Ui]) -> None:
        b_link: B_Ui_Prompt_Range = bMap[self.name_link]

        prompt_a = promptSanitized(self.prompt_a)
        prompt_b = promptSanitized(self.prompt_b)

        negative = bool(b_link.ui_negative.value)

        value = float(b_link.ui_range.value)
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

class B_Ui_Preset(B_Ui):
    @staticmethod
    def _paramsFromArgs(args: dict[str, str]) -> tuple[bool, bool]:
        additive = bool(int(args.get("is_additive", 0)))
        hidden = bool(int(args.get("hide", 0)))
        return additive, hidden
    
    @staticmethod
    def _fromArgs(args: dict[str, str], name: str = None):
        additive, hidden = B_Ui_Preset._paramsFromArgs(args)
        return B_Ui_Preset(
            name = name
            , additive = additive
            , hidden = hidden
        )
    
    @staticmethod
    def _getDefaultName() -> str:
        return "Preset"
    
    def __init__(
            self
            , name: str = None
            , mappings: dict[str, tuple] = None
            , additive: bool = False
            , hidden: bool = False
        ) -> None:
        super().__init__(name, hidden)

        self.mappings = mappings if mappings is not None else {}
        self.additive = additive
        
        self.mappings_temp: dict[str, dict[str, str]] = {}

        self.ui: Gr_Button = None
    
    def init(self, gr_outputs: list[Gr_Output], gr_outputs_extras: list[Gr_Output], bMap: dict[str, B_Ui]) -> None:
        self.ui = Gr_Button(self.name)
        gr_outputs_extras.append(self.ui)

        self.buildMappings(bMap)
    
    def validate(self, bMap: dict[str, B_Ui]) -> bool:
        valid = super().validate(bMap)

        if len(self.mappings) == 0:
            printWarning(self.__class__.__name__, self.name, "No entries set")
            return False
        
        for k in self.mappings:
            if k not in bMap:
                valid = False
                printWarning(self.__class__.__name__, self.name, f"Entry not found -> '{k}'")
            else:
                b = bMap[k]
                for v_valid, v_message in b.validateArgs(*self.mappings[k]):
                    if not v_valid:
                        valid = False
                        printWarning(self.__class__.__name__, self.name, v_message)
        
        return valid
    
    def buildUI(self) -> None:
        self.ui.initGr()
    
    def finalizeUI(self, bMap: dict[str, B_Ui]) -> None:
        inputs: list = []
        outputs: list = []

        bList: list[B_Ui] = None
        if self.additive:
            bList = []
            for b in bMap.values():
                if b.name in self.mappings:
                    bList.append(b)
        else:
            bList = list(bMap.values())
        
        for b in bList:
            for gr_input in b.getInput():
                inputs.append(gr_input.gr)
            for gr_output in b.getOutput(exclude_labeled_outputs = True):
                outputs.append(gr_output.gr)
        
        def _apply(*inputValues) -> list:
            updates: list = []
            offset: int = 0

            for x in bList:
                if x.name in self.mappings:
                    x.updateFromArgs(*self.mappings[x.name])
                    offset += x.getOutputUpdates(updates, False)
                else:
                    offset += x.getOutputUpdates(updates, not self.additive, *inputValues[offset:])
            
            return updates

        self.ui.gr.click(
            fn = _apply
            , inputs = inputs
            , outputs = outputs
        )
    
    def handlePrompt(self, p: StableDiffusionProcessing, bMap: dict[str, B_Ui]) -> None:
        pass

    def buildMappings(self, bMap: dict[str, B_Ui]) -> None:
        if len(self.mappings_temp) > 0:
            for k in self.mappings_temp:
                self.mappings[k] = bMap[k]._paramsFromArgs(self.mappings_temp[k])
            
            del self.mappings_temp #!

    def addMapping(self, name: str, args: dict[str, str]):
        if name in self.mappings_temp:
            printWarning(self.__class__.__name__, self.name, f"Duplicate entry ({name})")
        
        self.mappings_temp[name] = args

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

    def __init__(self) -> None:
        self.path_script_config = os.path.join(b_path_base, b_folder_name_scripts, b_folder_name_script_config)
        self.path_layout = os.path.join(self.path_script_config, b_file_name_layout)
        self.path_presets = os.path.join(self.path_script_config, b_file_name_presets)

        # PARSE
        self.layout = self.parseLayout()
        self.presets = self.parsePresets()

        # INIT
        self.map: dict[str, B_Ui] = {}
        self.buildMapRecursive(self.map, self.layout)

        for x in self.presets:
            x.init_main(self.map)

        for x in self.layout:
            x.init_main(self.map)
        
        # VALIDATE
        if not b_validate_skip:
            valid: bool = True

            for preset in self.presets:
                if not preset.validate(self.map):
                    valid = False

            for x in self.layout:
                if not x.validate(self.map):
                    valid = False
            
            if not valid:
                printWarning("B_Ui_Map", "validate()", "Invalid layout or presets")
    
    def parseLayout(self) -> list[B_Ui]:
        layout: list[B_Ui] = []
        
        stack_containers: list[B_Ui_Container] = []
        stack_selects: list[B_Ui_Prompt_Select] = []
        select_choice_has_preset: bool = False

        skip = 0
        
        def _build(item: B_Ui) -> None:
            if len(stack_selects) > 0:
                stack_selects[-1].addItem(item)
                return
            
            if len(stack_containers) > 0:
                stack_containers[-1].addItem(item)
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
                        if select_choice_has_preset:
                            select_choice_has_preset = False
                            continue

                        if len(stack_selects) > 0:
                            item_select = stack_selects.pop()
                            _build(item_select)
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

                        _build(B_Ui_Prompt_Single._fromArgs(l_args, l_name))
                    
                    case "DUAL":
                        if ignore:
                            continue

                        _build(B_Ui_Prompt_Dual._fromArgs(l_args, l_name))
                    
                    case "RANGE":
                        if ignore:
                            continue
                        
                        _build(B_Ui_Prompt_Range._fromArgs(l_args, l_name))
                    
                    case "RANGE_LINK":
                        if ignore:
                            continue
                        
                        _build(B_Ui_Prompt_Range_Link._fromArgs(l_args, l_name))
                    
                    case "SELECT":
                        if ignore:
                            skip += 1
                            continue
                        
                        stack_selects.append(B_Ui_Prompt_Select._fromArgs(l_args, l_name))
                    
                    case "CHOICES":
                        if ignore:
                            continue

                        stack_selects[-1].addChoices(l_args)
                    
                    case "SET":
                        select_choice_has_preset = True
                        stack_selects[-1].addChoicePresetMapping(stack_selects[-1].items[-1].name, l_name, l_args)
                    
                    case "GROUP":
                        if ignore:
                            skip += 1
                            continue

                        stack_containers.append(B_Ui_Container_Group._fromArgs(l_args, l_name))
                    
                    case "TAB":
                        if ignore:
                            skip += 1
                            continue
                        
                        stack_containers.append(B_Ui_Container_Tab._fromArgs(l_args, l_name))
                    
                    case "ROW":
                        if ignore:
                            skip += 1
                            continue
                        
                        stack_containers.append(B_Ui_Container_Row._fromArgs(l_args, l_name))
                    
                    case "COLUMN":
                        if ignore:
                            skip += 1
                            continue
                        
                        stack_containers.append(B_Ui_Container_Column._fromArgs(l_args, l_name))
                    
                    case "ACCORDION":
                        if ignore:
                            skip += 1
                            continue
                        
                        stack_containers.append(B_Ui_Container_Accordion._fromArgs(l_args, l_name))
                    
                    case "SEPARATOR":
                        if ignore:
                            continue
                        
                        _build(B_Ui_Separator._fromArgs(l_args))

                    case _:
                        print(f"WARNING: Invalid layout type -> {l_type}")
        
        return layout
    
    def parsePresets(self) -> list[B_Ui_Preset]:
        presets: list[B_Ui_Preset] = []
        
        preset_current: B_Ui_Preset = None
        
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
                        preset_current = B_Ui_Preset._fromArgs(l_args, l_name)
                    
                    case "SET":
                        preset_current.addMapping(l_name, l_args)
                    
                    case _:
                        print(f"WARNING: Invalid preset type -> {l_type}")
        
        return presets

    def buildMapRecursive(self, target: dict[str, B_Ui], layout: list[B_Ui]):
        for x in layout:
            x_type = type(x)

            if not issubclass(x_type, B_Ui_Collection) or x_type is B_Ui_Prompt_Select:
                if x.name in target:
                    printWarning("B_Ui_Map", "buildMapRecursive()", f"Duplicate B_Ui name -> '{x.name}'")
                
                target[x.name] = x
            
            if issubclass(x_type, B_Ui_Collection):
                x_collection: B_Ui_Collection = x
                self.buildMapRecursive(target, x_collection.items)
    
    def initUI(self) -> list[typing.Any]:
        gr_list: list[typing.Any] = []

        # PRESETS
        B_Ui_Separator._build()

        with gr.Accordion("Presets", open = False):
            i = 0
            for preset in self.presets:
                preset.initUI()
                gr_list += list(map(lambda gr_output: gr_output.gr, preset.getOutput(True)))

                i += 1
                if i < len(self.presets) and not preset.hidden:
                    B_Ui_Separator._build()

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
        
        # (FINALIZE)
        for preset in self.presets:
            preset.finalizeUI(self.map)

        for x in self.layout:
            x.finalizeUI(self.map)
        
        # - DONE -
        return gr_list
    
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

class Script(scripts.Script):
    bUiMap = B_Ui_Map()
    
    def title(self):
        return "B Prompt Builder"

    def show(self, is_img2img):
        return not is_img2img
    
    def ui(self, is_img2img):
        return self.bUiMap.initUI()

    def run(self, p, *outputValues):
        i = 0
        
        for preset in self.bUiMap.presets:
            i += preset.consumeOutputs(*outputValues[i:])
        
        for x in self.bUiMap.layout:
            i += x.consumeOutputs(*outputValues[i:])
            x.handlePrompt(p, self.bUiMap.map)
        
        proc = process_images(p)
        
        return proc
