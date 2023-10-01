import modules.scripts as scripts
import gradio as gr
import os
import typing
import random
import json

from abc import ABC, abstractmethod
from modules import scripts
from modules.processing import StableDiffusionProcessing, process_images

def addPrompt(promptExisting: str, promptToAdd: str) -> str:
    if len(promptToAdd) > 0:
        if len(promptExisting) > 0:
            promptExisting += ", " + promptToAdd
        else:
            promptExisting = promptToAdd
    
    return promptExisting

def buildRangePrompt(promptA: str, promptB: str, value: float) -> str:
    if len(promptA) == 0 or len(promptB) == 0:
        return ""
    
    if value < 0 or value > 1:
        return ""
    
    if value == 0:
        return promptA
    
    if value == 1:
        return promptB
        
    return f"[{promptB}:{promptA}:{value}]"

def printWarning(component: str, name: str, message: str):
    print(f"VALIDATE/{component}/{name} -> {message}")

class B_UI(ABC):
    _identifier: int = 0
    
    def __init__(self, name: str, visible: bool, isNamed: bool):
        self.identifier = self.getNextIdentifier()
        self.name = self.handleName(name)
        self.visible = visible
        self.isNamed = isNamed

        self.ui: typing.Any = None
        self.ui_extra_inputs: list[typing.Any] = []
        self.ui_extra_outputs: list[typing.Any] = []
    
    def getNextIdentifier(self) -> int:
        B_UI._identifier += 1
        return B_UI._identifier
    
    def handleName(self, name: str = None) -> str:
        if name is None or len(name) == 0:
            name = f"{self.identifier}_{self.getDefaultName()}"
        
        return name
    
    def getDefaultName(self) -> str:
        return "UI"
    
    def buildUI(self) -> list[typing.Any]:
        """Returns a list of Gradio components (first entry is self if named)"""
        self.ui = self.buildSelf()
        if self.isNamed:
            return [self.ui]
        else:
            return []
    
    def getExtraUiCount(self) -> int:
        return 0
    
    def setValue(self, *values) -> int:
        """Returns number of gradio components consumed in this instance"""
        return (0 if not self.isNamed else 1) + self.getExtraUiCount()
    
    def validate(self, componentMap: dict) -> bool:
        """Base function -> True"""
        return True
    
    def finalizeUI(self, componentMap: dict):
        """Bindings, etc."""
        pass

    def handlePrompt(self, p: StableDiffusionProcessing, componentMap: dict):
        pass

    def getUpdate(self, *values) -> tuple[list, int]:
        """Returns update values and number of inputs consumed"""
        return [], 0
    
    def getUpdateRandom(self, *currentValues) -> tuple[list, int]:
        """Returns update values and number of inputs consumed"""
        return self.getUpdate(*currentValues)
    
    @abstractmethod
    def buildSelf(self) -> typing.Any:
        pass

class B_UI_Markdown(B_UI):
    html_separator: str = "<hr style=\"margin: 0.5em 0; border-style: dotted; border-color: var(--border-color-primary);\" />"

    @staticmethod
    def _fromArgs(isSeparator: bool, **kwargs: str):
        return B_UI_Markdown(
            isSeparator = isSeparator
            , value = kwargs.get("v", None)
            , visible = not bool(int(kwargs.get("hide", 0)))
        )

    @staticmethod
    def _buildSeparator():
        markdown = B_UI_Markdown(isSeparator = True)
        markdown.buildUI()
        return markdown.ui
    
    def __init__(self, value: str = None, isSeparator: bool = False, visible: bool = True):
        super().__init__(None, visible, False)

        if isSeparator:
            value = self.html_separator
        
        self.value: str = value
    
    def getDefaultName(self) -> str:
        return "Markdown"
    
    def buildSelf(self) -> typing.Any:
        return gr.Markdown(
            value = self.value
            , visible = self.visible
        )

class B_UI_Component(B_UI, ABC):
    @staticmethod
    @abstractmethod
    def _fromArgs(name: str, *args, **kwargs: str) -> B_UI:
        pass
    
    def __init__(self, name: str = None, defaultValue = None, visible: bool = True):
        super().__init__(name, visible, True)

        self.value = defaultValue
        self.defaultValue = defaultValue
    
    def getDefaultName(self) -> str:
        return "Component"
    
    def setValue(self, *values) -> int:
        self.value = self.processValue(values[0])
        return super().setValue(*values)
    
    def validate(self, componentMap: dict) -> bool:
        """Base function -> Validates default value"""
        return super().validate(componentMap) and self.validateValue(self.defaultValue)
    
    def getUpdate(self, *values) -> tuple[list, int]:
        return [self.handleUpdateValue(values[0] if len(values) > 0 else None)], 1
    
    def processValue(self, value: typing.Any) -> typing.Any:
        return value
    
    def validateValue(self, value: typing.Any) -> bool:
        """Base function -> True"""
        return True
    
    def handleUpdateValue(self, value):
        return value if value is not None else self.defaultValue

class B_UI_Container(B_UI):
    @staticmethod
    @abstractmethod
    def _fromArgs(name: str, items: list[B_UI], **kwargs: str) -> B_UI:
        pass

    def __init__(self, name: str = None, items: list[B_UI] = [], visible: bool = True, buildCustomPromptInputs: bool = False, buildResetButton: bool = False, buildRandomButton: bool = False):
        super().__init__(name, visible, False)
        
        self.items = self.handleItems(items, buildCustomPromptInputs, name)
        self.buildResetButton = buildResetButton
        self.buildRandomButton = buildRandomButton

        self.bComponents = self.getBComponents()

        self.buttonReset: typing.Any = None
        self.buttonRandom: typing.Any = None
    
    def getDefaultName(self) -> str:
        return "Container"
    
    def setValue(self, *values) -> int:
        i: int = super().setValue(*values)
        
        for bUi in self.items:
            i += bUi.setValue(*values[i:])
        
        return i
    
    def buildUI(self) -> list[typing.Any]:
        built = super().buildUI()
        built_items: list[typing.Any] = []
        
        built_items_inputs: list[typing.Any] = []
        built_items_outputs: list[typing.Any] = []
        
        with self.ui:
            for item in self.items:
                item_ui = item.buildUI()
                
                if len(item_ui) > 0:
                    built_items += item_ui

                    if issubclass(type(item), B_UI_Component):
                        bComponent: B_UI_Component = item
                        built_items_inputs += [bComponent.ui] + bComponent.ui_extra_inputs
                        built_items_outputs += [bComponent.ui] + bComponent.ui_extra_outputs
                    elif issubclass(type(item), B_UI_Container):
                        bContainer: B_UI_Container = item
                        for bComponent in bContainer.bComponents:
                            built_items_inputs += [bComponent.ui] + bComponent.ui_extra_inputs
                            built_items_outputs += [bComponent.ui] + bComponent.ui_extra_outputs
            
            if self.buildResetButton or self.buildRandomButton:
                def _buildRandomButton() -> typing.Any:
                    btnRandom = gr.Button(value = f"Randomize {self.name}")
                    btnRandom.click(
                        fn = self.randomizeComponentsValues
                        , inputs = built_items_inputs
                        , outputs = built_items_outputs
                    )
                    self.buttonRandom = btnRandom
                    return btnRandom
                
                def _buildResetButton() -> typing.Any:
                    btnReset = gr.Button(value = f"Reset {self.name}")
                    btnReset.click(
                        fn = self.resetComponentsValues
                        , outputs = built_items_outputs
                    )
                    self.buttonReset = btnReset
                    return btnReset
                
                B_UI_Markdown._buildSeparator()

                if self.buildResetButton and self.buildRandomButton:
                    with gr.Row():
                        with gr.Column():
                            built.append(_buildRandomButton())
                        with gr.Column():
                            built.append(_buildResetButton())
                elif self.buildRandomButton:
                    built.append(_buildRandomButton())
                elif self.buildResetButton:
                    built.append(_buildResetButton())
        
        return built + built_items
    
    def getExtraUiCount(self) -> int:
        count = super().getExtraUiCount()

        if self.buttonRandom is not None:
            count += 1
        if self.buttonReset is not None:
            count += 1

        return count
    
    def validate(self, componentMap: dict) -> bool:
        valid: bool = super().validate(componentMap)

        for bUi in self.items:
            if not bUi.validate(componentMap):
                valid = False
                break
        
        return valid
    
    def finalizeUI(self, componentMap: dict):
        super().finalizeUI(componentMap)

        for bUi in self.items:
            bUi.finalizeUI(componentMap)
    
    def handlePrompt(self, p: StableDiffusionProcessing, componentMap: dict):
        super().handlePrompt(p, componentMap)

        for bUi in self.items:
            bUi.handlePrompt(p, componentMap)
    
    def getUpdate(self, *values) -> tuple[list, int]:
        updates: list = []
        consumed = 0

        for bUi in self.items:
            bUi_updates, bUi_consumed = bUi.getUpdate(*values[consumed:])
            updates += bUi_updates
            consumed += bUi_consumed
        
        return updates, consumed
    
    def getUpdateRandom(self, *currentValues) -> tuple[list, int]:
        updates: list = []
        consumed = 0

        for bUi in self.items:
            bUi_updates, bUi_consumed = bUi.getUpdateRandom(*currentValues[consumed:])
            updates += bUi_updates
            consumed += bUi_consumed
        
        return updates, consumed
    
    def handleItems(self, items: list[B_UI], buildCustomPromptInputs: bool, prefix: str) -> list[B_UI]:
        if buildCustomPromptInputs:
            items.append(B_UI_Component_Textbox(f"{prefix} - Positive prompt"))
            items.append(B_UI_Component_Textbox(f"{prefix} - Negative prompt", isNegativePrompt = True))
        
        return items
    
    def getBComponents(self) -> list[B_UI_Component]:
        bComponents: list[B_UI_Component] = []
        
        for item in self.items:
            item_type = type(item)
            
            if issubclass(item_type, B_UI_Component):
                item_bComponent: B_UI_Component = item
                bComponents.append(item_bComponent)
                continue
            
            if issubclass(item_type, B_UI_Container):
                item_bContainer: B_UI_Container = item
                bComponents += item_bContainer.bComponents
                continue
        
        return bComponents
    
    def resetComponentsValues(self) -> list[typing.Any]:
        updates, consumed = self.getUpdate()
        
        if len(updates) == 1:
            updates = updates[0]
        
        return updates
    
    def randomizeComponentsValues(self, *components) -> list[typing.Any]:
        updates, consumed = self.getUpdateRandom(*components)
        
        if len(updates) == 1:
            updates = updates[0]
        
        return updates

class B_UI_Preset(B_UI):
    @staticmethod
    def _fromArgs(name: str, mappings: dict[str, list[typing.Any]], **kwargs: str) -> B_UI:
        return B_UI_Preset(
            name = name
            , mappings = mappings
            , isAdditive = bool(int(kwargs.get("is_additive", 0)))
            , visible = not bool(int(kwargs.get("hide", 0)))
        )
    
    def __init__(self, name: str, mappings: dict[str, list[typing.Any]], isAdditive: bool = False, visible: bool = True):
        super().__init__(name, visible, True)

        self.mappings = mappings
        self.isAdditive = isAdditive
    
    def getDefaultName(self) -> str:
        return "Preset"
    
    def validate(self, componentMap: dict) -> bool:
        valid: bool = super().validate(componentMap)

        for k in self.mappings:
            if k not in componentMap:
                valid = False
                printWarning("Preset", self.name, "Key is not valid")
            else:
                bComponent: B_UI_Component = componentMap[k]
                if not bComponent.validateValue(self.mappings[k]):
                    valid = False
                    printWarning("Preset", f"{self.name}: {bComponent.name}", "Value is not valid")

        return valid
    
    def finalizeUI(self, componentMap: dict):
        super().finalizeUI(componentMap)

        bComponentMap: dict[str, B_UI_Component] = componentMap

        bComponents: list[B_UI_Component] = []
        if self.isAdditive:
            for bComponent in bComponentMap.values():
                if bComponent.name in self.mappings:
                    bComponents.append(bComponent)
        else:
            bComponents += list(bComponentMap.values())
        
        components_inputs: list[typing.Any] = []
        components_outputs: list[typing.Any] = []
        for bComponent in bComponents:
            components_inputs.append(bComponent.ui)
            components_outputs += [bComponent.ui] + bComponent.ui_extra_outputs
        
        def _applyPreset(*inputs):
            updates: list[typing.Any] = []

            i = 0
            for bComponent in bComponents:
                updates += self.getPresetValue(bComponent, inputs[i])[0]
                i += 1
            
            return updates
        
        self.ui.click(
            fn = _applyPreset
            , inputs = components_inputs
            , outputs = components_outputs
        )
    
    def buildSelf(self) -> typing.Any:
        return gr.Button(self.name, visible = self.visible)
    
    def getPresetValue(self, bComponent: B_UI_Component, componentValue) -> tuple[list, int]:
        """Returns update values and number of inputs consumed"""
        presetValue = componentValue

        if bComponent.name in self.mappings:
            presetValue = self.mappings[bComponent.name]

            if type(bComponent.ui) is not gr.Dropdown or not bComponent.ui.multiselect:
                presetValue = bComponent.defaultValue if len(presetValue) == 0 else presetValue[0]
        elif not self.isAdditive:
            presetValue = bComponent.defaultValue
        
        return bComponent.getUpdate(presetValue)

class B_Prompt(ABC):
    def __init__(self, name: str):
        self.name = name
    
    @abstractmethod
    def getPositive(self, componentMap: dict[str, B_UI_Component]) -> str:
        pass
    
    @abstractmethod
    def getNegative(self, componentMap: dict[str, B_UI_Component]) -> str:
        pass

class B_Prompt_Simple(B_Prompt):
    @staticmethod
    def _fromArgs(name: str, preset: B_UI_Preset, **kwargs: str):
        return B_Prompt_Simple(
            name = name
            , preset = preset
            , promptPositive = kwargs.get("p", "")
            , promptNegative = kwargs.get("n", "")
            , prefix = kwargs.get("prefix", "")
            , postfix = kwargs.get("postfix", "")
        )
    
    def __init__(self, name: str, promptPositive: str = "", promptNegative: str = "", prefix: str = "", postfix: str = "", preset: B_UI_Preset = None):
        super().__init__(name)

        self.positive = self.initPrompt(promptPositive, prefix, postfix)
        self.negative = self.initPrompt(promptNegative, prefix, postfix)
        self.preset = preset
    
    def initPrompt(self, prompt: str, prefix: str, postfix: str):
        if len(prompt) == 0:
            return prompt
        
        prefix = prefix.strip()
        postfix = postfix.strip()

        if len(prefix) > 0:
            prefix = f"{prefix} "
        if len(postfix) > 0:
            postfix = f" {postfix}"
        
        return f"{prefix}{prompt}{postfix}"
    
    def getPositive(self, componentMap: dict[str, B_UI_Component]) -> str:
        return self.positive
    
    def getNegative(self, componentMap: dict[str, B_UI_Component]) -> str:
        return self.negative

class B_Prompt_Link_Slider(B_Prompt):
    @staticmethod
    def _fromArgs(name: str, **kwargs: str):
        return B_Prompt_Link_Slider(
            name = name
            , linkedKey = kwargs["link_target"]
            , promptA = kwargs["a"]
            , promptB = kwargs["b"]
        )
    
    def __init__(self, name: str, linkedKey: str, promptA: str, promptB: str):
        super().__init__(name)

        self.linkedKey = linkedKey
        self.promptA = promptA
        self.promptB = promptB
        
        self.isNegativePrompt = False
    
    def buildPrompt(self, componentMap: dict[str, B_UI_Component]) -> str:
        component = componentMap.get(self.linkedKey)
        
        if component is None:
            print(f"B_Prompt_Link_Slider: Invalid key - '{self.linkedKey}'")
            return ""
        
        if type(component) is not B_UI_Component_Slider:
            print(f"B_Prompt_Link_Slider: Linked entry is not a slider - '{self.linkedKey}'")
            return ""
        
        return buildRangePrompt(self.promptA, self.promptB, component.value)
    
    def getPositive(self, componentMap: dict[str, B_UI_Component]) -> str:
        if self.isNegativePrompt:
            return ""
        
        return self.buildPrompt(componentMap)
    
    def getNegative(self, componentMap: dict[str, B_UI_Component]) -> str:
        if not self.isNegativePrompt:
            return ""
        
        return self.buildPrompt(componentMap)

class B_UI_Component_Textbox(B_UI_Component):
    @staticmethod
    def _fromArgs(name: str, *args, **kwargs: str) -> B_UI:
        return B_UI_Component_Textbox(
            name = name
            , defaultValue = kwargs.get("v", "")
            , isNegativePrompt = bool(int(kwargs.get("is_negative", 0)))
            , scale = int(kwargs.get("scale", 1))
            , visible = not bool(int(kwargs.get("hide", 0)))
        )
    
    def __init__(self, name: str = None, defaultValue: str = "", isNegativePrompt: bool = False, scale: int = None, visible: bool = True):
        super().__init__(name, defaultValue, visible)

        self.isNegativePrompt = isNegativePrompt
        self.scale = scale
    
    def getDefaultName(self) -> str:
        return "Textbox"
    
    def processValue(self, value: typing.Any) -> typing.Any:
        value_str: str = value
        return super().processValue(value_str.strip())
    
    def buildSelf(self) -> typing.Any:
        return gr.Textbox(
            label = self.name
            , value = self.value
            , scale = self.scale
            , visible = self.visible
        )

    def handlePrompt(self, p: StableDiffusionProcessing, componentMap: dict):
        super().handlePrompt(p, componentMap)

        if not self.isNegativePrompt:
            p.prompt = addPrompt(p.prompt, self.value)
        else:
            p.negative_prompt = addPrompt(p.negative_prompt, self.value)

class B_UI_Component_Dropdown(B_UI_Component):
    empty_choice: str = "-"
    advanced_defaultValue: float = 1
    advanced_step: float = 0.1
    random_maxChoices: int = 5

    @staticmethod
    def _fromArgs(name: str, *args, **kwargs: str) -> B_UI:
        choicesList: list[B_Prompt] = args[0]

        defaultValues = kwargs.get("v", "")
        if len(defaultValues) > 0:
            defaultValues = list(map(lambda v: v.strip(), defaultValues.split(",")))
        
        return B_UI_Component_Dropdown(
            name = name
            , choicesList = choicesList
            , defaultValues = defaultValues
            , multiselect = bool(int(kwargs.get("multi_select", 0)))
            , allowCustomValues = bool(int(kwargs.get("allow_custom", 1)))
            , sortChoices = bool(int(kwargs.get("sort", 1)))
            , advanced = not bool(int(kwargs.get("simple", 0)))
            , hideLabel = bool(int(kwargs.get("hide_label", 0)))
            , scale = int(kwargs.get("scale", 1))
            , visible = not bool(int(kwargs.get("hide", 0)))
        )
    
    @staticmethod
    def _buildColorChoicesList(postfixPrompt: str = "") -> list[B_Prompt_Simple]:
        return list(map(
            lambda text: B_Prompt_Simple(text, text.lower(), postfix = postfixPrompt)
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
        , choicesList: list[B_Prompt] = []
        , defaultValues: str | list[str] = None
        , multiselect: bool = False
        , allowCustomValues: bool = True
        , sortChoices: bool = True
        , advanced: bool = True
        , hideLabel: bool = False
        , scale: int = None
        , visible: bool = True
    ):
        choicesMap = self.buildChoicesMap(choicesList, not multiselect, sortChoices)
        defaultValuesFinal = self.buildDefaultValue(choicesMap, defaultValues, multiselect)
        
        super().__init__(name, defaultValuesFinal, visible)
        
        self.choicesMap = choicesMap
        self.multiselect = multiselect
        self.allowCustomValues = allowCustomValues if not multiselect else False
        self.advanced = advanced
        self.hideLabel = hideLabel
        self.scale = scale

        self.advanced_container: typing.Any = None
        self.advanced_options: dict[str, tuple[typing.Any, typing.Any]] = {} # key = choice, value = (number, column)
        self.advanced_values: dict[str, float] = {} # key = choice, value = number value
    
    def getDefaultName(self) -> str:
        return "Dropdown"
    
    def buildUI(self) -> list[typing.Any]:
        if not self.advanced:
            return super().buildUI()
        
        built: list[typing.Any] = []

        self.advanced_options = {}
        self.advanced_values = {}

        choices = self.getChoices(True)
        
        with gr.Column(
            scale = self.scale
            , min_width = 160
        ):
            built += super().buildUI()

            self.advanced_container = gr.Row(
                variant = "panel"
                , visible = self.value is not None and len(self.value) > 0 and self.value != self.empty_choice
            )
            self.ui_extra_outputs.append(self.advanced_container)
            with self.advanced_container:
                for k in choices:
                    column = gr.Column(
                        variant = "panel"
                        , visible = self.value is not None and (self.value == k or k in self.value)
                        , min_width = 160
                    )
                    with column:
                        number = gr.Number(
                            label = f"{k} (S)"
                            , value = self.advanced_defaultValue
                            , step = self.advanced_step
                            , minimum = 0
                        )

                        built.append(number)

                        self.ui_extra_inputs.append(number)
                        
                        self.ui_extra_outputs.append(number)
                        self.ui_extra_outputs.append(column)
                    
                    self.advanced_options[k] = number, column
                    self.advanced_values[k] = self.advanced_defaultValue
        
        def _update(choice: str | list[str], *numbers: float):
            choices_selected: list[str] = []
            
            if choice != self.empty_choice:
                if type(choice) is not list:
                    choices_selected.append(choice)
                else:
                    choices_selected += choice
            
            output: list = [self.advanced_container.update(visible = len(choices_selected) > 0)]
            i: int = 0
            for k in self.advanced_values:
                visible: bool = k in choices_selected
                value: float = numbers[i] if visible else self.advanced_defaultValue
                
                self.advanced_values[k] = value
                
                output.append(self.advanced_options[k][0].update(value = value, step = self.advanced_step))
                output.append(self.advanced_options[k][1].update(visible = visible))
                i += 1
            
            return output
        
        self.ui.input(
            fn = _update
            , inputs = [self.ui, *map(lambda t: t[0], self.advanced_options.values())]
            , outputs = [self.advanced_container] + sum(list(map(lambda t: list(t), self.advanced_options.values())), [])
        )

        return built
    
    def getExtraUiCount(self) -> int:
        count = super().getExtraUiCount()

        if self.advanced:
            count += len(self.advanced_values)
        
        return count
    
    def setValue(self, *values) -> int:
        if self.advanced:
            i = 1
            for k in self.advanced_values:
                self.advanced_values[k] = values[i]
                i += 1
        
        return super().setValue(*values)
    
    def buildSelf(self) -> typing.Any:
        return gr.Dropdown(
            label = self.name
            , choices = self.getChoices()
            , multiselect = self.multiselect
            , value = self.value
            , allow_custom_value = self.allowCustomValues
            , show_label = not self.hideLabel
            , scale = self.scale if not self.advanced else None
            , visible = self.visible
        )
    
    def validate(self, componentMap: dict) -> bool:
        valid = super().validate(componentMap)

        for choiceKey in self.getChoices(True):
            bPrompt = self.choicesMap[choiceKey]
            if issubclass(type(bPrompt), B_Prompt_Simple):
                bPrompt_simple: B_Prompt_Simple = bPrompt
                if bPrompt_simple.preset is not None and not bPrompt_simple.preset.validate(componentMap):
                    valid = False
        
        return valid
    
    def validateValue(self, value: typing.Any) -> bool:
        valid: bool = True

        if type(value) is not list:
            value: list[str] = [value]
        
        for v in value:
            if v not in self.choicesMap:
                valid = False
                printWarning("Dropdown", self.name, f"Invalid choice: '{v}'")

        return valid
    
    def finalizeUI(self, componentMap: dict):
        super().finalizeUI(componentMap)

        bComponentMap: dict[str, B_UI_Component] = componentMap
        
        bComponents: list[B_UI_Component] = []
        for bPrompt in self.choicesMap.values():
            if type(bPrompt) is B_Prompt_Simple and bPrompt.preset is not None:
                for k in bPrompt.preset.mappings:
                    bComponent = bComponentMap[k]
                    if bComponent not in bComponents:
                        bComponents.append(bComponent)
        
        if len(bComponents) > 0:
            components_inputs: list[typing.Any] = []
            components_outputs: list[typing.Any] = []
            for bComponent in bComponents:
                components_inputs.append(bComponent.ui)
                components_outputs += [bComponent.ui] + bComponent.ui_extra_outputs
            
            def _getPresetValues(choices: str | list[str], *inputs) -> list[typing.Any]:
                if type(choices) is not list:
                    choices = [choices]
                
                updatesMap: dict[str, list] = {}

                i = 0
                for bComponent in bComponents:
                    currentValue = inputs[i]
                    
                    for choice in choices:
                        bPrompt = self.choicesMap[choice]
                        if type(bPrompt) is B_Prompt_Simple and bPrompt.preset is not None:
                            updatesMap[bComponent.name] = bPrompt.preset.getPresetValue(bComponent, currentValue)[0]
                    
                    if bComponent.name not in updatesMap:
                        updatesMap[bComponent.name] = bComponent.getUpdate(currentValue)[0]
                    
                    i += 1
                
                updates: list[typing.Any] = []
                for bComponentUpdates in updatesMap.values():
                    updates += bComponentUpdates
                return updates
            
            self.ui.select(
                fn = _getPresetValues
                , inputs = [self.ui] + components_inputs
                , outputs = components_outputs
            )
    
    def buildChoicesMap(self, choicesList: list[B_Prompt], insertEmptyChoice: bool, sortChoices: bool) -> dict[str, B_Prompt]:
        choicesMap: dict[str, B_Prompt] = {}
        
        if insertEmptyChoice:
            choicesMap[self.empty_choice] = B_Prompt_Simple(self.empty_choice)
        
        if sortChoices:
            choicesList = sorted(choicesList, key = lambda bPrompt: bPrompt.name)
        
        for bPrompt in choicesList:
            choicesMap[bPrompt.name] = bPrompt
        
        return choicesMap
    
    def buildDefaultValue(self, choicesMap: dict[str, B_Prompt], defaultValues: str | list[str], multiselect: bool) -> str | list[str]:
        defaultValue: str | list[str] = None if not multiselect else []
        
        if (type(defaultValues) is str or defaultValues is None):
            defaultValues = [defaultValues]
        
        if len(choicesMap) > 0:
            choices = list(choicesMap)
            if multiselect:
                if defaultValues[0] is not None:
                    for v in defaultValues:
                        if v in choices:
                            defaultValue.append(v)
            else:
                defaultValueFirst = defaultValues[0]
                if defaultValueFirst is not None and defaultValueFirst in choices:
                    defaultValue = defaultValueFirst
                else:
                    defaultValue = choices[0]
        
        return defaultValue
    
    def getChoices(self, excludeEmpty: bool = False) -> list[str]:
        choices = list(self.choicesMap)
        
        if excludeEmpty and len(choices) > 0 and choices[0] == self.empty_choice:
            choices.pop(0)
        
        return choices
    
    def getBPromptsFromValue(self) -> list[B_Prompt]:
        bPrompts: list[B_Prompt] = []
        
        if type(self.value) is str:
            bPrompts.append(self.choicesMap.get(self.value, None if not self.allowCustomValues else B_Prompt_Simple(self.value, self.value)))
        elif type(self.value) is list and len(self.value) > 0:
            for k in self.value:
                bPrompts.append(self.choicesMap[k])
        
        return bPrompts
    
    def handlePrompt(self, p: StableDiffusionProcessing, componentMap: dict):
        super().handlePrompt(p, componentMap)

        bPrompts = self.getBPromptsFromValue()
        if len(bPrompts) > 0:
            for bPrompt in bPrompts:
                positive = bPrompt.getPositive(componentMap)
                negative = bPrompt.getNegative(componentMap)
                
                if self.advanced and bPrompt.name != self.empty_choice:
                    value = self.advanced_values[bPrompt.name]
                    
                    if value <= 0:
                        continue

                    if value != 1:
                        if len(positive) > 0:
                            positive = f"({positive}:{value})"
                        if len(negative) > 0:
                            negative = f"({negative}:{value})"

                if bPrompt is not None:
                    p.prompt = addPrompt(p.prompt, positive)
                    p.negative_prompt = addPrompt(p.negative_prompt, negative)
    
    def getUpdate(self, *values) -> tuple[list, int]:
        updates, consumed = super().getUpdate(*values)
        
        if self.advanced:
            selfValue: str | list[str] = updates[0]
            if type(selfValue) is not list:
                selfValue = [selfValue]
            
            choices: list[str] = []
            for v in selfValue:
                choices.append(self.choicesMap[v].name)
            
            updates.append(self.advanced_container.update(visible = len(choices) > 0 and choices[0] != self.empty_choice))
            for k in self.advanced_options:
                number, column = self.advanced_options[k]
                updates.append(values[consumed] if len(values) > consumed else self.advanced_defaultValue)
                updates.append(column.update(visible = k in choices))
                consumed += 1
        
        return updates, consumed
    
    def getUpdateRandom(self, *currentValues) -> tuple[list, int]:
        newValues: list = [currentValues[0]]
        consumed: int = 1

        choiceKeys = list(self.choicesMap.keys())
        
        if len(choiceKeys) > 0:
            if self.multiselect:
                value: list[str] = []

                cMax = len(choiceKeys)
                if self.random_maxChoices > 0 and self.random_maxChoices < cMax:
                    cMax = self.random_maxChoices
                
                r = random.randint(0, cMax)
                if r > 0:
                    for c in range(r):
                        i = random.randint(0, len(choiceKeys) - 1)
                        value.append(choiceKeys.pop(i))
                
                newValues[0] = value
            else:
                r = random.randint(0, len(choiceKeys) - 1)
                newValues[0] = choiceKeys[r]
        
        if self.advanced:
            selfValue: str | list[str] = newValues[0]
            if type(selfValue) is not list:
                selfValue = [selfValue]
            
            choices: list[str] = []
            for v in selfValue:
                choices.append(self.choicesMap[v].name)
            
            newValues.append(self.advanced_container.update(visible = len(choices) > 0 and choices[0] != self.empty_choice))
            for k in self.advanced_options:
                number, column = self.advanced_options[k]
                newValues.append(self.advanced_defaultValue)
                newValues.append(column.update(visible = k in choices))
                consumed += 1
        
        return newValues, consumed

class B_UI_Component_Slider(B_UI_Component):
    @staticmethod
    def _fromArgs(name: str, *args, **kwargs: str) -> B_UI:
        return B_UI_Component_Slider(
            name = name
            , promptA = kwargs.get("a", "")
            , promptB = kwargs.get("b", "")
            , defaultValue = float(kwargs.get("v", -1))
            , isRequired = bool(int(kwargs.get("is_required", 0)))
            , visible = not bool(int(kwargs.get("hide", 0)))
            , buildButtons = bool(int(kwargs.get("build_buttons", 1)))
            , promptAButton = kwargs.get("a_button", "")
            , promptBButton = kwargs.get("b_button", "")
        )
    
    def __init__(self, name: str = None, promptA: str = "", promptB: str = "", defaultValue: float = None, isRequired: bool = False, visible: bool = True, buildButtons: bool = True, promptAButton: str = "", promptBButton: str = ""):
        super().__init__(name, defaultValue, visible)
        
        self.promptA = promptA.strip()
        self.promptB = promptB.strip()
        self.isRequired = isRequired
        self.buildButtons = buildButtons
        self.promptAButton = promptAButton
        self.promptBButton = promptBButton
        
        self.isNegativePrompt = False

        self.buttonA: typing.Any = None
        self.buttonB: typing.Any = None
    
    def getDefaultName(self) -> str:
        return "Slider"
    
    def buildUI(self) -> list[typing.Any]:
        built = super().buildUI()

        if self.buildButtons:
            with gr.Row():
                self.buttonA = self.buildButton(self.promptAButton, 0)
                self.buttonB = self.buildButton(self.promptBButton, self.getMaximum())

                built.append(self.buttonA)
                built.append(self.buttonB)

        return built
    
    def processValue(self, value: typing.Any) -> typing.Any:
        value_float = float(value)
        value_float = round(value_float / self.getMaximum(), 2) if value_float > -1 else value_float
        return super().processValue(value_float)
    
    def getExtraUiCount(self) -> int:
        count = super().getExtraUiCount()

        if self.buttonA is not None:
            count += 1
        if self.buttonB is not None:
            count += 1
        
        return count
    
    def getUpdateRandom(self, *currentValues) -> tuple[list, int]:
        return [float(random.randint(self.getMinimum(), self.getMaximum()))], 1

    def buildSelf(self) -> typing.Any:
        return gr.Slider(
            label = self.name
            , minimum = self.getMinimum()
            , maximum = self.getMaximum()
            , value = self.value
            , step = self.getStep()
            , visible = self.visible
        )
    
    def validateValue(self, value: typing.Any) -> bool:
        valuePrintStr: str = "Value" if value != self.defaultValue else "Default value"

        value_final: float | None = None
        if type(value) is list and len(value) > 0:
            value_final = float(value[0])
        
        if value_final is not None and value_final < self.getMinimum():
            printWarning("Slider", self.name, f"{valuePrintStr} exceeds minimum")
        elif value_final is not None and value_final > self.getMaximum():
            printWarning("Slider", self.name, f"{valuePrintStr} exceeds maximum")
        else:
            return True
        
        return False
    
    def getMinimum(self) -> float:
        return -1 if not self.isRequired else 0
    
    def getMaximum(self) -> float:
        return 100
    
    def getStep(self) -> float:
        return 1
    
    def buildButton(self, text: str, value: float) -> typing.Any:
        btn = gr.Button(value = text)
        btn.click(
            fn = lambda btnText: self.getUpdate(value)[0][0]
            , inputs = btn
            , outputs = self.ui
        )
        return btn

    def handlePrompt(self, p: StableDiffusionProcessing, componentMap: dict):
        super().handlePrompt(p, componentMap)
        
        promptToAdd = buildRangePrompt(self.promptA, self.promptB, self.value)
        
        if not self.isNegativePrompt:
            p.prompt = addPrompt(p.prompt, promptToAdd)
        else:
            p.negative_prompt = addPrompt(p.prompt, promptToAdd)

class B_UI_Container_Tab(B_UI_Container):
    @staticmethod
    def _fromArgs(name: str, items: list[B_UI], **kwargs: str) -> B_UI:
        return B_UI_Container_Tab(
            name = name
            , items = items
            , visible = not bool(int(kwargs.get("hide", 0)))
            , buildResetButton = bool(int(kwargs.get("build_reset_button", 1)))
            , buildRandomButton = bool(int(kwargs.get("build_random_button", 0)))
        )
    
    def __init__(self, name: str = None, items: list[B_UI] = [], visible: bool = True, buildResetButton: bool = True, buildRandomButton: bool = False):
        super().__init__(name, items, visible, False, buildResetButton, buildRandomButton)
    
    def getDefaultName(self) -> str:
        return "Tab"
    
    def buildSelf(self) -> typing.Any:
        return gr.Tab(self.name)

class B_UI_Container_Row(B_UI_Container):
    @staticmethod
    def _fromArgs(name: str, items: list[B_UI], **kwargs: str) -> B_UI:
        return B_UI_Container_Row(
            items = items
            , visible = not bool(int(kwargs.get("hide", 0)))
            , buildResetButton = bool(int(kwargs.get("build_reset_button", 0)))
            , name = name
        )
    
    def __init__(self, items: list[B_UI] = [], visible: bool = True, buildResetButton: bool = False, name: str = None):
        super().__init__(name, items, visible, False, buildResetButton)
    
    def getDefaultName(self) -> str:
        return "Row"
    
    def buildSelf(self) -> typing.Any:
        return gr.Row(visible = self.visible)

class B_UI_Container_Column(B_UI_Container):
    @staticmethod
    def _fromArgs(name: str, items: list[B_UI], **kwargs: str) -> B_UI:
        return B_UI_Container_Column(
            items = items
            , scale = int(kwargs.get("scale", 1))
            , visible = not bool(int(kwargs.get("hide", 0)))
            , buildResetButton = bool(int(kwargs.get("build_reset_button", 0)))
            , name = name
        )
    
    def __init__(self, items: list[B_UI] = [], scale: int = 1, visible: bool = True, buildResetButton: bool = False, name: str = None):
        super().__init__(name, items, visible, False, buildResetButton)
        
        self.scale = scale
    
    def getDefaultName(self) -> str:
        return "Column"
    
    def buildSelf(self) -> typing.Any:
        return gr.Column(scale = self.scale, visible = self.visible)

class B_UI_Container_Group(B_UI_Container):
    @staticmethod
    def _fromArgs(name: str, items: list[B_UI], **kwargs: str) -> B_UI:
        return B_UI_Container_Group(
            items = items
            , visible = not bool(int(kwargs.get("hide", 0)))
            , buildResetButton = bool(int(kwargs.get("build_reset_button", 0)))
            , buildRandomButton = bool(int(kwargs.get("build_random_button", 0)))
            , name = name
        )
    
    def __init__(self, items: list[B_UI] = [], visible: bool = True, buildResetButton: bool = False, buildRandomButton: bool = False, name: str = None):
        super().__init__(name, items, visible, False, buildResetButton, buildRandomButton)
    
    def getDefaultName(self) -> str:
        return "Group"
    
    def buildSelf(self) -> typing.Any:
        return gr.Group(visible = self.visible)

class B_UI_Container_Accordion(B_UI_Container):
    @staticmethod
    def _fromArgs(name: str, items: list[B_UI], **kwargs: str) -> B_UI:
        return B_UI_Container_Accordion(
            items = items
            , visible = not bool(int(kwargs.get("hide", 0)))
            , buildResetButton = bool(int(kwargs.get("build_reset_button", 0)))
            , name = name
        )
    
    def __init__(self, name: str = None, items: list[B_UI] = [], visible: bool = True, buildResetButton: bool = False):
        super().__init__(name, items, visible, False, buildResetButton)
    
    def getDefaultName(self) -> str:
        return "Accordion"
    
    def buildSelf(self) -> typing.Any:
        return gr.Accordion(label = self.name, visible = self.visible)

class B_UI_Builder(ABC):
    def __init__(self, name: str, **kwargs: str):
        self.name = name
        self.args = kwargs

        self.parent: B_UI_Builder = None
        self.builtChildren: list[B_UI] = []
    
    @abstractmethod
    def build(self) -> B_UI:
        pass

class B_UI_Builder_WithParent(B_UI_Builder):
    def __init__(self, name: str, parent: B_UI_Builder, **kwargs: str):
        super().__init__(name, **kwargs)
        
        self.parent = parent
    
    def build(self) -> B_UI:
        built = self.buildExtended()

        if self.parent is not None:
            self.parent.builtChildren.append(built)
        
        return built

    @abstractmethod
    def buildExtended(self) -> B_UI:
        pass

class B_UI_Preset_Builder(B_UI_Builder):
    def __init__(self, name: str, **kwargs: str):
        super().__init__(name, **kwargs)
        
        self.mappings: dict[str, list[str]] = {}
    
    def addMapping(self, name: str, **kwargs: str):
        value = kwargs.get("v", "")
        
        if len(value) > 0:
            value = list(map(lambda v: v.strip(), value.split(",")))
        else:
            value = list[str]([])
        
        self.mappings[name] = value
    
    def build(self) -> B_UI:
        return None if len(self.mappings) == 0 else B_UI_Preset._fromArgs(self.name, self.mappings, **self.args)

class B_UI_Container_Builder(B_UI_Builder_WithParent):
    def __init__(self, t: type[B_UI_Container], name: str, parent: B_UI_Builder, **kwargs: str):
        super().__init__(name, parent, **kwargs)
        
        self.t = t
    
    def buildExtended(self) -> B_UI:
        return self.t._fromArgs(self.name, self.builtChildren, **self.args)

class B_UI_Markdown_Builder(B_UI_Builder_WithParent):
    def __init__(self, isSeparator: bool, parent: B_UI_Container_Builder, **kwargs: str):
        super().__init__(None, parent, **kwargs)

        self.isSeparator = isSeparator
    
    def buildExtended(self) -> B_UI:
        return B_UI_Markdown._fromArgs(self.isSeparator, **self.args)

class B_UI_Component_Builder(B_UI_Builder_WithParent):
    def __init__(self, t: type[B_UI_Component], name: str, parent: B_UI_Container_Builder, **kwargs: str):
        super().__init__(name, parent, **kwargs)
        
        self.t = t
    
    def buildExtended(self) -> B_UI:
        return self.t._fromArgs(self.name, **self.args)

class B_UI_Component_Dropdown_Builder(B_UI_Builder_WithParent):
    def __init__(self, name: str, parent: B_UI_Container_Builder, **kwargs: str):
        super().__init__(name, parent, **kwargs)
        
        self.choicesList: list[B_Prompt] = []
    
    def addChoice(self, text: str, preset_builder: B_UI_Preset_Builder, **bPromptKwargs: str):
        bPrompt: B_Prompt = None
        
        link_type = bPromptKwargs.get("link_type", "")
        match link_type:
            case "SLIDER":
                bPrompt = B_Prompt_Link_Slider._fromArgs(text, **bPromptKwargs)
            case _:
                preset: B_UI_Preset = preset_builder.build()
                bPrompt = B_Prompt_Simple._fromArgs(text, preset, **bPromptKwargs)
        
        if any(map(lambda bPrompt_existing: bPrompt_existing.name == bPrompt.name, self.choicesList)):
            print(f"WARNING: Duplicate CHOICE in {self.name} -> {text}")
        
        self.choicesList.append(bPrompt)
    
    def addChoices(self, **choicesKwargs: str):
        choicesList: list[B_Prompt] = []
        
        special_type = choicesKwargs["type"]
        match special_type:
            case "COLOR":
                choicesList = B_UI_Component_Dropdown._buildColorChoicesList(choicesKwargs["postfix"])
            case _:
                print(f"Invalid CHOICES type in {self.name} -> {special_type}")
        
        self.choicesList += choicesList
    
    def buildExtended(self) -> B_UI:
        return B_UI_Component_Dropdown._fromArgs(self.name, self.choicesList, **self.args)

class B_UI_Map():
    def __init__(self, path_base: str, file_name_layout: str, file_name_presets: str, tagged_show: bool = True, validate: bool = True):
        self.layout = self.parseLayout(os.path.join(path_base, file_name_layout), tagged_show)
        self.presets = self.parsePresets(os.path.join(path_base, file_name_presets))
        
        self.componentMap = self.buildComponentMap(self.layout, {})

        if validate:
            self.validate()
    
    def readLine(self, l: str) -> tuple[str, str, dict[str, str]]:
        # TODO: Fix empty str l_name
        l = l.strip()
        
        l_type = l
        l_name = None
        l_args = {}
        
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
    
    def parseLayout(self, file_path_layout: str, tagged_show: bool) -> list[B_UI]:
        layout: list[B_UI] = []
        
        builder_current_container: B_UI_Container_Builder = None
        builder_current_dropdown: B_UI_Component_Dropdown_Builder = None
        dropdown_current_choice: tuple[str, B_UI_Preset_Builder, dict[str, str]] = None # (choice key, choice preset, choice args)
        
        skip = 0
        
        def _build(builder: B_UI_Builder):
            built = builder.build()
            if builder.parent is None:
                layout.append(built)
            return built
        
        def _buildDropdownChoice(dropdown_choice: tuple[str, B_UI_Preset_Builder, dict[str, str]]) -> bool:
            """Returns True if choice is not None and had preset mappings"""
            if dropdown_choice is not None:
                l_choice_name, l_choice_preset_builder, l_choice_args = dropdown_choice

                l_choice_args["prefix"] = builder_current_dropdown.args.get("prefix", "")
                l_choice_args["postfix"] = builder_current_dropdown.args.get("postfix", "")

                builder_current_dropdown.addChoice(l_choice_name, l_choice_preset_builder, **l_choice_args)
                return len(l_choice_preset_builder.mappings) > 0
            
            return False
        
        with open(file_path_layout) as file_layout:
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
                        if dropdown_current_choice is not None:
                            had_mappings = _buildDropdownChoice(dropdown_current_choice)
                            dropdown_current_choice = None
                            if had_mappings:
                                continue

                        if builder_current_dropdown is not None:
                            built = _build(builder_current_dropdown)
                            builder_current_dropdown = None
                            continue
                        
                        if builder_current_container is not None:
                            built = _build(builder_current_container)
                            builder_current_container = builder_current_container.parent
                            continue
                    
                    skip -= 1
                    
                    continue
                
                if l_args.get("x", "") == "1":
                    if not tagged_show:
                        skip += 1
                        continue
                
                match l_type:
                    case "TEXTBOX":
                        built = _build(B_UI_Component_Builder(B_UI_Component_Textbox, l_name, builder_current_container, **l_args))
                    
                    case "SLIDER":
                        built = _build(B_UI_Component_Builder(B_UI_Component_Slider, l_name, builder_current_container, **l_args))
                    
                    case "DROPDOWN":
                        if skip > 0:
                            skip += 1
                            continue
                        
                        builder_current_dropdown = B_UI_Component_Dropdown_Builder(l_name, builder_current_container, **l_args)
                    
                    case "CHOICE":
                        if skip > 0:
                            continue
                        
                        _buildDropdownChoice(dropdown_current_choice)
                        
                        dropdown_current_choice = (l_name, B_UI_Preset_Builder(f"{builder_current_dropdown.name}_{l_name}_PRESET", **{ "is_additive": "1", "hide": "1" }), l_args)
                    
                    case "SET":
                        dropdown_current_choice[1].addMapping(l_name, **l_args)
                    
                    case "CHOICES":
                        if skip > 0:
                            continue
                        
                        builder_current_dropdown.addChoices(**l_args)
                    
                    case "GROUP":
                        if skip > 0:
                            skip += 1
                            continue
                        
                        builder_current_container = B_UI_Container_Builder(B_UI_Container_Group, l_name, builder_current_container, **l_args)
                    
                    case "TAB":
                        if skip > 0:
                            skip += 1
                            continue
                        
                        builder_current_container = B_UI_Container_Builder(B_UI_Container_Tab, l_name, builder_current_container, **l_args)
                    
                    case "ROW":
                        if skip > 0:
                            skip += 1
                            continue
                        
                        builder_current_container = B_UI_Container_Builder(B_UI_Container_Row, l_name, builder_current_container, **l_args)
                    
                    case "COLUMN":
                        if skip > 0:
                            skip += 1
                            continue
                        
                        builder_current_container = B_UI_Container_Builder(B_UI_Container_Column, l_name, builder_current_container, **l_args)
                    
                    case "ACCORDION":
                        if skip > 0:
                            skip += 1
                            continue
                        
                        builder_current_container = B_UI_Container_Builder(B_UI_Container_Accordion, l_name, builder_current_container, **l_args)
                    
                    case "SEPARATOR":
                        if skip > 0:
                            continue
                        
                        built = _build(B_UI_Markdown_Builder(True, builder_current_container, **l_args))

                    case _:
                        print(f"Invalid layout type: {l_type}")
        
        return layout
    
    def parsePresets(self, file_path_presets: str) -> dict[str, B_UI_Preset]:
        presets: dict[str, B_UI_Preset] = {}
        
        builder_current: B_UI_Preset_Builder = None
        
        with open(file_path_presets) as file_presets:
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
                    presets[builder_current.name] = builder_current.build()
                    builder_current = None
                    continue
                
                match l_type:
                    case "PRESET":
                        builder_current = B_UI_Preset_Builder(l_name, **l_args)
                    
                    case "SET":
                        builder_current.addMapping(l_name, **l_args)
                    
                    case _:
                        print(f"Invalid preset type: {l_type}")
        
        return presets
    
    def buildComponentMap(self, layout: list[B_UI], target: dict[str, B_UI_Component]) -> dict[str, B_UI_Component]:
        for bUi in layout:
            bUi_type = type(bUi)

            if issubclass(bUi_type, B_UI_Component):
                bComponent: B_UI_Component = bUi

                if bComponent.name in target:
                    print(f"WARNING: Duplicate component -> {bUi.name}")
                target[bComponent.name] = bComponent

            elif issubclass(bUi_type, B_UI_Container):
                bContainer: B_UI_Container = bUi
                target = self.buildComponentMap(bContainer.items, target)
        
        return target
    
    def validate(self):
        valid: bool = True

        for bUi in self.layout:
            if not bUi.validate(self.componentMap):
                valid = False
        
        for presetKey in self.presets:
            if not self.presets[presetKey].validate(self.componentMap):
                valid = False
        
        if not valid:
            print("WARNING: Invalid layout or presets")
    
    def buildUI(self) -> list[typing.Any]:
        grComponents: list[typing.Any] = []

        # PRESETS
        B_UI_Markdown._buildSeparator()

        with gr.Accordion("Presets", open = False):
            i = 0
            for preset in self.presets.values():
                grComponents += preset.buildUI()

                i += 1
                if i < len(self.presets) and preset.visible:
                    B_UI_Markdown._buildSeparator()

        # LAYOUT
        B_UI_Markdown._buildSeparator()
        
        for bUi in self.layout:
            grComponents += bUi.buildUI()
        
        # SETTINGS
        B_UI_Markdown._buildSeparator()

        with gr.Accordion("Settings", open = False):
            btnClearConfig = gr.Button("Clear config")
            btnClearConfig.click(fn = self.clearConfigFile)
            
            grComponents.append(btnClearConfig)
        
        return grComponents
    
    def finalizeUI(self):
        for bUi in self.layout:
            bUi.finalizeUI(self.componentMap)
        
        for preset in self.presets.values():
            preset.finalizeUI(self.componentMap)
    
    def clearConfigFile(self):
        path = os.path.join(scripts.basedir(), "ui-config.json")
        with open(path, "r+", encoding = "utf-8") as file_config:
            config: dict[str, typing.Any] = json.load(file_config)
            
            config_keys = filter(lambda k: k.find("b_prompt_builder") == -1, config.keys())

            config_new: dict[str, typing.Any] = {}
            for k in config_keys:
                config_new[k] = config[k]
            
            file_config.seek(0)
            json.dump(config_new, file_config, indent = 4)
            file_config.truncate()

b_layout = B_UI_Map(
    path_base = os.path.join(scripts.basedir(), "scripts", "b_prompt_builder")
    , file_name_layout = "layout.txt"
    , file_name_presets = "presets.txt"
    , tagged_show = True
    , validate = True
)

class Script(scripts.Script):
    def title(self):
        return "B Prompt Builder"

    def show(self, is_img2img):
        return True

    def ui(self, is_img2img):
        built = b_layout.buildUI()
        b_layout.finalizeUI()
        return built

    def run(self, p, *args):
        i = 0
        
        for preset in b_layout.presets.values():
            i += preset.setValue(*args[i:])
        
        for bUi in b_layout.layout:
            i += bUi.setValue(*args[i:])
            bUi.handlePrompt(p, b_layout.componentMap)
        
        proc = process_images(p)
        
        return proc
