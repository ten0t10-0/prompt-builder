import modules.scripts as scripts
import gradio as gr
import os
import typing
import random

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
    
    def __init__(self, name: str, visible: bool):
        self.identifier = self.getNextIdentifier()
        self.name = self.handleName(name)
        self.visible = visible

        self.ui: typing.Any = None
    
    def getNextIdentifier(self) -> int:
        B_UI._identifier += 1
        return B_UI._identifier
    
    def handleName(self, name: str = None) -> str:
        if name is None or len(name) == 0:
            name = f"{self.identifier}_{self.getDefaultName()}"
        
        return name
    
    @abstractmethod
    def getDefaultName(self) -> str:
        pass

    @abstractmethod
    def validate(self, componentMap: dict) -> bool:
        pass
    
    @abstractmethod
    def buildUI(self) -> list[typing.Any]:
        """Returns a list of Gradio components"""
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
        super().__init__(None, visible)

        if isSeparator:
            value = self.html_separator
        
        self.value: str = value
    
    def getDefaultName(self) -> str:
        return "Markdown"
    
    def validate(self, componentMap: dict) -> bool:
        return True
    
    def buildUI(self) -> list[typing.Any]:
        markdown = gr.Markdown(
            value = self.value
            , visible = self.visible
        )
        
        self.ui = markdown

        return []

class B_UI_Component(B_UI, ABC):
    @staticmethod
    @abstractmethod
    def _fromArgs(name: str, *args, **kwargs: str):
        pass
    
    def __init__(self, name: str = None, defaultValue = None, visible: bool = True):
        super().__init__(name, visible)

        self.value = defaultValue
        self.defaultValue = defaultValue
    
    def handleUpdateValue(self, value):
        return value if value is not None else self.defaultValue
    
    def buildUI(self) -> list[typing.Any]:
        self.ui = self.buildComponent()
        return [self.ui]
    
    @abstractmethod
    def getDefaultName(self) -> str:
        pass

    @abstractmethod
    def validate(self, componentMap: dict) -> bool:
        pass

    @abstractmethod
    def validateValue(self, value: typing.Any) -> bool:
        pass
    
    @abstractmethod
    def buildComponent(self) -> typing.Any:
        pass
    
    @abstractmethod
    def setValue(self, value):
        pass
    
    @abstractmethod
    def handlePrompt(self, p: StableDiffusionProcessing, componentMap: dict):
        pass
    
    @abstractmethod
    def getUpdate(self, value = None) -> typing.Any:
        pass

    @abstractmethod
    def getUpdateRandom(self, currentValue) -> typing.Any:
        pass

    @abstractmethod
    def finalizeComponent(self, componentMap: dict):
        pass

class B_UI_Container(B_UI):
    @staticmethod
    @abstractmethod
    def _fromArgs(name: str, items: list[B_UI], **kwargs: str):
        pass

    def __init__(self, name: str = None, items: list[B_UI] = [], visible: bool = True, buildCustomPromptInputs: bool = False, buildResetButton: bool = False, buildRandomButton: bool = False):
        super().__init__(name, visible)
        
        self.items = self.handleItems(items, buildCustomPromptInputs, name)
        self.buildResetButton = buildResetButton
        self.buildRandomButton = buildRandomButton

        self.bComponents = self.getBComponents()
    
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
        updates = []
        
        bComponents = self.getBComponents()
        for x in bComponents:
            updates.append(x.getUpdate())
        
        if len(updates) == 1:
            updates = updates[0]
        
        return updates
    
    def randomizeComponentsValues(self, *components) -> list[typing.Any]:
        updates = []
        
        bComponents = self.getBComponents()
        i = 0
        for x in bComponents:
            updates.append(x.getUpdateRandom(components[i]))
            i += 1
        
        if len(updates) == 1:
            updates = updates[0]
        
        return updates
    
    def validate(self, componentMap: dict) -> bool:
        for bUi in self.items:
            if not bUi.validate(componentMap):
                return False
        
        return True
    
    def buildUI(self) -> list[typing.Any]:
        components: list[typing.Any] = []
        
        self.ui = self.buildContainer()
        with self.ui:
            for item in self.items:
                item_ui = item.buildUI()
                
                if len(item_ui) > 0:
                    components += item_ui
            
            if self.buildResetButton or self.buildRandomButton:
                def _buildResetButton() -> typing.Any:
                    btnReset = gr.Button(value = f"Reset {self.name}")
                    btnReset.click(
                        fn = self.resetComponentsValues
                        , outputs = components
                    )
                    return btnReset
                
                def _buildRandomButton() -> typing.Any:
                    btnRandom = gr.Button(value = f"Randomize {self.name}")
                    btnRandom.click(
                        fn = self.randomizeComponentsValues
                        , inputs = components
                        , outputs = components
                    )
                    return btnRandom
                
                B_UI_Markdown._buildSeparator()

                if self.buildResetButton and self.buildRandomButton:
                    with gr.Row():
                        with gr.Column():
                            _buildRandomButton()
                        with gr.Column():
                            _buildResetButton()
                elif self.buildResetButton:
                    _buildResetButton()
                elif self.buildRandomButton:
                    _buildRandomButton()
        
        return components
    
    @abstractmethod
    def getDefaultName(self) -> str:
        pass
    
    @abstractmethod
    def buildContainer(self) -> typing.Any:
        pass

class B_UI_Preset():
    @staticmethod
    def _fromArgs(mappings: dict[str, list[typing.Any]], **kwargs: str):
        return B_UI_Preset(
            mappings = mappings
            , isAdditive = bool(int(kwargs.get("is_additive", 0)))
        )
    
    def __init__(self, mappings: dict[str, list[typing.Any]], isAdditive: bool = False):
        self.mappings = mappings
        self.isAdditive = isAdditive
    
    def getPresetValue(self, bComponent: B_UI_Component, componentValue):
        component = bComponent.ui
        
        hasPresetValue = component.label in self.mappings
        if not hasPresetValue:
            return bComponent.defaultValue if not self.isAdditive else componentValue
        
        presetValue: typing.Any = self.mappings[component.label]
        
        if type(component) is not gr.Dropdown or not component.multiselect:
            presetValue = bComponent.defaultValue if len(presetValue) == 0 else presetValue[0]
        
        return presetValue
    
    def validate(self, name: str, componentMap: dict[str, B_UI_Component]) -> bool:
        valid: bool = True

        for k in self.mappings:
            if k not in componentMap:
                valid = False
                printWarning("Preset", name, "Key is not valid")
            elif not componentMap[k].validateValue(self.mappings[k]):
                valid = False
                printWarning("Preset", f"{name}: {componentMap[k].name}", "Value is not valid")

        return valid

class B_Prompt(ABC):
    def __init__(self):
        pass
    
    @abstractmethod
    def getPositive(self, componentMap: dict[str, B_UI_Component]) -> str:
        pass
    
    @abstractmethod
    def getNegative(self, componentMap: dict[str, B_UI_Component]) -> str:
        pass

class B_Prompt_Simple(B_Prompt):
    @staticmethod
    def _fromArgs(preset: B_UI_Preset, **kwargs: str):
        return B_Prompt_Simple(
            preset = preset
            , promptPositive = kwargs.get("p", "")
            , promptNegative = kwargs.get("n", "")
            , prefix = kwargs.get("prefix", "")
            , postfix = kwargs.get("postfix", "")
        )
    
    @staticmethod
    def _createEmpty():
        return B_Prompt_Simple("", "")
    
    def __init__(self, promptPositive: str = "", promptNegative: str = "", prefix: str = "", postfix: str = "", preset: B_UI_Preset = None):
        super().__init__()

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
    def _fromArgs(**kwargs: str):
        return B_Prompt_Link_Slider(
            linkedKey = kwargs["link_target"]
            , promptA = kwargs["a"]
            , promptB = kwargs["b"]
        )
    
    def __init__(self, linkedKey: str, promptA: str, promptB: str):
        super().__init__()

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
    def _fromArgs(name: str, *args, **kwargs: str):
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
    
    def validate(self, componentMap: dict) -> bool:
        return True
    
    def validateValue(self, value: typing.Any) -> bool:
        return True
    
    def buildComponent(self) -> typing.Any:
        return gr.Textbox(
            label = self.name
            , value = self.value
            , scale = self.scale
            , visible = self.visible
        )
    
    def setValue(self, value):
        self.value = value.strip()
    
    def handlePrompt(self, p: StableDiffusionProcessing, componentMap: dict[str, B_UI_Component]):
        if not self.isNegativePrompt:
            p.prompt = addPrompt(p.prompt, self.value)
        else:
            p.negative_prompt = addPrompt(p.negative_prompt, self.value)
    
    def getUpdate(self, value = None) -> typing.Any:
        return gr.Textbox.update(value = self.handleUpdateValue(value))
    
    def getUpdateRandom(self, currentValue) -> typing.Any:
        return self.getUpdate(currentValue)
    
    def finalizeComponent(self, componentMap: dict):
        pass

class B_UI_Component_Dropdown(B_UI_Component):
    empty_choice: str = "-"
    advanced_defaultValue: float = 1
    advanced_step: float = 0.1
    random_maxChoices: int = 5

    @staticmethod
    def _fromArgs(name: str, *args, **kwargs: str):
        choicesMap: dict[str, B_Prompt] = args[0]

        defaultValues = kwargs.get("v", "")
        if len(defaultValues) > 0:
            defaultValues = list(map(lambda v: v.strip(), defaultValues.split(",")))
        
        return B_UI_Component_Dropdown(
            name = name
            , choicesMap = choicesMap
            , defaultValues = defaultValues
            , multiselect = bool(int(kwargs.get("multi_select", 0)))
            , allowCustomValues = bool(int(kwargs.get("allow_custom", 1)))
            , sortChoices = bool(int(kwargs.get("sort", 1)))
            , advanced = bool(int(kwargs.get("advanced", 0)))
            , hideLabel = bool(int(kwargs.get("hide_label", 0)))
            , scale = int(kwargs.get("scale", 1))
            , visible = not bool(int(kwargs.get("hide", 0)))
        )
    
    @staticmethod
    def _buildColorChoicesMap(postfixPrompt: str = "") -> dict[str, B_Prompt_Simple]:
        return {
            "Dark": B_Prompt_Simple(f"dark", postfix = postfixPrompt)
            , "Light": B_Prompt_Simple(f"light", postfix = postfixPrompt)
            , "Black": B_Prompt_Simple(f"black", postfix = postfixPrompt)
            , "Grey": B_Prompt_Simple(f"grey", postfix = postfixPrompt)
            , "White": B_Prompt_Simple(f"white", postfix = postfixPrompt)
            , "Brown": B_Prompt_Simple(f"brown", postfix = postfixPrompt)
            , "Blue": B_Prompt_Simple(f"blue", postfix = postfixPrompt)
            , "Green": B_Prompt_Simple(f"green", postfix = postfixPrompt)
            , "Red": B_Prompt_Simple(f"red", postfix = postfixPrompt)
            , "Blonde": B_Prompt_Simple(f"blonde", postfix = postfixPrompt)
            , "Rainbow": B_Prompt_Simple(f"rainbow", postfix = postfixPrompt)
            , "Pink": B_Prompt_Simple(f"pink", postfix = postfixPrompt)
            , "Purple": B_Prompt_Simple(f"purple", postfix = postfixPrompt)
            , "Orange": B_Prompt_Simple(f"orange", postfix = postfixPrompt)
            , "Yellow": B_Prompt_Simple(f"yellow", postfix = postfixPrompt)
            , "Multicolored": B_Prompt_Simple(f"multicolored", postfix = postfixPrompt)
        }
    
    def __init__(
        self
        , name: str = None
        , choicesMap: dict[str, B_Prompt] = {}
        , defaultValues: str | list[str] = None
        , multiselect: bool = False
        , allowCustomValues: bool = True
        , sortChoices: bool = True
        , advanced: bool = False
        , hideLabel: bool = False
        , scale: int = None
        , visible: bool = True
    ):
        choicesMapFinal = self.buildChoicesMap(choicesMap, not multiselect)
        defaultValueFinal = self.buildDefaultValue(choicesMap, defaultValues, multiselect, list(choicesMapFinal)[0])
        
        super().__init__(name, defaultValueFinal, visible)
        
        self.choicesMap = choicesMapFinal
        self.multiselect = multiselect
        self.allowCustomValues = allowCustomValues if not multiselect else False
        self.sortChoices = sortChoices
        self.advanced = advanced
        self.hideLabel = hideLabel
        self.scale = scale
    
    def buildChoicesMap(self, choicesMap: dict[str, B_Prompt], insertEmptyChoice: bool) -> dict[str, B_Prompt]:
        choicesMapFinal = choicesMap
        
        if insertEmptyChoice:
            choicesMapFinal = {
                self.empty_choice: B_Prompt_Simple._createEmpty()
            }
            
            choicesMapFinal.update(choicesMap)
        
        return choicesMapFinal
    
    def buildDefaultValue(self, choicesMap: dict[str, B_Prompt], defaultValues: str | list[str], multiselect: bool, fallbackValue: str) -> str | list[str]:
        defaultValue: str | list[str] = None if not multiselect else []
        
        if (type(defaultValues) is str or defaultValues is None):
            defaultValues = [defaultValues]
        
        if len(choicesMap) > 0:
            choices = list(choicesMap)
            if multiselect:
                if defaultValues[0] is not None:
                    defaultValue = []
                    for v in defaultValues:
                        if v in choices:
                            defaultValue.append(v)
            else:
                defaultValueFirst = defaultValues[0]
                if defaultValueFirst is not None and defaultValueFirst in choices:
                    defaultValue = defaultValueFirst
                else:
                    defaultValue = fallbackValue
        
        return defaultValue
    
    def getChoices(self) -> list[str]:
        choices = list(self.choicesMap)
        
        if not self.sortChoices:
            return choices
        
        choicesSorted: list[str] = []
        
        if choices[0] == self.empty_choice:
            choicesSorted.append(choices.pop(0))
        
        choices.sort()
        choicesSorted += choices
        
        return choicesSorted
    
    def getBPromptsFromValue(self) -> list[tuple[str, B_Prompt]]:
        bPrompts: list[tuple[str, B_Prompt]] = []
        
        if type(self.value) is str:
            bPrompts.append((self.value, self.choicesMap.get(self.value, None if not self.allowCustomValues else B_Prompt_Simple(self.value))))
        elif type(self.value) is list and len(self.value) > 0:
            for k in list(self.choicesMap):
                if k not in self.value:
                    continue
                
                bPrompts.append((k, self.choicesMap[k]))
        
        return bPrompts
    
    def setAdvancedValue(self, choice: str, value: float):
        self.advanced_values[choice] = value
    
    def getDefaultName(self) -> str:
        return "Dropdown"
    
    def validate(self, componentMap: dict) -> bool:
        valid: bool = True

        for choiceKey in self.choicesMap:
            if choiceKey == self.empty_choice:
                continue
            
            choiceValue = self.choicesMap[choiceKey]
            if issubclass(type(choiceValue), B_Prompt_Simple):
                choiceValue_simple: B_Prompt_Simple = choiceValue
                if choiceValue_simple.preset is not None and choiceValue_simple.preset.validate(f"{self.name}: {choiceKey}", componentMap):
                    valid = False
        
        return valid
    
    def validateValue(self, value: typing.Any) -> bool:
        valid: bool = True

        if type(value) is not list:
            value: list = [value]
        
        for v in value:
            if v not in self.choicesMap:
                valid = False
                printWarning("Dropdown", self.name, f"Invalid choice: '{v}'")

        return valid
    
    def buildComponent(self) -> list[typing.Any]:
        def _build(useScale: bool = True):
            return gr.Dropdown(
                label = self.name
                , choices = self.getChoices()
                , multiselect = self.multiselect
                , value = self.value
                , allow_custom_value = self.allowCustomValues
                , show_label = not self.hideLabel
                , scale = self.scale if useScale else None
                , visible = self.visible
            )

        if not self.advanced:
            component = _build()
        else:
            self.advanced_options: dict[str, tuple[typing.Any, typing.Any]] = {}
            self.advanced_values: dict[str, float] = {}
            
            with gr.Column(
                scale = self.scale
                , min_width = 160
            ):
                component = _build(False)
                row = gr.Row(
                    variant = "panel"
                    , visible = self.value is not None and len(self.value) > 0 and self.value != self.empty_choice
                )
                with row:
                    for k in self.choicesMap:
                        if k == self.empty_choice:
                            continue

                        column = gr.Column(
                            variant = "panel"
                            , visible = self.value is not None and (self.value == k or k in self.value)
                            , min_width = 160
                        )
                        with column:
                            markdown = gr.Markdown(
                                value = f"{k}"
                                , visible = False
                            )

                            number = gr.Number(
                                label = f"{k} (S)"
                                , value = self.advanced_defaultValue
                                , step = self.advanced_step
                                , minimum = 0
                            )
                            number.input(
                                fn = self.setAdvancedValue
                                , inputs = [markdown, number]
                            )
                        
                        self.advanced_options[k] = number, column
                        self.advanced_values[k] = self.advanced_defaultValue
            
            self.advanced_container = row
            
            def _update(choice: str | list[str]):
                outputMap: dict[str, bool] = {}

                for k in self.choicesMap:
                    if k == self.empty_choice:
                        continue
                    
                    outputMap[k] = False
                
                choices: list[str] = []
                if type(choice) is str:
                    choices.append(choice)
                else:
                    choices += choice
                
                for k in choices:
                    if k == self.empty_choice:
                        continue
                    
                    outputMap[k] = True
                
                output: list = [self.advanced_container.update(visible = any(list(outputMap.values())))]
                for k in outputMap:
                    self.advanced_values[k] = self.advanced_defaultValue
                    output.append(self.advanced_options[k][0].update(value = self.advanced_values[k], step = self.advanced_step))
                    output.append(self.advanced_options[k][1].update(visible = outputMap[k]))
                
                return output

            component.change(
                fn = _update
                , inputs = component
                , outputs = [self.advanced_container] + sum(list(map(lambda t: list(t), self.advanced_options.values())), [])
            )
        
        return component
    
    def setValue(self, value):
        self.value = value
    
    def handlePrompt(self, p: StableDiffusionProcessing, componentMap: dict[str, B_UI_Component]):
        bPrompts = self.getBPromptsFromValue()
        if len(bPrompts) > 0:
            for choice, bPrompt in bPrompts:
                positive = bPrompt.getPositive(componentMap)
                negative = bPrompt.getNegative(componentMap)
                
                if self.advanced and choice != self.empty_choice:
                    value = self.advanced_values[choice]
                    
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
    
    def getUpdate(self, value = None) -> typing.Any:
        return gr.Dropdown.update(value = self.handleUpdateValue(value))
    
    def getUpdateRandom(self, currentValue) -> typing.Any:
        value: str | list[str]

        choiceKeys = list(self.choicesMap.keys())
        
        if len(choiceKeys) == 0:
            value = currentValue
        else:
            if self.multiselect:
                value = []

                cMax = len(choiceKeys)
                if self.random_maxChoices > 0 and self.random_maxChoices < cMax:
                    cMax = self.random_maxChoices
                
                r = random.randint(0, cMax)
                if r > 0:
                    for c in range(r):
                        i = random.randint(0, len(choiceKeys) - 1)
                        value.append(choiceKeys.pop(i))
            else:
                r = random.randint(0, len(choiceKeys) - 1)
                value = choiceKeys[r]
        
        return self.getUpdate(value)
    
    def finalizeComponent(self, componentMap: dict):
        bComponents: list[B_UI_Component] = list(componentMap.values())
        bComponents.remove(self)
        
        components: list[typing.Any] = list(map(lambda bComponent: bComponent.ui, bComponents))

        anyChoiceMapHasPreset = any(
            map(
                lambda bPrompt: False if type(bPrompt) is not B_Prompt_Simple else bPrompt.preset is not None and len(bPrompt.preset.mappings) > 0
                , self.choicesMap.values()
            )
        )
        if anyChoiceMapHasPreset:
            def getPresetValues(choices: str | list[str], *args):
                if type(choices) is str:
                    choices = [choices]
                
                updatedValues: list[typing.Any] = list(args)
                
                if len(choices) > 0:
                    for choice in choices:
                        bPrompt = self.choicesMap[choice]
                        
                        if not issubclass(type(bPrompt), B_Prompt_Simple) or bPrompt.preset == None:
                            continue
                        
                        bPrompt: B_Prompt_Simple = bPrompt
                        
                        i = 0
                        for bComponent in bComponents:
                            updatedValues[i] = bPrompt.preset.getPresetValue(bComponent, updatedValues[i])
                            i += 1
                
                return updatedValues
            
            self.ui.select(
                fn = getPresetValues
                , inputs = [self.ui] + components
                , outputs = components
            )

class B_UI_Component_Slider(B_UI_Component):
    @staticmethod
    def _fromArgs(name: str, *args, **kwargs: str):
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
    
    def getMinimum(self) -> float:
        return -1 if not self.isRequired else 0
    
    def getMaximum(self) -> float:
        return 100
    
    def getStep(self) -> float:
        return 1
    
    def buildButton(self, component: typing.Any, text: str, value: float) -> typing.Any:
        btn = gr.Button(value = text)
        btn.click(
            fn = lambda component: self.getUpdate(value)
            , inputs = component
            , outputs = component
        )
        return btn
    
    def getDefaultName(self) -> str:
        return "Slider"
    
    def validate(self) -> bool:
        if self.validateValue(self.defaultValue):
            return True
        
        return False
    
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
    
    def buildComponent(self) -> list[typing.Any]:
        slider = gr.Slider(
            label = self.name
            , minimum = self.getMinimum()
            , maximum = self.getMaximum()
            , value = self.value
            , step = self.getStep()
            , visible = self.visible
        )
        
        if self.buildButtons:
            with gr.Row():
                self.buildButton(slider, self.promptAButton, 0)
                self.buildButton(slider, self.promptBButton, self.getMaximum())
        
        return slider
    
    def setValue(self, value: typing.Any):
        value = float(value)
        self.value = round(value / self.getMaximum(), 2) if value > -1 else value
    
    def handlePrompt(self, p: StableDiffusionProcessing, componentMap: dict[str, B_UI_Component]):
        promptToAdd = buildRangePrompt(self.promptA, self.promptB, self.value)
        
        if not self.isNegativePrompt:
            p.prompt = addPrompt(p.prompt, promptToAdd)
        else:
            p.negative_prompt = addPrompt(p.prompt, promptToAdd)
    
    def getUpdate(self, value = None) -> typing.Any:
        return gr.Slider.update(value = self.handleUpdateValue(value))
    
    def getUpdateRandom(self, currentValue) -> typing.Any:
        value = float(random.randint(self.getMinimum(), self.getMaximum()))
        return self.getUpdate(value)
    
    def finalizeComponent(self, componentMap: dict):
        pass

class B_UI_Container_Tab(B_UI_Container):
    @staticmethod
    def _fromArgs(name: str, items: list[B_UI], **kwargs: str):
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
    
    def buildContainer(self) -> typing.Any:
        return gr.Tab(self.name)

class B_UI_Container_Row(B_UI_Container):
    @staticmethod
    def _fromArgs(name: str, items: list[B_UI], **kwargs: str):
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
    
    def buildContainer(self) -> typing.Any:
        return gr.Row(visible = self.visible)

class B_UI_Container_Column(B_UI_Container):
    @staticmethod
    def _fromArgs(name: str, items: list[B_UI], **kwargs: str):
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
    
    def buildContainer(self) -> typing.Any:
        return gr.Column(scale = self.scale, visible = self.visible)

class B_UI_Container_Group(B_UI_Container):
    @staticmethod
    def _fromArgs(name: str, items: list[B_UI], **kwargs: str):
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
    
    def buildContainer(self) -> typing.Any:
        return gr.Group(visible = self.visible)

class B_UI_Container_Accordion(B_UI_Container):
    @staticmethod
    def _fromArgs(name: str, items: list[B_UI], **kwargs: str):
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
    
    def buildContainer(self) -> typing.Any:
        return gr.Accordion(label = self.name, visible = self.visible)

class B_UI_Builder(ABC):
    def __init__(self, name: str, **kwargs: str):
        self.name = name
        self.args = kwargs
    
    @abstractmethod
    def build(self) -> B_UI | B_UI_Preset:
        pass

class B_UI_Builder_WithChildren(B_UI_Builder):
    def __init__(self, name: str, **kwargs: str):
        super().__init__(name, **kwargs)
        
        self.builtChildren: list[B_UI] = []
    
    @abstractmethod
    def build(self) -> B_UI:
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
    
    def build(self) -> B_UI_Preset:
        return None if len(self.mappings) == 0 else B_UI_Preset._fromArgs(self.mappings, **self.args)

class B_UI_Container_Builder(B_UI_Builder_WithChildren):
    def __init__(self, t: type[B_UI_Container], name: str, parent: B_UI_Builder_WithChildren, **kwargs: str):
        super().__init__(name, **kwargs)
        
        self.t = t
        self.parent = parent
    
    def finalizeBuilt(self, built: B_UI):
        if self.parent is not None:
            self.parent.builtChildren.append(built)
        
        return built
    
    def build(self) -> B_UI:
        return self.finalizeBuilt(self.t._fromArgs(self.name, self.builtChildren, **self.args))

class B_UI_Markdown_Builder(B_UI_Builder):
    def __init__(self, isSeparator: bool, parent: B_UI_Container_Builder, **kwargs: str):
        super().__init__(None, **kwargs)

        self.isSeparator = isSeparator
        self.parent = parent
    
    def build(self) -> B_UI:
        builtSelf = B_UI_Markdown._fromArgs(self.isSeparator, **self.args)

        if self.parent is not None:
            self.parent.builtChildren.append(builtSelf)
        
        return builtSelf

class B_UI_Component_Builder(B_UI_Builder):
    def __init__(self, t: type[B_UI_Component], name: str, parent: B_UI_Container_Builder, **kwargs: str):
        super().__init__(name, **kwargs)
        
        self.t = t
        self.parent = parent
    
    def build(self) -> B_UI:
        builtSelf = self.t._fromArgs(self.name, **self.args)
        
        if self.parent is not None:
            self.parent.builtChildren.append(builtSelf)
        
        return builtSelf

class B_UI_Component_Dropdown_Builder(B_UI_Builder):
    def __init__(self, name: str, parent: B_UI_Container_Builder, **kwargs: str):
        super().__init__(name, **kwargs)
        
        self.parent = parent
        
        self.choicesMap: dict[str, B_Prompt] = {}
    
    def addChoice(self, text: str, preset_builder: B_UI_Preset_Builder, **kwargs: str):
        bPrompt: B_Prompt = None
        
        link_type = kwargs.get("link_type", "")
        match link_type:
            case "SLIDER":
                bPrompt = B_Prompt_Link_Slider._fromArgs(**kwargs)
            case _:
                preset = preset_builder.build()
                bPrompt = B_Prompt_Simple._fromArgs(preset, **kwargs)
        
        self.choicesMap[text] = bPrompt
    
    def addChoices(self, **kwargs: str):
        choicesMap: dict[str, B_Prompt] = {}
        
        special_type = kwargs["type"]
        match special_type:
            case "COLOR":
                choicesMap = B_UI_Component_Dropdown._buildColorChoicesMap(kwargs["postfix"])
            case _:
                print(f"Invalid CHOICES type: {special_type}")
        
        self.choicesMap = choicesMap
    
    def build(self) -> B_UI:
        builtSelf = B_UI_Component_Dropdown._fromArgs(self.name, self.choicesMap, **self.args)
        
        if self.parent is not None:
            self.parent.builtChildren.append(builtSelf)
        
        return builtSelf

class B_UI_Map():
    def __init__(self, path_base: str, file_name_layout: str, file_name_presets: str, tagged_show: bool = True, validate: bool = True):
        self.layout = self.parseLayout(os.path.join(path_base, file_name_layout), tagged_show)
        self.presets = self.parsePresets(os.path.join(path_base, file_name_presets))
        
        self.componentMap = self.buildComponentMap()

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
        dropdown_current_choice: tuple[str, B_UI_Preset_Builder, dict[str, str]] = None
        
        skip = 0
        
        def _build(builder: B_UI_Markdown_Builder | B_UI_Container_Builder | B_UI_Component_Builder | B_UI_Component_Dropdown_Builder):
            built = builder.build()
            if builder.parent is None:
                layout.append(built)
            return built
        
        def _buildDropdownChoice(dropdown_choice: tuple[str, B_UI_Preset_Builder, dict[str, str]]) -> bool:
            if dropdown_choice is not None:
                l_choice_name, l_choice_preset_builder, l_choice_args = dropdown_choice

                l_choice_args["prefix"] = builder_current_dropdown.args.get("prefix", "")
                l_choice_args["postfix"] = builder_current_dropdown.args.get("postfix", "")

                builder_current_dropdown.addChoice(l_choice_name, l_choice_preset_builder, **l_choice_args)
                return len(l_choice_preset_builder.mappings) > 0
            
            return False
        
        with open(file_path_layout) as file_layout:
            for l in file_layout:
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
                        
                        dropdown_current_choice = (l_name, B_UI_Preset_Builder(l_name, **{ "is_additive": "1" }), l_args)
                    
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
            for l in file_presets:
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
    
    def buildComponentMap(self) -> dict[str, B_UI_Component]:
        componentMap: dict[str, B_UI_Component] = {}

        def mapBComponent(bComponent: B_UI_Component):
            if bComponent.name in componentMap:
                print(f"WARNING: Duplicate attribute -> {bComponent.name}")
            
            componentMap[bComponent.name] = bComponent
        
        for x in self.layout:
            x_type = type(x)
            
            if issubclass(x_type, B_UI_Component):
                mapBComponent(x)
                continue

            if issubclass(x_type, B_UI_Container):
                x_bContainer: B_UI_Container = x
                if len(x_bContainer.bComponents) == 0:
                    continue

                for bComponent in x_bContainer.bComponents:
                    mapBComponent(bComponent)
        
        return componentMap
    
    def validate(self):
        for bUi in self.layout:
            bUi.validate(self.componentMap)
        
        for presetKey in self.presets:
            self.presets[presetKey].validate(presetKey, self.componentMap)
    
    def buildUI(self) -> list[typing.Any]:
        B_UI_Markdown._buildSeparator()
        
        components: list[typing.Any] = []
        
        for item in self.layout:
            item_ui = item.buildUI()
            
            if len(item_ui) > 0:
                components += item_ui
        
        self.buildPresetsUI()

        for bComponent in self.componentMap.values():
            bComponent.finalizeComponent(self.componentMap)
        
        return components
    
    def buildPresetsUI(self):
        if len(self.presets) == 0:
            return
        
        bComponents = self.componentMap.values()
        components: list[typing.Any] = list(map(lambda bComponent: bComponent.ui, bComponents))
        
        presetKeys = self.presets.keys()
        
        def applyPreset(presetKey: str, *args):
            preset = self.presets[presetKey]
            return list(map(preset.getPresetValue, bComponents, args))
        
        B_UI_Markdown._buildSeparator()
        with gr.Accordion("Presets", open = False):
            i = 0
            for presetKey in presetKeys:
                button_preset = gr.Button(presetKey)
                button_preset.click(
                    fn = applyPreset
                    , inputs = [button_preset] + components
                    , outputs = components
                )
                
                i += 1
                if i < len(presetKeys):
                    B_UI_Markdown._buildSeparator()

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
        return b_layout.buildUI()

    def run(self, p, *args):
        i = 0
        for k in list(b_layout.componentMap):
            component = b_layout.componentMap[k]
            component.setValue(args[i])
            component.handlePrompt(p, b_layout.componentMap)
            i = i+1
        
        proc = process_images(p)
        
        return proc
